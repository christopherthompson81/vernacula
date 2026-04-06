using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// Chunked streaming denoiser using DeepFilterNet3 ONNX models.
///
/// Pipeline (mirrors libdf Rust implementation):
///   1. Frame-by-frame STFT (vorbis window, fft=960, hop=480 @ 48 kHz)
///   2. ERB features: mean-norm'd log power in 32 ERB bands
///   3. Spec features: unit-norm'd complex first 96 bins
///   4. ONNX inference (chunked, ChunkFrames per call):
///      enc → erb_dec + df_dec, GRU state carried across chunks
///   5. Apply ERB mask to spectrum
///   6. Apply deep-filter coefficients to first 96 bins (order=5, lookahead=2)
///   7. ISTFT with overlap-add
///
/// Streaming models expose GRU hidden state as explicit inputs/outputs,
/// allowing chunked inference (ChunkFrames ≈ 1 s) for better GPU utilisation.
/// </summary>
public sealed class DeepFilterNet3Denoiser : IDisposable
{
    // ── DSP constants (from config.ini) ──────────────────────────────────────
    private const int   Sr          = 48_000;
    private const int   FftSize     = 960;
    private const int   HopSize     = 480;   // 50% overlap
    private const int   FreqBins    = FftSize / 2 + 1;  // 481
    private const int   NbErb       = 32;
    private const int   NbDf        = 96;
    private const int   DfOrder     = 5;
    private const int   DfLookahead = 2;
    private const float NormAlpha   = 0.99f;
    private const float WNorm       = 1f / FftSize;
    private const float LogEps      = 1e-10f;

    // ── Chunked streaming inference ───────────────────────────────────────────
    // GRU hidden state shapes (from export_df3_streaming.py):
    //   h_enc: [1, 1, 256]  — enc emb_gru,   1 layer
    //   h_erb: [2, 1, 256]  — erb_dec emb_gru, 2 layers
    //   h_df:  [2, 1, 256]  — df_dec df_gru,  2 layers
    private const int ChunkFrames  = 500;    // ~5 s per chunk @ 48 kHz / 480 hop
    private const int HSize        = 256;
    private const int HEncElems    = 1 * 1 * HSize;   // 256 floats
    private const int HErbElems    = 2 * 1 * HSize;   // 512 floats
    private const int HDfElems     = 2 * 1 * HSize;   // 512 floats

    // ERB band widths computed by libdf's erb_fb(48000, 960, 32, 2).
    // Verified via df_state.erb_widths() → sum = 481 = FreqBins.
    private static readonly int[] ErbWidths =
    [
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20,
        24, 28, 31, 37, 42, 50, 56, 67,
    ];

    // Normalization initial states (mirror libdf MEAN_NORM_INIT / UNIT_NORM_INIT)
    private static readonly float[] MeanNormInit = Linspace(-60f, -90f, NbErb);
    private static readonly float[] UnitNormInit = Linspace(0.001f, 0.0001f, NbDf);

    private readonly InferenceSession _enc;
    private readonly InferenceSession _erbDec;
    private readonly InferenceSession _dfDec;

    // Vorbis window never changes — compute once for the process lifetime.
    private static readonly float[] _window = MakeVorbisWindow(FftSize);

    // ── Session cache ────────────────────────────────────────────────────────
    // InferenceSession loading is expensive (~1–5 s per model).  Cache the three
    // sessions across Denoise() calls; reload only if the models directory or EP changes.

    private static string?           _cachedModelsDir;
    private static ExecutionProvider _cachedEp = ExecutionProvider.Auto;
    private static InferenceSession? _cachedEnc;
    private static InferenceSession? _cachedErbDec;
    private static InferenceSession? _cachedDfDec;
    private static readonly object   _sessionLock = new();

    private static SessionOptions MakeSessionOptions(ExecutionProvider ep)
    {
        var opts = new SessionOptions();
        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                    try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                try { opts.AppendExecutionProvider_DML(0); } catch { }
                break;
            case ExecutionProvider.Cuda:
                try { opts.AppendExecutionProvider_CUDA(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("CUDA EP not available in current OnnxRuntime build."); }
                break;
            case ExecutionProvider.DirectML:
                try { opts.AppendExecutionProvider_DML(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("DirectML EP not available. Build with -p:UseDirectML=true."); }
                break;
            case ExecutionProvider.Cpu:
                break;
        }
        return opts;
    }

    private static (InferenceSession enc, InferenceSession erbDec, InferenceSession dfDec)
        GetOrLoadSessions(string modelsDir, ExecutionProvider ep)
    {
        lock (_sessionLock)
        {
            if (_cachedModelsDir != modelsDir || _cachedEp != ep || _cachedEnc is null)
            {
                _cachedEnc?.Dispose();
                _cachedErbDec?.Dispose();
                _cachedDfDec?.Dispose();

                var opts = MakeSessionOptions(ep);
                _cachedEnc    = new InferenceSession(Path.Combine(modelsDir, Config.Dfn3EncFile),    opts);
                _cachedErbDec = new InferenceSession(Path.Combine(modelsDir, Config.Dfn3ErbDecFile), opts);
                _cachedDfDec  = new InferenceSession(Path.Combine(modelsDir, Config.Dfn3DfDecFile),  opts);
                _cachedModelsDir = modelsDir;
                _cachedEp        = ep;
            }
            return (_cachedEnc!, _cachedErbDec!, _cachedDfDec!);
        }
    }

    // ── Constructor / Dispose ────────────────────────────────────────────────

    public DeepFilterNet3Denoiser(string modelsDir, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        (_enc, _erbDec, _dfDec) = GetOrLoadSessions(modelsDir, ep);
    }

    public void Dispose()
    {
        // Sessions are held in the static cache; do not dispose here.
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Denoise mono audio at 48 kHz.  Returns enhanced audio at 48 kHz.
    /// Input length must be a multiple of HopSize (480 samples).
    /// <paramref name="progress"/> receives (current, total) where total = nChunks + 2.
    /// </summary>
    public float[] Denoise(float[] audio48k, IProgress<(int current, int total)>? progress = null)
    {
        int nFrames = audio48k.Length / HopSize;
        if (nFrames == 0) return audio48k;

        // ── Analysis STFT ────────────────────────────────────────────────────
        var (specRe, specIm) = AnalysisStft(audio48k, nFrames);

        // ── Feature extraction ────────────────────────────────────────────────
        float[] featErbFlat    = new float[nFrames * NbErb];
        float[] featSpecReFlat = new float[nFrames * NbDf];
        float[] featSpecImFlat = new float[nFrames * NbDf];

        float[] meanNormState = (float[])MeanNormInit.Clone();
        float[] unitNormState = (float[])UnitNormInit.Clone();

        for (int t = 0; t < nFrames; t++)
        {
            ComputeErbFeatures(specRe, specIm, t, nFrames, meanNormState, featErbFlat);
            ComputeSpecFeatures(specRe, specIm, t, nFrames, unitNormState,
                                featSpecReFlat, featSpecImFlat);
        }

        // ── Chunked ONNX inference ────────────────────────────────────────────
        // Process in chunks of ChunkFrames, carrying GRU state across chunks.
        // Assemble full mask and coef arrays; masks are applied to the full spectrum.
        int nChunks   = (nFrames + ChunkFrames - 1) / ChunkFrames;
        int totalSteps = nChunks + 2;    // STFT(1) + chunks(nChunks) + synthesis(1)
        progress?.Report((1, totalSteps));

        float[] mAll     = new float[nFrames * NbErb];
        float[] coefsAll = new float[nFrames * NbDf * DfOrder * 2];

        // Zero-initialised GRU hidden states
        float[] hEnc = new float[HEncElems];
        float[] hErb = new float[HErbElems];
        float[] hDf  = new float[HDfElems];

        for (int chunk = 0; chunk < nChunks; chunk++)
        {
            int frameStart = chunk * ChunkFrames;
            int chunkT     = Math.Min(ChunkFrames, nFrames - frameStart);

            // Slice feature arrays for this chunk
            float[] cFeatErb   = new float[chunkT * NbErb];
            float[] cFeatSpecRe = new float[chunkT * NbDf];
            float[] cFeatSpecIm = new float[chunkT * NbDf];
            Array.Copy(featErbFlat,    frameStart * NbErb, cFeatErb,    0, chunkT * NbErb);
            Array.Copy(featSpecReFlat, frameStart * NbDf,  cFeatSpecRe, 0, chunkT * NbDf);
            Array.Copy(featSpecImFlat, frameStart * NbDf,  cFeatSpecIm, 0, chunkT * NbDf);

            // Encoder
            var (e0, e1, e2, e3, emb, c0, hEncOut) =
                RunEnc(cFeatErb, cFeatSpecRe, cFeatSpecIm, chunkT, hEnc);
            hEnc = hEncOut;

            // ERB decoder
            var (m, hErbOut) = RunErbDec(emb, e3, e2, e1, e0, chunkT, hErb);
            hErb = hErbOut;

            // DF decoder
            var (coefs, hDfOut) = RunDfDec(emb, c0, chunkT, hDf);
            hDf = hDfOut;

            // Accumulate results
            Array.Copy(m,     0, mAll,     frameStart * NbErb,           chunkT * NbErb);
            Array.Copy(coefs, 0, coefsAll, frameStart * NbDf * DfOrder * 2,
                                           chunkT * NbDf * DfOrder * 2);

            progress?.Report((chunk + 2, totalSteps));
        }

        // ── Apply masks (full spectrum) ───────────────────────────────────────
        ApplyErbMask(specRe, specIm, mAll, nFrames);
        ApplyDeepFilter(specRe, specIm, coefsAll, nFrames);

        // ── Synthesis ISTFT ───────────────────────────────────────────────────
        var result = SynthesisIstft(specRe, specIm, nFrames);
        progress?.Report((totalSteps, totalSteps));
        return result;
    }

    // ── STFT analysis ────────────────────────────────────────────────────────

    /// <summary>
    /// Frame-by-frame STFT exactly mirroring libdf's frame_analysis.
    /// Each frame assembles buf = [analysis_mem × w[0..hop] | input × w[hop..fft]]
    /// then FFT and multiplies by wnorm.
    /// Returns (specRe, specIm), each float[nFrames * FreqBins] in row-major order.
    /// </summary>
    private (float[] specRe, float[] specIm) AnalysisStft(float[] audio, int nFrames)
    {
        float[] specRe  = new float[nFrames * FreqBins];
        float[] specIm  = new float[nFrames * FreqBins];
        float[] analysisMem = new float[HopSize]; // previous frame's samples
        var     buf         = new Complex32[FftSize];

        for (int t = 0; t < nFrames; t++)
        {
            int audioOffset = t * HopSize;

            // buf[0..HopSize] = analysisMem × w[0..HopSize]
            for (int j = 0; j < HopSize; j++)
                buf[j] = new Complex32(analysisMem[j] * _window[j], 0f);

            // buf[HopSize..FftSize] = audio[offset..offset+HopSize] × w[HopSize..FftSize]
            for (int j = 0; j < HopSize; j++)
                buf[HopSize + j] = new Complex32(audio[audioOffset + j] * _window[HopSize + j], 0f);

            // Update analysis memory with current frame
            Array.Copy(audio, audioOffset, analysisMem, 0, HopSize);

            // Forward FFT (unnormalized)
            Fourier.Forward(buf, FourierOptions.NoScaling);

            // Copy first FreqBins bins, apply wnorm
            int baseIdx = t * FreqBins;
            for (int k = 0; k < FreqBins; k++)
            {
                specRe[baseIdx + k] = buf[k].Real      * WNorm;
                specIm[baseIdx + k] = buf[k].Imaginary * WNorm;
            }
        }

        return (specRe, specIm);
    }

    // ── Feature extraction ────────────────────────────────────────────────────

    /// <summary>
    /// ERB power features for one frame, updating the running mean-norm state.
    /// Mirrors libdf feat_erb: compute_band_corr → 10*log10 → band_mean_norm_erb.
    /// Output written to featErbFlat[t*NbErb .. (t+1)*NbErb].
    /// </summary>
    private static void ComputeErbFeatures(
        float[] specRe, float[] specIm, int t, int nFrames,
        float[] state, float[] featErbFlat)
    {
        int baseSpec = t * FreqBins;
        int baseOut  = t * NbErb;
        int binOff   = 0;

        for (int b = 0; b < NbErb; b++)
        {
            int width = ErbWidths[b];
            float power = 0f;
            for (int j = 0; j < width; j++)
            {
                int idx = baseSpec + binOff + j;
                power += specRe[idx] * specRe[idx] + specIm[idx] * specIm[idx];
            }
            // Average power then convert to dB (matches compute_band_corr + 10*log10)
            float dB = MathF.Log10(power / width + LogEps) * 10f;

            // band_mean_norm_erb: state update then subtract and scale by 1/40
            state[b] = dB * (1f - NormAlpha) + state[b] * NormAlpha;
            featErbFlat[baseOut + b] = (dB - state[b]) / 40f;

            binOff += width;
        }
    }

    /// <summary>
    /// Unit-norm complex spec features for first NbDf=96 bins of one frame.
    /// Mirrors libdf band_unit_norm: state = |x|*(1-α) + state*α; output = x/sqrt(state).
    /// Outputs written to featSpecReFlat/ImFlat[t*NbDf .. (t+1)*NbDf].
    /// </summary>
    private static void ComputeSpecFeatures(
        float[] specRe, float[] specIm, int t, int nFrames,
        float[] state, float[] featReFlat, float[] featImFlat)
    {
        int baseSpec = t * FreqBins;
        int baseOut  = t * NbDf;

        for (int j = 0; j < NbDf; j++)
        {
            float re  = specRe[baseSpec + j];
            float im  = specIm[baseSpec + j];
            float mag = MathF.Sqrt(re * re + im * im);
            state[j] = mag * (1f - NormAlpha) + state[j] * NormAlpha;
            float invSqrtState = 1f / MathF.Sqrt(state[j]);
            featReFlat[baseOut + j] = re * invSqrtState;
            featImFlat[baseOut + j] = im * invSqrtState;
        }
    }

    // ── ONNX inference ────────────────────────────────────────────────────────

    // ── Streaming ONNX inference (explicit GRU state I/O) ────────────────────

    private (float[] e0, float[] e1, float[] e2, float[] e3, float[] emb, float[] c0,
             float[] hEncOut)
        RunEnc(float[] featErbFlat, float[] featSpecRe, float[] featSpecIm, int T,
               float[] hEnc)
    {
        // feat_erb: (1, 1, T, 32)
        var erbTensor  = new DenseTensor<float>(featErbFlat, [1, 1, T, NbErb]);

        // feat_spec: (1, 2, T, 96) — re channel then im channel
        float[] featSpecFlat = new float[2 * T * NbDf];
        Array.Copy(featSpecRe, 0, featSpecFlat, 0,        T * NbDf);
        Array.Copy(featSpecIm, 0, featSpecFlat, T * NbDf, T * NbDf);
        var specTensor = new DenseTensor<float>(featSpecFlat, [1, 2, T, NbDf]);

        // h_enc: (1, 1, 256)
        var hEncTensor = new DenseTensor<float>(hEnc, [1, 1, HSize]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("feat_erb",  erbTensor),
            NamedOnnxValue.CreateFromTensor("feat_spec", specTensor),
            NamedOnnxValue.CreateFromTensor("h_enc",     hEncTensor),
        };

        using var outputs = _enc.Run(inputs);
        return (
            e0:      outputs.First(o => o.Name == "e0").AsEnumerable<float>().ToArray(),
            e1:      outputs.First(o => o.Name == "e1").AsEnumerable<float>().ToArray(),
            e2:      outputs.First(o => o.Name == "e2").AsEnumerable<float>().ToArray(),
            e3:      outputs.First(o => o.Name == "e3").AsEnumerable<float>().ToArray(),
            emb:     outputs.First(o => o.Name == "emb").AsEnumerable<float>().ToArray(),
            c0:      outputs.First(o => o.Name == "c0").AsEnumerable<float>().ToArray(),
            hEncOut: outputs.First(o => o.Name == "h_enc_out").AsEnumerable<float>().ToArray()
        );
    }

    private (float[] m, float[] hErbOut)
        RunErbDec(float[] emb, float[] e3, float[] e2, float[] e1, float[] e0, int T,
                  float[] hErb)
    {
        // emb: (1,T,512) — note: emb dim=512 from enc output (emb_out_dim)
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("emb",   new DenseTensor<float>(emb,  [1, T, 512])),
            NamedOnnxValue.CreateFromTensor("e3",    new DenseTensor<float>(e3,   [1, 64, T, 8])),
            NamedOnnxValue.CreateFromTensor("e2",    new DenseTensor<float>(e2,   [1, 64, T, 8])),
            NamedOnnxValue.CreateFromTensor("e1",    new DenseTensor<float>(e1,   [1, 64, T, 16])),
            NamedOnnxValue.CreateFromTensor("e0",    new DenseTensor<float>(e0,   [1, 64, T, 32])),
            NamedOnnxValue.CreateFromTensor("h_erb", new DenseTensor<float>(hErb, [2, 1, HSize])),
        };

        using var outputs = _erbDec.Run(inputs);
        return (
            m:       outputs.First(o => o.Name == "m").AsEnumerable<float>().ToArray(),
            hErbOut: outputs.First(o => o.Name == "h_erb_out").AsEnumerable<float>().ToArray()
        );
    }

    private (float[] coefs, float[] hDfOut)
        RunDfDec(float[] emb, float[] c0, int T, float[] hDf)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("emb",  new DenseTensor<float>(emb, [1, T, 512])),
            NamedOnnxValue.CreateFromTensor("c0",   new DenseTensor<float>(c0,  [1, 64, T, NbDf])),
            NamedOnnxValue.CreateFromTensor("h_df", new DenseTensor<float>(hDf, [2, 1, HSize])),
        };

        using var outputs = _dfDec.Run(inputs);
        return (
            coefs:   outputs.First(o => o.Name == "coefs").AsEnumerable<float>().ToArray(),
            hDfOut:  outputs.First(o => o.Name == "h_df_out").AsEnumerable<float>().ToArray()
        );
    }

    // ── Mask and filter application ───────────────────────────────────────────

    /// <summary>
    /// Apply ERB mask to the spectrum in-place.
    /// Mirrors libdf apply_interp_band_gain: multiply all bins in each ERB band
    /// by the corresponding mask gain m[b].
    /// m is shaped (1, 1, T, 32) row-major.
    /// </summary>
    private static void ApplyErbMask(float[] specRe, float[] specIm, float[] m, int nFrames)
    {
        for (int t = 0; t < nFrames; t++)
        {
            int baseSpec = t * FreqBins;
            int baseMask = t * NbErb;   // m layout: [batch=1, ch=1, T, NbErb]
            int binOff   = 0;

            for (int b = 0; b < NbErb; b++)
            {
                float gain = m[baseMask + b];
                int   w    = ErbWidths[b];
                for (int j = 0; j < w; j++)
                {
                    int idx       = baseSpec + binOff + j;
                    specRe[idx] *= gain;
                    specIm[idx] *= gain;
                }
                binOff += w;
            }
        }
    }

    /// <summary>
    /// Apply multi-frame deep filter to the first NbDf=96 frequency bins.
    /// Mirrors libdf DF.forward with lookahead=2, order=5.
    ///
    /// For output frame t, bin f:
    ///   out[t,f] = sum_{n=0}^{4} spec[t - 2 + n, f] * coefs[n, t, f]
    ///   (zero-padding for out-of-bounds t)
    ///
    /// coefs layout from df_dec output (1, T, 96, 10) is [re, im] interleaved per tap.
    /// After DfOutputReshapeMF: coefs[tap, t, f] complex.
    /// </summary>
    private static void ApplyDeepFilter(float[] specRe, float[] specIm, float[] coefs, int nFrames)
    {
        // coefs raw shape: (1, T, NbDf, DfOrder*2) = (1, T, 96, 10)
        // coefs[t, f, n] real = coefs_raw[t * NbDf * DfOrder*2 + f * DfOrder*2 + n*2]
        // coefs[t, f, n] imag = coefs_raw[... + n*2 + 1]
        // After DfOutputReshapeMF: view (T, NbDf, DfOrder, 2) → permute (DfOrder, T, NbDf)
        // But we can work directly with the raw layout.

        float[] outRe = new float[nFrames * FreqBins];
        float[] outIm = new float[nFrames * FreqBins];

        // Copy the untouched upper bins (96..480) from original spectrum
        for (int t = 0; t < nFrames; t++)
        {
            int baseSpec = t * FreqBins;
            Array.Copy(specRe, baseSpec + NbDf, outRe, baseSpec + NbDf, FreqBins - NbDf);
            Array.Copy(specIm, baseSpec + NbDf, outIm, baseSpec + NbDf, FreqBins - NbDf);
        }

        for (int t = 0; t < nFrames; t++)
        {
            int baseOut   = t * FreqBins;
            // coefs for frame t: flat index base = t * NbDf * DfOrder * 2
            int baseCoefs = t * NbDf * DfOrder * 2;

            for (int f = 0; f < NbDf; f++)
            {
                float sumRe = 0f, sumIm = 0f;
                for (int n = 0; n < DfOrder; n++)
                {
                    int srcT = t - DfLookahead + n;  // uses frames [t-2, t-1, t, t+1, t+2]
                    if (srcT < 0 || srcT >= nFrames) continue;

                    float sRe = specRe[srcT * FreqBins + f];
                    float sIm = specIm[srcT * FreqBins + f];

                    // coefs_raw shape (1, T, NbDf, DfOrder*2):
                    // tap n at frame t bin f: [baseCoefs + f*DfOrder*2 + n*2] = re
                    //                         [baseCoefs + f*DfOrder*2 + n*2 + 1] = im
                    float cRe = coefs[baseCoefs + f * DfOrder * 2 + n * 2];
                    float cIm = coefs[baseCoefs + f * DfOrder * 2 + n * 2 + 1];

                    // complex multiply: (sRe + i*sIm) * (cRe + i*cIm)
                    sumRe += sRe * cRe - sIm * cIm;
                    sumIm += sRe * cIm + sIm * cRe;
                }
                outRe[baseOut + f] = sumRe;
                outIm[baseOut + f] = sumIm;
            }
        }

        Array.Copy(outRe, specRe, nFrames * FreqBins);
        Array.Copy(outIm, specIm, nFrames * FreqBins);
    }

    // ── ISTFT synthesis ───────────────────────────────────────────────────────

    /// <summary>
    /// Frame-by-frame ISTFT with overlap-add, exactly mirroring libdf frame_synthesis.
    /// IFFT (unnormalized) → multiply by Vorbis window → OLA.
    /// </summary>
    private float[] SynthesisIstft(float[] specRe, float[] specIm, int nFrames)
    {
        float[] output       = new float[nFrames * HopSize];
        float[] synthesisMem = new float[HopSize];
        var     buf          = new Complex32[FftSize];

        for (int t = 0; t < nFrames; t++)
        {
            int baseSpec = t * FreqBins;

            // Reconstruct full Hermitian-symmetric spectrum for real IFFT
            // bins 0..FreqBins-1 directly, bins FreqBins..FftSize-1 by conjugate symmetry
            for (int k = 0; k < FreqBins; k++)
                buf[k] = new Complex32(specRe[baseSpec + k], specIm[baseSpec + k]);

            // X[N-k] = conj(X[k]) for k = 1..N/2-1
            for (int k = 1; k < FftSize / 2; k++)
                buf[FftSize - k] = new Complex32(specRe[baseSpec + k], -specIm[baseSpec + k]);

            // Unnormalized inverse FFT: output = N * IDFT(X)
            Fourier.Inverse(buf, FourierOptions.NoScaling);

            // Apply synthesis window, then OLA
            int audioOffset = t * HopSize;
            for (int j = 0; j < HopSize; j++)
                output[audioOffset + j] = buf[j].Real * _window[j] + synthesisMem[j];

            // Update synthesis memory with second half of windowed IFFT output
            for (int j = 0; j < HopSize; j++)
                synthesisMem[j] = buf[HopSize + j].Real * _window[HopSize + j];
        }

        return output;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Vorbis window: w[i] = sin(π/2 × sin²(π×(i+0.5)/(N/2)))
    /// Satisfies WOLA property at 50% overlap: sum_k w(n + k×hop)² = 1.
    /// </summary>
    private static float[] MakeVorbisWindow(int n)
    {
        var w    = new float[n];
        int half = n / 2;
        for (int i = 0; i < n; i++)
        {
            double s = Math.Sin(0.5 * Math.PI * (i + 0.5) / half);
            w[i] = (float)Math.Sin(0.5 * Math.PI * s * s);
        }
        return w;
    }

    private static float[] Linspace(float start, float stop, int count)
    {
        var result = new float[count];
        if (count == 1) { result[0] = start; return result; }
        float step = (stop - start) / (count - 1);
        for (int i = 0; i < count; i++)
            result[i] = start + i * step;
        return result;
    }

    // ── Static resampling helpers ─────────────────────────────────────────────

    /// <summary>
    /// Resample mono audio from <paramref name="fromHz"/> to 48 kHz using WDL resampler.
    /// </summary>
    public static float[] ResampleTo48k(float[] mono, int fromHz)
    {
        if (fromHz == Sr) return mono;
        return Resample(mono, fromHz, Sr);
    }

    /// <summary>
    /// Resample mono audio from 48 kHz to <paramref name="toHz"/> using WDL resampler.
    /// </summary>
    public static float[] ResampleFrom48k(float[] mono, int toHz)
    {
        if (toHz == Sr) return mono;
        return Resample(mono, Sr, toHz);
    }

    private static float[] Resample(float[] mono, int fromHz, int toHz)
    {
        var monoFormat   = WaveFormat.CreateIeeeFloatWaveFormat(fromHz, 1);
        var waveProvider = new FloatArraySampleProvider(mono, monoFormat);
        var resampler    = new WdlResamplingSampleProvider(waveProvider, toHz);
        var outList      = new List<float>((int)((long)mono.Length * toHz / fromHz + 1024));
        var outBuffer    = new float[8192];
        int outRead;
        while ((outRead = resampler.Read(outBuffer, 0, outBuffer.Length)) > 0)
            for (int i = 0; i < outRead; i++) outList.Add(outBuffer[i]);
        return outList.ToArray();
    }
}
