using System.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Parakeet.Base.Models;

namespace Parakeet.Base;

/// <summary>
/// Speaker embedding extraction using the pyannote WeSpeaker-ResNet34-LM ONNX model.
///
/// <para><strong>Pipeline:</strong></para>
/// <list type="number">
/// <item><description>Scale audio to int16 range (×32768, Kaldi convention)</description></item>
/// <item><description>Compute Kaldi Fbank: 80-bin log Mel filterbank, 25ms/10ms window/hop, Hamming window</description></item>
/// <item><description>Subtract per-utterance mean (CMVN-style normalisation)</description></item>
/// <item><description>Run ResNet34 ONNX inference → 256-dim L2-normalised embedding</description></item>
/// <item><description>Optionally apply LDA transform (256→128) for PLDA-space clustering</description></item>
/// </list>
///
/// <para><strong>Fbank parameters (matches torchaudio.compliance.kaldi.fbank defaults):</strong></para>
/// <list type="bullet">
/// <item><description>sample_rate = 16 000 Hz</description></item>
/// <item><description>frame_length = 25 ms (400 samples)</description></item>
/// <item><description>frame_shift = 10 ms (160 samples)</description></item>
/// <item><description>num_mel_bins = 80, low_freq = 20 Hz, high_freq = 8 000 Hz</description></item>
/// <item><description>preemphasis_coefficient = 0.97, remove_dc_offset = true, snip_edges = true</description></item>
/// </list>
/// </summary>
public sealed class WeSpeakerEmbedder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly float[] _window;
    private readonly float[,] _melFilters;
    private bool _disposed;

    // Full PLDA transform parameters (optional, loaded from plda/ directory)
    private readonly float[]? _ldaMean1;    // (256,) — xvec_transform mean1
    private readonly float[,]? _ldaMatrix;  // (256, 128) — xvec_transform lda
    private readonly float[]? _ldaMean2;    // (128,) — xvec_transform mean2
    private readonly float[,]? _pldaTr;    // (128, 128) — PLDA whitening rows
    private readonly float[]?  _pldaPsi;   // (128,) — PLDA eigenvalues (Phi for VBx)

    /// <summary>PLDA eigenvalues (Phi) needed by VBx, or null if PLDA not loaded.</summary>
    public float[]? VbxPhi => _pldaPsi;

    // ── Constants ──────────────────────────────────────────────────────────

    public const int EmbeddingDim    = 256;
    public const int SampleRate      = 16_000;
    public const int NumMelBins      = 80;
    public const int FrameLengthSamples = 400;   // 25 ms × 16 kHz
    public const int FrameShiftSamples  = 160;   // 10 ms × 16 kHz
    public const int FftSize            = 512;   // next power of 2 ≥ 400
    public const float LowFreqHz        = 20f;
    public const float HighFreqHz       = 8_000f;
    public const float PreEmphCoeff     = 0.97f;
    public const float WaveformScale    = 32_768f;  // Kaldi int16 convention
    /// <summary>Maximum audio (in seconds) taken from the centre of a region for embedding.</summary>
    public const int MaxEmbedWindowSec  = 5;

    // ── Construction ───────────────────────────────────────────────────────

    /// <summary>
    /// Create a WeSpeaker embedder.
    /// </summary>
    /// <param name="modelPath">Path to wespeaker_pyannote.onnx.</param>
    /// <param name="ldaDir">
    /// Optional directory containing <c>mean1.bin</c>, <c>lda.bin</c>, <c>mean2.bin</c>
    /// (exported from plda/xvec_transform.npz by export_lda_transform.py).
    /// When supplied, embeddings are projected 256→128 before clustering.
    /// </param>
    /// <param name="ep">ONNX execution provider.</param>
    public WeSpeakerEmbedder(string modelPath, string? ldaDir = null,
        ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _window     = MakeHammingWindow(FrameLengthSamples);
        _melFilters = MakeMelFilters(NumMelBins, FftSize, SampleRate, LowFreqHz, HighFreqHz);

        // Load optional PLDA transform (xvec_tf + plda_tf)
        if (ldaDir != null && Directory.Exists(ldaDir))
        {
            string m1  = Path.Combine(ldaDir, "mean1.bin");
            string ld  = Path.Combine(ldaDir, "lda.bin");
            string m2  = Path.Combine(ldaDir, "mean2.bin");
            string ptr = Path.Combine(ldaDir, "plda_tr.bin");
            string pps = Path.Combine(ldaDir, "plda_psi.bin");
            if (File.Exists(m1) && File.Exists(ld) && File.Exists(m2))
            {
                _ldaMean1  = ReadFloatBin(m1);               // (256,)
                _ldaMatrix = ReadFloat2dBin(ld, 256, 128);   // (256, 128)
                _ldaMean2  = ReadFloatBin(m2);               // (128,)
            }
            if (File.Exists(ptr) && File.Exists(pps))
            {
                _pldaTr  = ReadFloat2dBin(ptr, 128, 128);    // (128, 128)
                _pldaPsi = ReadFloatBin(pps);                // (128,)
            }
        }

        var opts = new SessionOptions();
        opts.IntraOpNumThreads = 2;
        switch (ep)
        {
            case ExecutionProvider.Auto:
                try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                try { opts.AppendExecutionProvider_DML(0); }  catch { }
                break;
            case ExecutionProvider.Cuda:
                opts.AppendExecutionProvider_CUDA(0);
                break;
            case ExecutionProvider.DirectML:
                opts.AppendExecutionProvider_DML(0);
                break;
        }
        _session = new InferenceSession(modelPath, opts);
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>Returns the output dimension: 256 (raw) or 128 (if PLDA transform loaded).</summary>
    public int OutputDim => _ldaMatrix != null ? 128 : EmbeddingDim;

    /// <summary>Whether the full PLDA transform (xvec_tf + plda_tf) is available.</summary>
    public bool HasPlda => _ldaMatrix != null && _pldaTr != null;

    /// <summary>
    /// Compute a speaker embedding for one audio region.
    /// Returns 256-dim L2-normalised embedding, or 128-dim LDA-projected if transform is loaded.
    /// </summary>
    /// <param name="audio">Full recording at 16 kHz mono.</param>
    /// <param name="startSample">Inclusive start sample of the region.</param>
    /// <param name="endSample">Exclusive end sample of the region.</param>
    public float[] ComputeEmbedding(float[] audio, int startSample, int endSample)
    {
        int maxSamples = SampleRate * MaxEmbedWindowSec;
        int len = endSample - startSample;

        // If region is longer than the cap, take the centre window
        if (len > maxSamples)
        {
            int centre = startSample + len / 2;
            startSample = centre - maxSamples / 2;
            endSample   = startSample + maxSamples;
        }

        startSample = Math.Max(0, startSample);
        endSample   = Math.Min(audio.Length, endSample);
        int regionLen = endSample - startSample;

        if (regionLen < FrameLengthSamples)
            return new float[OutputDim];

        var segment = new float[regionLen];
        Array.Copy(audio, startSample, segment, 0, regionLen);

        var fbank = ComputeFbank(segment);
        if (fbank.GetLength(0) == 0) return new float[OutputDim];

        float[] raw = RunInference(fbank);

        if (_ldaMatrix != null)
            return ApplyLda(raw);

        // No LDA: L2-normalise the raw embedding before returning
        float norm0 = 0f;
        for (int i = 0; i < raw.Length; i++) norm0 += raw[i] * raw[i];
        norm0 = MathF.Sqrt(norm0);
        if (norm0 > 0) for (int i = 0; i < raw.Length; i++) raw[i] /= norm0;
        return raw;
    }

    /// <summary>
    /// Full PLDA transform pipeline: xvec_tf then plda_tf.
    /// <list type="number">
    /// <item>centered = l2_norm(raw − mean1)</item>
    /// <item>lda_normed = l2_norm(centered @ lda − mean2)</item>
    /// <item>fea = l2_norm(plda_tr @ lda_normed)  — final embedding for VBx</item>
    /// </list>
    /// When plda_tr is not loaded, stops after step 2.
    /// </summary>
    private float[] ApplyLda(float[] embedding)
    {
        int inDim  = _ldaMean1!.Length;   // 256
        int ldaDim = _ldaMean2!.Length;   // 128

        // Step 1: (raw - mean1) → L2-normalise
        var centered = new float[inDim];
        float n1 = 0f;
        for (int i = 0; i < inDim; i++) { centered[i] = embedding[i] - _ldaMean1[i]; n1 += centered[i] * centered[i]; }
        n1 = MathF.Sqrt(n1);
        if (n1 > 0) for (int i = 0; i < inDim; i++) centered[i] /= n1;

        // Step 2: centered @ lda − mean2 → L2-normalise
        var lda_normed = new float[ldaDim];
        for (int j = 0; j < ldaDim; j++)
        {
            float v = -_ldaMean2[j];
            for (int i = 0; i < inDim; i++) v += centered[i] * _ldaMatrix![i, j];
            lda_normed[j] = v;
        }
        float n2 = 0f;
        for (int j = 0; j < ldaDim; j++) n2 += lda_normed[j] * lda_normed[j];
        n2 = MathF.Sqrt(n2);
        if (n2 > 0) for (int j = 0; j < ldaDim; j++) lda_normed[j] /= n2;

        if (_pldaTr == null) return lda_normed;

        // Step 3: plda_tr (128×128) @ lda_normed → L2-normalise (plda_tf)
        int pldaDim = _pldaTr.GetLength(0);
        var fea = new float[pldaDim];
        for (int j = 0; j < pldaDim; j++)
        {
            float v = 0f;
            for (int i = 0; i < ldaDim; i++) v += _pldaTr[j, i] * lda_normed[i];
            fea[j] = v;
        }
        float n3 = 0f;
        for (int j = 0; j < pldaDim; j++) n3 += fea[j] * fea[j];
        n3 = MathF.Sqrt(n3);
        if (n3 > 0) for (int j = 0; j < pldaDim; j++) fea[j] /= n3;

        return fea;
    }

    // ── Fbank computation ──────────────────────────────────────────────────

    private float[,] ComputeFbank(float[] waveform)
    {
        int n         = waveform.Length;
        int numFrames = 1 + (n - FrameLengthSamples) / FrameShiftSamples;
        if (numFrames <= 0) return new float[0, NumMelBins];

        int numFreqBins = FftSize / 2 + 1;
        var result     = new float[numFrames, NumMelBins];
        var complexBuf = new Complex[FftSize];
        var frame      = new float[FrameLengthSamples];

        for (int fi = 0; fi < numFrames; fi++)
        {
            int offset = fi * FrameShiftSamples;

            // Scale and compute DC offset
            float dcSum = 0f;
            for (int i = 0; i < FrameLengthSamples; i++)
            {
                frame[i] = waveform[offset + i] * WaveformScale;
                dcSum += frame[i];
            }

            float dc = dcSum / FrameLengthSamples;
            for (int i = 0; i < FrameLengthSamples; i++)
                frame[i] -= dc;

            // Pre-emphasis: x[0] *= (1-α),  x[i] -= α·x[i-1]
            float prev = frame[0];
            frame[0] *= (1f - PreEmphCoeff);
            for (int i = 1; i < FrameLengthSamples; i++)
            {
                float curr = frame[i];
                frame[i] -= PreEmphCoeff * prev;
                prev = curr;
            }

            // Hamming window + copy into FFT buffer (zero-padded to FftSize)
            Array.Clear(complexBuf, 0, FftSize);
            for (int i = 0; i < FrameLengthSamples; i++)
                complexBuf[i] = new Complex(frame[i] * _window[i], 0.0);

            // Forward FFT (no scaling – matches NumPy/PyTorch convention)
            Fourier.Forward(complexBuf, FourierOptions.AsymmetricScaling);

            // Mel filterbank + log
            for (int m = 0; m < NumMelBins; m++)
            {
                float energy = 0f;
                for (int k = 0; k < numFreqBins; k++)
                {
                    float re = (float)complexBuf[k].Real;
                    float im = (float)complexBuf[k].Imaginary;
                    energy += _melFilters[m, k] * (re * re + im * im);
                }
                result[fi, m] = MathF.Log(MathF.Max(energy, 1e-10f));
            }
        }

        // No per-utterance mean normalisation: pyannote's WeSpeakerResNet34 pipeline
        // (torchaudio.compliance.kaldi.fbank) does not apply CMVN before the ResNet.
        return result;
    }

    // ── ONNX inference ─────────────────────────────────────────────────────

    private float[] RunInference(float[,] fbank)
    {
        int numFrames = fbank.GetLength(0);
        var data = new float[numFrames * NumMelBins];
        for (int fi = 0; fi < numFrames; fi++)
            for (int m = 0; m < NumMelBins; m++)
                data[fi * NumMelBins + m] = fbank[fi, m];

        var tensor = new DenseTensor<float>(data, new[] { 1, numFrames, NumMelBins });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("fbank", tensor) };

        using var results = _session.Run(inputs);
        var output = results.First(r => r.Name == "embedding").AsTensor<float>();

        int outSize = output.Dimensions[1];
        var embedding = new float[outSize];
        for (int i = 0; i < outSize; i++)
            embedding[i] = output[0, i];
        return embedding;
    }

    // ── LDA binary file loaders ────────────────────────────────────────────

    private static float[] ReadFloatBin(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var result = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }

    private static float[,] ReadFloat2dBin(string path, int rows, int cols)
    {
        var bytes = File.ReadAllBytes(path);
        var flat  = new float[rows * cols];
        Buffer.BlockCopy(bytes, 0, flat, 0, bytes.Length);
        var result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r, c] = flat[r * cols + c];
        return result;
    }

    // ── Static DSP helpers ─────────────────────────────────────────────────

    private static float[] MakeHammingWindow(int size)
    {
        var w = new float[size];
        double twoPiOverN = 2.0 * Math.PI / (size - 1);
        for (int i = 0; i < size; i++)
            w[i] = (float)(0.54 - 0.46 * Math.Cos(i * twoPiOverN));
        return w;
    }

    private static float[,] MakeMelFilters(int numMelBins, int fftSize, int sampleRate,
        float lowFreqHz, float highFreqHz)
    {
        int numFreqBins = fftSize / 2 + 1;
        var filters = new float[numMelBins, numFreqBins];

        double melLow  = HzToMel(lowFreqHz);
        double melHigh = HzToMel(highFreqHz);

        // numMelBins + 2 equally spaced Mel-scale centre points
        int   nPoints   = numMelBins + 2;
        var   melPts    = new double[nPoints];
        var   hzPts     = new double[nPoints];
        var   binPts    = new int[nPoints];

        for (int i = 0; i < nPoints; i++)
        {
            melPts[i] = melLow + i * (melHigh - melLow) / (numMelBins + 1);
            hzPts[i]  = MelToHz(melPts[i]);
            binPts[i] = (int)Math.Floor((fftSize + 1) * hzPts[i] / sampleRate);
        }

        for (int m = 0; m < numMelBins; m++)
        {
            int left   = binPts[m];
            int centre = binPts[m + 1];
            int right  = binPts[m + 2];

            for (int k = left; k < centre && k < numFreqBins; k++)
                if (centre > left)
                    filters[m, k] = (k - left) / (float)(centre - left);

            for (int k = centre; k <= right && k < numFreqBins; k++)
                if (right > centre)
                    filters[m, k] = (right - k) / (float)(right - centre);
        }

        return filters;
    }

    private static double HzToMel(double hz) => 1127.0 * Math.Log(1.0 + hz / 700.0);
    private static double MelToHz(double mel) => 700.0 * (Math.Exp(mel / 1127.0) - 1.0);

    // ── IDisposable ────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
