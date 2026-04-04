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
    private readonly bool _supportsWeights;
    private readonly string _fbankInputName;
    private readonly string? _weightsInputName;
    private readonly string _outputName;
    private readonly float[] _window;
    private readonly float[] _melFilters;
    private bool _disposed;

    // Full PLDA transform parameters (optional, loaded from plda/ directory)
    private readonly float[]? _ldaMean1;    // (256,) — xvec_transform mean1
    private readonly float[,]? _ldaMatrix;  // (256, 128) — xvec_transform lda
    private readonly float[]? _ldaMean2;    // (128,) — xvec_transform mean2
    private readonly float[]?  _pldaMu;    // (128,) — PLDA mean (subtract before plda_tr)
    private readonly float[,]? _pldaTr;    // (128, 128) — PLDA eigenspace matrix
    private readonly float[]?  _pldaPsi;   // (128,) — PLDA eigenvalues (Phi for VBx)

    /// <summary>PLDA eigenvalues (Phi) needed by VBx, or null if PLDA not loaded.</summary>
    public float[]? VbxPhi => _pldaPsi;

    /// <summary>PLDA mean vector (128-dim), subtract from xvec before plda_tr projection.</summary>
    public float[]? PldaMu => _pldaMu;

    /// <summary>PLDA eigenspace matrix (128×128), needed for batch plda_tf, or null.</summary>
    public float[,]? PldaTr => _pldaTr;

    /// <summary>Whether the loaded ONNX embedder supports native frame weights.</summary>
    public bool SupportsWeights => _supportsWeights;

    /// <summary>
    /// Minimum waveform length accepted by the weighted pyannote WeSpeaker path.
    /// This matches the reference pipeline's `min_num_samples` behavior.
    /// </summary>
    public int MinNumSamples => FrameLengthSamples;

    // ── Constants ──────────────────────────────────────────────────────────

    public const int EmbeddingDim    = 256;
    public const int SampleRate      = 16_000;
    public const int NumMelBins      = 80;
    public const int FrameLengthSamples = 400;   // 25 ms × 16 kHz
    public const int FrameShiftSamples  = 160;   // 10 ms × 16 kHz
    public const int FftSize            = 512;   // next power of 2 ≥ 400
    public const int NumFreqBins        = FftSize / 2 + 1;
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
    /// <param name="modelPath">Path to wespeaker_pyannote_weighted.onnx or wespeaker_pyannote.onnx.</param>
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
            string pmu = Path.Combine(ldaDir, "plda_mu.bin");
            string ptr = Path.Combine(ldaDir, "plda_tr.bin");
            string pps = Path.Combine(ldaDir, "plda_psi.bin");
            if (File.Exists(m1) && File.Exists(ld) && File.Exists(m2))
            {
                _ldaMean1  = ReadFloatBin(m1);               // (256,)
                _ldaMatrix = ReadFloat2dBin(ld, 256, 128);   // (256, 128)
                _ldaMean2  = ReadFloatBin(m2);               // (128,)
            }
            if (File.Exists(pmu) && File.Exists(ptr) && File.Exists(pps))
            {
                _pldaMu  = ReadFloatBin(pmu);                // (128,)
                _pldaTr  = ReadFloat2dBin(ptr, 128, 128);    // (128, 128)
                _pldaPsi = ReadFloatBin(pps);                // (128,)
            }
        }

        var opts = new SessionOptions();
        // Keep per-session threading low and use outer parallelism across
        // multiple embedding jobs instead of oversubscribing CPU cores.
        opts.IntraOpNumThreads = 1;
        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                {
                    try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                }
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
        var inputNames = _session.InputMetadata.Keys.ToArray();
        _fbankInputName = inputNames.First();
        _weightsInputName = inputNames.FirstOrDefault(name =>
            string.Equals(name, "weights", StringComparison.OrdinalIgnoreCase));
        _supportsWeights = _weightsInputName != null;
        _outputName = _session.OutputMetadata.Keys.First();
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>Whether the full PLDA transform (xvec_tf + plda_tf) is available.</summary>
    public bool HasPlda => _ldaMatrix != null && _pldaTr != null && _pldaMu != null;

    /// <summary>
    /// Compute embeddings for one audio region.
    /// </summary>
    /// <param name="audio">Full recording at 16 kHz mono.</param>
    /// <param name="startSample">Inclusive start sample.</param>
    /// <param name="endSample">Exclusive end sample.</param>
    /// <param name="rawEmbedding">
    /// Output: raw un-normalised 256-dim ONNX output, needed for <see cref="TransformToPlda"/>.
    /// </param>
    /// <returns>L2-normalised 256-dim embedding (for AHC initialisation).</returns>
    public float[] ComputeEmbedding(float[] audio, int startSample, int endSample,
        out float[] rawEmbedding)
    {
        int maxSamples = SampleRate * MaxEmbedWindowSec;
        int len = endSample - startSample;
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
        {
            rawEmbedding = new float[EmbeddingDim];
            return new float[EmbeddingDim];
        }

        var segment = new float[regionLen];
        Array.Copy(audio, startSample, segment, 0, regionLen);

        var fbank = ComputeFbank(segment);
        return ComputeEmbeddingFromFbank(fbank, out rawEmbedding);
    }

    /// <summary>
    /// Compute an embedding from a chunk waveform using a frame mask defined at
    /// diarization frame resolution.
    ///
    /// When the ONNX model supports a `weights` input, this mirrors pyannote's
    /// native weighted pooling semantics by feeding the full fbank sequence plus
    /// an upsampled 0/1 frame-weight vector.
    ///
    /// Otherwise, it falls back to the older masked-frame approximation where
    /// only selected frames are passed to the embedding model.
    /// </summary>
    public float[] ComputeEmbedding(float[] waveform, bool[] frameMask, out float[] rawEmbedding)
    {
        if (waveform.Length < FrameLengthSamples || frameMask.Length == 0)
        {
            rawEmbedding = new float[EmbeddingDim];
            return new float[EmbeddingDim];
        }

        var fbank = ComputeFbank(waveform);
        int numFrames = fbank.GetLength(0);
        if (fbank.GetLength(0) == 0)
        {
            rawEmbedding = new float[EmbeddingDim];
            return new float[EmbeddingDim];
        }

        int selectedCount = 0;
        var selected = new bool[numFrames];
        var frameWeights = new float[numFrames];
        for (int fi = 0; fi < numFrames; fi++)
        {
            int srcIndex = Math.Clamp(
                (int)Math.Floor(fi * frameMask.Length / (double)numFrames),
                0,
                frameMask.Length - 1);
            bool keep = frameMask[srcIndex];
            selected[fi] = keep;
            frameWeights[fi] = keep ? 1f : 0f;
            if (keep) selectedCount++;
        }

        if (selectedCount == 0)
        {
            rawEmbedding = new float[EmbeddingDim];
            return new float[EmbeddingDim];
        }

        if (_supportsWeights)
            return ComputeEmbeddingFromFbank(fbank, frameWeights, out rawEmbedding);

        var maskedFbank = new float[selectedCount, NumMelBins];
        int dst = 0;
        for (int fi = 0; fi < numFrames; fi++)
        {
            if (!selected[fi]) continue;
            for (int m = 0; m < NumMelBins; m++)
                maskedFbank[dst, m] = fbank[fi, m];
            dst++;
        }

        return ComputeEmbeddingFromFbank(maskedFbank, out rawEmbedding);
    }

    private float[] ComputeEmbeddingFromFbank(float[,] fbank, out float[] rawEmbedding)
    {
        if (fbank.GetLength(0) == 0)
        {
            rawEmbedding = new float[EmbeddingDim];
            return new float[EmbeddingDim];
        }

        rawEmbedding = RunInference(fbank);  // raw un-normalised 256-dim

        // L2-normalise for AHC (matches Python: train_embeddings_normed = emb / ||emb||)
        return NormalizeEmbedding(rawEmbedding);
    }

    private float[] ComputeEmbeddingFromFbank(float[,] fbank, float[] frameWeights, out float[] rawEmbedding)
    {
        if (fbank.GetLength(0) == 0)
        {
            rawEmbedding = new float[EmbeddingDim];
            return new float[EmbeddingDim];
        }

        rawEmbedding = RunInference(fbank, frameWeights);  // raw un-normalised 256-dim
        return NormalizeEmbedding(rawEmbedding);
    }

    /// <summary>
    /// Apply xvec_tf to a raw un-normalised 256-dim embedding → 128-dim xvec.
    /// Must then pass the full batch through <see cref="ApplyPldaTfBatch"/> for VBx.
    /// Returns null if PLDA files were not loaded.
    /// </summary>
    public float[]? ComputeXvec(float[] rawEmbedding)
    {
        if (_ldaMatrix == null) return null;
        return ApplyXvecTf(rawEmbedding);
    }

    /// <summary>
    /// xvec_tf: transforms one raw un-normalised 256-dim embedding to 128-dim xvec space.
    /// Matches Python:
    ///   sqrt(128) * l2_norm( lda.T @ (sqrt(256) * l2_norm(x - mean1).T).T - mean2 )
    /// </summary>
    private float[] ApplyXvecTf(float[] raw)
    {
        int inDim  = _ldaMean1!.Length;   // 256
        int ldaDim = _ldaMean2!.Length;   // 128
        double sqrtIn  = Math.Sqrt(inDim);   // sqrt(256)
        double sqrtOut = Math.Sqrt(ldaDim);  // sqrt(128)

        // Step 1: l2_norm(raw - mean1) * sqrt(256)
        double n1 = 0;
        var c = new double[inDim];
        for (int i = 0; i < inDim; i++) { c[i] = raw[i] - _ldaMean1[i]; n1 += c[i]*c[i]; }
        n1 = Math.Sqrt(n1);
        for (int i = 0; i < inDim; i++) c[i] = n1 > 0 ? c[i] / n1 * sqrtIn : 0;

        // Step 2: lda.T @ c − mean2 = c @ lda − mean2  (lda is inDim×ldaDim)
        var v2 = new double[ldaDim];
        for (int j = 0; j < ldaDim; j++)
        {
            double v = -_ldaMean2[j];
            for (int i = 0; i < inDim; i++) v += c[i] * _ldaMatrix![i, j];
            v2[j] = v;
        }

        // Step 3: l2_norm(v2) * sqrt(128)
        double n2 = 0;
        for (int j = 0; j < ldaDim; j++) n2 += v2[j]*v2[j];
        n2 = Math.Sqrt(n2);
        var xvec = new float[ldaDim];
        for (int j = 0; j < ldaDim; j++) xvec[j] = n2 > 0 ? (float)(v2[j] / n2 * sqrtOut) : 0f;
        return xvec;
    }

    /// <summary>
    /// plda_tf applied to a batch. Matches Python:
    ///   (xvecs - plda_mu) @ plda_tr.T
    /// No normalization — output norms are ~19 (PLDA eigenspace scale).
    /// </summary>
    public static double[][] ApplyPldaTfBatch(float[,] pldaTr, float[] pldaMu, float[][] xvecs)
    {
        int n   = xvecs.Length;
        int dim = pldaTr.GetLength(0);  // 128

        // fea[i] = (xvecs[i] - pldaMu) @ pldaTr.T
        var fea = new double[n][];
        for (int i = 0; i < n; i++)
        {
            fea[i] = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                double v = 0;
                for (int j = 0; j < dim; j++) v += pldaTr[d, j] * (xvecs[i][j] - pldaMu[j]);
                fea[i][d] = v;
            }
        }
        return fea;
    }

    /// <summary>
    /// Convenience: apply xvec_tf to a single raw embedding.
    /// Use <see cref="ApplyPldaTfBatch"/> on the full batch for plda_tf.
    /// </summary>
    private float[] ApplyLda(float[] rawEmbedding) => ApplyXvecTf(rawEmbedding);

    // ── Fbank computation ──────────────────────────────────────────────────

    private float[,] ComputeFbank(float[] waveform)
    {
        int n         = waveform.Length;
        int numFrames = 1 + (n - FrameLengthSamples) / FrameShiftSamples;
        if (numFrames <= 0) return new float[0, NumMelBins];

        var result     = new float[numFrames, NumMelBins];
        var complexBuf = new Complex[FftSize];
        var frame      = new float[FrameLengthSamples];
        var power      = new float[NumFreqBins];

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

            for (int k = 0; k < NumFreqBins; k++)
            {
                float re = (float)complexBuf[k].Real;
                float im = (float)complexBuf[k].Imaginary;
                power[k] = re * re + im * im;
            }

            // Mel filterbank + log
            for (int m = 0; m < NumMelBins; m++)
            {
                float energy = DotProduct(
                    _melFilters.AsSpan(m * NumFreqBins, NumFreqBins),
                    power);
                result[fi, m] = MathF.Log(MathF.Max(energy, 1e-10f));
            }
        }

        // Per-utterance mean normalisation: subtract mean per mel bin across all frames.
        // Matches pyannote WeSpeakerResNet34.compute_fbank:
        //   features = features - features.mean(dim=1, keepdim=True)
        var means = new float[NumMelBins];
        for (int fi = 0; fi < numFrames; fi++)
        {
            for (int m = 0; m < NumMelBins; m++)
                means[m] += result[fi, m];
        }

        for (int m = 0; m < NumMelBins; m++)
            means[m] /= numFrames;

        for (int fi = 0; fi < numFrames; fi++)
        {
            for (int m = 0; m < NumMelBins; m++)
                result[fi, m] -= means[m];
        }

        return result;
    }

    // ── ONNX inference ─────────────────────────────────────────────────────

    private float[] RunInference(float[,] fbank, float[]? frameWeights = null)
    {
        int numFrames = fbank.GetLength(0);
        var data = new float[numFrames * NumMelBins];
        Buffer.BlockCopy(fbank, 0, data, 0, data.Length * sizeof(float));

        var tensor = new DenseTensor<float>(data, new[] { 1, numFrames, NumMelBins });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_fbankInputName, tensor)
        };

        if (_supportsWeights && frameWeights != null)
        {
            var weightTensor = new DenseTensor<float>(frameWeights, new[] { 1, numFrames });
            inputs.Add(NamedOnnxValue.CreateFromTensor(_weightsInputName!, weightTensor));
        }

        using var results = _session.Run(inputs);
        var output = results.First(r => r.Name == _outputName).AsTensor<float>();

        int outSize = output.Dimensions[1];
        var embedding = new float[outSize];
        for (int i = 0; i < outSize; i++)
            embedding[i] = output[0, i];
        return embedding;
    }

    private static float[] NormalizeEmbedding(float[] rawEmbedding)
    {
        float[] l2 = (float[])rawEmbedding.Clone();
        float norm = 0f;
        for (int i = 0; i < l2.Length; i++) norm += l2[i] * l2[i];
        norm = MathF.Sqrt(norm);
        if (norm > 0) for (int i = 0; i < l2.Length; i++) l2[i] /= norm;
        return l2;
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

    private static float[] MakeMelFilters(int numMelBins, int fftSize, int sampleRate,
        float lowFreqHz, float highFreqHz)
    {
        int numFreqBins = fftSize / 2 + 1;
        var filters = new float[numMelBins * numFreqBins];

        double melLow  = HzToMel(lowFreqHz);
        double melHigh = HzToMel(highFreqHz);

        // numMelBins + 2 equally spaced Mel-scale centre points
        int   nPoints = numMelBins + 2;
        var   hzPts   = new double[nPoints];
        for (int i = 0; i < nPoints; i++)
            hzPts[i] = MelToHz(melLow + i * (melHigh - melLow) / (numMelBins + 1));

        // Evaluate each triangular filter at the exact FFT bin centre frequency
        // (k * sr / fftSize Hz).  Matches torchaudio's get_mel_banks behaviour:
        // no integer rounding of bin boundaries, so every filter has non-zero
        // support and narrow low-frequency filters are not accidentally collapsed
        // to zero.
        for (int m = 0; m < numMelBins; m++)
        {
            double fl = hzPts[m];
            double fc = hzPts[m + 1];
            double fh = hzPts[m + 2];

            for (int k = 0; k < numFreqBins; k++)
            {
                double fk = k * (double)sampleRate / fftSize;
                float w;
                if (fk >= fl && fk < fc && fc > fl)
                    w = (float)((fk - fl) / (fc - fl));
                else if (fk >= fc && fk <= fh && fh > fc)
                    w = (float)((fh - fk) / (fh - fc));
                else
                    w = 0f;
                filters[m * numFreqBins + k] = w;
            }
        }

        return filters;
    }

    private static float DotProduct(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        int width = Vector<float>.Count;
        int i = 0;
        Vector<float> sum = Vector<float>.Zero;
        for (; i <= left.Length - width; i += width)
            sum += new Vector<float>(left.Slice(i, width)) * new Vector<float>(right.Slice(i, width));

        float total = 0f;
        for (int lane = 0; lane < Vector<float>.Count; lane++)
            total += sum[lane];

        for (; i < left.Length; i++)
            total += left[i] * right[i];

        return total;
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
