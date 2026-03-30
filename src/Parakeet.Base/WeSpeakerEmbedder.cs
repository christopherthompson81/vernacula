using System.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Parakeet.Base.Models;

namespace Parakeet.Base;

/// <summary>
/// Speaker embedding extraction using the WeSpeaker-ResNet34 ONNX model.
///
/// <para><strong>Pipeline:</strong></para>
/// <list type="number">
/// <item><description>Scale audio to int16 range (×32768, Kaldi convention)</description></item>
/// <item><description>Compute Kaldi Fbank: 40-bin log Mel filterbank, 25ms/10ms window/hop, Hamming window</description></item>
/// <item><description>Subtract per-utterance mean (CMVN-style normalisation)</description></item>
/// <item><description>Run ResNet34 ONNX inference → 512-dim embedding</description></item>
/// <item><description>The ONNX wrapper already L2-normalises the output</description></item>
/// </list>
///
/// <para><strong>Fbank parameters (must match the Python training config):</strong></para>
/// <list type="bullet">
/// <item><description>sample_rate = 16 000 Hz</description></item>
/// <item><description>frame_length = 25 ms (400 samples)</description></item>
/// <item><description>frame_shift = 10 ms (160 samples)</description></item>
/// <item><description>num_mel_bins = 40, low_freq = 20 Hz, high_freq = 8 000 Hz</description></item>
/// <item><description>pre_emphasis = 0.97, remove_dc_offset = true, snip_edges = true</description></item>
/// </list>
/// </summary>
public sealed class WeSpeakerEmbedder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly float[] _window;
    private readonly float[,] _melFilters;
    private bool _disposed;

    // ── Constants ──────────────────────────────────────────────────────────

    public const int EmbeddingDim    = 512;
    public const int SampleRate      = 16_000;
    public const int NumMelBins      = 40;
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

    public WeSpeakerEmbedder(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _window     = MakeHammingWindow(FrameLengthSamples);
        _melFilters = MakeMelFilters(NumMelBins, FftSize, SampleRate, LowFreqHz, HighFreqHz);

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

    /// <summary>
    /// Compute a 512-dim L2-normalised speaker embedding for one audio region.
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
            return new float[EmbeddingDim];

        var segment = new float[regionLen];
        Array.Copy(audio, startSample, segment, 0, regionLen);

        var fbank = ComputeFbank(segment);
        return fbank.GetLength(0) == 0 ? new float[EmbeddingDim] : RunInference(fbank);
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

        // Per-utterance mean normalisation (subtract mean per mel bin)
        for (int m = 0; m < NumMelBins; m++)
        {
            float mean = 0f;
            for (int fi = 0; fi < numFrames; fi++) mean += result[fi, m];
            mean /= numFrames;
            for (int fi = 0; fi < numFrames; fi++) result[fi, m] -= mean;
        }

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

        var embedding = new float[EmbeddingDim];
        for (int i = 0; i < EmbeddingDim; i++)
            embedding[i] = output[0, i];
        return embedding;
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
