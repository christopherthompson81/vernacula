using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// Whisper large-v3-turbo backend.  Phase 2a scope: Whisper-style log-mel
/// frontend and encoder-only inference.  Decoder pair, greedy loop, tokenizer,
/// and language-token handling land in Phase 2b.
///
/// ONNX layout (fp16 internal, fp32 graph boundaries — Optimum default export):
///   encoder_model_fp16.onnx
///     in  input_features [B, 128, 3000] float32   (30-second chunks, padded)
///     out last_hidden_state [B, 1500, 1280] float32
///
/// Files are downloaded from the onnx-community pre-export — see
/// <c>docs/whisper_turbo_investigation.md</c> for why we're not running our
/// own exporter.
/// </summary>
public sealed class WhisperTurbo : IDisposable
{
    // ── File layout (download manifest lives in ModelManagerService) ────────
    public const string EncoderFile            = "encoder_model_fp16.onnx";
    public const string DecoderInitFile        = "decoder_model_fp16.onnx";
    public const string DecoderStepFile        = "decoder_with_past_model_fp16.onnx";
    public const string TokenizerFile          = "tokenizer.json";
    public const string ConfigFile             = "config.json";
    public const string GenerationConfigFile   = "generation_config.json";
    public const string PreprocessorConfigFile = "preprocessor_config.json";
    public const string ModelName              = "openai/whisper-large-v3-turbo";

    // ── DSP constants (match preprocessor_config.json) ──────────────────────
    private const int   SampleRate   = 16_000;
    private const int   NFft         = 400;
    private const int   HopLength    = 160;
    private const int   NMels        = 128;
    private const int   ChunkSeconds = 30;
    public  const int   ChunkSamples = ChunkSeconds * SampleRate;  // 480 000
    public  const int   ChunkFrames  = ChunkSamples / HopLength;   // 3 000
    private const float LogFloor     = -10.0f;
    private const float LogClampSpan = 8.0f;
    private const float LogOffset    = 4.0f;

    // ── Encoder output shape (fixed for 30 s chunks) ────────────────────────
    public const int EncoderOutFrames = ChunkFrames / 2;  // 1500, conv 2× downsampling
    public const int HiddenSize       = 1280;

    // ── Static precomputed ──────────────────────────────────────────────────
    // Note: Qwen3Asr.cs has an identical Whisper-style mel frontend inline.
    // Dedup deferred — touching Qwen3 now would risk regressing a validated
    // backend for the sake of a backend that hasn't shipped yet. Revisit
    // after Phase 6 validation lands.
    private static readonly float[,] MelFilterbank = CreateMelFilterbank();
    private static readonly double[] HannWindow    = Window.HannPeriodic(NFft);

    private readonly InferenceSession _encoder;

    public WhisperTurbo(string modelsDir, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = MakeSessionOptions(ep);
        _encoder = new InferenceSession(Path.Combine(modelsDir, EncoderFile), opts);
    }

    public void Dispose() => _encoder.Dispose();

    // ── Public API ──────────────────────────────────────────────────────────

    /// <summary>
    /// Prepare a chunk-sized log-mel spectrogram ready to feed the encoder.
    /// Zero-pads (or truncates) the waveform to 30 s first — matches the
    /// HuggingFace WhisperFeatureExtractor convention, which avoids the
    /// edge-reflection artefact you'd get from padding the mel frames
    /// afterwards.
    ///
    /// Returns a flat float[<see cref="NMels"/> * <see cref="ChunkFrames"/>]
    /// in row-major <c>[mel, frame]</c> order.
    /// </summary>
    public static float[] PrepareChunkMel(float[] audio16k)
    {
        var chunk = new float[ChunkSamples];
        int copy  = Math.Min(audio16k.Length, ChunkSamples);
        Array.Copy(audio16k, chunk, copy);
        var (mel, frames) = ComputeLogMel(chunk, 0, ChunkSamples);
        if (frames != ChunkFrames)
            throw new InvalidOperationException(
                $"Expected {ChunkFrames} mel frames for {ChunkSamples}-sample input, got {frames}.");
        return mel;
    }

    /// <summary>
    /// Run the encoder on a padded chunk mel (<c>float[128 * 3000]</c>).
    /// Returns last_hidden_state as a flat float[1 * 1500 * 1280] in
    /// row-major <c>[batch, encoder_frame, hidden]</c> order.
    /// </summary>
    public float[] RunEncoder(float[] paddedMel)
    {
        if (paddedMel.Length != NMels * ChunkFrames)
            throw new ArgumentException(
                $"Expected mel length {NMels * ChunkFrames}, got {paddedMel.Length}.",
                nameof(paddedMel));

        var input = new DenseTensor<float>(paddedMel, [1, NMels, ChunkFrames]);
        using var outputs = _encoder.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_features", input),
        ]);
        var tensor = outputs.First(o => o.Name == "last_hidden_state").AsTensor<float>();
        if (tensor is DenseTensor<float> dense)
            return dense.Buffer.ToArray();
        var result = new float[tensor.Length];
        int i = 0;
        foreach (float v in tensor) result[i++] = v;
        return result;
    }

    // ── DSP internals ───────────────────────────────────────────────────────

    /// <summary>
    /// Whisper log-mel spectrogram (Hann STFT → power → Slaney-mel → log10 →
    /// clamp(max − 8) → (x + 4) / 4).  Output length: <c>keptFrames = max(1,
    /// ((length + pad·2 − n_fft) / hop_length))</c>.
    /// </summary>
    internal static (float[] mel, int frames) ComputeLogMel(float[] signal, int start, int length)
    {
        int pad       = NFft / 2;
        float[] padded = ReflectPad(signal, start, length, pad);
        int frameCount = ((padded.Length - NFft) / HopLength) + 1;
        int keptFrames = Math.Max(frameCount - 1, 1);
        int freqBins   = (NFft / 2) + 1;
        var mel        = new float[NMels * keptFrames];

        Parallel.For(
            0,
            frameCount,
            () => new Complex32[NFft],
            (frame, _, fft) =>
            {
                int startIndex = frame * HopLength;
                Array.Clear(fft, 0, fft.Length);
                for (int i = 0; i < NFft; i++)
                    fft[i] = new Complex32((float)(padded[startIndex + i] * HannWindow[i]), 0f);
                Fourier.Forward(fft, FourierOptions.NoScaling);

                if (frame < keptFrames)
                {
                    for (int m = 0; m < NMels; m++)
                    {
                        double sum = 0;
                        for (int k = 0; k < freqBins; k++)
                        {
                            float re = fft[k].Real;
                            float im = fft[k].Imaginary;
                            sum += MelFilterbank[m, k] * (re * re + im * im);
                        }
                        mel[m * keptFrames + frame] = MathF.Log10(MathF.Max((float)sum, 1e-10f));
                    }
                }
                return fft;
            },
            _ => { });

        float maxLog = float.NegativeInfinity;
        for (int i = 0; i < mel.Length; i++)
            if (mel[i] > maxLog) maxLog = mel[i];

        float floor = MathF.Max(maxLog - LogClampSpan, LogFloor);
        for (int i = 0; i < mel.Length; i++)
            mel[i] = (MathF.Max(mel[i], floor) + LogOffset) / LogOffset;

        return (mel, keptFrames);
    }

    private static float[] ReflectPad(float[] signal, int start, int length, int pad)
    {
        if (length == 0) return new float[pad * 2];
        var padded = new float[length + pad * 2];
        Array.Copy(signal, start, padded, pad, length);
        for (int i = 0; i < pad; i++)
        {
            int leftSrc  = Math.Min(length - 1, pad - i);
            int rightSrc = Math.Max(0, length - 2 - i);
            padded[i]                     = signal[start + leftSrc];
            padded[pad + length + i]      = signal[start + rightSrc];
        }
        return padded;
    }

    private static float[,] CreateMelFilterbank()
    {
        int freqBins = (NFft / 2) + 1;
        var fb       = new float[NMels, freqBins];
        var fftFreqs = new double[freqBins];
        for (int k = 0; k < freqBins; k++)
            fftFreqs[k] = (double)k * SampleRate / NFft;

        double fminMel = AudioUtils.HzToMelSlaney(0.0);
        double fmaxMel = AudioUtils.HzToMelSlaney(SampleRate / 2.0);
        var melF = new double[NMels + 2];
        for (int i = 0; i <= NMels + 1; i++)
        {
            double m = fminMel + (fmaxMel - fminMel) * i / (NMels + 1);
            melF[i]  = AudioUtils.MelToHzSlaney(m);
        }

        var fdiff = new double[NMels + 1];
        for (int i = 0; i <= NMels; i++)
            fdiff[i] = melF[i + 1] - melF[i];

        for (int i = 0; i < NMels; i++)
        {
            for (int k = 0; k < freqBins; k++)
            {
                double lower = (fftFreqs[k] - melF[i])     / fdiff[i];
                double upper = (melF[i + 2] - fftFreqs[k]) / fdiff[i + 1];
                fb[i, k]     = (float)Math.Max(0.0, Math.Min(lower, upper));
            }
        }

        for (int i = 0; i < NMels; i++)
        {
            float enorm = (float)(2.0 / (melF[i + 2] - melF[i]));
            for (int k = 0; k < freqBins; k++)
                fb[i, k] *= enorm;
        }
        return fb;
    }

    // ── Session options ─────────────────────────────────────────────────────

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
                opts.AppendExecutionProvider_CUDA(0);
                break;
            case ExecutionProvider.DirectML:
                opts.AppendExecutionProvider_DML(0);
                break;
            case ExecutionProvider.Cpu:
                break;
        }
        return opts;
    }
}
