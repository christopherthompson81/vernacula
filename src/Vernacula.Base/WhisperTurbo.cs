using System.Text;
using System.Text.Json;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Base;

public readonly record struct WhisperTranscript(string Text, IReadOnlyList<int> Tokens);

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

    // ── Decoder geometry (large-v3-turbo) ───────────────────────────────────
    private const int NumDecoderLayers = 4;      // large-v3-turbo distillation
    private const int NumHeads         = 20;
    private const int HeadDim          = 64;     // 20 * 64 = 1280
    private const int MaxDecoderLength = 448;    // Whisper hard context limit

    // ── Whisper special tokens (from generation_config.json) ────────────────
    // Regular BPE vocab spans [0, 50256]. All special tokens are >= 50257.
    private const int SotToken           = 50258;  // <|startoftranscript|>
    private const int EotToken           = 50257;  // <|endoftext|> (also BOS/EOS/pad)
    private const int NoTimestampsToken  = 50364;
    private const int TranscribeToken    = 50360;
    private const int TranslateToken     = 50359;
    private const int SpecialTokenFloor  = 50257;  // everything >= this is a special token
    // Suppress these at the FIRST content step only: 220 = leading space, 50257 = immediate EOT.
    // Prevents the two degenerate argmax outcomes in the initial decode position.
    private static readonly int[] BeginSuppressTokens = [220, EotToken];

    // Language ISO → Whisper <|lang|> token id. Populated from
    // generation_config.json's lang_to_id dict on load.
    private readonly Dictionary<string, int> _langToId;

    // ── Static precomputed ──────────────────────────────────────────────────
    // Note: Qwen3Asr.cs has an identical Whisper-style mel frontend inline.
    // Dedup deferred — touching Qwen3 now would risk regressing a validated
    // backend for the sake of a backend that hasn't shipped yet. Revisit
    // after Phase 6 validation lands.
    private static readonly float[,] MelFilterbank = CreateMelFilterbank();
    private static readonly double[] HannWindow    = Window.HannPeriodic(NFft);

    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoderInit;
    private readonly InferenceSession _decoderStep;
    private readonly string?[] _idToToken;
    private readonly Dictionary<char, byte> _byteLevelDecode;

    public WhisperTurbo(string modelsDir, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = MakeSessionOptions(ep);
        _encoder     = new InferenceSession(Path.Combine(modelsDir, EncoderFile), opts);
        _decoderInit = new InferenceSession(Path.Combine(modelsDir, DecoderInitFile), opts);
        _decoderStep = new InferenceSession(Path.Combine(modelsDir, DecoderStepFile), opts);

        _idToToken       = LoadTokenizerVocab(Path.Combine(modelsDir, TokenizerFile));
        _byteLevelDecode = BuildByteLevelDecode();
        _langToId        = LoadLangToId(Path.Combine(modelsDir, GenerationConfigFile));
    }

    public void Dispose()
    {
        _decoderStep.Dispose();
        _decoderInit.Dispose();
        _encoder.Dispose();
    }

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
        return ExtractFloat(outputs.First(o => o.Name == "last_hidden_state"));
    }

    /// <summary>
    /// Transcribe a single 30-second chunk (or less) of 16 kHz mono audio
    /// with greedy decode. Longer inputs are truncated to the first 30 s —
    /// multi-chunk handling lands in a later phase.
    /// <paramref name="languageIso"/> is an ISO 639-1 code; <c>"en"</c> by default.
    /// </summary>
    public WhisperTranscript Transcribe(float[] audio16k, string languageIso = "en")
    {
        float[] mel    = PrepareChunkMel(audio16k);
        float[] hidden = RunEncoder(mel);

        int langToken = ResolveLanguageToken(languageIso);
        int[] prefix  = [SotToken, langToken, TranscribeToken, NoTimestampsToken];

        // ── Decoder-init: consumes the full prefix in one pass, produces
        //     initial KV cache (cross-attn encoder KV is computed here and
        //     reused unchanged across every subsequent step — matches the
        //     Cohere cross-KV reuse pattern). ──────────────────────────────
        var (initLogits, kv) = RunDecoderInit(prefix, hidden);

        // First content token: argmax over logits at the last prefix position,
        // with BeginSuppressTokens masked to -inf to prevent the two degenerate
        // greedy outcomes (leading space or immediate EOT).
        int nextToken = ArgmaxLastPosition(initLogits, prefix.Length, BeginSuppressTokens);

        // ── Step loop: each call takes one token + the grown KV cache,
        //     returns logits at the single new position and an updated KV. ─
        var outTokens = new List<int>(64);
        while (nextToken != EotToken && outTokens.Count + prefix.Length < MaxDecoderLength)
        {
            outTokens.Add(nextToken);
            (float[] stepLogits, kv) = RunDecoderStep(nextToken, kv);
            nextToken = ArgmaxLastPosition(stepLogits, 1, suppress: null);
        }

        // Filter out any stray special tokens that slipped through; BPE-decode
        // the remainder through the byte-level map.
        var contentTokens = outTokens.Where(t => t < SpecialTokenFloor).ToList();
        string text = DecodeTokens(contentTokens);
        return new WhisperTranscript(text, outTokens);
    }

    // ── Decoder helpers ─────────────────────────────────────────────────────

    /// <summary>
    /// One KV-cache tensor. Kept as flat float[] + explicit shape so we can
    /// round-trip through DenseTensor&lt;float&gt; each step without
    /// overcomplicating storage. 16 of these per decoder state (4 layers × 4
    /// tensors per layer).
    /// </summary>
    private sealed record KvTensor(string Name, float[] Data, int[] Shape);

    private (float[] logits, List<KvTensor> kv)
        RunDecoderInit(int[] prefix, float[] encoderHidden)
    {
        var inputIds = new long[prefix.Length];
        for (int i = 0; i < prefix.Length; i++) inputIds[i] = prefix[i];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<long>(inputIds, [1, prefix.Length])),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states",
                new DenseTensor<float>(encoderHidden, [1, EncoderOutFrames, HiddenSize])),
        };

        using var outputs = _decoderInit.Run(inputs);

        float[] logits = ExtractFloat(outputs.First(o => o.Name == "logits"));

        // Build the KV cache in the exact order the step model expects as
        // past_key_values.N.{decoder|encoder}.{key|value}. All 16 tensors.
        var kv = new List<KvTensor>(NumDecoderLayers * 4);
        for (int layer = 0; layer < NumDecoderLayers; layer++)
        {
            foreach (bool encoder in new[] { false, true })
            {
                foreach (bool isValue in new[] { false, true })
                {
                    string presentName = $"present.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                    string pastName    = $"past_key_values.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                    var tensor = outputs.First(o => o.Name == presentName).AsTensor<float>();
                    int seqLen = encoder ? EncoderOutFrames : prefix.Length;
                    kv.Add(new KvTensor(pastName, ExtractFloatFromTensor(tensor),
                                        [1, NumHeads, seqLen, HeadDim]));
                }
            }
        }
        return (logits, kv);
    }

    private (float[] logits, List<KvTensor> updatedKv)
        RunDecoderStep(int nextToken, List<KvTensor> pastKv)
    {
        var inputs = new List<NamedOnnxValue>(1 + pastKv.Count)
        {
            NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<long>(new long[] { nextToken }, [1, 1])),
        };
        foreach (var t in pastKv)
            inputs.Add(NamedOnnxValue.CreateFromTensor(t.Name, new DenseTensor<float>(t.Data, t.Shape)));

        using var outputs = _decoderStep.Run(inputs);

        float[] logits = ExtractFloat(outputs.First(o => o.Name == "logits"));

        // Step outputs only decoder KVs (8 tensors) — encoder KVs are constant
        // across steps and stay in pastKv. Rebuild the 16-entry list with the
        // grown decoder KVs and the unchanged encoder KVs.
        var updatedKv = new List<KvTensor>(NumDecoderLayers * 4);
        for (int layer = 0; layer < NumDecoderLayers; layer++)
        {
            foreach (bool encoder in new[] { false, true })
            {
                foreach (bool isValue in new[] { false, true })
                {
                    string pastName = $"past_key_values.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                    if (encoder)
                    {
                        // Reuse unchanged encoder KV from the previous state.
                        updatedKv.Add(pastKv.First(t => t.Name == pastName));
                    }
                    else
                    {
                        string presentName = $"present.{layer}.decoder.{(isValue ? "value" : "key")}";
                        var tensor = outputs.First(o => o.Name == presentName).AsTensor<float>();
                        // Decoder seq grew by 1. Derive from tensor dims rather than
                        // tracking separately — avoids a second source of truth.
                        var dims = tensor.Dimensions;
                        updatedKv.Add(new KvTensor(pastName, ExtractFloatFromTensor(tensor),
                                                    [dims[0], dims[1], dims[2], dims[3]]));
                    }
                }
            }
        }
        return (logits, updatedKv);
    }

    /// <summary>
    /// Argmax over the vocab at the last token position of a logits tensor
    /// shaped [1, seqLen, vocabSize]. Optional <paramref name="suppress"/>
    /// ids get -inf before argmax.
    /// </summary>
    private static int ArgmaxLastPosition(float[] logits, int seqLen, int[]? suppress)
    {
        int vocab = logits.Length / seqLen;
        int baseIdx = (seqLen - 1) * vocab;

        if (suppress is not null)
            foreach (int id in suppress)
                if (id >= 0 && id < vocab)
                    logits[baseIdx + id] = float.NegativeInfinity;

        int best = 0;
        float bestVal = logits[baseIdx];
        for (int i = 1; i < vocab; i++)
        {
            float v = logits[baseIdx + i];
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    private int ResolveLanguageToken(string languageIso)
    {
        string key = $"<|{languageIso.ToLowerInvariant()}|>";
        if (_langToId.TryGetValue(key, out int id)) return id;
        throw new ArgumentException(
            $"Unsupported Whisper language code '{languageIso}'. "
            + $"Expected one of the {_langToId.Count} languages in generation_config.json.");
    }

    // ── Tokenizer ───────────────────────────────────────────────────────────

    /// <summary>
    /// BPE-decode a token-id sequence to UTF-8 text. Regular vocab tokens
    /// (id &lt; 50257) get byte-level-BPE mapped; higher ids are Whisper
    /// specials and should be filtered by the caller before calling this.
    /// </summary>
    public string DecodeTokens(IReadOnlyList<int> tokens)
    {
        var bytes = new List<byte>(tokens.Count * 4);
        foreach (int token in tokens)
        {
            if (token < 0 || token >= _idToToken.Length) continue;
            string? raw = _idToToken[token];
            if (raw is null) continue;
            foreach (char ch in raw)
                if (_byteLevelDecode.TryGetValue(ch, out byte value))
                    bytes.Add(value);
        }
        string text = Encoding.UTF8.GetString(bytes.ToArray());
        return text.Length > 0 && text[0] == ' ' ? text[1..] : text;
    }

    /// <summary>
    /// Load tokenizer.json into a flat id → token string lookup. Handles both
    /// the regular BPE vocab under model.vocab and the added_tokens list
    /// (which is where all Whisper specials live).
    /// </summary>
    private static string?[] LoadTokenizerVocab(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var vocab = root.GetProperty("model").GetProperty("vocab");
        int maxId = -1;
        foreach (var kv in vocab.EnumerateObject())
            maxId = Math.Max(maxId, kv.Value.GetInt32());
        if (root.TryGetProperty("added_tokens", out var addedTokens))
            foreach (var tok in addedTokens.EnumerateArray())
                maxId = Math.Max(maxId, tok.GetProperty("id").GetInt32());

        var idToToken = new string?[maxId + 1];
        foreach (var kv in vocab.EnumerateObject())
            idToToken[kv.Value.GetInt32()] = kv.Name;
        if (root.TryGetProperty("added_tokens", out var added))
            foreach (var tok in added.EnumerateArray())
                idToToken[tok.GetProperty("id").GetInt32()] =
                    tok.GetProperty("content").GetString() ?? "";
        return idToToken;
    }

    /// <summary>
    /// GPT-2 byte-level BPE decode map: unicode surrogate char → underlying byte.
    /// Mirrors the canonical bytes_to_unicode table from OpenAI's GPT-2 repo,
    /// which Whisper inherits. Same as Qwen3Asr's copy; they use the same
    /// pre-tokenizer family.
    /// </summary>
    private static Dictionary<char, byte> BuildByteLevelDecode()
    {
        var bs = new List<int>();
        for (int i = (int)'!'; i <= (int)'~'; i++) bs.Add(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);
        var cs = new List<int>(bs);
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (bs.Contains(b)) continue;
            bs.Add(b);
            cs.Add(256 + extra);
            extra++;
        }
        var map = new Dictionary<char, byte>(256);
        for (int i = 0; i < bs.Count; i++)
            map[(char)cs[i]] = (byte)bs[i];
        return map;
    }

    private static Dictionary<string, int> LoadLangToId(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var dict = new Dictionary<string, int>(StringComparer.Ordinal);
        if (root.TryGetProperty("lang_to_id", out var langToId))
            foreach (var kv in langToId.EnumerateObject())
                dict[kv.Name] = kv.Value.GetInt32();
        return dict;
    }

    // ── Tensor extraction helpers ───────────────────────────────────────────

    private static float[] ExtractFloat(DisposableNamedOnnxValue o)
        => ExtractFloatFromTensor(o.AsTensor<float>());

    private static float[] ExtractFloatFromTensor(Tensor<float> tensor)
    {
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
