using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// Cohere Transcribe 03-2026 — encoder-decoder ASR with KV-cache greedy decoder.
///
/// Uses four ONNX models in the model directory:
///   mel.onnx          waveforms [1,T] → features [1,128,F]
///   encoder.onnx      features [1,128,F] → encoder_hidden_states [1,T',1280]
///   decoder_init.onnx BOS token + enc_hidden → logits + 32 KV tensors
///   decoder_step.onnx single token + past self-KV + fixed cross-KV → next logit + updated self-KV
///
/// Vocab is loaded from vocab.json (array of 16384 token strings indexed by ID).
/// </summary>
public sealed class CohereTranscribe : IDisposable
{
    private const int NumLayers = 8;
    private const int NumHeads  = 8;
    private const int HeadDim   = 128;

    public const string MelFile         = "mel.onnx";
    public const string EncoderFile     = "encoder.onnx";
    public const string DecoderInitFile = "decoder_init.onnx";
    public const string DecoderStepFile = "decoder_step.onnx";
    public const string VocabFile       = "vocab.json";
    public const string ConfigFile      = "config.json";

    private readonly InferenceSession _mel;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoderInit;
    private readonly InferenceSession _decoderStep;

    private readonly string[] _vocab;      // index → token string
    private readonly int _bosTokenId;      // decoder_start_token_id (13764)
    private readonly int _eosTokenId;      // eos_token_id (3)
    private readonly int _padTokenId;      // pad_token_id (2)
    private readonly int _startTokenId;    // bos_token_id (4)

    // Byte-fallback tokens occupy indices 255..510 (0x00..0xFF).
    private const int ByteFallbackOffset = 255;

    // Language token ID range in the vocab.
    private const int LangIdMin = 22;
    private const int LangIdMax = 204;

    // The Cohere conformer encoder specializes batch=1 internally — encoder
    // batching is not supported by this model architecture.

    // ── Construction ─────────────────────────────────────────────────────────

    public CohereTranscribe(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var cpuOpts = new SessionOptions();
        _mel = new InferenceSession(Path.Combine(modelPath, MelFile), cpuOpts);

        var gpuOpts = MakeSessionOptions(ep);
        _encoder      = new InferenceSession(Path.Combine(modelPath, EncoderFile),     gpuOpts);
        _decoderInit  = new InferenceSession(Path.Combine(modelPath, DecoderInitFile), gpuOpts);
        _decoderStep  = new InferenceSession(Path.Combine(modelPath, DecoderStepFile), gpuOpts);

        // Load vocab
        string vocabJson = File.ReadAllText(Path.Combine(modelPath, VocabFile));
        _vocab = JsonSerializer.Deserialize<string[]>(vocabJson)
            ?? throw new InvalidDataException("Failed to deserialize vocab.json");

        // Load token IDs from config
        string cfgJson = File.ReadAllText(Path.Combine(modelPath, ConfigFile));
        using var doc  = JsonDocument.Parse(cfgJson);
        var root       = doc.RootElement;
        _bosTokenId    = root.GetProperty("decoder_start_token_id").GetInt32();
        _eosTokenId    = root.GetProperty("eos_token_id").GetInt32();
        _padTokenId    = root.GetProperty("pad_token_id").GetInt32();
        _startTokenId  = root.GetProperty("bos_token_id").GetInt32();
    }

    private static SessionOptions MakeSessionOptions(ExecutionProvider ep)
    {
        var opts = new SessionOptions();
        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                {
                    try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                }
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
                { throw new InvalidOperationException("DirectML EP not available."); }
                break;
            case ExecutionProvider.Cpu:
                break;
        }
        return opts;
    }

    // ── Language token lookup ─────────────────────────────────────────────────

    /// <summary>
    /// Returns the vocab token ID for an ISO 639-1 language code (e.g. "en", "fr"),
    /// or -1 if the language is not in the vocab.
    /// </summary>
    public int LookupLanguageTokenId(string isoCode)
    {
        string tag = $"<|{isoCode.ToLowerInvariant()}|>";
        for (int i = LangIdMin; i <= LangIdMax; i++)
            if (i < _vocab.Length && _vocab[i] == tag)
                return i;
        return -1;
    }

    // ── Inference pipeline ───────────────────────────────────────────────────

    /// <summary>
    /// Mel preprocessing for a single waveform: waveforms [1,T] → features [1,128,F].
    /// Returns the flat mel features and the valid frame count.
    /// </summary>
    private (float[] features, int nMels, int F) RunMel(float[] waveform)
    {
        int T = waveform.Length;
        var waveT = new DenseTensor<float>(waveform, new[] { 1, T });
        var waveL = new DenseTensor<long>(new long[] { T }, new[] { 1 });

        using var results = _mel.Run(
        [
            NamedOnnxValue.CreateFromTensor("waveforms",      waveT),
            NamedOnnxValue.CreateFromTensor("waveforms_lens", waveL),
        ]);

        var featT = results.First(r => r.Name == "features").AsTensor<float>();
        int nMels = featT.Dimensions[1];
        int F     = featT.Dimensions[2];

        var features = new float[nMels * F];
        for (int m = 0; m < nMels; m++)
            for (int f = 0; f < F; f++)
                features[m * F + f] = featT[0, m, f];

        return (features, nMels, F);
    }

    /// <summary>
    /// Encoder: mel features [1, 128, F] → encoder_hidden_states [1, T', 1280].
    /// Returns the hidden states flat [T', dModel] and T'.
    /// </summary>
    private (float[] hidden, int T_enc) RunEncoder(float[] features, int nMels, int F)
    {
        var featT = new DenseTensor<float>(features, new[] { 1, nMels, F });

        using var results = _encoder.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_features", featT),
        ]);

        var hidT  = results.First(r => r.Name == "encoder_hidden_states").AsTensor<float>();
        int T_enc = hidT.Dimensions[1];
        int dMod  = hidT.Dimensions[2];

        var hidden = new float[T_enc * dMod];
        for (int t = 0; t < T_enc; t++)
            for (int d = 0; d < dMod; d++)
                hidden[t * dMod + d] = hidT[0, t, d];

        return (hidden, T_enc);
    }

    /// <summary>
    /// Greedy KV-cache decoder.
    /// Returns the full token list including decoder_start token.
    /// When <paramref name="forcedLangTokenId"/> is ≥ 0, any context-block token in the
    /// language ID range (22–204) is replaced with the forced language token ID so that
    /// all subsequent decoding is conditioned on the specified language.
    /// </summary>
    private List<int> GreedyDecode(float[] encoderHidden, int encT,
                                   int maxTokens = 256, int forcedLangTokenId = -1)
    {
        const int dModel = 1280;
        var encT_tensor = new DenseTensor<float>(encoderHidden, new[] { 1, encT, dModel });
        var bosT        = new DenseTensor<long>(new long[] { _bosTokenId }, new[] { 1, 1 });

        // ── Decoder init ─────────────────────────────────────────────────────
        float[] logits;
        float[][] selfKey  = new float[NumLayers][];
        float[][] selfVal  = new float[NumLayers][];
        float[][] crossKey = new float[NumLayers][];
        float[][] crossVal = new float[NumLayers][];

        using (var initResults = _decoderInit.Run(
        [
            NamedOnnxValue.CreateFromTensor("decoder_input_ids",    bosT),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encT_tensor),
        ]))
        {
            // Outputs: logits, sk0..7, sv0..7, ck0..7, cv0..7
            logits = ExtractLogits(initResults, "logits");

            for (int i = 0; i < NumLayers; i++)
            {
                selfKey[i]  = ExtractTensorFlat(initResults, $"self_key_{i}");
                selfVal[i]  = ExtractTensorFlat(initResults, $"self_val_{i}");
                crossKey[i] = ExtractTensorFlat(initResults, $"cross_key_{i}");
                crossVal[i] = ExtractTensorFlat(initResults, $"cross_val_{i}");
            }
        }

        int firstToken = ForceLang(ArgMax(logits), forcedLangTokenId);
        var tokens     = new List<int>(maxTokens + 2) { _bosTokenId, firstToken };
        if (firstToken == _eosTokenId)
            return tokens;

        int tPast = 1;  // number of tokens in self-KV (= 1 after init)

        // ── Step loop ────────────────────────────────────────────────────────
        for (int step = 1; step < maxTokens; step++)
        {
            int lastToken = tokens[tokens.Count - 1];
            var inputs = BuildStepInputs(lastToken, step, tPast, encT,
                                         selfKey, selfVal, crossKey, crossVal);

            using var stepResults = _decoderStep.Run(inputs);

            logits = ExtractLogits(stepResults, "logits");
            tPast++;

            for (int i = 0; i < NumLayers; i++)
            {
                selfKey[i] = ExtractTensorFlat(stepResults, $"new_self_key_{i}");
                selfVal[i] = ExtractTensorFlat(stepResults, $"new_self_val_{i}");
            }

            int nextToken = ForceLang(ArgMax(logits), forcedLangTokenId);
            tokens.Add(nextToken);
            if (nextToken == _eosTokenId)
                break;
        }

        return tokens;
    }

    // If the decoded token is a language tag and we have a forced language, substitute it.
    private static int ForceLang(int token, int forcedLangTokenId) =>
        forcedLangTokenId >= 0 && token >= LangIdMin && token <= LangIdMax
            ? forcedLangTokenId
            : token;

    private List<NamedOnnxValue> BuildStepInputs(
        int lastToken, int position, int tPast, int encT,
        float[][] selfKey, float[][] selfVal,
        float[][] crossKey, float[][] crossVal)
    {
        var inputs = new List<NamedOnnxValue>(2 + NumLayers * 4);

        inputs.Add(NamedOnnxValue.CreateFromTensor("decoder_input_ids",
            new DenseTensor<long>(new long[] { lastToken }, new[] { 1, 1 })));
        inputs.Add(NamedOnnxValue.CreateFromTensor("positions",
            new DenseTensor<long>(new long[] { position }, new[] { 1, 1 })));

        for (int i = 0; i < NumLayers; i++)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor($"self_key_{i}",
                new DenseTensor<float>(selfKey[i], new[] { 1, NumHeads, tPast, HeadDim })));
            inputs.Add(NamedOnnxValue.CreateFromTensor($"self_val_{i}",
                new DenseTensor<float>(selfVal[i], new[] { 1, NumHeads, tPast, HeadDim })));
        }
        for (int i = 0; i < NumLayers; i++)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor($"cross_key_{i}",
                new DenseTensor<float>(crossKey[i], new[] { 1, NumHeads, encT, HeadDim })));
            inputs.Add(NamedOnnxValue.CreateFromTensor($"cross_val_{i}",
                new DenseTensor<float>(crossVal[i], new[] { 1, NumHeads, encT, HeadDim })));
        }

        return inputs;
    }

    // ── Token decoding ───────────────────────────────────────────────────────

    /// <summary>
    /// Decodes a list of token IDs to a UTF-8 string using the BPE vocab.
    ///
    /// Skips special tokens, handles byte-fallback tokens (&lt;0xNN&gt;),
    /// and converts the SentencePiece word-boundary marker '▁' to space.
    /// </summary>
    public string DecodeTokens(IReadOnlyList<int> tokens)
    {
        var bytes = new List<byte>(tokens.Count * 4);

        foreach (int id in tokens)
        {
            if (id < 0 || id >= _vocab.Length) continue;
            // IDs 0..254 are all special/control tokens (language tags, emotion, etc.) — skip.
            // ID ByteFallbackOffset (13764) is decoder_start — skip.
            if (id < ByteFallbackOffset || id == _bosTokenId) continue;

            string token = _vocab[id];

            // Byte-fallback: <0xNN> → single byte 0xNN
            if (token.Length == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x'
                && token[5] == '>')
            {
                if (TryParseHexByte(token[3], token[4], out byte b))
                {
                    bytes.Add(b);
                    continue;
                }
            }

            // Normal BPE token: replace ▁ (U+2581) with space, then UTF-8 encode
            string expanded = token.Replace('\u2581', ' ');
            byte[] tokenBytes = Encoding.UTF8.GetBytes(expanded);
            bytes.AddRange(tokenBytes);
        }

        string text = Encoding.UTF8.GetString(bytes.ToArray());

        // Strip up to one leading space (mirrors HF tokenizer's Strip decoder)
        if (text.Length > 0 && text[0] == ' ')
            text = text[1..];

        return text;
    }

    private static bool TryParseHexByte(char hi, char lo, out byte value)
    {
        int h = HexVal(hi);
        int l = HexVal(lo);
        if (h < 0 || l < 0) { value = 0; return false; }
        value = (byte)((h << 4) | l);
        return true;
    }

    private static int HexVal(char c) => c switch
    {
        >= '0' and <= '9' => c - '0',
        >= 'a' and <= 'f' => c - 'a' + 10,
        >= 'A' and <= 'F' => c - 'A' + 10,
        _ => -1,
    };

    // ── Context-block parsing ─────────────────────────────────────────────────

    /// <summary>
    /// Extracts structured metadata from the context tokens the model prepends
    /// before the text (IDs 0–254: language tags, emotion, formatting flags).
    /// </summary>
    private CohereSegmentMeta ParseContextBlock(IReadOnlyList<int> tokens)
    {
        string? language   = null;
        string? emotion    = null;
        bool?   pnc        = null;
        bool?   itn        = null;
        bool?   timestamps = null;
        bool?   diarize    = null;

        foreach (int id in tokens)
        {
            if (id <= 0 || id >= ByteFallbackOffset) continue;  // skip BOS, text, byte-fallback

            switch (id)
            {
                case  5: pnc        = true;        break;
                case  6: pnc        = false;       break;
                case  8: itn        = true;        break;
                case  9: itn        = false;       break;
                case 10: timestamps = true;        break;
                case 11: timestamps = false;       break;
                case 12: diarize    = true;        break;
                case 13: diarize    = false;       break;
                case 16: emotion    = "undefined"; break;
                case 17: emotion    = "neutral";   break;
                case 18: emotion    = "happy";     break;
                case 19: emotion    = "sad";       break;
                case 20: emotion    = "angry";     break;
                case 21: language ??= "unknown";   break;
                default:
                    // Language tokens 22–204: vocab string is "<|xx|>" → extract "xx"
                    if (id >= 22 && id <= 204 && language is null)
                    {
                        string tok = _vocab[id];
                        if (tok.Length > 4 && tok.StartsWith("<|") && tok.EndsWith("|>"))
                            language = tok[2..^2];
                    }
                    break;
            }
        }

        return new CohereSegmentMeta(language, emotion, pnc, itn, timestamps, diarize);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Transcribes each segment from <paramref name="segs"/> and yields
    /// <c>(segId, text, meta)</c> in order as each segment completes.
    /// </summary>
    /// <param name="forceLanguage">
    /// Optional ISO 639-1 language code (e.g. "en").  When set, any language token
    /// the model would emit during context-block decoding is replaced with this
    /// language's token, conditioning the rest of the decode on the specified language.
    /// </param>
    public IEnumerable<(int segId, string text, CohereSegmentMeta meta)> Recognize(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256,
        string? forceLanguage = null)
    {
        int forcedLangTokenId = -1;
        if (forceLanguage is not null)
        {
            forcedLangTokenId = LookupLanguageTokenId(forceLanguage);
            if (forcedLangTokenId < 0)
                throw new ArgumentException(
                    $"Language '{forceLanguage}' not found in Cohere vocab. " +
                    "Use an ISO 639-1 code such as 'en', 'fr', 'de'.");
        }

        for (int i = 0; i < segs.Count; i++)
        {
            var (start, end, _) = segs[i];
            float[] waveform = ExtractSegment(audio, start, end);

            if (waveform.Length < Config.SampleRate / 10)  // < 100 ms
            {
                yield return (i, string.Empty, CohereSegmentMeta.Empty);
                continue;
            }

            var (features, nMels, F) = RunMel(waveform);
            var (encHidden, encT)    = RunEncoder(features, nMels, F);
            var tokens               = GreedyDecode(encHidden, encT, maxNewTokens, forcedLangTokenId);
            var meta                 = ParseContextBlock(tokens);
            string text              = DecodeTokens(tokens);

            yield return (i, text, meta);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static float[] ExtractSegment(float[] audio, double start, double end)
    {
        int s = Math.Max((int)(start * Config.SampleRate), 0);
        int e = Math.Min((int)(end   * Config.SampleRate), audio.Length);
        int len = Math.Max(e - s, 0);
        var seg = new float[len];
        if (len > 0) Array.Copy(audio, s, seg, 0, len);
        return seg;
    }

    private static float[] ExtractLogits(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        string name)
    {
        var tensor = results.First(r => r.Name == name).AsTensor<float>();
        // shape [1, 1, vocab_size] — return the last (only) position
        int vocabSize = tensor.Dimensions[2];
        var logits = new float[vocabSize];
        for (int v = 0; v < vocabSize; v++)
            logits[v] = tensor[0, 0, v];
        return logits;
    }

    private static float[] ExtractTensorFlat(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        string name)
    {
        var tensor = results.First(r => r.Name == name).AsTensor<float>();
        int total = 1;
        for (int d = 0; d < tensor.Rank; d++)
            total *= tensor.Dimensions[d];
        var flat = new float[total];
        for (int i = 0; i < total; i++)
            flat[i] = tensor.GetValue(i);
        return flat;
    }

    private static int ArgMax(float[] arr)
    {
        int idx = 0;
        float max = float.NegativeInfinity;
        for (int i = 0; i < arr.Length; i++)
            if (arr[i] > max) { max = arr[i]; idx = i; }
        return idx;
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose()
    {
        _mel.Dispose();
        _encoder.Dispose();
        _decoderInit.Dispose();
        _decoderStep.Dispose();
    }
}

/// <summary>
/// Structured metadata extracted from the Cohere Transcribe context-token block.
/// All fields are nullable — null means the model did not emit a token for that category.
/// </summary>
public sealed record CohereSegmentMeta(
    string? Language,    // ISO 639-1 code e.g. "en", "fr"; "unknown" if <|unklang|>; null if absent
    string? Emotion,     // "neutral" | "happy" | "sad" | "angry" | "undefined" | null
    bool?   Pnc,         // punctuation and capitalisation applied (true) or suppressed (false)
    bool?   Itn,         // inverse text normalisation applied (true) or suppressed (false)
    bool?   Timestamps,  // word-level timestamps in output
    bool?   Diarize      // model's own speaker-change tracking active
)
{
    public static readonly CohereSegmentMeta Empty = new(null, null, null, null, null, null);

    /// <summary>Serialises all fields to a compact JSON string for the asr_meta DB column.</summary>
    public string ToJson() => JsonSerializer.Serialize(new
    {
        language   = Language,
        emotion    = Emotion,
        pnc        = Pnc,
        itn        = Itn,
        timestamps = Timestamps,
        diarize    = Diarize,
    });
}
