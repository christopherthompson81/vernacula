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
///   encoder.onnx      features [B,128,F] → encoder_hidden_states [B,T',1280]
///   decoder_init.onnx BOS [B,1] + enc_hidden [B,T',1280] → logits [B,1,V] + 32 KV tensors
///   decoder_step.onnx tokens [B,1] + past self-KV [B,H,t,d] + fixed cross-KV → logits + updated self-KV
///
/// All four models are batch-dynamic.  Mel runs B=1 serially (per-segment);
/// encoder and decoder run over the full segment batch together.
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

    // Fixed context-block token IDs (from vocab.json).
    // When a language is forced the entire context block is known upfront;
    // decoder_init is called once with a 10-token prefix instead of stepping
    // through the context block one token at a time (saves 9 decoder_step calls).
    private const int TokStartOfContext    = 7;   // <|startofcontext|>
    private const int TokStartOfTranscript = 4;   // <|startoftranscript|>
    private const int TokEmoNeutral        = 17;  // <|emo:neutral|>
    private const int TokPncOn             = 5;   // <|pnc|>
    private const int TokNoItn             = 9;   // <|noitn|>
    private const int TokNoTimestamp       = 11;  // <|notimestamp|>
    private const int TokNoDiarize         = 13;  // <|nodiarize|>

    // ── Batch sizing ─────────────────────────────────────────────────────────
    // Both encoder.onnx and decoder_init/step.onnx are fully batch-dynamic.
    // Batch size is determined dynamically per-batch from the VRAM budget and the
    // estimated peak KV-cache size of the longest segment in that batch.
    //
    // Segments are sorted by ascending audio duration before batching so that
    // segments of similar length land in the same batch, minimising the "straggler"
    // waste where shorter segments keep stepping after they have emitted EOS.

    // Hard upper limit on segments per batch.  The VRAM constraint (EstimateKvBytes vs
    // _vramBudgetForKvBytes) is the primary limiter; this cap handles pathological cases
    // (thousands of very short segments) or CPU-only builds where VRAM estimation falls
    // back to 3 GB.
    //
    // encoder.onnx accepts input_lengths [B] int64 so the Conformer's self-attention masks
    // padded positions per item — segments of different lengths can be freely batched.
    // decoder_init intermediate activations scale as B × encTMax; peak VRAM is estimated
    // conservatively and the budget check prevents OOM.
    private const int MaxBatchSize = 32;

    // VRAM budget for KV-cache tensors AND intermediate activations during the
    // decoder_init forward pass.  Computed at construction time: free GPU VRAM
    // (measured after all four ONNX sessions are loaded) minus a 2 GB safety
    // buffer for CUDA runtime overhead and activation peaks.
    // Falls back to 3 GB if the CUDA query is unavailable (CPU-only builds).
    private const long VramSafetyBufferBytes = 2_000_000_000L;
    private const long VramBudgetFallbackBytes = 3_000_000_000L;
    private readonly long _vramBudgetForKvBytes;

    // Rough token rate used to estimate the maximum decode depth from audio duration.
    // ~8 text tokens/s + context-block overhead; capped by maxNewTokens.
    private const float TokensPerAudioSecond = 10f;

    // Mel spectrogram frame rate and encoder temporal downsampling factor.
    private const float MelFramesPerSec  = 100f;
    private const float EncoderDownsample = 8f;

    private static int EstimateEncFrames(double durSec)
        => Math.Max(1, (int)Math.Ceiling(durSec * MelFramesPerSec / EncoderDownsample));

    private static int EstimateDecSteps(double durSec, int maxNewTokens)
        => Math.Min(maxNewTokens, (int)Math.Ceiling(durSec * TokensPerAudioSecond) + 16);

    // Peak VRAM for KV tensors at the end of a batch decode.
    //   Self-KV:  B × NumLayers × 2 × NumHeads × maxDecSteps × HeadDim × 4 bytes
    //   Cross-KV: B × NumLayers × 2 × NumHeads × maxEncFrames × HeadDim × 4 bytes
    private static long EstimateKvBytes(int batchSize, int maxEncFrames, int maxDecSteps)
    {
        const long bytesPerFloat = 4L;
        return (long)batchSize * NumLayers * 2 * NumHeads * HeadDim * bytesPerFloat
               * (maxDecSteps + maxEncFrames);
    }

    // ── Construction ─────────────────────────────────────────────────────────

    public CohereTranscribe(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var cpuOpts = new SessionOptions();
        _mel = new InferenceSession(Path.Combine(modelPath, MelFile), cpuOpts);

        var gpuOpts = MakeSessionOptions(ep);
        _encoder     = new InferenceSession(Path.Combine(modelPath, EncoderFile),     gpuOpts);
        _decoderInit = new InferenceSession(Path.Combine(modelPath, DecoderInitFile), gpuOpts);
        _decoderStep = new InferenceSession(Path.Combine(modelPath, DecoderStepFile), gpuOpts);

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

        // Query free VRAM now that all four sessions are loaded so model weights
        // are already resident on the device.
        _vramBudgetForKvBytes = QueryVramBudget();
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
    /// mel.onnx is B=1 only; callers run this per segment.
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
        // Batch=1, so the flat layout is already [nMels, F]; bulk copy.
        var features = ExtractTensorFlatDirect(featT);
        return (features, nMels, F);
    }

    /// <summary>
    /// Batched encoder: mel features [B, 128, F_max] + lengths [B] → encoder_hidden_states [B, T_enc_max, 1280].
    ///
    /// The encoder.onnx graph accepts an <c>input_lengths</c> tensor (int64 [B]) giving the
    /// actual mel-frame count for each batch item.  The Conformer's self-attention uses this
    /// to build a padding mask, so zero-padded positions beyond each segment's true length are
    /// masked out and cannot contaminate the attention of valid positions.  This means segments
    /// of different lengths can be safely batched without hallucinations or repetition loops.
    /// </summary>
    private (float[] batchHidden, int T_enc_max) RunEncoderBatch(
        IReadOnlyList<(float[] features, int nMels, int F)> melResults)
    {
        int B     = melResults.Count;
        int nMels = melResults[0].Item2;
        int F_max = 0;
        foreach (var (_, _, F) in melResults) if (F > F_max) F_max = F;

        // Build [B, 128, F_max] — zero-initialised so padded positions are silent.
        var batchData = new float[B * nMels * F_max];
        var lengths   = new long[B];
        for (int b = 0; b < B; b++)
        {
            var (features, _, F) = melResults[b];
            for (int m = 0; m < nMels; m++)
                Array.Copy(features, m * F, batchData, (b * nMels + m) * F_max, F);
            lengths[b] = F;   // actual mel frames; encoder uses this for the padding mask
        }

        var batchT   = new DenseTensor<float>(batchData, new[] { B, nMels, F_max });
        var lengthsT = new DenseTensor<long>(lengths, new[] { B });
        using var encResults = _encoder.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_features", batchT),
            NamedOnnxValue.CreateFromTensor("input_lengths",  lengthsT),
        ]);

        var hidT      = encResults.First(r => r.Name == "encoder_hidden_states").AsTensor<float>();
        int T_enc_max = hidT.Dimensions[1];
        var batchHidden = ExtractTensorFlatDirect(hidT);
        return (batchHidden, T_enc_max);
    }

    /// <summary>
    /// Batched KV-cache greedy decoder with ORT IOBinding.
    ///
    /// Key optimisations vs the previous implementation:
    ///
    /// 1. IOBinding — cross-KV tensors ([B, H, encTMax, d]) are returned by
    ///    decoder_init directly into CUDA device memory and bound once to the
    ///    decoder_step session.  They never cross the PCIe bus.  Self-KV tensors
    ///    are similarly kept on the GPU between steps (step-output OrtValues bound
    ///    directly as next-step inputs).
    ///
    /// 2. Context-block prefill — when a language is forced, the full 10-token
    ///    context prefix [BOS, startofcontext, startoftranscript, emo:neutral,
    ///    lang, lang, pnc, noitn, notimestamp, nodiarize] is fed to decoder_init
    ///    in one call, collapsing 9 serial decoder_step calls into zero.
    ///    Without forced language the prefix is just BOS (existing behaviour).
    ///
    /// Returns one token list per segment (includes BOS and all context tokens).
    /// </summary>
    private List<int>[] GreedyDecodeBatch(
        float[] encoderHiddenBatch, int B, int encTMax,
        int maxTokens = 256, int forcedLangTokenId = -1)
    {
        const int encDim = 1280;

        // ── Build decoder_init input ──────────────────────────────────────────
        // Always send BOS only and let the model generate the context block
        // token-by-token in the step loop.  ForceLang intercepts any language
        // token the model emits (IDs 22–204) and substitutes the forced language;
        // all other context tokens (emotion, ITN flag, etc.) are left to the model
        // so it can choose appropriate values for each segment's content.
        //
        // NOTE: The full 10-token prefill was tried but caused the model to
        // produce Chinese annotations ([笑], 嗯) for short non-speech segments
        // (laughter, filled pauses) because hard-coding noitn + nodiarize etc.
        // over-constrains the decoding for ambiguous audio.  Auto-generating the
        // context block with only the language forced gives significantly better
        // results for those segments.
        bool usePrefill   = false;
        int  contextLen   = 1;   // always BOS only for decoder_init

        var initTokens = new long[B * contextLen];
        for (int b = 0; b < B; b++)
        {
            int o = b * contextLen;
            initTokens[o] = _bosTokenId;
            if (usePrefill)
            {
                initTokens[o + 1] = TokStartOfContext;
                initTokens[o + 2] = TokStartOfTranscript;
                initTokens[o + 3] = TokEmoNeutral;
                initTokens[o + 4] = forcedLangTokenId;
                initTokens[o + 5] = forcedLangTokenId;
                initTokens[o + 6] = TokPncOn;
                initTokens[o + 7] = TokNoItn;
                initTokens[o + 8] = TokNoTimestamp;
                initTokens[o + 9] = TokNoDiarize;
            }
        }

        // ── Memory infos ─────────────────────────────────────────────────────
        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cpuMemInfo  = new OrtMemoryInfo("Cpu",  OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts     = new RunOptions();

        // ── decoder_init with IOBinding ───────────────────────────────────────
        // OrtIoBinding.GetOutputValues() returns tensors in BINDING ORDER (the order
        // BindOutputToDevice was called), not model output order.  Bind in four grouped
        // passes so the indices the step loop reads match the actual positions:
        //   [0]       logits
        //   [1..8]    self_key_0..7
        //   [9..16]   self_val_0..7
        //   [17..24]  cross_key_0..7
        //   [25..32]  cross_val_0..7
        using var initTokValue = OrtValue.CreateTensorValueFromMemory(
            initTokens, new long[] { B, contextLen });
        using var encHidValue  = OrtValue.CreateTensorValueFromMemory(
            encoderHiddenBatch, new long[] { B, encTMax, encDim });

        using var initBinding = _decoderInit.CreateIoBinding();
        initBinding.BindInput("decoder_input_ids",     initTokValue);
        initBinding.BindInput("encoder_hidden_states", encHidValue);
        initBinding.BindOutputToDevice("logits", cpuMemInfo);   // logits read on CPU
        for (int i = 0; i < NumLayers; i++) initBinding.BindOutputToDevice($"self_key_{i}",  cudaMemInfo);
        for (int i = 0; i < NumLayers; i++) initBinding.BindOutputToDevice($"self_val_{i}",  cudaMemInfo);
        for (int i = 0; i < NumLayers; i++) initBinding.BindOutputToDevice($"cross_key_{i}", cudaMemInfo);
        for (int i = 0; i < NumLayers; i++) initBinding.BindOutputToDevice($"cross_val_{i}", cudaMemInfo);
        _decoderInit.RunWithBinding(runOpts, initBinding);

        // Keep initOutputs alive for the full decode loop — cross-KV is bound
        // from it for every step.  Disposed after the loop.
        var initOutputs = initBinding.GetOutputValues();

        // ── Read first tokens from init logits ────────────────────────────────
        // Shape: [B, contextLen, vocabSize]; last position = index contextLen-1.
        int vocabSize;
        int[] firstTokens;
        {
            var shape     = initOutputs[0].GetTensorTypeAndShape().Shape;
            vocabSize     = (int)shape[2];
            var logitsSpan = initOutputs[0].GetTensorDataAsSpan<float>();
            firstTokens   = new int[B];
            int lastPos   = contextLen - 1;
            for (int b = 0; b < B; b++)
            {
                var slice   = logitsSpan.Slice((b * contextLen + lastPos) * vocabSize, vocabSize);
                firstTokens[b] = ForceLang(ArgMaxSpan(slice), forcedLangTokenId);
            }
        }

        // ── Initialise per-segment token lists ───────────────────────────────
        var tokens        = new List<int>[B];
        var finished      = new bool[B];
        var nextTok       = new long[B];
        int finishedCount = 0;

        for (int b = 0; b < B; b++)
        {
            tokens[b] = new List<int>(maxTokens + contextLen + 1);
            tokens[b].Add(_bosTokenId);
            if (usePrefill)
            {
                tokens[b].Add(TokStartOfContext);
                tokens[b].Add(TokStartOfTranscript);
                tokens[b].Add(TokEmoNeutral);
                tokens[b].Add(forcedLangTokenId);
                tokens[b].Add(forcedLangTokenId);
                tokens[b].Add(TokPncOn);
                tokens[b].Add(TokNoItn);
                tokens[b].Add(TokNoTimestamp);
                tokens[b].Add(TokNoDiarize);
            }
            tokens[b].Add(firstTokens[b]);
            nextTok[b] = firstTokens[b];
            if (firstTokens[b] == _eosTokenId) { finished[b] = true; finishedCount++; }
        }

        int tPast = contextLen;

        // ── Step loop ─────────────────────────────────────────────────────────
        // decoder_step output layout (17 tensors, binding order = grouped):
        //   [0]       logits              float32 CPU
        //   [1..8]    new_self_key_0..7   float16 CUDA  (grow by 1 each step)
        //   [9..16]   new_self_val_0..7   float16 CUDA
        //
        // A FRESH OrtIoBinding is created each step.  ORT caches the CUDA buffer
        // allocated by BindOutputToDevice and reuses it on subsequent RunWithBinding
        // calls; when the Concat output grows by one token the cached shape mismatches
        // and ORT throws.  Creating a fresh binding each step forces a new allocation
        // at the correct shape without any caching from the previous step.
        //
        // prevStepOutputs keeps the previous step's self-KV OrtValues alive while
        // they are bound as inputs for the current step.  They are disposed only
        // after the current step has run and the new self-KV are in hand.
        IDisposableReadOnlyCollection<OrtValue>? prevStepOutputs = null;

        for (int step = 1; step < maxTokens && finishedCount < B; step++)
        {
            var stepToks = new long[B];
            var stepPos  = new long[B];
            for (int b = 0; b < B; b++)
                stepToks[b] = finished[b] ? _eosTokenId : nextTok[b];
            Array.Fill(stepPos, (long)tPast);

            using var tokVal = OrtValue.CreateTensorValueFromMemory(stepToks, new long[] { B, 1 });
            using var posVal = OrtValue.CreateTensorValueFromMemory(stepPos,  new long[] { B, 1 });

            // Fresh binding each step — avoids buffer-reuse shape failures on self-KV.
            // Outputs bound in two grouped passes so GetOutputValues() returns them in the
            // expected grouped order: [1..8]=new_sk0..7, [9..16]=new_sv0..7.
            using var stepBinding = _decoderStep.CreateIoBinding();
            stepBinding.BindOutputToDevice("logits", cpuMemInfo);
            for (int i = 0; i < NumLayers; i++) stepBinding.BindOutputToDevice($"new_self_key_{i}", cudaMemInfo);
            for (int i = 0; i < NumLayers; i++) stepBinding.BindOutputToDevice($"new_self_val_{i}", cudaMemInfo);
            stepBinding.BindInput("decoder_input_ids", tokVal);
            stepBinding.BindInput("positions",         posVal);
            for (int i = 0; i < NumLayers; i++)
            {
                stepBinding.BindInput($"cross_key_{i}", initOutputs[17 + i]);
                stepBinding.BindInput($"cross_val_{i}", initOutputs[25 + i]);
                // self-KV: from init on step 1, from previous step's outputs thereafter.
                var skSrc = prevStepOutputs is null ? initOutputs[1 + i]             : prevStepOutputs[1 + i];
                var svSrc = prevStepOutputs is null ? initOutputs[9 + i]             : prevStepOutputs[1 + NumLayers + i];
                stepBinding.BindInput($"self_key_{i}", skSrc);
                stepBinding.BindInput($"self_val_{i}", svSrc);
            }

            _decoderStep.RunWithBinding(runOpts, stepBinding);
            var curOutputs = stepBinding.GetOutputValues();
            tPast++;

            // Logits are CPU-resident; read argmax without copying.
            var logitsSpan = curOutputs[0].GetTensorDataAsSpan<float>();

            // Dispose previous self-KV — no longer needed as inputs (new ones are in curOutputs).
            prevStepOutputs?.Dispose();
            prevStepOutputs = curOutputs;

            for (int b = 0; b < B; b++)
            {
                if (finished[b]) { nextTok[b] = _eosTokenId; continue; }
                var stepLogits = logitsSpan.Slice(b * vocabSize, vocabSize);
                int tok = ForceLang(ArgMaxSpan(stepLogits), forcedLangTokenId);
                nextTok[b] = tok;
                tokens[b].Add(tok);
                if (tok == _eosTokenId) { finished[b] = true; finishedCount++; }
            }
        }

        prevStepOutputs?.Dispose();
        initOutputs.Dispose();
        return tokens;
    }

    // If the decoded token is a language tag and we have a forced language, substitute it.
    private static int ForceLang(int token, int forcedLangTokenId) =>
        forcedLangTokenId >= 0 && token >= LangIdMin && token <= LangIdMax
            ? forcedLangTokenId
            : token;

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

        // Sort segment indices by ascending audio duration so that similar-length
        // segments land in the same batch, minimising straggler waste (shorter segments
        // that have emitted EOS still step until the longest segment finishes).
        int[] order = Enumerable.Range(0, segs.Count)
            .OrderBy(i => segs[i].end - segs[i].start)
            .ToArray();

        int pos = 0;
        while (pos < order.Length)
        {
            // ── Build a variable-size batch bounded by VRAM and MaxBatchSize ──────
            // Grow the batch one segment at a time, stopping when the estimated peak
            // KV-cache size would exceed VramBudgetForKvBytes or MaxBatchSize is hit.
            // Segments are sorted ascending by duration so the worst-case (longest)
            // segment drives the KV estimate.
            // Note: there is no encoder-frame ratio limit because the encoder is now
            // run per-segment (B=1 each), so Conformer self-attention contamination
            // from batch padding is no longer an issue.  Padded positions in the
            // stacked decoder input are zero, contributing zero to cross-attention output.
            int batchSize    = 0;
            int maxEncFrames = 0;
            int maxDecSteps  = 0;

            while (pos + batchSize < order.Length && batchSize < MaxBatchSize)
            {
                int candidateIdx    = order[pos + batchSize];
                var (cs, ce, _)     = segs[candidateIdx];
                double candDur      = ce - cs;
                int candEncFrames   = EstimateEncFrames(candDur);
                int candDecSteps    = EstimateDecSteps(candDur, maxNewTokens);

                // Worst-case across all segments in the prospective batch.
                int newMaxEnc = Math.Max(maxEncFrames, candEncFrames);
                int newMaxDec = Math.Max(maxDecSteps,  candDecSteps);

                long kvBytes = EstimateKvBytes(batchSize + 1, newMaxEnc, newMaxDec);
                if (batchSize > 0 && kvBytes > _vramBudgetForKvBytes)
                    break;  // adding this segment would exceed the VRAM budget

                maxEncFrames = newMaxEnc;
                maxDecSteps  = newMaxDec;
                batchSize++;
            }

            // ── Extract waveforms; flag segments too short to encode ──────────────
            var waveforms = new float[batchSize][];
            var skipped   = new bool[batchSize];
            var segIds    = new int[batchSize];

            for (int b = 0; b < batchSize; b++)
            {
                segIds[b] = order[pos + b];
                var (start, end, _) = segs[segIds[b]];
                waveforms[b] = ExtractSegment(audio, start, end);
                skipped[b]   = waveforms[b].Length < Config.SampleRate / 10;
            }

            // ── Mel (serial, B=1 only) ────────────────────────────────────────────
            var melResults = new (float[] features, int nMels, int F)[batchSize];
            for (int b = 0; b < batchSize; b++)
                if (!skipped[b])
                    melResults[b] = RunMel(waveforms[b]);

            // ── Encoder + batched decoder ────────────────────────────────────────
            // Only pass valid (non-skipped) segments to the encoder.
            var validMel    = new List<(float[], int, int)>(batchSize);
            var validBSlots = new List<int>(batchSize);  // b-indices of valid segments
            for (int b = 0; b < batchSize; b++)
                if (!skipped[b]) { validMel.Add(melResults[b]); validBSlots.Add(b); }

            List<int>[] batchTokens = [];
            if (validMel.Count > 0)
            {
                var (batchHidden, T_enc_max) = RunEncoderBatch(validMel);
                batchTokens = GreedyDecodeBatch(
                    batchHidden, validMel.Count, T_enc_max, maxNewTokens, forcedLangTokenId);
            }

            // ── Yield results ────────────────────────────────────────────────────
            int encIdx = 0;
            for (int b = 0; b < batchSize; b++)
            {
                int segId = segIds[b];
                if (skipped[b])
                {
                    yield return (segId, string.Empty, CohereSegmentMeta.Empty);
                    continue;
                }

                var tokens = batchTokens[encIdx++];
                var meta   = ParseContextBlock(tokens);
                string text = DecodeTokens(tokens);
                yield return (segId, text, meta);
            }

            pos += batchSize;
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

    // Bulk-copy a Tensor<float> to a flat float[] via DenseTensor buffer when available.
    private static float[] ExtractTensorFlatDirect(Tensor<float> tensor)
    {
        int total = 1;
        for (int d = 0; d < tensor.Rank; d++) total *= tensor.Dimensions[d];
        var flat = new float[total];
        if (tensor is DenseTensor<float> dense)
            dense.Buffer.Span.CopyTo(flat);
        else
            for (int i = 0; i < total; i++) flat[i] = tensor.GetValue(i);
        return flat;
    }

    private static int ArgMaxSpan(ReadOnlySpan<float> span)
    {
        int idx = 0;
        float max = float.NegativeInfinity;
        for (int i = 0; i < span.Length; i++)
            if (span[i] > max) { max = span[i]; idx = i; }
        return idx;
    }

    // ── CUDA VRAM query ───────────────────────────────────────────────────────

    // cudaMemGetInfo returns free and total device memory in bytes.
    // The DllImport name resolves to libcudart.so on Linux / cudart.dll on Windows.
    [System.Runtime.InteropServices.DllImport("cudart",
        EntryPoint            = "cudaMemGetInfo",
        ExactSpelling         = true,
        CallingConvention     = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern int CudaMemGetInfo(out ulong free, out ulong total);

    /// <summary>
    /// Queries GPU free memory after all ONNX sessions are loaded and subtracts
    /// <see cref="VramSafetyBufferBytes"/> as headroom for activations and CUDA
    /// runtime overhead.  Returns <see cref="VramBudgetFallbackBytes"/> if the
    /// CUDA runtime is not available (CPU-only builds or non-CUDA EPs).
    /// </summary>
    private static long QueryVramBudget()
    {
        try
        {
            int rc = CudaMemGetInfo(out ulong free, out _);
            if (rc == 0 && free > VramSafetyBufferBytes)
                return (long)(free - VramSafetyBufferBytes);
        }
        catch (DllNotFoundException) { }
        catch (EntryPointNotFoundException) { }
        return VramBudgetFallbackBytes;
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
