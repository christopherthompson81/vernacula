using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// IBM Granite Speech 4.1 — speech encoder + 1.84 B-param Granite LLM
/// decoder, with audio embedding fused into the prompt via a Q-Former
/// projector. Decoder-only attention; no encoder-decoder cross-KV.
///
/// Uses four ONNX models in the model directory:
///   mel.onnx           audio [1, samples] → input_features [1, T_stacked, 160]
///   encoder.onnx       input_features [B, T, 160] → encoder_hidden [B, T, 1024]
///   projector.onnx     encoder_hidden [B, T, 1024] → audio_embeds [B, A, 2048]
///   decoder.onnx       UNIFIED prefill + step graph:
///                       inputs:  input_ids, audio_embeds, attention_mask, cache_position,
///                                past_key_&lt;L&gt;/past_value_&lt;L&gt; for L in 0..39
///                       outputs: logits + present_key_&lt;L&gt;/present_value_&lt;L&gt; for L in 0..39
///                       Prefill is signalled by passing zero-length past_kv inputs;
///                       step is signalled by populated past_kv. Same graph either way.
///
/// The decode loop uses the **chained Run-with-OrtValue** pattern: each step's
/// output OrtValues become the next step's input OrtValues directly, so the KV
/// cache stays GPU-resident across steps without ever touching managed memory.
/// See docs/dev/granite_speech_perf_investigation.md Run 7 for why this beats
/// `OrtIoBinding.RunWithBinding` on this graph.
///
/// First C# integration: serial single-segment dispatch, no batching, fixed
/// English ASR prompt. Matches Cohere's <see cref="CohereTranscribe.Recognize"/>
/// public surface so <c>Vernacula.CLI</c> can dispatch the same way.
/// VRAM-budgeted batching via <see cref="BatchSizer"/> is a follow-up.
/// </summary>
public sealed class GraniteSpeech : IDisposable
{
    public const string MelFile           = "mel.onnx";
    public const string EncoderFile       = "encoder.onnx";
    public const string ProjectorFile     = "projector.onnx";
    public const string DecoderFile       = "decoder.onnx";
    public const string TokenizerFile     = "tokenizer.json";
    public const string VocabFile         = "vocab.json";
    public const string MergesFile        = "merges.txt";
    public const string AddedTokensFile   = "added_tokens.json";
    public const string SpecialTokensFile = "special_tokens_map.json";
    public const string TokenizerCfgFile  = "tokenizer_config.json";

    // ── Architecture constants ──────────────────────────────────────────
    // Mirror what the Python export wrote into config.json. Hardcoded here
    // (rather than read at runtime) because they're load-bearing for the
    // KV-cache shapes and don't change across model variants without a
    // re-export.
    private const int NumDecoderLayers = 40;
    private const int NumKvHeads       = 4;     // GQA — text_config.num_key_value_heads
    private const int HeadDim          = 128;   // text_config.hidden_size / num_attention_heads
    private const int NumAttnHeads     = 16;    // text_config.num_attention_heads (for activation sizing)
    private const int VocabSize        = 100353;
    private const int AudioTokenId     = 100352;
    private const int PadTokenId       = 100256; // text_config.pad_token_id
    private const int EosTokenId       = 100257; // bos_token_id == eos_token_id
    private const int MaxContextLen    = 4096;   // text_config.max_position_embeddings

    // Mel frontend constants — driving the audio-token-count formula. Match
    // GraniteSpeechFeatureExtractor: hop_length=160, projector_window_size=15,
    // projector_downsample_rate=5 → effective_window = 3.
    private const int HopLength             = 160;
    private const int ProjectorWindowSize   = 15;
    private const int EffectiveWindowSize   = 3;  // = ProjectorWindowSize / projector_downsample_rate

    // ── Prompt template ─────────────────────────────────────────────────
    // Tokens for the standard ASR chat-template prompt:
    //   "USER: <|audio|>transcribe the speech with proper punctuation and capitalization.\n ASSISTANT:"
    // Generated once via the Granite tokenizer (see scripts/granite_export/
    // dump_inputs_for_csharp_smoke.py for the source command). We splice in N
    // copies of AudioTokenId between Prefix and Suffix where N is the
    // audio-token count derived from sample count.
    private static readonly long[] AsrPromptPrefix = { 6584, 25, 220 };
    private static readonly long[] AsrPromptSuffix =
    {
        1485, 3191, 279, 8982, 449, 6300, 62603, 323, 6864, 2065, 627,
        36660, 3931, 2891, 25,
    };

    // ── Batching budget ─────────────────────────────────────────────────
    // Hard cap on rows per batch even if VRAM is plentiful. Granite Speech
    // is a 1.84 B-param LM; activation pressure scales fast with B.
    private const int MaxBatchSize = 16;

    // VRAM safety buffer reserved for CUDA runtime + headroom for the LM
    // logits spike at long prompts (B * S_max * 100 353 * 4 bytes peaks
    // around 1.5 GB at B=4 / S=900 / fp32). Falls back to 3 GB on CPU-only
    // builds where cudaMemGetInfo is unavailable.
    private const long VramSafetyBufferBytes  = 3_000_000_000L;
    private const long VramBudgetFallbackBytes = 3_000_000_000L;
    private readonly long _vramBudgetForKvBytes;

    // ── ONNX sessions + tokenizer ───────────────────────────────────────
    private readonly InferenceSession _mel;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _projector;
    private readonly InferenceSession _decoder;

    // Past-KV dtype detected from the decoder's input metadata. The fp32
    // bundle uses tensor(float); the BF16 mixed-precision bundle uses
    // tensor(bfloat16) for KV across the chained Run loop. We need this
    // to construct the empty-prefill past_kv OrtValues in the right
    // dtype — KV inputs and outputs across step calls are otherwise
    // GPU-resident and never touch managed memory.
    private readonly TensorElementType _pastKvDtype;

    // Static-shape bundle plumbing (Run 15 phase 2c, issue #41).
    // _isStaticBundle = true when the decoder graph pins past_kv / audio_embeds
    // to fixed dims (the --static-shapes-unified export). The pinned values
    // come from the graph's input metadata; we never hard-code 768 / 375 / 16
    // here so that re-exports with different sizes work without a rebuild.
    // The shared zero past_kv buffer is allocated once and referenced by all
    // 2 × NumDecoderLayers prefill past_kv inputs — they're all-zero and
    // ORT only reads, so sharing avoids 80 × 12 MB of redundant allocation.
    private readonly bool _isStaticBundle;
    private readonly bool _isGqaBundle;     // Set when the static bundle was
                                            // produced by `rewrite_attention_to_mha.py
                                            // --mode gqa`. Detected by the
                                            // presence of `seqlens_k` as a
                                            // decoder graph input. Selects a
                                            // left-aligned cache + compute-skip
                                            // code path (Run 20).
    private readonly bool _hasPositionIdsInput;  // Set when the decoder graph
                                            // has a `position_ids` input (the
                                            // Run 20 phase 2 export). Lets the
                                            // C# driver feed per-row RoPE
                                            // positions and restore batched
                                            // dispatch.
    private readonly bool _isDynamicGqaBundle;   // GQA bundle with FULLY
                                            // dynamic shapes (no past_len
                                            // pinning). Past_kv grows each
                                            // step; no buffer-sharing
                                            // IOBinding possible. Run 20
                                            // phase 4.
    private readonly bool _hasNextTokenOutput;   // Bundle was rewritten with
                                            // --add-argmax: the LM head has
                                            // an on-GPU Slice(last) + ArgMax
                                            // appended and `logits` is no
                                            // longer a graph output —
                                            // `next_token` (int64 [B, 1])
                                            // is. Eliminates ~6 MB of
                                            // GPU→CPU logits transfer per
                                            // step (Run 20 phase 9).
    private readonly int _staticBatchSize;
    private readonly int _staticPastLen;
    private readonly int _staticAudioLen;
    private readonly Array? _staticZeroPastKvBuffer;
    private readonly OrtValue[]? _staticZeroPastKv;

    private readonly string?[] _idToToken;
    private readonly Dictionary<int, string> _addedTokens;
    private readonly Dictionary<char, byte> _byteLevelDecode;

    // ── Construction ────────────────────────────────────────────────────

    public GraniteSpeech(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = OrtSessionBuilder.Create(ep);
        _mel       = new InferenceSession(Path.Combine(modelPath, MelFile),       opts);
        _encoder   = new InferenceSession(Path.Combine(modelPath, EncoderFile),   opts);
        _projector = new InferenceSession(Path.Combine(modelPath, ProjectorFile), opts);
        _decoder   = new InferenceSession(Path.Combine(modelPath, DecoderFile),   opts);

        (_idToToken, _addedTokens) = LoadTokenizerVocab(Path.Combine(modelPath, TokenizerFile));
        _byteLevelDecode = BuildByteLevelDecodeTable();

        _pastKvDtype = _decoder.InputMetadata.TryGetValue("past_key_0", out var pkMeta)
            ? pkMeta.ElementDataType
            : TensorElementType.Float;

        // Static-shape bundle detection (Run 15 phase 2c, issue #41). The
        // sliding-window static export pins past_kv to a fixed shape and
        // slices KV outputs to match, eliminating Run 13's per-step
        // CPU-resident shape island. Detect by reading past_key_0's
        // dimensions from input metadata: positive integers across
        // batch / past_len / head_dim => static; symbolic (-1) anywhere
        // => dynamic, and we use the legacy TranscribeBatch path.
        _isStaticBundle = false;
        _isGqaBundle = _decoder.InputMetadata.ContainsKey("seqlens_k");
        _hasPositionIdsInput = _decoder.InputMetadata.ContainsKey("position_ids");
        _hasNextTokenOutput = _decoder.OutputMetadata.ContainsKey("next_token");
        // Dynamic GQA: GQA bundle with past_kv batch dim symbolic
        // (no static batch pinning).
        _isDynamicGqaBundle = false;
        if (_isGqaBundle
            && _hasPositionIdsInput
            && _decoder.InputMetadata.TryGetValue("past_key_0", out var pkDynMeta)
            && pkDynMeta.Dimensions.Length == 4
            && pkDynMeta.Dimensions[0] <= 0)
        {
            _isDynamicGqaBundle = true;
        }
        _staticBatchSize = 0;
        _staticPastLen = 0;
        _staticAudioLen = 0;
        if (pkMeta is not null)
        {
            var dims = pkMeta.Dimensions;
            // Expected static layout: [B, NumKvHeads, past_len, HeadDim].
            if (dims.Length == 4 && dims[0] > 0 && dims[2] > 0)
            {
                _isStaticBundle = true;
                _staticBatchSize = dims[0];
                _staticPastLen   = dims[2];
                if (_decoder.InputMetadata.TryGetValue("audio_embeds", out var aeMeta)
                    && aeMeta.Dimensions.Length == 3 && aeMeta.Dimensions[1] > 0)
                {
                    _staticAudioLen = aeMeta.Dimensions[1];
                }
                else
                {
                    throw new InvalidOperationException(
                        "Static-shape decoder bundle declared past_kv with fixed past_len but "
                      + "audio_embeds is not statically shaped. Re-export with --static-shapes-unified.");
                }
                // Pre-allocate the shared zero past_kv buffer once and bind it
                // to 2 * NumDecoderLayers OrtValues (one buffer, many views).
                // ORT only reads the past_kv at prefill; sharing is safe.
                long total = (long)_staticBatchSize * NumKvHeads * _staticPastLen * HeadDim;
                long[] shape = { _staticBatchSize, NumKvHeads, _staticPastLen, HeadDim };
                _staticZeroPastKvBuffer = AllocateZeroPastKvBuffer(total);
                _staticZeroPastKv = new OrtValue[2 * NumDecoderLayers];
                for (int L = 0; L < NumDecoderLayers; L++)
                {
                    _staticZeroPastKv[2 * L]     = CreateZeroPastKvView(_staticZeroPastKvBuffer, shape);
                    _staticZeroPastKv[2 * L + 1] = CreateZeroPastKvView(_staticZeroPastKvBuffer, shape);
                }
            }
        }

        _vramBudgetForKvBytes = QueryVramBudget();
    }

    public void Dispose()
    {
        if (_staticZeroPastKv is not null)
        {
            foreach (var ov in _staticZeroPastKv) ov.Dispose();
        }
        _mel.Dispose();
        _encoder.Dispose();
        _projector.Dispose();
        _decoder.Dispose();
    }

    // ── Static-shape helpers (Run 15 phase 2c) ───────────────────────────

    private Array AllocateZeroPastKvBuffer(long totalElements) => _pastKvDtype switch
    {
        TensorElementType.Float    => new float[totalElements],
        TensorElementType.BFloat16 => new BFloat16[totalElements],
        TensorElementType.Float16  => new Float16[totalElements],
        _ => throw new InvalidOperationException(
            $"Unsupported past_kv dtype {_pastKvDtype}."),
    };

    private OrtValue CreateZeroPastKvView(Array sharedBuffer, long[] shape) => _pastKvDtype switch
    {
        TensorElementType.Float    => OrtValue.CreateTensorValueFromMemory((float[])sharedBuffer, shape),
        TensorElementType.BFloat16 => OrtValue.CreateTensorValueFromMemory((BFloat16[])sharedBuffer, shape),
        TensorElementType.Float16  => OrtValue.CreateTensorValueFromMemory((Float16[])sharedBuffer, shape),
        _ => throw new InvalidOperationException(
            $"Unsupported past_kv dtype {_pastKvDtype}."),
    };

    // ── Audio-token-count derivation ────────────────────────────────────

    /// <summary>
    /// Compute the number of audio tokens (projector output frames) that
    /// will be produced for an audio segment of the given sample count.
    /// Mirrors `GraniteSpeechFeatureExtractor._get_num_audio_features`:
    ///   mel_length     = samples / hop_length + 1
    ///   encoder_length = mel_length / 2          (frame stack)
    ///   nblocks        = ceil(encoder_length / projector_window_size)
    ///   audio_tokens   = nblocks * effective_window_size  (=3)
    /// </summary>
    public static int NumAudioTokens(int sampleCount)
    {
        int melLength     = sampleCount / HopLength + 1;
        int encoderLength = melLength / 2;
        int nblocks       = (encoderLength + ProjectorWindowSize - 1) / ProjectorWindowSize;
        return nblocks * EffectiveWindowSize;
    }

    private static long[] BuildPromptIds(int audioTokens)
    {
        var ids = new long[AsrPromptPrefix.Length + audioTokens + AsrPromptSuffix.Length];
        Array.Copy(AsrPromptPrefix, ids, AsrPromptPrefix.Length);
        for (int i = 0; i < audioTokens; i++)
            ids[AsrPromptPrefix.Length + i] = AudioTokenId;
        Array.Copy(AsrPromptSuffix, 0, ids, AsrPromptPrefix.Length + audioTokens,
                   AsrPromptSuffix.Length);
        return ids;
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// <summary>
    /// Transcribes each segment from <paramref name="segs"/> and yields
    /// <c>(segId, text, speaker)</c> in order as each batch completes. Speaker
    /// labels are passed through from the input segments — Granite Speech
    /// itself doesn't do speaker diarization.
    ///
    /// Thin wrapper over <see cref="RecognizeDetailed"/> that drops the
    /// per-segment token IDs. Callers that want token-level data — for
    /// editor word-sync via synthetic linear timestamps, or for any
    /// per-token UI — should use <see cref="RecognizeDetailed"/> directly.
    /// </summary>
    /// <param name="segs">Segment list as <c>(start_seconds, end_seconds, speaker_label)</c>.</param>
    /// <param name="audio">Mono 16 kHz waveform covering all segments.</param>
    /// <param name="maxNewTokens">Cap on generated tokens per segment.</param>
    public IEnumerable<(int segId, string text, string speaker)> Recognize(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        foreach (var (segId, text, _, speaker) in RecognizeDetailed(segs, audio, maxNewTokens))
            yield return (segId, text, speaker);
    }

    /// <summary>
    /// Like <see cref="Recognize"/> but also surfaces the post-trim BPE token
    /// IDs that produced the text. Used by the GUI's transcript editor to
    /// drive word-level click-to-position via <c>BuildSyntheticTokenTimestamps</c>:
    /// per-token linear timestamps over the segment duration give finer
    /// granularity than per-word splitting and match what Cohere/Qwen3
    /// surface today.
    ///
    /// Tokens are the same sequence used to produce the text: trailing EOS
    /// stripped, periodic-loop tail trimmed when the runaway-loop detector
    /// fired. Token count therefore matches the visible text's information
    /// content, not the raw decode length.
    ///
    /// Segments are sorted by ascending duration and packed into VRAM-budgeted
    /// batches via <see cref="BatchSizer.Plan"/>. Within a batch, all segments
    /// run through a single batched forward of mel + encoder + projector +
    /// unified decoder; per-row EOS tracking lets shorter segments stop while
    /// the longest finishes its decode.
    /// </summary>
    public IEnumerable<(int segId, string text, IReadOnlyList<long> tokens, string speaker)> RecognizeDetailed(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        if (segs.Count == 0) yield break;

        var durations = new double[segs.Count];
        for (int i = 0; i < segs.Count; i++) durations[i] = segs[i].end - segs[i].start;

        var plan = BatchSizer.Plan(
            durations,
            new GraniteBatchCostModel(maxNewTokens, KvBytesPerElement),
            _vramBudgetForKvBytes,
            MaxBatchSize);

        foreach (var batch in plan)
        {
            int B = batch.Count;
            int[] segIds = batch.SegmentIndices;

            // Extract waveforms; mark short ones to skip without dropping the
            // batch slot (we yield an empty result for them).
            var waveforms = new float[B][];
            var skipped = new bool[B];
            for (int b = 0; b < B; b++)
            {
                var (start, end, _) = segs[segIds[b]];
                waveforms[b] = ExtractSegment(audio, start, end);
                skipped[b] = waveforms[b].Length < Config.SampleRate / 10;
            }

            // Compact to the rows we'll actually transcribe.
            var realIdx = new List<int>(B);
            var realWaves = new List<float[]>(B);
            for (int b = 0; b < B; b++)
                if (!skipped[b]) { realIdx.Add(b); realWaves.Add(waveforms[b]); }

            (string text, long[] tokens)[] realRows = realWaves.Count > 0
                ? TranscribeBatch(realWaves.ToArray(), maxNewTokens)
                : Array.Empty<(string, long[])>();

            // Yield in the batch order; reinflate the skipped slots.
            for (int b = 0; b < B; b++)
            {
                int segId = segIds[b];
                string text = string.Empty;
                IReadOnlyList<long> tokens = Array.Empty<long>();
                if (!skipped[b])
                {
                    int realPos = realIdx.IndexOf(b);
                    text = realRows[realPos].text;
                    tokens = realRows[realPos].tokens;
                }
                yield return (segId, text, tokens, segs[segId].spk);
            }
        }
    }

    /// <summary>Convenience entry point for transcribing a single waveform.</summary>
    public string Transcribe(float[] audio16kMono, int maxNewTokens = 256)
        => TranscribeBatch(new[] { audio16kMono }, maxNewTokens)[0].text;

    // ── Batched pipeline ────────────────────────────────────────────────

    /// <summary>
    /// Runs mel + encoder + projector + unified decoder over a batch of
    /// waveforms, generating greedy text for each row in lockstep until
    /// every row hits EOS or <paramref name="maxNewTokens"/>.
    ///
    /// Key shape decisions:
    ///   - Mel/encoder/projector run **serially per-row**: the encoder uses
    ///     full attention with no padding mask, and batching variable-length
    ///     audio with zero padding contaminates the attention output of real
    ///     positions. Per-row keeps each forward at its actual length.
    ///   - Decoder is **batched with LEFT-padded input_ids**: cache_position
    ///     is shared [S] across the batch, so left-padding aligns every row's
    ///     real last token at S-1 — rotary positions for generated tokens are
    ///     identical across rows.
    ///   - audio_embeds is right-aligned in the [B, A_max, 2048] buffer:
    ///     the cumsum-gather merge picks the first N_audio[b] entries per
    ///     row, so they must sit at indices [0, N_audio[b]) of audio_embeds[b].
    ///
    /// EOS handling: once a row emits EOS we substitute <see cref="EosTokenId"/>
    /// for its step input and freeze its output token, but keep stepping the
    /// batch until the longest row finishes (straggler waste — minimised by
    /// <see cref="BatchSizer.Plan"/>'s ascending-duration packing).
    /// </summary>
    // ── Per-stage timing accumulators (opt-in via VERNACULA_GRANITE_PROFILE=1) ─
    private static readonly bool _profile =
        Environment.GetEnvironmentVariable("VERNACULA_GRANITE_PROFILE") == "1";
    internal static long MelMs, EncMs, ProjMs, PrefillMs, StepLoopMs, OverheadMs;
    internal static long BatchCount, RowCount, StepCount;
    internal static int MaxBatchSeen;
    private static readonly object _profileLock = new();

    private (string text, long[] tokens)[] TranscribeBatch(float[][] waveforms, int maxNewTokens)
    {
        // Dynamic GQA bundle: B=1 per segment (Run 20 phase 4). Phase 5
        // attempted true batched dispatch but the CUDA GroupQueryAttention
        // kernel doesn't implement the `attention_bias` input (spec lists
        // it; runtime errors with "attention_bias is not supported in
        // GroupQueryAttention cuda kernel"). Without that, batched
        // variable-length prompts can't mask their pad slots, and pad K
        // contamination breaks transcripts even when BatchSizer groups
        // similar-duration segments.
        if (_isDynamicGqaBundle)
        {
            var dynRows = new (string text, long[] tokens)[waveforms.Length];
            for (int i = 0; i < waveforms.Length; i++)
                dynRows[i] = TranscribeDynamicGqaSingle(waveforms[i], maxNewTokens);
            return dynRows;
        }

        // Static-shape bundle (Run 15 phase 2c) takes the padded-to-fixed-B
        // path; the legacy dynamic-shape bundle keeps the original loop.
        if (_isStaticBundle)
            return TranscribeBatchStatic(waveforms, maxNewTokens);

        int B = waveforms.Length;
        if (B == 0) return Array.Empty<(string, long[])>();

        var swBatch = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        long melLocal = 0, encLocal = 0, projLocal = 0;

        // ── Per-row mel + encoder + projector ───────────────────────────
        // audioEmbeds[b] is the row's [N_audio[b] * 2048] flattened buffer.
        var audioEmbeds = new float[B][];
        var nAudio      = new int[B];
        long audioDim   = 2048;
        for (int b = 0; b < B; b++)
        {
            int audioTokens = NumAudioTokens(waveforms[b].Length);
            int promptLen   = AsrPromptPrefix.Length + audioTokens + AsrPromptSuffix.Length;
            if (promptLen + maxNewTokens > MaxContextLen)
            {
                int allowedAudio = MaxContextLen - maxNewTokens
                                 - AsrPromptPrefix.Length - AsrPromptSuffix.Length;
                int allowedSeconds = (int)(allowedAudio
                    / (double)EffectiveWindowSize * ProjectorWindowSize * 2 * HopLength
                    / Config.SampleRate);
                throw new InvalidOperationException(
                    $"Segment too long: prompt+max_new_tokens ({promptLen + maxNewTokens}) " +
                    $"exceeds Granite Speech context ({MaxContextLen}). " +
                    $"Pre-segment audio to ≤{allowedSeconds} s.");
            }

            // Mel
            var swStage = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
            var audioT = new DenseTensor<float>(waveforms[b], [1, waveforms[b].Length]);
            using var melResults = _mel.Run([NamedOnnxValue.CreateFromTensor("audio", audioT)]);
            var inputFeatures = melResults[0].AsTensor<float>();
            var ifShape = inputFeatures.Dimensions.ToArray();
            var ifData = inputFeatures is DenseTensor<float> dT ? dT.ToArray() : inputFeatures.ToArray();
            if (swStage != null) { swStage.Stop(); melLocal += swStage.ElapsedMilliseconds; swStage.Restart(); }

            // Encoder
            using var encResults = _encoder.Run([NamedOnnxValue.CreateFromTensor(
                "input_features", new DenseTensor<float>(ifData, ifShape))]);
            var encoderHidden = encResults[0].AsTensor<float>();
            var encShape = encoderHidden.Dimensions.ToArray();
            var encData = encoderHidden.ToArray();
            if (swStage != null) { swStage.Stop(); encLocal += swStage.ElapsedMilliseconds; swStage.Restart(); }

            // Projector
            using var projResults = _projector.Run([NamedOnnxValue.CreateFromTensor(
                "encoder_hidden", new DenseTensor<float>(encData, encShape))]);
            var ae = projResults[0].AsTensor<float>();
            var projShape = ae.Dimensions.ToArray();
            if (projShape[1] != audioTokens)
                throw new InvalidOperationException(
                    $"Audio token count mismatch (row {b}): formula predicts {audioTokens}, "
                  + $"projector produced {projShape[1]}.");

            audioEmbeds[b] = ae.ToArray();
            nAudio[b]      = audioTokens;
            audioDim       = projShape[2];
            if (swStage != null) { swStage.Stop(); projLocal += swStage.ElapsedMilliseconds; }
        }

        // ── Build batched, left-padded prompt ───────────────────────────
        var realLen = new int[B];
        int sMax    = 0;
        int aMax    = 0;
        for (int b = 0; b < B; b++)
        {
            realLen[b] = AsrPromptPrefix.Length + nAudio[b] + AsrPromptSuffix.Length;
            if (realLen[b] > sMax) sMax = realLen[b];
            if (nAudio[b]  > aMax) aMax = nAudio[b];
        }

        var inputIdsBatch = new long[B * sMax];
        var attnMaskBatch = new long[B * sMax];
        for (int b = 0; b < B; b++)
        {
            int padLen = sMax - realLen[b];
            int rowOff = b * sMax;
            for (int s = 0; s < padLen; s++)
            {
                inputIdsBatch[rowOff + s] = PadTokenId;
                // attnMaskBatch already 0 by default
            }
            int p = rowOff + padLen;
            // Prefix
            for (int i = 0; i < AsrPromptPrefix.Length; i++) inputIdsBatch[p++] = AsrPromptPrefix[i];
            // Audio token slots
            for (int i = 0; i < nAudio[b]; i++)              inputIdsBatch[p++] = AudioTokenId;
            // Suffix
            for (int i = 0; i < AsrPromptSuffix.Length; i++) inputIdsBatch[p++] = AsrPromptSuffix[i];
            // Real positions get attention=1
            for (int s = padLen; s < sMax; s++) attnMaskBatch[rowOff + s] = 1L;
        }

        var audioEmbedsBatch = new float[B * aMax * audioDim];
        for (int b = 0; b < B; b++)
        {
            int rowOff = b * aMax * (int)audioDim;
            int n = nAudio[b] * (int)audioDim;
            if (n > 0) Array.Copy(audioEmbeds[b], 0, audioEmbedsBatch, rowOff, n);
            // Remaining positions stay zero — never selected by cumsum-gather
            // because is_audio=False at non-AudioTokenId positions.
        }

        var cachePosPrefill = new long[sMax];
        for (int i = 0; i < sMax; i++) cachePosPrefill[i] = i;

        // Run 20 phase 6: dynamic-MHA-fused bundles also pass through this
        // path, but their graph signature includes a `position_ids` input
        // (the --unified-position-ids export adds it so the C# driver can
        // feed per-row RoPE positions). Build per-row position_ids that
        // map left-padded slot indices to the row's semantic positions
        // [0, realLen[b]).
        long[]? positionIdsPrefill = null;
        if (_hasPositionIdsInput)
        {
            positionIdsPrefill = new long[B * sMax];
            for (int b = 0; b < B; b++)
            {
                int padLen = sMax - realLen[b];
                int rowOff = b * sMax;
                for (int s = 0; s < sMax; s++)
                    positionIdsPrefill[rowOff + s] = Math.Max(0, s - padLen);
            }
        }

        // ── Decoder: prefill + chained Run-with-OrtValue step loop ──────
        int baseInputCount = _hasPositionIdsInput ? 5 : 4;
        var decInputNames = new List<string>(baseInputCount + 2 * NumDecoderLayers)
        { "input_ids", "audio_embeds", "attention_mask", "cache_position" };
        if (_hasPositionIdsInput) decInputNames.Add("position_ids");
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            decInputNames.Add($"past_key_{L}");
            decInputNames.Add($"past_value_{L}");
        }
        // If the bundle was rewritten with --add-argmax, the LM head emits
        // `next_token` (int64 [B, 1]) instead of `logits` (fp32 [B, S, V]).
        // The argmax happens on the GPU and we only transfer 8 bytes per
        // row per step instead of ~6 MB.
        string headOutputName = _hasNextTokenOutput ? "next_token" : "logits";
        var decOutputNames = new List<string>(1 + 2 * NumDecoderLayers) { headOutputName };
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_key_{L}");
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_value_{L}");

        var generated = new List<long>[B];
        for (int b = 0; b < B; b++) generated[b] = new List<long>(maxNewTokens);
        var finished      = new bool[B];
        var loopDetected  = new bool[B];   // true iff IsRepetitionLoop fired for row b
        int finishedCount = 0;
        var nextTok       = new long[B];

        long prefillLocal = 0, stepLocal = 0;
        long stepCountLocal = 0;
        using (var runOpts = new RunOptions())
        {
            var swPrefill = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
            // ── Prefill ─────────────────────────────────────────────────
            using var inputIdsVal = OrtValue.CreateTensorValueFromMemory(
                inputIdsBatch, new long[] { B, sMax });
            using var audioVal = OrtValue.CreateTensorValueFromMemory(
                audioEmbedsBatch, new long[] { B, Math.Max(aMax, 1), audioDim });
            using var attnVal = OrtValue.CreateTensorValueFromMemory(
                attnMaskBatch, new long[] { B, sMax });
            using var cpVal = OrtValue.CreateTensorValueFromMemory(
                cachePosPrefill, new long[] { sMax });

            // Audio_embeds requires A >= 1 even when nobody has audio.
            // Provide a 1-row zero buffer if aMax == 0 (shouldn't happen but
            // keeps the graph valid).
            if (aMax == 0)
                audioEmbedsBatch = new float[B * 1 * (int)audioDim];

            OrtValue? posPrefillVal = positionIdsPrefill is null
                ? null
                : OrtValue.CreateTensorValueFromMemory(
                    positionIdsPrefill, new long[] { B, sMax });

            var emptyPasts = new List<OrtValue>(2 * NumDecoderLayers);
            var prefillInputs = new List<OrtValue>(baseInputCount + 2 * NumDecoderLayers)
            { inputIdsVal, audioVal, attnVal, cpVal };
            if (posPrefillVal is not null) prefillInputs.Add(posPrefillVal);
            long[] emptyPastShape = { B, NumKvHeads, 0, HeadDim };
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                var k = CreateEmptyPastKv(emptyPastShape);
                var v = CreateEmptyPastKv(emptyPastShape);
                emptyPasts.Add(k);
                emptyPasts.Add(v);
                prefillInputs.Add(k);
                prefillInputs.Add(v);
            }

            var prefillOutputs = _decoder.Run(
                runOpts, decInputNames, prefillInputs, decOutputNames);

            // Argmax per row at the LAST position (S-1) — left-padding
            // means real last token sits there for every row. With
            // --add-argmax the head emits `next_token` (int64 [B, 1]),
            // so we just read tokens directly; otherwise argmax the
            // float logits on CPU.
            if (_hasNextTokenOutput)
            {
                var prefillTokSpan = prefillOutputs[0].GetTensorDataAsSpan<long>();
                for (int b = 0; b < B; b++)
                {
                    long tok = prefillTokSpan[b];
                    nextTok[b] = tok;
                    generated[b].Add(tok);
                    if (tok == EosTokenId) { finished[b] = true; finishedCount++; }
                }
            }
            else
            {
                var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
                for (int b = 0; b < B; b++)
                {
                    long tok = ArgmaxRowLastPosition(prefillLogitsSpan, b, sMax, VocabSize);
                    nextTok[b] = tok;
                    generated[b].Add(tok);
                    if (tok == EosTokenId) { finished[b] = true; finishedCount++; }
                }
            }
            prefillOutputs[0].Dispose();
            posPrefillVal?.Dispose();
            foreach (var p in emptyPasts) p.Dispose();

            // pastKvs = prefill output OrtValues, GPU-resident, chained forward.
            var pastKvs = new OrtValue[2 * NumDecoderLayers];
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                pastKvs[2 * L]     = prefillOutputs[1 + L];
                pastKvs[2 * L + 1] = prefillOutputs[1 + NumDecoderLayers + L];
            }

            if (swPrefill != null) { swPrefill.Stop(); prefillLocal = swPrefill.ElapsedMilliseconds; }
            var swStep = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
            // ── Step loop ───────────────────────────────────────────────
            // Reusable buffers. attn mask grows by 1 each step; pre-size to
            // the maximum we'll ever need (sMax + maxNewTokens + 1).
            long[] stepInputIds = new long[B];
            long[] cachePosStep = new long[1];
            long[] stepAttnBuf  = new long[B * (sMax + maxNewTokens + 1)];
            // Initialise step attention: row b has zeros at the LEFT pad
            // positions [0, sMax-realLen[b]) and ones from sMax-realLen[b]
            // onward. We append a new "1" each step.
            for (int b = 0; b < B; b++)
            {
                int padLen = sMax - realLen[b];
                int rowOff = b * (sMax + maxNewTokens + 1);
                for (int s = padLen; s < sMax; s++) stepAttnBuf[rowOff + s] = 1L;
            }
            // ── Step-loop IOBinding (Run 20 phase 8) ───────────────────────
            // Pre-allocate the constant-shape inputs (input_ids, audio_embeds,
            // cache_position, position_ids) and the logits OUTPUT buffer
            // once, then re-use them every iteration. attention_mask and
            // past_kv still vary shape per step so they get fresh
            // OrtValues, but everything else stops re-allocating ~60 times.
            // Binding `logits` to a pre-allocated CPU buffer also skips the
            // per-step GetTensorDataAsSpan readout that pulls logits across
            // the GPU→CPU bus implicitly.
            float[] dummyAudio = new float[B * 1 * (int)audioDim];
            // With --add-argmax bundles the head output is `next_token`
            // int64 [B, 1] — pre-allocate a small CPU buffer for it; the
            // float logits buffer is unused in that case.
            float[]? stepLogitsBuf = _hasNextTokenOutput ? null : new float[B * 1 * VocabSize];
            long[]? stepNextTokenBuf = _hasNextTokenOutput ? new long[B * 1] : null;
            long[] stepPositionIds = new long[B];

            using var stepInputIdVal = OrtValue.CreateTensorValueFromMemory(
                stepInputIds, new long[] { B, 1 });
            using var stepAudioVal = OrtValue.CreateTensorValueFromMemory(
                dummyAudio, new long[] { B, 1, audioDim });
            using var cachePosVal = OrtValue.CreateTensorValueFromMemory(
                cachePosStep, new long[] { 1 });
            using var stepPosVal = _hasPositionIdsInput
                ? OrtValue.CreateTensorValueFromMemory(stepPositionIds, new long[] { B, 1 })
                : null;
            using var stepLogitsVal = stepLogitsBuf is null
                ? null
                : OrtValue.CreateTensorValueFromMemory(
                    stepLogitsBuf, new long[] { B, 1, VocabSize });
            using var stepNextTokenVal = stepNextTokenBuf is null
                ? null
                : OrtValue.CreateTensorValueFromMemory(
                    stepNextTokenBuf, new long[] { B, 1 });

            // Lifetime management across iterations follows WhisperTurbo's
            // pattern (see WhisperTurbo.cs step loop): pastKvs holds borrowed
            // references into either prefillOutputs or the previous step's
            // bound output collection. `prevStepOutputs` keeps that
            // collection alive until the NEXT iteration has consumed (re-bound)
            // its OrtValues — at which point we can release it.
            using var cudaMemInfo = new OrtMemoryInfo(
                "Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
            IDisposableReadOnlyCollection<OrtValue>? prevStepOutputs = null;
            int pastLen = sMax;
            for (int step = 0; step < maxNewTokens && finishedCount < B; step++)
            {
                int totalLen = pastLen + 1;

                for (int b = 0; b < B; b++)
                {
                    stepInputIds[b] = finished[b] ? EosTokenId : nextTok[b];
                    int rowOff = b * (sMax + maxNewTokens + 1);
                    stepAttnBuf[rowOff + pastLen] = 1L;
                    if (_hasPositionIdsInput) stepPositionIds[b] = realLen[b] + step;
                }
                cachePosStep[0] = pastLen;

                var stepAttnFlat = new long[B * totalLen];
                for (int b = 0; b < B; b++)
                {
                    int srcOff = b * (sMax + maxNewTokens + 1);
                    int dstOff = b * totalLen;
                    Array.Copy(stepAttnBuf, srcOff, stepAttnFlat, dstOff, totalLen);
                }
                using var stepAttnVal = OrtValue.CreateTensorValueFromMemory(
                    stepAttnFlat, new long[] { B, totalLen });

                using var stepBinding = _decoder.CreateIoBinding();
                stepBinding.BindInput("input_ids",      stepInputIdVal);
                stepBinding.BindInput("audio_embeds",   stepAudioVal);
                stepBinding.BindInput("attention_mask", stepAttnVal);
                stepBinding.BindInput("cache_position", cachePosVal);
                if (stepPosVal is not null)
                    stepBinding.BindInput("position_ids", stepPosVal);
                for (int L = 0; L < NumDecoderLayers; L++)
                {
                    stepBinding.BindInput($"past_key_{L}",   pastKvs[2 * L]);
                    stepBinding.BindInput($"past_value_{L}", pastKvs[2 * L + 1]);
                }
                // Bind output order matches `decOutputNames` so
                // GetOutputValues() indices line up: [0]=head (logits or
                // next_token), [1..40]=present_key_<L>, [41..80]=
                // present_value_<L>.
                if (_hasNextTokenOutput)
                    stepBinding.BindOutput("next_token", stepNextTokenVal!);
                else
                    stepBinding.BindOutput("logits", stepLogitsVal!);
                for (int L = 0; L < NumDecoderLayers; L++)
                    stepBinding.BindOutputToDevice($"present_key_{L}", cudaMemInfo);
                for (int L = 0; L < NumDecoderLayers; L++)
                    stepBinding.BindOutputToDevice($"present_value_{L}", cudaMemInfo);

                _decoder.RunWithBinding(runOpts, stepBinding);
                var curOutputs = stepBinding.GetOutputValues();

                pastKvs = new OrtValue[2 * NumDecoderLayers];
                for (int L = 0; L < NumDecoderLayers; L++)
                {
                    pastKvs[2 * L]     = curOutputs[1 + L];
                    pastKvs[2 * L + 1] = curOutputs[1 + NumDecoderLayers + L];
                }

                // The previous iteration's collection no longer has any
                // active borrowers (we just swapped pastKvs off it), so it
                // can go. On the FIRST step, prevStepOutputs is null —
                // pastKvs was pointing into prefillOutputs there, which we
                // intentionally don't dispose because it owns the per-step
                // logits OrtValue we already disposed earlier.
                prevStepOutputs?.Dispose();
                prevStepOutputs = curOutputs;

                // Read next token: either from the GPU-argmax output
                // (`next_token`, int64) or by argmaxing the float logits
                // on CPU.
                if (_hasNextTokenOutput)
                {
                    for (int b = 0; b < B; b++)
                    {
                        if (finished[b]) { nextTok[b] = EosTokenId; continue; }
                        long tok = stepNextTokenBuf![b];
                        nextTok[b] = tok;
                        generated[b].Add(tok);
                        if (tok == EosTokenId) { finished[b] = true; finishedCount++; }
                        else if (IsRepetitionLoop(generated[b]))
                        { finished[b] = true; loopDetected[b] = true; finishedCount++; }
                    }
                }
                else
                {
                    var stepLogitsSpan = new ReadOnlySpan<float>(stepLogitsBuf);
                    for (int b = 0; b < B; b++)
                    {
                        if (finished[b]) { nextTok[b] = EosTokenId; continue; }
                        long tok = ArgmaxRowLastPosition(stepLogitsSpan, b, 1, VocabSize);
                        nextTok[b] = tok;
                        generated[b].Add(tok);
                        if (tok == EosTokenId) { finished[b] = true; finishedCount++; }
                        else if (IsRepetitionLoop(generated[b]))
                        { finished[b] = true; loopDetected[b] = true; finishedCount++; }
                    }
                }
                pastLen = totalLen;
                stepCountLocal++;
            }

            prevStepOutputs?.Dispose();
            if (swStep != null) { swStep.Stop(); stepLocal = swStep.ElapsedMilliseconds; }
        }

        // Trim trailing EOS from every row, but ONLY trim the periodic loop
        // tail on rows where the runtime detector classified the row as a
        // runaway. Otherwise natural emphatic repetition (e.g. a speaker
        // saying "no, no, no!") would be silently truncated.
        // Both the decoded text AND the trimmed token list are surfaced so
        // the GUI's synthetic-linear-timestamp logic in TranscriptionService
        // can use real BPE tokens for editor word-sync (token count drives
        // BuildSyntheticTokenTimestamps' interpolation granularity).
        var rows = new (string text, long[] tokens)[B];
        for (int b = 0; b < B; b++)
        {
            var toks = generated[b];
            int realCount = toks.Count;
            while (realCount > 0 && toks[realCount - 1] == EosTokenId) realCount--;
            int trimmed = loopDetected[b]
                ? TrimRepetitionTail(toks, realCount)
                : realCount;
            if (trimmed == toks.Count)
                rows[b] = (DecodeTokens(toks), toks.ToArray());
            else
            {
                var trimmedToks = toks.GetRange(0, trimmed);
                rows[b] = (DecodeTokens(trimmedToks), trimmedToks.ToArray());
            }
        }

        if (swBatch != null)
        {
            swBatch.Stop();
            long total = swBatch.ElapsedMilliseconds;
            long accounted = melLocal + encLocal + projLocal + prefillLocal + stepLocal;
            long overhead = Math.Max(0, total - accounted);
            lock (_profileLock)
            {
                MelMs += melLocal; EncMs += encLocal; ProjMs += projLocal;
                PrefillMs += prefillLocal; StepLoopMs += stepLocal; OverheadMs += overhead;
                BatchCount++; RowCount += B; StepCount += stepCountLocal;
                if (B > MaxBatchSeen) MaxBatchSeen = B;
            }
            Console.Error.WriteLine(
                $"[granite-prof] B={B} total={total}ms (mel={melLocal} enc={encLocal} "
              + $"proj={projLocal} prefill={prefillLocal} step={stepLocal}ms x{stepCountLocal} "
              + $"overhead={overhead}ms)");
        }
        return rows;
    }

    // ── Static-shape decoder path (Run 15 phase 2c, issue #41) ─────────────
    /// <summary>
    /// Transcribes a batch through the static-shape decoder bundle. The
    /// graph pins past_kv to (StaticBatchSize, NumKvHeads, StaticPastLen,
    /// HeadDim) and audio_embeds to (StaticBatchSize, StaticAudioLen,
    /// audioDim); the wrapper-level slice in the export keeps present_kv
    /// the same shape as past_kv so the chained Run-with-OrtValue pattern
    /// works verbatim, GPU-resident throughout. Compared to the dynamic
    /// path (TranscribeBatch above) the differences are:
    ///
    ///   - Real batch size B_real may be less than StaticBatchSize; we
    ///     pad to StaticBatchSize with PAD-token rows whose attention
    ///     mask is fully zero. They get computed but discarded.
    ///   - audio_embeds is right-zero-padded to StaticAudioLen.
    ///   - past_kv comes in at past_len = StaticPastLen (not 0); the
    ///     attention mask gates all StaticPastLen positions out at
    ///     prefill. The shared all-zero past_kv buffer is allocated once
    ///     in the ctor and reused.
    ///   - cache_position values are StaticPastLen, StaticPastLen+1, ...
    ///     because the empty-prefix cache positions [0, StaticPastLen)
    ///     are conceptually before-the-start.
    ///   - At each step the attention mask shifts left: the cache
    ///     positions that were "real" creep one slot earlier as the
    ///     sliding window absorbs the new token.
    ///
    /// All other logic (per-row mel/encoder/projector, repetition-loop
    /// detection, EOS handling, profile bookkeeping) mirrors TranscribeBatch.
    ///
    /// **Status (Run 20):** The static-shape contract is a 3.7× wall-time
    /// regression vs the dynamic path. Kept for back-compat with any
    /// static-shape bundle still on disk; prefer the dynamic export +
    /// MHA rewriter + IOBinding path (the default TranscribeBatch route
    /// when the bundle's past_kv batch dim is symbolic).
    /// </summary>
    private (string text, long[] tokens)[] TranscribeBatchStatic(float[][] waveforms, int maxNewTokens)
    {
        int realB = waveforms.Length;
        if (realB == 0) return Array.Empty<(string, long[])>();
        if (realB > _staticBatchSize)
            throw new InvalidOperationException(
                $"Static-shape decoder requires batch size <= {_staticBatchSize}; got {realB}.");

        // GQA bundles split by capability:
        //   - phase-2 (has `position_ids` input): per-row RoPE positions
        //     let the whole batch run together. Use TranscribeStaticGqaBatched.
        //   - phase-1 (no position_ids): cache_position is shared across the
        //     batch, so we can't mix variable-length prompts. Fall back to
        //     serialising one segment per call.
        if (_isGqaBundle && _hasPositionIdsInput)
            return TranscribeStaticGqaBatched(waveforms, maxNewTokens);

        if (_isGqaBundle)
        {
            var gqaRows = new (string text, long[] tokens)[realB];
            for (int i = 0; i < realB; i++)
                gqaRows[i] = TranscribeStaticGqaSingle(waveforms[i], maxNewTokens);
            return gqaRows;
        }

        int B = _staticBatchSize;
        int A = _staticAudioLen;
        int P = _staticPastLen;

        var swBatch = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        long melLocal = 0, encLocal = 0, projLocal = 0;

        // ── Per-row mel + encoder + projector (only for real rows) ──────
        var audioEmbeds = new float[realB][];
        var nAudio      = new int[realB];
        long audioDim   = 2048;
        for (int b = 0; b < realB; b++)
        {
            int audioTokens = NumAudioTokens(waveforms[b].Length);
            int promptLen   = AsrPromptPrefix.Length + audioTokens + AsrPromptSuffix.Length;
            if (promptLen + maxNewTokens > P)
            {
                throw new InvalidOperationException(
                    $"Segment too long for static-shape bundle: prompt+max_new_tokens "
                  + $"({promptLen + maxNewTokens}) exceeds pinned past_len ({P}). "
                  + $"Re-export with a larger --static-past-len, or pre-segment audio.");
            }
            if (audioTokens > A)
            {
                throw new InvalidOperationException(
                    $"Segment too long for static-shape bundle: audio_tokens ({audioTokens}) "
                  + $"exceeds pinned audio_len ({A}). Re-export with a larger "
                  + $"--static-audio-len, or pre-segment audio.");
            }

            var swStage = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
            var audioT = new DenseTensor<float>(waveforms[b], [1, waveforms[b].Length]);
            using var melResults = _mel.Run([NamedOnnxValue.CreateFromTensor("audio", audioT)]);
            var inputFeatures = melResults[0].AsTensor<float>();
            var ifShape = inputFeatures.Dimensions.ToArray();
            var ifData = inputFeatures is DenseTensor<float> dT ? dT.ToArray() : inputFeatures.ToArray();
            if (swStage != null) { swStage.Stop(); melLocal += swStage.ElapsedMilliseconds; swStage.Restart(); }

            using var encResults = _encoder.Run([NamedOnnxValue.CreateFromTensor(
                "input_features", new DenseTensor<float>(ifData, ifShape))]);
            var encoderHidden = encResults[0].AsTensor<float>();
            var encShape = encoderHidden.Dimensions.ToArray();
            var encData = encoderHidden.ToArray();
            if (swStage != null) { swStage.Stop(); encLocal += swStage.ElapsedMilliseconds; swStage.Restart(); }

            using var projResults = _projector.Run([NamedOnnxValue.CreateFromTensor(
                "encoder_hidden", new DenseTensor<float>(encData, encShape))]);
            var ae = projResults[0].AsTensor<float>();
            var projShape = ae.Dimensions.ToArray();
            if (projShape[1] != audioTokens)
                throw new InvalidOperationException(
                    $"Audio token count mismatch (row {b}): formula predicts {audioTokens}, "
                  + $"projector produced {projShape[1]}.");
            audioEmbeds[b] = ae.ToArray();
            nAudio[b]      = audioTokens;
            audioDim       = projShape[2];
            if (swStage != null) { swStage.Stop(); projLocal += swStage.ElapsedMilliseconds; }
        }

        // ── Build padded prompt at fixed B ───────────────────────────────
        var realLen = new int[B];   // 0 for padding rows (b >= realB)
        int sMaxReal = 0;
        for (int b = 0; b < realB; b++)
        {
            realLen[b] = AsrPromptPrefix.Length + nAudio[b] + AsrPromptSuffix.Length;
            if (realLen[b] > sMaxReal) sMaxReal = realLen[b];
        }
        int S = sMaxReal;  // graph keeps seq dynamic; use real max across the batch

        var inputIdsBatch = new long[B * S];
        for (int b = 0; b < realB; b++)
        {
            int padLen = S - realLen[b];
            int rowOff = b * S;
            for (int s = 0; s < padLen; s++) inputIdsBatch[rowOff + s] = PadTokenId;
            int p = rowOff + padLen;
            for (int i = 0; i < AsrPromptPrefix.Length; i++) inputIdsBatch[p++] = AsrPromptPrefix[i];
            for (int i = 0; i < nAudio[b]; i++)              inputIdsBatch[p++] = AudioTokenId;
            for (int i = 0; i < AsrPromptSuffix.Length; i++) inputIdsBatch[p++] = AsrPromptSuffix[i];
        }
        // Padding rows (b >= realB): all PadTokenId. attention_mask stays 0.
        for (int b = realB; b < B; b++)
        {
            int rowOff = b * S;
            for (int s = 0; s < S; s++) inputIdsBatch[rowOff + s] = PadTokenId;
        }

        // attention_mask at prefill: shape [B, P + S]. First P positions are
        // empty cache (mask=0). Remaining S per row: padLen zeros + realLen ones.
        var attnMaskPrefill = new long[B * (P + S)];
        for (int b = 0; b < realB; b++)
        {
            int padLen = S - realLen[b];
            int rowOff = b * (P + S);
            for (int s = padLen; s < S; s++) attnMaskPrefill[rowOff + P + s] = 1L;
        }

        // audio_embeds at prefill: [B, A, audioDim], real rows zero-padded
        // up to A on the audio_len axis. Padding rows are all zero.
        var audioEmbedsBatch = new float[B * A * audioDim];
        for (int b = 0; b < realB; b++)
        {
            int rowOff = b * A * (int)audioDim;
            int n = nAudio[b] * (int)audioDim;
            if (n > 0) Array.Copy(audioEmbeds[b], 0, audioEmbedsBatch, rowOff, n);
        }

        // cache_position at prefill: positions P, P+1, ..., P+S-1.
        // The first P cache slots are conceptually before-the-start (zero
        // past + masked) so the new tokens land at "real" positions starting
        // at index P. Granite's RoPE uses cache_position as the absolute
        // position; with our right-aligned cache layout this matches the
        // model's expected position numbering.
        var cachePosPrefill = new long[S];
        for (int i = 0; i < S; i++) cachePosPrefill[i] = P + i;

        // ── Decoder I/O names ────────────────────────────────────────────
        var decInputNames = new List<string>(4 + 2 * NumDecoderLayers)
        { "input_ids", "audio_embeds", "attention_mask", "cache_position" };
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            decInputNames.Add($"past_key_{L}");
            decInputNames.Add($"past_value_{L}");
        }
        var decOutputNames = new List<string>(1 + 2 * NumDecoderLayers) { "logits" };
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_key_{L}");
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_value_{L}");

        var generated = new List<long>[B];
        for (int b = 0; b < B; b++) generated[b] = new List<long>(maxNewTokens);
        var finished      = new bool[B];
        var loopDetected  = new bool[B];
        int finishedCount = 0;
        var nextTok       = new long[B];

        // Mark padding rows as finished — their decoded output is discarded
        // anyway, and we don't want them dragging the rest of the batch
        // toward maxNewTokens.
        for (int b = realB; b < B; b++) { finished[b] = true; finishedCount++; }

        long prefillLocal = 0, stepLocal = 0;
        long stepCountLocal = 0;
        using (var runOpts = new RunOptions())
        {
            var swPrefill = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;

            // ── Prefill ──────────────────────────────────────────────────
            using var inputIdsVal = OrtValue.CreateTensorValueFromMemory(
                inputIdsBatch, new long[] { B, S });
            using var audioVal = OrtValue.CreateTensorValueFromMemory(
                audioEmbedsBatch, new long[] { B, A, audioDim });
            using var attnVal = OrtValue.CreateTensorValueFromMemory(
                attnMaskPrefill, new long[] { B, P + S });
            using var cpVal = OrtValue.CreateTensorValueFromMemory(
                cachePosPrefill, new long[] { S });

            var prefillInputs = new List<OrtValue>(4 + 2 * NumDecoderLayers)
            { inputIdsVal, audioVal, attnVal, cpVal };
            // Reuse the shared zero past_kv from the ctor — read-only at prefill.
            prefillInputs.AddRange(_staticZeroPastKv!);

            var prefillOutputs = _decoder.Run(runOpts, decInputNames, prefillInputs, decOutputNames);

            // Argmax per real row at the LAST position (S-1); padding rows
            // don't contribute. Left-padding inside the row means the real
            // last token always sits at S-1 after the prompt.
            var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
            for (int b = 0; b < realB; b++)
            {
                long tok = ArgmaxRowLastPosition(prefillLogitsSpan, b, S, VocabSize);
                nextTok[b] = tok;
                generated[b].Add(tok);
                if (tok == EosTokenId) { finished[b] = true; finishedCount++; }
            }
            prefillOutputs[0].Dispose();

            // pastKvs = prefill's sliced output. Already shape (B, H, P, D).
            var pastKvs = new OrtValue[2 * NumDecoderLayers];
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                pastKvs[2 * L]     = prefillOutputs[1 + L];
                pastKvs[2 * L + 1] = prefillOutputs[1 + NumDecoderLayers + L];
            }

            if (swPrefill != null) { swPrefill.Stop(); prefillLocal = swPrefill.ElapsedMilliseconds; }
            var swStep = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;

            // ── Step loop ────────────────────────────────────────────────
            // cache_position monotonically increases: prefill used [P..P+S-1],
            // so step 0 starts at P+S and grows by 1 per step. Granite uses
            // RoPE only; relative-position dependence dominates so the
            // absolute offset (starting at P instead of 0) is shift-
            // invariant in attention math.
            long[] cachePosStep = new long[1];
            // Static graph pins audio_embeds shape to (B, A, audioDim). At
            // step time the audio fuse collapses to a no-op (no AudioTokenId
            // in the next-token input), so we pass an A-sized zero buffer.
            // Allocated once and the OrtValue recreated each step pointing
            // at the same buffer.
            float[] dummyAudio = new float[B * A * (int)audioDim];

            for (int step = 0; step < maxNewTokens && finishedCount < B; step++)
            {
                cachePosStep[0] = P + S + step;
                long[] stepInputIds = new long[B];
                for (int b = 0; b < B; b++)
                {
                    stepInputIds[b] = (b >= realB || finished[b]) ? EosTokenId : nextTok[b];
                }

                // Build attention mask for this step: shape (B, P + 1).
                // For row b: (P - realLen[b] - step) zeros + (realLen[b] +
                // step + 1) ones. As step advances, the "real" prefix
                // length grows. The pinning P >= prompt + max_new_tokens
                // (checked above) keeps realStart >= 0.
                var stepMask = new long[B * (P + 1)];
                for (int b = 0; b < realB; b++)
                {
                    int realStart = P - realLen[b] - step;
                    int rowOff = b * (P + 1);
                    for (int t = realStart; t < P + 1; t++) stepMask[rowOff + t] = 1L;
                }
                // Padding rows: leave mask all zeros so the decoder treats
                // them as completely-padded; their output is discarded.

                using var stepInputIdVal = OrtValue.CreateTensorValueFromMemory(
                    stepInputIds, new long[] { B, 1 });
                using var stepAudioVal = OrtValue.CreateTensorValueFromMemory(
                    dummyAudio, new long[] { B, A, audioDim });
                using var stepAttnVal = OrtValue.CreateTensorValueFromMemory(
                    stepMask, new long[] { B, P + 1 });
                using var cachePosVal = OrtValue.CreateTensorValueFromMemory(
                    cachePosStep, new long[] { 1 });

                var stepInputs = new List<OrtValue>(4 + 2 * NumDecoderLayers)
                { stepInputIdVal, stepAudioVal, stepAttnVal, cachePosVal };
                stepInputs.AddRange(pastKvs);

                var outputs = _decoder.Run(runOpts, decInputNames, stepInputs, decOutputNames);

                var oldPast = pastKvs;
                pastKvs = new OrtValue[2 * NumDecoderLayers];
                for (int L = 0; L < NumDecoderLayers; L++)
                {
                    pastKvs[2 * L]     = outputs[1 + L];
                    pastKvs[2 * L + 1] = outputs[1 + NumDecoderLayers + L];
                }

                var stepLogits = outputs[0];
                var stepLogitsSpan = stepLogits.GetTensorDataAsSpan<float>();
                for (int b = 0; b < realB; b++)
                {
                    if (finished[b]) { nextTok[b] = EosTokenId; continue; }
                    long tok = ArgmaxRowLastPosition(stepLogitsSpan, b, 1, VocabSize);
                    nextTok[b] = tok;
                    generated[b].Add(tok);
                    if (tok == EosTokenId)
                    {
                        finished[b] = true;
                        finishedCount++;
                    }
                    else if (IsRepetitionLoop(generated[b]))
                    {
                        finished[b] = true;
                        loopDetected[b] = true;
                        finishedCount++;
                    }
                }
                stepLogits.Dispose();
                foreach (var ov in oldPast) ov.Dispose();
                stepCountLocal++;
            }

            foreach (var ov in pastKvs) ov.Dispose();
            if (swStep != null) { swStep.Stop(); stepLocal = swStep.ElapsedMilliseconds; }
        }

        // Trim trailing EOS / runaway-loop tail per row (real rows only).
        var rows = new (string text, long[] tokens)[realB];
        for (int b = 0; b < realB; b++)
        {
            var toks = generated[b];
            int realCount = toks.Count;
            while (realCount > 0 && toks[realCount - 1] == EosTokenId) realCount--;
            int trimmed = loopDetected[b]
                ? TrimRepetitionTail(toks, realCount)
                : realCount;
            if (trimmed == toks.Count)
                rows[b] = (DecodeTokens(toks), toks.ToArray());
            else
            {
                var trimmedToks = toks.GetRange(0, trimmed);
                rows[b] = (DecodeTokens(trimmedToks), trimmedToks.ToArray());
            }
        }

        if (swBatch != null)
        {
            swBatch.Stop();
            long total = swBatch.ElapsedMilliseconds;
            long accounted = melLocal + encLocal + projLocal + prefillLocal + stepLocal;
            long overhead = Math.Max(0, total - accounted);
            lock (_profileLock)
            {
                MelMs += melLocal; EncMs += encLocal; ProjMs += projLocal;
                PrefillMs += prefillLocal; StepLoopMs += stepLocal; OverheadMs += overhead;
                BatchCount++; RowCount += realB; StepCount += stepCountLocal;
                if (realB > MaxBatchSeen) MaxBatchSeen = realB;
            }
            Console.Error.WriteLine(
                $"[granite-prof] B={realB}/{B} total={total}ms (mel={melLocal} enc={encLocal} "
              + $"proj={projLocal} prefill={prefillLocal} step={stepLocal}ms x{stepCountLocal} "
              + $"overhead={overhead}ms) static");
        }
        return rows;
    }

    /// <summary>
    /// Single-segment static-shape decode against a GQA-rewritten bundle
    /// (Run 20). Padded to B = _staticBatchSize with dummy rows so the
    /// pre-built zero past-KV buffer can be reused; the dummies share Q's
    /// position embeddings with row 0 (correct because realLen is uniform
    /// across the batch by construction) and contribute nothing to row 0's
    /// output since GQA is per-row.
    ///
    /// Cache layout: left-aligned in a fixed (B, kv_num_heads, max_seq,
    /// head_dim) buffer. Real K/V live at [0, realLen) after prefill and
    /// grow rightward by 1 per step. GQA's seqlens_k input tells the kernel
    /// where the real K ends, so attention compute scales with the actual
    /// past length, not the buffer length — the static-padding compute-skip
    /// win this whole migration is for.
    ///
    /// **Status (Run 20 phase 4 onwards):** Superseded by the dynamic
    /// MHA-fused + IOBinding path (~3.6 s vs this path's ~13 s on the
    /// 60 s test clip). Kept for back-compat with phase 1 GQA-static
    /// bundles. New exports should use dynamic shapes.
    /// </summary>
    private (string text, long[] tokens) TranscribeStaticGqaSingle(
        float[] waveform, int maxNewTokens)
    {
        int B = _staticBatchSize;
        int A = _staticAudioLen;
        int P = _staticPastLen;

        int audioTokens = NumAudioTokens(waveform.Length);
        int realLen = AsrPromptPrefix.Length + audioTokens + AsrPromptSuffix.Length;
        if (realLen + maxNewTokens > P)
            throw new InvalidOperationException(
                $"Segment too long for static-shape GQA bundle: prompt+max_new_tokens "
              + $"({realLen + maxNewTokens}) exceeds pinned past_len ({P}).");
        if (audioTokens > A)
            throw new InvalidOperationException(
                $"Segment too long for static-shape GQA bundle: audio_tokens "
              + $"({audioTokens}) exceeds pinned audio_len ({A}).");

        var swBatch = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        long melLocal = 0, encLocal = 0, projLocal = 0;

        // ── Mel + encoder + projector ──────────────────────────────────
        var swStage = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        var audioT = new DenseTensor<float>(waveform, [1, waveform.Length]);
        using var melResults = _mel.Run(
            [NamedOnnxValue.CreateFromTensor("audio", audioT)]);
        var inputFeatures = melResults[0].AsTensor<float>();
        var ifShape = inputFeatures.Dimensions.ToArray();
        var ifData = inputFeatures is DenseTensor<float> dT
            ? dT.ToArray() : inputFeatures.ToArray();
        if (swStage != null) { swStage.Stop(); melLocal = swStage.ElapsedMilliseconds; swStage.Restart(); }

        using var encResults = _encoder.Run(
            [NamedOnnxValue.CreateFromTensor(
                "input_features", new DenseTensor<float>(ifData, ifShape))]);
        var encoderHidden = encResults[0].AsTensor<float>();
        var encShape = encoderHidden.Dimensions.ToArray();
        var encData = encoderHidden.ToArray();
        if (swStage != null) { swStage.Stop(); encLocal = swStage.ElapsedMilliseconds; swStage.Restart(); }

        using var projResults = _projector.Run(
            [NamedOnnxValue.CreateFromTensor(
                "encoder_hidden", new DenseTensor<float>(encData, encShape))]);
        var ae = projResults[0].AsTensor<float>();
        long audioDim = ae.Dimensions[2];
        var audioEmbedsRow = ae.ToArray();
        if (swStage != null) { swStage.Stop(); projLocal = swStage.ElapsedMilliseconds; }

        // ── Build prefill inputs ───────────────────────────────────────
        // input_ids [B, realLen]: row 0 is the real prompt, dummies are PAD.
        var inputIdsBatch = new long[B * realLen];
        int p = 0;
        for (int i = 0; i < AsrPromptPrefix.Length; i++)
            inputIdsBatch[p++] = AsrPromptPrefix[i];
        for (int i = 0; i < audioTokens; i++)
            inputIdsBatch[p++] = AudioTokenId;
        for (int i = 0; i < AsrPromptSuffix.Length; i++)
            inputIdsBatch[p++] = AsrPromptSuffix[i];
        for (int b = 1; b < B; b++)
        {
            int off = b * realLen;
            for (int i = 0; i < realLen; i++) inputIdsBatch[off + i] = PadTokenId;
        }

        // audio_embeds [B, A, audioDim]: row 0 fills the first audioTokens
        // entries; rest of A and all dummies remain zero.
        var audioEmbedsBatch = new float[B * A * (int)audioDim];
        Array.Copy(audioEmbedsRow, 0, audioEmbedsBatch, 0,
                   audioTokens * (int)audioDim);

        // attention_mask [B, realLen]: row 0 all 1s, dummies all 0s. (Where
        // this propagates to where_2 inside the wrapper is dead code in the
        // GQA-rewritten graph, but the input is still part of the graph
        // signature so we must pass a valid tensor.)
        var attnMaskBatch = new long[B * realLen];
        for (int i = 0; i < realLen; i++) attnMaskBatch[i] = 1L;

        // cache_position [realLen]: left-aligned positions [0..realLen-1].
        // HF's RoPE uses these to look up cos/sin caches.
        var cachePosBuf = new long[realLen];
        for (int i = 0; i < realLen; i++) cachePosBuf[i] = i;

        // seqlens_k [B]: total_seq - 1 = realLen - 1 (uniform across rows).
        var seqlensKBuf = new int[B];
        for (int b = 0; b < B; b++) seqlensKBuf[b] = realLen - 1;
        var totalSeqBuf = new int[] { realLen };

        // ── Decoder I/O names ──────────────────────────────────────────
        var decInputNames = new List<string>(6 + 2 * NumDecoderLayers)
        { "input_ids", "audio_embeds", "attention_mask", "cache_position" };
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            decInputNames.Add($"past_key_{L}");
            decInputNames.Add($"past_value_{L}");
        }
        decInputNames.Add("seqlens_k");
        decInputNames.Add("total_sequence_length");

        var decOutputNames = new List<string>(1 + 2 * NumDecoderLayers) { "logits" };
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_key_{L}");
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_value_{L}");

        var generated = new List<long>(maxNewTokens);
        bool finished = false;
        bool looped = false;

        long prefillLocal = 0, stepLocal = 0;
        long stepCountLocal = 0;

        using (var runOpts = new RunOptions())
        {
            var swPrefill = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;

            // ── Prefill ────────────────────────────────────────────────
            using var inputIdsVal = OrtValue.CreateTensorValueFromMemory(
                inputIdsBatch, new long[] { B, realLen });
            using var audioVal = OrtValue.CreateTensorValueFromMemory(
                audioEmbedsBatch, new long[] { B, A, audioDim });
            using var attnVal = OrtValue.CreateTensorValueFromMemory(
                attnMaskBatch, new long[] { B, realLen });
            using var cpVal = OrtValue.CreateTensorValueFromMemory(
                cachePosBuf, new long[] { realLen });
            using var seqlensVal = OrtValue.CreateTensorValueFromMemory(
                seqlensKBuf, new long[] { B });
            using var totalSeqVal = OrtValue.CreateTensorValueFromMemory(
                totalSeqBuf, Array.Empty<long>());

            var prefillInputs = new List<OrtValue>(6 + 2 * NumDecoderLayers)
            { inputIdsVal, audioVal, attnVal, cpVal };
            prefillInputs.AddRange(_staticZeroPastKv!);
            prefillInputs.Add(seqlensVal);
            prefillInputs.Add(totalSeqVal);

            var prefillOutputs = _decoder.Run(
                runOpts, decInputNames, prefillInputs, decOutputNames);

            var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
            long nextTok = ArgmaxRowLastPosition(prefillLogitsSpan, 0, realLen, VocabSize);
            generated.Add(nextTok);
            if (nextTok == EosTokenId) finished = true;
            prefillOutputs[0].Dispose();

            var pastKvs = new OrtValue[2 * NumDecoderLayers];
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                pastKvs[2 * L]     = prefillOutputs[1 + L];
                pastKvs[2 * L + 1] = prefillOutputs[1 + NumDecoderLayers + L];
            }

            if (swPrefill != null) { swPrefill.Stop(); prefillLocal = swPrefill.ElapsedMilliseconds; }
            var swStep = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;

            // ── Step loop ──────────────────────────────────────────────
            long[] stepInputIds = new long[B];
            float[] dummyAudio = new float[B * A * (int)audioDim];
            for (int step = 0; step < maxNewTokens && !finished; step++)
            {
                int totalLenCurrent = realLen + step + 1;
                int seqlensK = realLen + step;  // (total - 1)

                stepInputIds[0] = nextTok;
                for (int b = 1; b < B; b++) stepInputIds[b] = EosTokenId;

                // attention_mask [B, totalLenCurrent]: row 0 attends to all
                // positions [0..totalLenCurrent), dummies see nothing.
                var stepMaskBatch = new long[B * totalLenCurrent];
                for (int i = 0; i < totalLenCurrent; i++) stepMaskBatch[i] = 1L;

                // cache_position [1]: RoPE position of the new token.
                var cachePosStep = new long[] { seqlensK };

                for (int b = 0; b < B; b++) seqlensKBuf[b] = seqlensK;
                totalSeqBuf[0] = totalLenCurrent;

                using var stepInputIdVal = OrtValue.CreateTensorValueFromMemory(
                    stepInputIds, new long[] { B, 1 });
                using var stepAudioVal = OrtValue.CreateTensorValueFromMemory(
                    dummyAudio, new long[] { B, A, audioDim });
                using var stepAttnVal = OrtValue.CreateTensorValueFromMemory(
                    stepMaskBatch, new long[] { B, totalLenCurrent });
                using var cachePosVal = OrtValue.CreateTensorValueFromMemory(
                    cachePosStep, new long[] { 1 });
                using var stepSeqlensVal = OrtValue.CreateTensorValueFromMemory(
                    seqlensKBuf, new long[] { B });
                using var stepTotalSeqVal = OrtValue.CreateTensorValueFromMemory(
                    totalSeqBuf, Array.Empty<long>());

                var stepInputs = new List<OrtValue>(6 + 2 * NumDecoderLayers)
                { stepInputIdVal, stepAudioVal, stepAttnVal, cachePosVal };
                stepInputs.AddRange(pastKvs);
                stepInputs.Add(stepSeqlensVal);
                stepInputs.Add(stepTotalSeqVal);

                var outputs = _decoder.Run(
                    runOpts, decInputNames, stepInputs, decOutputNames);

                var oldPast = pastKvs;
                pastKvs = new OrtValue[2 * NumDecoderLayers];
                for (int L = 0; L < NumDecoderLayers; L++)
                {
                    pastKvs[2 * L]     = outputs[1 + L];
                    pastKvs[2 * L + 1] = outputs[1 + NumDecoderLayers + L];
                }

                var stepLogits = outputs[0];
                var stepLogitsSpan = stepLogits.GetTensorDataAsSpan<float>();
                nextTok = ArgmaxRowLastPosition(stepLogitsSpan, 0, 1, VocabSize);
                generated.Add(nextTok);

                if (nextTok == EosTokenId) finished = true;
                else if (IsRepetitionLoop(generated)) { finished = true; looped = true; }

                stepLogits.Dispose();
                foreach (var ov in oldPast) ov.Dispose();
                stepCountLocal++;
            }

            foreach (var ov in pastKvs) ov.Dispose();
            if (swStep != null) { swStep.Stop(); stepLocal = swStep.ElapsedMilliseconds; }
        }

        // Trim trailing EOS and loop tails.
        int realCount = generated.Count;
        while (realCount > 0 && generated[realCount - 1] == EosTokenId) realCount--;
        int trimmed = looped
            ? TrimRepetitionTail(generated, realCount)
            : realCount;
        long[] outputTokens;
        string text;
        if (trimmed == generated.Count)
        {
            outputTokens = generated.ToArray();
            text = DecodeTokens(generated);
        }
        else
        {
            var trimmedToks = generated.GetRange(0, trimmed);
            outputTokens = trimmedToks.ToArray();
            text = DecodeTokens(trimmedToks);
        }

        if (swBatch != null)
        {
            swBatch.Stop();
            long total = swBatch.ElapsedMilliseconds;
            long accounted = melLocal + encLocal + projLocal + prefillLocal + stepLocal;
            long overhead = Math.Max(0, total - accounted);
            lock (_profileLock)
            {
                MelMs += melLocal; EncMs += encLocal; ProjMs += projLocal;
                PrefillMs += prefillLocal; StepLoopMs += stepLocal; OverheadMs += overhead;
                BatchCount++; RowCount++; StepCount += stepCountLocal;
                if (B > MaxBatchSeen) MaxBatchSeen = B;
            }
            Console.Error.WriteLine(
                $"[granite-prof] B=1/{B} total={total}ms (mel={melLocal} enc={encLocal} "
              + $"proj={projLocal} prefill={prefillLocal} step={stepLocal}ms x{stepCountLocal} "
              + $"overhead={overhead}ms) static-gqa");
        }

        return (text, outputTokens);
    }

    /// <summary>
    /// Batched static-shape decode against a GQA-rewritten bundle that
    /// also exposes a `position_ids` graph input (Run 20 phase 2 export
    /// with `--unified-position-ids`).
    ///
    /// Prompts are right-padded to S = max(realLen): each real row's
    /// prompt sits at [0, realLen[b]) of the row, and PAD tokens fill
    /// [realLen[b], S). Dummy rows (b ≥ realB) are all PAD.
    ///
    /// At prefill we use a UNIFORM seqlens_k = S - 1 — GQA's write
    /// region is derived from seqlens_k - new_S + 1 and won't accept
    /// per-row write positions inside one call. This writes pad K into
    /// past_kv[b, realLen[b], S) for shorter rows; that contamination
    /// is paid by the prefill's own attention compute and would also
    /// poison step-time attention if we left it there.
    ///
    /// At STEP time we switch to per-row seqlens_k[b] = realLen[b] + k,
    /// so GQA's write position is exactly past_kv[b, realLen[b]+k].
    /// Step k=0 overwrites the first pad slot at past_kv[b, realLen[b]],
    /// step k=1 overwrites the next, etc. — the per-row attention bound
    /// [0, realLen[b]+k+1) never reaches the un-overwritten pad slots,
    /// so step-time attention is clean regardless of what prefill wrote
    /// to the pad region.
    ///
    /// position_ids are per-row throughout: real tokens get [0..realLen[b]),
    /// pad tokens get [realLen[b]..S) (continuing the sequence so RoPE
    /// is stable across the prefill→step boundary), and step k's new
    /// token uses position realLen[b] + k.
    ///
    /// **Status (Run 20 phase 4 onwards):** Superseded by the dynamic
    /// MHA-fused + IOBinding path. Kept for back-compat with phase 2
    /// GQA-static bundles that include the position_ids input.
    /// </summary>
    private (string text, long[] tokens)[] TranscribeStaticGqaBatched(
        float[][] waveforms, int maxNewTokens)
    {
        int realB = waveforms.Length;
        int B = _staticBatchSize;
        int A = _staticAudioLen;
        int P = _staticPastLen;

        var swBatch = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        long melLocal = 0, encLocal = 0, projLocal = 0;

        // ── Per-row mel + encoder + projector ────────────────────────────
        var audioEmbeds = new float[realB][];
        var nAudio      = new int[realB];
        long audioDim   = 2048;
        for (int b = 0; b < realB; b++)
        {
            int audioTokens = NumAudioTokens(waveforms[b].Length);
            int promptLen   = AsrPromptPrefix.Length + audioTokens + AsrPromptSuffix.Length;
            if (promptLen + maxNewTokens > P)
                throw new InvalidOperationException(
                    $"Segment too long for static-shape GQA bundle: prompt+max_new_tokens "
                  + $"({promptLen + maxNewTokens}) exceeds pinned past_len ({P}).");
            if (audioTokens > A)
                throw new InvalidOperationException(
                    $"Segment too long: audio_tokens ({audioTokens}) > pinned A ({A}).");

            var swStage = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
            var audioT = new DenseTensor<float>(waveforms[b], [1, waveforms[b].Length]);
            using var melResults = _mel.Run([NamedOnnxValue.CreateFromTensor("audio", audioT)]);
            var inputFeatures = melResults[0].AsTensor<float>();
            var ifShape = inputFeatures.Dimensions.ToArray();
            var ifData = inputFeatures is DenseTensor<float> dT ? dT.ToArray() : inputFeatures.ToArray();
            if (swStage != null) { swStage.Stop(); melLocal += swStage.ElapsedMilliseconds; swStage.Restart(); }

            using var encResults = _encoder.Run([NamedOnnxValue.CreateFromTensor(
                "input_features", new DenseTensor<float>(ifData, ifShape))]);
            var encoderHidden = encResults[0].AsTensor<float>();
            var encShape = encoderHidden.Dimensions.ToArray();
            var encData = encoderHidden.ToArray();
            if (swStage != null) { swStage.Stop(); encLocal += swStage.ElapsedMilliseconds; swStage.Restart(); }

            using var projResults = _projector.Run([NamedOnnxValue.CreateFromTensor(
                "encoder_hidden", new DenseTensor<float>(encData, encShape))]);
            var ae = projResults[0].AsTensor<float>();
            audioDim = ae.Dimensions[2];
            audioEmbeds[b] = ae.ToArray();
            nAudio[b] = audioTokens;
            if (swStage != null) { swStage.Stop(); projLocal += swStage.ElapsedMilliseconds; }
        }

        // ── Build padded prompt at fixed B ───────────────────────────────
        var realLen = new int[B];   // 0 for dummies
        int sMax = 0;
        for (int b = 0; b < realB; b++)
        {
            realLen[b] = AsrPromptPrefix.Length + nAudio[b] + AsrPromptSuffix.Length;
            if (realLen[b] > sMax) sMax = realLen[b];
        }
        if (sMax == 0) sMax = 1;
        int S = sMax;

        // input_ids [B, S]: right-padded. Real rows: prompt at [0, realLen[b]),
        // PAD at [realLen[b], S). Dummies: all PAD.
        var inputIdsBatch = new long[B * S];
        for (int b = 0; b < realB; b++)
        {
            int rowOff = b * S;
            int q = rowOff;
            for (int i = 0; i < AsrPromptPrefix.Length; i++) inputIdsBatch[q++] = AsrPromptPrefix[i];
            for (int i = 0; i < nAudio[b]; i++) inputIdsBatch[q++] = AudioTokenId;
            for (int i = 0; i < AsrPromptSuffix.Length; i++) inputIdsBatch[q++] = AsrPromptSuffix[i];
            for (int i = realLen[b]; i < S; i++) inputIdsBatch[rowOff + i] = PadTokenId;
        }
        for (int b = realB; b < B; b++)
        {
            int rowOff = b * S;
            for (int i = 0; i < S; i++) inputIdsBatch[rowOff + i] = PadTokenId;
        }

        // audio_embeds [B, A, audioDim]: real audio at the start of each
        // real row, zero elsewhere.
        var audioEmbedsBatch = new float[B * A * (int)audioDim];
        for (int b = 0; b < realB; b++)
        {
            int rowOff = b * A * (int)audioDim;
            int n = nAudio[b] * (int)audioDim;
            if (n > 0) Array.Copy(audioEmbeds[b], 0, audioEmbedsBatch, rowOff, n);
        }

        // attention_mask [B, S]: real rows have 1s at [0, realLen[b]).
        // Dead-code in the GQA-rewritten graph but still part of the
        // graph signature.
        var attnMaskBatch = new long[B * S];
        for (int b = 0; b < realB; b++)
        {
            int rowOff = b * S;
            for (int i = 0; i < realLen[b]; i++) attnMaskBatch[rowOff + i] = 1L;
        }

        // cache_position [S]: shared sequence index 0..S-1. Used by the
        // wrapper's now-dead RoPE/mask construction; harmless.
        var cachePosBuf = new long[S];
        for (int i = 0; i < S; i++) cachePosBuf[i] = i;

        // position_ids [B, S]: per-row. Real positions at [0..realLen[b]),
        // pad positions continue [realLen[b]..S-1] so RoPE is stable.
        // Dummies use the same scheme (they don't matter).
        var positionIdsBatch = new long[B * S];
        for (int b = 0; b < B; b++)
        {
            int rowOff = b * S;
            for (int i = 0; i < S; i++) positionIdsBatch[rowOff + i] = i;
        }

        // seqlens_k [B] int32: at prefill, UNIFORM = S - 1 (GQA can't accept
        // per-row write positions inside one call). Per-row is restored at
        // step time.
        var seqlensKBuf = new int[B];
        for (int b = 0; b < B; b++) seqlensKBuf[b] = S - 1;
        var totalSeqBuf = new int[] { S };

        // ── Decoder I/O names ────────────────────────────────────────────
        var decInputNames = new List<string>(7 + 2 * NumDecoderLayers)
        { "input_ids", "audio_embeds", "attention_mask", "cache_position", "position_ids" };
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            decInputNames.Add($"past_key_{L}");
            decInputNames.Add($"past_value_{L}");
        }
        decInputNames.Add("seqlens_k");
        decInputNames.Add("total_sequence_length");

        var decOutputNames = new List<string>(1 + 2 * NumDecoderLayers) { "logits" };
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_key_{L}");
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_value_{L}");

        var generated = new List<long>[B];
        for (int b = 0; b < B; b++) generated[b] = new List<long>(maxNewTokens);
        var finished     = new bool[B];
        var loopDetected = new bool[B];
        int finishedCount = 0;
        var nextTok      = new long[B];
        for (int b = realB; b < B; b++) { finished[b] = true; finishedCount++; }

        long prefillLocal = 0, stepLocal = 0;
        long stepCountLocal = 0;

        using (var runOpts = new RunOptions())
        {
            var swPrefill = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;

            using var inputIdsVal = OrtValue.CreateTensorValueFromMemory(
                inputIdsBatch, new long[] { B, S });
            using var audioVal = OrtValue.CreateTensorValueFromMemory(
                audioEmbedsBatch, new long[] { B, A, audioDim });
            using var attnVal = OrtValue.CreateTensorValueFromMemory(
                attnMaskBatch, new long[] { B, S });
            using var cpVal = OrtValue.CreateTensorValueFromMemory(
                cachePosBuf, new long[] { S });
            using var posVal = OrtValue.CreateTensorValueFromMemory(
                positionIdsBatch, new long[] { B, S });
            using var seqlensVal = OrtValue.CreateTensorValueFromMemory(
                seqlensKBuf, new long[] { B });
            using var totalSeqVal = OrtValue.CreateTensorValueFromMemory(
                totalSeqBuf, Array.Empty<long>());

            var prefillInputs = new List<OrtValue>(7 + 2 * NumDecoderLayers)
            { inputIdsVal, audioVal, attnVal, cpVal, posVal };
            prefillInputs.AddRange(_staticZeroPastKv!);
            prefillInputs.Add(seqlensVal);
            prefillInputs.Add(totalSeqVal);

            var prefillOutputs = _decoder.Run(
                runOpts, decInputNames, prefillInputs, decOutputNames);

            // Per-row argmax at realLen[b]-1 (last real token).
            var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
            for (int b = 0; b < realB; b++)
            {
                long tok = ArgmaxRowAtPosition(prefillLogitsSpan, b, S, realLen[b] - 1, VocabSize);
                nextTok[b] = tok;
                generated[b].Add(tok);
                if (tok == EosTokenId) { finished[b] = true; finishedCount++; }
            }
            prefillOutputs[0].Dispose();

            var pastKvs = new OrtValue[2 * NumDecoderLayers];
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                pastKvs[2 * L]     = prefillOutputs[1 + L];
                pastKvs[2 * L + 1] = prefillOutputs[1 + NumDecoderLayers + L];
            }

            if (swPrefill != null) { swPrefill.Stop(); prefillLocal = swPrefill.ElapsedMilliseconds; }
            var swStep = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;

            // ── Step loop with IOBinding (Run 20 phase 3) ─────────────────
            // Bind past_key_<L> input and present_key_<L> output to the SAME
            // OrtValue per layer. GQA detects the shared buffer (same memory
            // address for past and present) and writes new K in place at
            // past_kv[seqlens_k+1] without copying the rest of the cache.
            // Per-step memcpy elimination is 80 layers × 16 × 4 × 512 × 128
            // × 2 (bf16) = ~6.4 GB of avoided KV traffic per step iteration.
            //
            // Logits is bound to a pre-allocated CPU buffer so we can argmax
            // directly without an extra GetTensorDataAsSpan round-trip.
            float[] dummyAudio = new float[B * A * (int)audioDim];
            long[] stepInputIds = new long[B];
            long[] cachePosStep = new long[1];
            long[] stepPositionIds = new long[B];
            float[] stepLogitsBuf = new float[B * 1 * VocabSize];

            using var cudaMemInfo = new OrtMemoryInfo(
                "Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);

            for (int step = 0; step < maxNewTokens && finishedCount < B; step++)
            {
                // Per-row seqlens_k: realLen[b] + step (= write position).
                int maxTotalLen = 0;
                for (int b = 0; b < B; b++)
                {
                    int sl = (b < realB ? realLen[b] : realLen[Math.Max(0, realB - 1)]) + step;
                    seqlensKBuf[b] = sl;
                    int rowTotal = sl + 1;
                    if (rowTotal > maxTotalLen) maxTotalLen = rowTotal;
                    stepPositionIds[b] = sl;
                    stepInputIds[b] = (b >= realB || finished[b]) ? EosTokenId : nextTok[b];
                }
                int totalLen = maxTotalLen;
                totalSeqBuf[0] = totalLen;
                cachePosStep[0] = totalLen - 1;

                var stepMaskBatch = new long[B * totalLen];
                for (int b = 0; b < realB; b++)
                {
                    int rowOff = b * totalLen;
                    int end = realLen[b] + step + 1;
                    if (end > totalLen) end = totalLen;
                    for (int i = 0; i < end; i++) stepMaskBatch[rowOff + i] = 1L;
                }

                using var stepInputIdVal = OrtValue.CreateTensorValueFromMemory(
                    stepInputIds, new long[] { B, 1 });
                using var stepAudioVal = OrtValue.CreateTensorValueFromMemory(
                    dummyAudio, new long[] { B, A, audioDim });
                using var stepAttnVal = OrtValue.CreateTensorValueFromMemory(
                    stepMaskBatch, new long[] { B, totalLen });
                using var stepCpVal = OrtValue.CreateTensorValueFromMemory(
                    cachePosStep, new long[] { 1 });
                using var stepPosVal = OrtValue.CreateTensorValueFromMemory(
                    stepPositionIds, new long[] { B, 1 });
                using var stepSeqlensVal = OrtValue.CreateTensorValueFromMemory(
                    seqlensKBuf, new long[] { B });
                using var stepTotalSeqVal = OrtValue.CreateTensorValueFromMemory(
                    totalSeqBuf, Array.Empty<long>());
                using var stepLogitsVal = OrtValue.CreateTensorValueFromMemory(
                    stepLogitsBuf, new long[] { B, 1, VocabSize });

                using var stepBinding = _decoder.CreateIoBinding();
                stepBinding.BindInput("input_ids",             stepInputIdVal);
                stepBinding.BindInput("audio_embeds",          stepAudioVal);
                stepBinding.BindInput("attention_mask",        stepAttnVal);
                stepBinding.BindInput("cache_position",        stepCpVal);
                stepBinding.BindInput("position_ids",          stepPosVal);
                stepBinding.BindInput("seqlens_k",             stepSeqlensVal);
                stepBinding.BindInput("total_sequence_length", stepTotalSeqVal);
                for (int L = 0; L < NumDecoderLayers; L++)
                {
                    stepBinding.BindInput($"past_key_{L}",   pastKvs[2 * L]);
                    stepBinding.BindInput($"past_value_{L}", pastKvs[2 * L + 1]);
                    // Same OrtValue as output → GQA writes new K in place.
                    stepBinding.BindOutput($"present_key_{L}",   pastKvs[2 * L]);
                    stepBinding.BindOutput($"present_value_{L}", pastKvs[2 * L + 1]);
                }
                stepBinding.BindOutput("logits", stepLogitsVal);

                _decoder.RunWithBinding(runOpts, stepBinding);

                // Logits are now in stepLogitsBuf (CPU). pastKvs OrtValues
                // are unchanged references; their underlying GPU buffers
                // have been updated in place.
                var stepLogitsSpan = new ReadOnlySpan<float>(stepLogitsBuf);
                for (int b = 0; b < realB; b++)
                {
                    if (finished[b]) { nextTok[b] = EosTokenId; continue; }
                    long tok = ArgmaxRowLastPosition(stepLogitsSpan, b, 1, VocabSize);
                    nextTok[b] = tok;
                    generated[b].Add(tok);
                    if (tok == EosTokenId) { finished[b] = true; finishedCount++; }
                    else if (IsRepetitionLoop(generated[b]))
                    {
                        finished[b] = true; loopDetected[b] = true; finishedCount++;
                    }
                }
                stepCountLocal++;
            }

            foreach (var ov in pastKvs) ov.Dispose();
            if (swStep != null) { swStep.Stop(); stepLocal = swStep.ElapsedMilliseconds; }
        }

        // Trim + decode per real row.
        var rows = new (string text, long[] tokens)[realB];
        for (int b = 0; b < realB; b++)
        {
            var toks = generated[b];
            int realCount = toks.Count;
            while (realCount > 0 && toks[realCount - 1] == EosTokenId) realCount--;
            int trimmed = loopDetected[b] ? TrimRepetitionTail(toks, realCount) : realCount;
            if (trimmed == toks.Count)
                rows[b] = (DecodeTokens(toks), toks.ToArray());
            else
            {
                var trimmedToks = toks.GetRange(0, trimmed);
                rows[b] = (DecodeTokens(trimmedToks), trimmedToks.ToArray());
            }
        }

        if (swBatch != null)
        {
            swBatch.Stop();
            long total = swBatch.ElapsedMilliseconds;
            long accounted = melLocal + encLocal + projLocal + prefillLocal + stepLocal;
            long overhead = Math.Max(0, total - accounted);
            lock (_profileLock)
            {
                MelMs += melLocal; EncMs += encLocal; ProjMs += projLocal;
                PrefillMs += prefillLocal; StepLoopMs += stepLocal; OverheadMs += overhead;
                BatchCount++; RowCount += realB; StepCount += stepCountLocal;
                if (realB > MaxBatchSeen) MaxBatchSeen = realB;
            }
            Console.Error.WriteLine(
                $"[granite-prof] B={realB}/{B} total={total}ms (mel={melLocal} enc={encLocal} "
              + $"proj={projLocal} prefill={prefillLocal} step={stepLocal}ms x{stepCountLocal} "
              + $"overhead={overhead}ms) static-gqa-batched");
        }
        return rows;
    }

    /// <summary>
    /// Single-segment dynamic-shape decode against a GQA-rewritten bundle
    /// where the decoder is fully dynamic (no static B/A/past_len pinning).
    /// Run 20 phase 4.
    ///
    /// One segment per call: B=1, no dummies, audio_embeds sized to the
    /// segment's actual audio tokens, input_ids sized to the segment's
    /// actual prompt length, past_kv starts empty and grows by one slot
    /// per step. Eliminates the static-shape padding overhead (16× batch,
    /// up-to-12× audio_len, up-to-6× past_len) that bounded the static
    /// GQA path's perf.
    /// </summary>
    private (string text, long[] tokens) TranscribeDynamicGqaSingle(
        float[] waveform, int maxNewTokens)
    {
        int audioTokens = NumAudioTokens(waveform.Length);
        int realLen = AsrPromptPrefix.Length + audioTokens + AsrPromptSuffix.Length;
        if (realLen + maxNewTokens > MaxContextLen)
            throw new InvalidOperationException(
                $"Segment too long: prompt+max_new_tokens ({realLen + maxNewTokens}) "
              + $"exceeds Granite Speech context ({MaxContextLen}).");

        var swBatch = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        long melLocal = 0, encLocal = 0, projLocal = 0;

        // ── Mel + encoder + projector (B=1) ──────────────────────────────
        var swStage = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        var audioT = new DenseTensor<float>(waveform, [1, waveform.Length]);
        using var melResults = _mel.Run(
            [NamedOnnxValue.CreateFromTensor("audio", audioT)]);
        var inputFeatures = melResults[0].AsTensor<float>();
        var ifShape = inputFeatures.Dimensions.ToArray();
        var ifData = inputFeatures is DenseTensor<float> dT ? dT.ToArray() : inputFeatures.ToArray();
        if (swStage != null) { swStage.Stop(); melLocal = swStage.ElapsedMilliseconds; swStage.Restart(); }

        using var encResults = _encoder.Run(
            [NamedOnnxValue.CreateFromTensor(
                "input_features", new DenseTensor<float>(ifData, ifShape))]);
        var encoderHidden = encResults[0].AsTensor<float>();
        var encShape = encoderHidden.Dimensions.ToArray();
        var encData = encoderHidden.ToArray();
        if (swStage != null) { swStage.Stop(); encLocal = swStage.ElapsedMilliseconds; swStage.Restart(); }

        using var projResults = _projector.Run(
            [NamedOnnxValue.CreateFromTensor(
                "encoder_hidden", new DenseTensor<float>(encData, encShape))]);
        var ae = projResults[0].AsTensor<float>();
        long audioDim = ae.Dimensions[2];
        if (ae.Dimensions[1] != audioTokens)
            throw new InvalidOperationException(
                $"Audio token count mismatch: formula predicts {audioTokens}, "
              + $"projector produced {ae.Dimensions[1]}.");
        var audioEmbedsRow = ae.ToArray();
        if (swStage != null) { swStage.Stop(); projLocal = swStage.ElapsedMilliseconds; }

        // ── Build prefill inputs at actual sizes ────────────────────────
        var inputIds = new long[realLen];
        int p = 0;
        for (int i = 0; i < AsrPromptPrefix.Length; i++) inputIds[p++] = AsrPromptPrefix[i];
        for (int i = 0; i < audioTokens; i++) inputIds[p++] = AudioTokenId;
        for (int i = 0; i < AsrPromptSuffix.Length; i++) inputIds[p++] = AsrPromptSuffix[i];

        var attentionMask = new long[realLen];
        for (int i = 0; i < realLen; i++) attentionMask[i] = 1L;

        var cachePos = new long[realLen];
        for (int i = 0; i < realLen; i++) cachePos[i] = i;

        var positionIds = new long[realLen];
        for (int i = 0; i < realLen; i++) positionIds[i] = i;

        var seqlensKBuf = new int[] { realLen - 1 };
        var totalSeqBuf = new int[] { realLen };

        // Input/output name lists.
        var decInputNames = new List<string>(7 + 2 * NumDecoderLayers)
        { "input_ids", "audio_embeds", "attention_mask", "cache_position", "position_ids" };
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            decInputNames.Add($"past_key_{L}");
            decInputNames.Add($"past_value_{L}");
        }
        decInputNames.Add("seqlens_k");
        decInputNames.Add("total_sequence_length");

        var decOutputNames = new List<string>(1 + 2 * NumDecoderLayers) { "logits" };
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_key_{L}");
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_value_{L}");

        var generated = new List<long>(maxNewTokens);
        bool finished = false;
        bool looped = false;
        long prefillLocal = 0, stepLocal = 0;
        long stepCountLocal = 0;

        using (var runOpts = new RunOptions())
        {
            var swPrefill = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;

            // ── Prefill ────────────────────────────────────────────────
            using var inputIdsVal = OrtValue.CreateTensorValueFromMemory(
                inputIds, new long[] { 1, realLen });
            using var audioVal = OrtValue.CreateTensorValueFromMemory(
                audioEmbedsRow, new long[] { 1, audioTokens, audioDim });
            using var attnVal = OrtValue.CreateTensorValueFromMemory(
                attentionMask, new long[] { 1, realLen });
            using var cpVal = OrtValue.CreateTensorValueFromMemory(
                cachePos, new long[] { realLen });
            using var posVal = OrtValue.CreateTensorValueFromMemory(
                positionIds, new long[] { 1, realLen });
            using var seqlensVal = OrtValue.CreateTensorValueFromMemory(
                seqlensKBuf, new long[] { 1 });
            using var totalSeqVal = OrtValue.CreateTensorValueFromMemory(
                totalSeqBuf, Array.Empty<long>());

            // Empty past_kv: shape [1, NumKvHeads, 0, HeadDim]. ORT accepts
            // 0-element tensors via a dummy non-empty backing buffer.
            var emptyShape = new long[] { 1, NumKvHeads, 0, HeadDim };
            var emptyPasts = new List<OrtValue>(2 * NumDecoderLayers);
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                emptyPasts.Add(CreateEmptyPastKv(emptyShape));
                emptyPasts.Add(CreateEmptyPastKv(emptyShape));
            }

            var prefillInputs = new List<OrtValue>(7 + 2 * NumDecoderLayers)
            { inputIdsVal, audioVal, attnVal, cpVal, posVal };
            prefillInputs.AddRange(emptyPasts);
            prefillInputs.Add(seqlensVal);
            prefillInputs.Add(totalSeqVal);

            var prefillOutputs = _decoder.Run(
                runOpts, decInputNames, prefillInputs, decOutputNames);

            var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
            long nextTok = ArgmaxRowLastPosition(prefillLogitsSpan, 0, realLen, VocabSize);
            generated.Add(nextTok);
            if (nextTok == EosTokenId) finished = true;
            prefillOutputs[0].Dispose();
            foreach (var ov in emptyPasts) ov.Dispose();

            var pastKvs = new OrtValue[2 * NumDecoderLayers];
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                pastKvs[2 * L]     = prefillOutputs[1 + L];
                pastKvs[2 * L + 1] = prefillOutputs[1 + NumDecoderLayers + L];
            }

            if (swPrefill != null) { swPrefill.Stop(); prefillLocal = swPrefill.ElapsedMilliseconds; }
            var swStep = _profile ? System.Diagnostics.Stopwatch.StartNew() : null;

            // ── Step loop ──────────────────────────────────────────────
            // audio_embeds at step is a [1, 1, audioDim] zero placeholder —
            // graph requires audio_embeds.shape[1] >= 1 even when nothing
            // is being merged.
            var dummyAudio = new float[1 * 1 * (int)audioDim];
            var stepInputIds = new long[1];

            for (int step = 0; step < maxNewTokens && !finished; step++)
            {
                int seqlensK = realLen + step;       // total_seq - 1 after this call
                int totalLen = seqlensK + 1;

                stepInputIds[0] = nextTok;

                var stepMask = new long[totalLen];
                for (int i = 0; i < totalLen; i++) stepMask[i] = 1L;

                var stepCachePos = new long[] { (long)seqlensK };
                var stepPositionIds = new long[] { (long)seqlensK };
                seqlensKBuf[0] = seqlensK;
                totalSeqBuf[0] = totalLen;

                using var stepInputIdVal = OrtValue.CreateTensorValueFromMemory(
                    stepInputIds, new long[] { 1, 1 });
                using var stepAudioVal = OrtValue.CreateTensorValueFromMemory(
                    dummyAudio, new long[] { 1, 1, audioDim });
                using var stepAttnVal = OrtValue.CreateTensorValueFromMemory(
                    stepMask, new long[] { 1, totalLen });
                using var stepCpVal = OrtValue.CreateTensorValueFromMemory(
                    stepCachePos, new long[] { 1 });
                using var stepPosVal = OrtValue.CreateTensorValueFromMemory(
                    stepPositionIds, new long[] { 1, 1 });
                using var stepSeqlensVal = OrtValue.CreateTensorValueFromMemory(
                    seqlensKBuf, new long[] { 1 });
                using var stepTotalSeqVal = OrtValue.CreateTensorValueFromMemory(
                    totalSeqBuf, Array.Empty<long>());

                var stepInputs = new List<OrtValue>(7 + 2 * NumDecoderLayers)
                { stepInputIdVal, stepAudioVal, stepAttnVal, stepCpVal, stepPosVal };
                stepInputs.AddRange(pastKvs);
                stepInputs.Add(stepSeqlensVal);
                stepInputs.Add(stepTotalSeqVal);

                var outputs = _decoder.Run(
                    runOpts, decInputNames, stepInputs, decOutputNames);

                var oldPast = pastKvs;
                pastKvs = new OrtValue[2 * NumDecoderLayers];
                for (int L = 0; L < NumDecoderLayers; L++)
                {
                    pastKvs[2 * L]     = outputs[1 + L];
                    pastKvs[2 * L + 1] = outputs[1 + NumDecoderLayers + L];
                }

                var stepLogits = outputs[0];
                var stepLogitsSpan = stepLogits.GetTensorDataAsSpan<float>();
                nextTok = ArgmaxRowLastPosition(stepLogitsSpan, 0, 1, VocabSize);
                generated.Add(nextTok);
                if (nextTok == EosTokenId) finished = true;
                else if (IsRepetitionLoop(generated)) { finished = true; looped = true; }

                stepLogits.Dispose();
                foreach (var ov in oldPast) ov.Dispose();
                stepCountLocal++;
            }

            foreach (var ov in pastKvs) ov.Dispose();
            if (swStep != null) { swStep.Stop(); stepLocal = swStep.ElapsedMilliseconds; }
        }

        int realCount = generated.Count;
        while (realCount > 0 && generated[realCount - 1] == EosTokenId) realCount--;
        int trimmed = looped ? TrimRepetitionTail(generated, realCount) : realCount;
        long[] outputTokens;
        string text;
        if (trimmed == generated.Count)
        {
            outputTokens = generated.ToArray();
            text = DecodeTokens(generated);
        }
        else
        {
            var trimmedToks = generated.GetRange(0, trimmed);
            outputTokens = trimmedToks.ToArray();
            text = DecodeTokens(trimmedToks);
        }

        if (swBatch != null)
        {
            swBatch.Stop();
            long total = swBatch.ElapsedMilliseconds;
            long accounted = melLocal + encLocal + projLocal + prefillLocal + stepLocal;
            long overhead = Math.Max(0, total - accounted);
            lock (_profileLock)
            {
                MelMs += melLocal; EncMs += encLocal; ProjMs += projLocal;
                PrefillMs += prefillLocal; StepLoopMs += stepLocal; OverheadMs += overhead;
                BatchCount++; RowCount++; StepCount += stepCountLocal;
                if (1 > MaxBatchSeen) MaxBatchSeen = 1;
            }
            Console.Error.WriteLine(
                $"[granite-prof] B=1 total={total}ms (mel={melLocal} enc={encLocal} "
              + $"proj={projLocal} prefill={prefillLocal} step={stepLocal}ms x{stepCountLocal} "
              + $"overhead={overhead}ms) dynamic-gqa");
        }

        return (text, outputTokens);
    }


    /// <summary>Print accumulated profile stats and reset. Returns true if profiling was on.</summary>
    public static bool DumpProfile(System.IO.TextWriter? w = null)
    {
        if (!_profile) return false;
        w ??= Console.Error;
        lock (_profileLock)
        {
            long total = MelMs + EncMs + ProjMs + PrefillMs + StepLoopMs + OverheadMs;
            w.WriteLine();
            w.WriteLine("[granite-prof] === Aggregate ===");
            w.WriteLine($"  batches:    {BatchCount}  rows: {RowCount}  max_B: {MaxBatchSeen}  steps: {StepCount}");
            w.WriteLine($"  mel:        {MelMs} ms  ({Pct(MelMs, total)})");
            w.WriteLine($"  encoder:    {EncMs} ms  ({Pct(EncMs, total)})");
            w.WriteLine($"  projector:  {ProjMs} ms  ({Pct(ProjMs, total)})");
            w.WriteLine($"  prefill:    {PrefillMs} ms  ({Pct(PrefillMs, total)})");
            w.WriteLine($"  step-loop:  {StepLoopMs} ms  ({Pct(StepLoopMs, total)})");
            w.WriteLine($"  overhead:   {OverheadMs} ms  ({Pct(OverheadMs, total)})");
            w.WriteLine($"  total in TranscribeBatch: {total} ms");
        }
        return true;
    }

    private static string Pct(long part, long total) =>
        total > 0 ? $"{100.0 * part / total:F1}%" : "0%";

    // ── Helpers ─────────────────────────────────────────────────────────

    private static float[] ExtractSegment(float[] audio, double start, double end)
    {
        int s = Math.Max((int)(start * Config.SampleRate), 0);
        int e = Math.Min((int)(end   * Config.SampleRate), audio.Length);
        if (e <= s) return Array.Empty<float>();
        var seg = new float[e - s];
        Array.Copy(audio, s, seg, 0, e - s);
        return seg;
    }

    /// <summary>
    /// Creates a zero-length past-KV OrtValue in the dtype the decoder
    /// expects. The fp32 bundle uses Float; the BF16 mixed-precision
    /// bundle uses BFloat16. ORT requires the dtype on the input
    /// OrtValue to match the graph's declared type — mismatched fp32
    /// fed to a BF16 graph fails with "Unexpected input data type"
    /// before the run starts.
    /// </summary>
    private OrtValue CreateEmptyPastKv(long[] shape)
    {
        return _pastKvDtype switch
        {
            TensorElementType.Float    => OrtValue.CreateTensorValueFromMemory(
                Array.Empty<float>(), shape),
            TensorElementType.BFloat16 => OrtValue.CreateTensorValueFromMemory(
                Array.Empty<BFloat16>(), shape),
            TensorElementType.Float16  => OrtValue.CreateTensorValueFromMemory(
                Array.Empty<Float16>(), shape),
            _ => throw new InvalidOperationException(
                $"Unsupported past_kv dtype {_pastKvDtype}. Decoder bundle should be Float, BFloat16, or Float16."),
        };
    }

    /// <summary>
    /// Detects greedy decode loops where the model has fallen into a fixed
    /// short cycle (e.g. <c>"we, we, we, …"</c>). Returns true when the
    /// most recent <c>3 × period</c> tokens are exactly periodic for any
    /// period in [1..4]. 3 cycles is conservative — natural text rarely
    /// repeats a 1-4-token motif three times in a row, but a stuck decode
    /// always does. Once true, the row is forced to EOS so the rest of the
    /// batch isn't dragged to <c>maxNewTokens</c> waiting on it.
    /// </summary>
    private static bool IsRepetitionLoop(List<long> tokens)
    {
        const int Cycles = 3;
        for (int p = 1; p <= 4; p++)
        {
            int needed = Cycles * p;
            if (tokens.Count < needed) continue;
            int start = tokens.Count - needed;
            bool periodic = true;
            for (int i = 0; i < (Cycles - 1) * p && periodic; i++)
                if (tokens[start + i] != tokens[start + i + p]) periodic = false;
            if (periodic) return true;
        }
        return false;
    }

    /// <summary>
    /// Walks back from <paramref name="end"/> and removes the longest
    /// trailing periodic tail (period 1-4, ≥2 cycles). Used post-decode to
    /// strip the start of a detected loop from the output text. Conservative:
    /// only trims when there's a clear repeating motif.
    /// </summary>
    private static int TrimRepetitionTail(List<long> tokens, int end)
    {
        for (int p = 1; p <= 4; p++)
        {
            if (end < 2 * p) continue;
            int cycles = 0;
            int pos = end - p;
            while (pos - p >= 0)
            {
                bool match = true;
                for (int i = 0; i < p; i++)
                    if (tokens[pos + i] != tokens[pos - p + i]) { match = false; break; }
                if (!match) break;
                cycles++;
                pos -= p;
            }
            if (cycles >= 2) return pos + p;  // keep one cycle as the canonical motif
        }
        return end;
    }

    /// <summary>
    /// Argmax over the last position of row <paramref name="b"/> in a
    /// [B, seqLen, vocab] logits tensor. Used both at prefill (seqLen=S_max,
    /// last real token aligned at S_max-1 thanks to left-padding) and at
    /// step time (seqLen=1).
    /// </summary>
    private static long ArgmaxRowLastPosition(
        ReadOnlySpan<float> logits, int b, int seqLen, int vocab)
    {
        return ArgmaxRowAtPosition(logits, b, seqLen, seqLen - 1, vocab);
    }

    private static long ArgmaxRowAtPosition(
        ReadOnlySpan<float> logits, int b, int seqLen, int pos, int vocab)
    {
        int offset = (b * seqLen + pos) * vocab;
        long best = 0;
        float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = logits[offset + v];
            if (val > bestVal) { bestVal = val; best = v; }
        }
        return best;
    }

    /// <summary>
    /// Decode a sequence of token IDs to a UTF-8 string using the GPT-2
    /// ByteLevel BPE scheme (the Granite tokenizer is GPT-2 family).
    /// Special / added tokens use their `content` field directly; regular
    /// BPE tokens go through the byte-level decode table.
    /// </summary>
    public string DecodeTokens(IReadOnlyList<long> tokenIds)
    {
        var bytes = new List<byte>(tokenIds.Count * 4);
        foreach (long id in tokenIds)
        {
            int iid = (int)id;
            if (_addedTokens.TryGetValue(iid, out string? special))
            {
                bytes.AddRange(Encoding.UTF8.GetBytes(special));
                continue;
            }
            string? raw = iid >= 0 && iid < _idToToken.Length ? _idToToken[iid] : null;
            if (raw is null) continue;
            foreach (char ch in raw)
                if (_byteLevelDecode.TryGetValue(ch, out byte b))
                    bytes.Add(b);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    // ── Tokenizer loading + ByteLevel decode (port from VibeVoice) ──────

    private static (string?[] idToToken, Dictionary<int, string> addedContent)
        LoadTokenizerVocab(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var vocab = root.GetProperty("model").GetProperty("vocab");

        int maxId = 0;
        foreach (var kv in vocab.EnumerateObject())
            if (kv.Value.GetInt32() > maxId) maxId = kv.Value.GetInt32();

        var idToToken = new string?[maxId + 1];
        foreach (var kv in vocab.EnumerateObject())
            idToToken[kv.Value.GetInt32()] = kv.Name;

        var addedContent = new Dictionary<int, string>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var at in addedTokens.EnumerateArray())
            {
                int atId = at.GetProperty("id").GetInt32();
                string atContent = at.GetProperty("content").GetString() ?? "";
                addedContent[atId] = atContent;
            }
        }

        return (idToToken, addedContent);
    }

    // ── VRAM cost model ─────────────────────────────────────────────────

    // Audio-token count derived purely from duration (samples = durSec * 16000).
    // Mirrors NumAudioTokens but takes seconds for the cost-model API.
    private static int EstimateAudioTokens(double durSec)
    {
        int samples       = (int)Math.Ceiling(durSec * Config.SampleRate);
        int melLength     = samples / HopLength + 1;
        int encoderLength = melLength / 2;
        int nblocks       = (encoderLength + ProjectorWindowSize - 1) / ProjectorWindowSize;
        return nblocks * EffectiveWindowSize;
    }

    private static int EstimatePromptLen(double durSec) =>
        AsrPromptPrefix.Length + EstimateAudioTokens(durSec) + AsrPromptSuffix.Length;

    private static int EstimateEncoderFrames(double durSec)
    {
        int samples   = (int)Math.Ceiling(durSec * Config.SampleRate);
        int melLength = samples / HopLength + 1;
        return melLength / 2;
    }

    /// <summary>
    /// Bytes per KV-cache element — 4 for FP32 bundles, 2 for the BF16
    /// mixed-precision bundle. Derived once from <see cref="_pastKvDtype"/>
    /// and passed into the cost model so smaller GPUs unlock a higher
    /// batch cap on the BF16 bundle.
    /// </summary>
    private long KvBytesPerElement => _pastKvDtype switch
    {
        TensorElementType.BFloat16 => 2L,
        TensorElementType.Float16  => 2L,
        _                          => 4L,
    };

    /// <summary>
    /// Peak-VRAM cost model for batched Granite Speech decode. Compares
    /// three transient allocations and returns the worst case:
    ///   1. Decoder KV at end of decode  (B × 80 × 4 × seqLen × 128 × kvBytes)
    ///   2. Prefill logits spike         (B × promptLen × VocabSize × 4 bytes
    ///      — logits are always fp32 on the C# boundary regardless of
    ///      bundle precision; the decoder graph casts BF16 → fp32 on
    ///      output for CPU argmax)
    ///   3. Encoder full-attention bias  (num_heads × T_enc² × 4 bytes, B=1
    ///      — mel/encoder/projector run serially and stay fp32 in both
    ///      bundles)
    /// </summary>
    private sealed class GraniteBatchCostModel(int maxNewTokens, long kvBytesPerElement) : IBatchCostModel
    {
        public long EstimatePeakBytes(int batchSize, double maxDurationSec)
        {
            const long bytesPerFloat = 4L;
            int promptLen = EstimatePromptLen(maxDurationSec);
            int seqLen    = promptLen + maxNewTokens;
            int encFrames = EstimateEncoderFrames(maxDurationSec);

            // KV cache: 2 (K+V) × 40 layers × B × 4 KV-heads × seqLen × 128 head dim.
            // BF16 bundle halves this — the dominant per-row cost on long decodes,
            // so on smaller GPUs the BF16 bundle unlocks meaningfully higher B.
            long kvBytes = 2L * NumDecoderLayers * batchSize * NumKvHeads
                         * seqLen * HeadDim * kvBytesPerElement;

            // Prefill logits: B × promptLen × VocabSize × float — peaks at start of decode.
            long logitsBytes = (long)batchSize * promptLen * VocabSize * bytesPerFloat;

            // Encoder full-attention bias [H, T, T] — runs serially per row so B=1.
            // num_heads_encoder = 16 (Granite Speech encoder).
            long encAttnBytes = 16L * encFrames * encFrames * bytesPerFloat;

            return Math.Max(kvBytes, Math.Max(logitsBytes, encAttnBytes));
        }
    }

    // ── CUDA VRAM query (mirrors Cohere's pattern) ──────────────────────

    [System.Runtime.InteropServices.DllImport("cudart",
        EntryPoint        = "cudaMemGetInfo",
        ExactSpelling     = true,
        CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
    private static extern int CudaMemGetInfo(out ulong free, out ulong total);

    private static long QueryVramBudget()
    {
        try
        {
            int rc = CudaMemGetInfo(out ulong free, out _);
            if (rc == 0 && free > (ulong)VramSafetyBufferBytes)
                return (long)(free - (ulong)VramSafetyBufferBytes);
        }
        catch (DllNotFoundException) { }
        catch (EntryPointNotFoundException) { }
        return VramBudgetFallbackBytes;
    }

    private static Dictionary<char, byte> BuildByteLevelDecodeTable()
    {
        var printable = new HashSet<int>();
        for (int b = 33; b <= 126; b++) printable.Add(b);
        for (int b = 161; b <= 172; b++) printable.Add(b);
        for (int b = 174; b <= 255; b++) printable.Add(b);

        var dict = new Dictionary<char, byte>(280);
        foreach (int b in printable) dict[(char)b] = (byte)b;
        int extra = 0;
        for (int b = 0; b < 256; b++)
            if (!printable.Contains(b))
                dict[(char)(0x100 + extra++)] = (byte)b;
        return dict;
    }
}
