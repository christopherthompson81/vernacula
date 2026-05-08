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
    public const string MelFile       = "mel.onnx";
    public const string EncoderFile   = "encoder.onnx";
    public const string ProjectorFile = "projector.onnx";
    public const string DecoderFile   = "decoder.onnx";
    public const string TokenizerFile = "tokenizer.json";

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

        _vramBudgetForKvBytes = QueryVramBudget();
    }

    public void Dispose()
    {
        _mel.Dispose();
        _encoder.Dispose();
        _projector.Dispose();
        _decoder.Dispose();
    }

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
    /// Segments are sorted by ascending duration and packed into VRAM-budgeted
    /// batches via <see cref="BatchSizer.Plan"/>. Within a batch, all segments
    /// run through a single batched forward of mel + encoder + projector +
    /// unified decoder; per-row EOS tracking lets shorter segments stop while
    /// the longest finishes its decode.
    /// </summary>
    /// <param name="segs">Segment list as <c>(start_seconds, end_seconds, speaker_label)</c>.</param>
    /// <param name="audio">Mono 16 kHz waveform covering all segments.</param>
    /// <param name="maxNewTokens">Cap on generated tokens per segment.</param>
    public IEnumerable<(int segId, string text, string speaker)> Recognize(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        if (segs.Count == 0) yield break;

        var durations = new double[segs.Count];
        for (int i = 0; i < segs.Count; i++) durations[i] = segs[i].end - segs[i].start;

        var plan = BatchSizer.Plan(
            durations,
            new GraniteBatchCostModel(maxNewTokens),
            _vramBudgetForKvBytes,
            MaxBatchSize);

        foreach (var batch in plan)
        {
            int B = batch.Count;
            int[] segIds = batch.SegmentIndices;

            // Extract waveforms; mark short ones to skip without dropping the
            // batch slot (we yield an empty string for them).
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

            string[] realTexts = realWaves.Count > 0
                ? TranscribeBatch(realWaves.ToArray(), maxNewTokens)
                : Array.Empty<string>();

            // Yield in the batch order; reinflate the skipped slots.
            for (int b = 0; b < B; b++)
            {
                int segId = segIds[b];
                string text = string.Empty;
                if (!skipped[b])
                {
                    int realPos = realIdx.IndexOf(b);
                    text = realTexts[realPos];
                }
                yield return (segId, text, segs[segId].spk);
            }
        }
    }

    /// <summary>Convenience entry point for transcribing a single waveform.</summary>
    public string Transcribe(float[] audio16kMono, int maxNewTokens = 256)
        => TranscribeBatch(new[] { audio16kMono }, maxNewTokens)[0];

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

    private string[] TranscribeBatch(float[][] waveforms, int maxNewTokens)
    {
        int B = waveforms.Length;
        if (B == 0) return Array.Empty<string>();

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

        // ── Decoder: prefill + chained Run-with-OrtValue step loop ──────
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

            var emptyPasts = new List<OrtValue>(2 * NumDecoderLayers);
            var prefillInputs = new List<OrtValue>(4 + 2 * NumDecoderLayers)
            { inputIdsVal, audioVal, attnVal, cpVal };
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                var k = OrtValue.CreateTensorValueFromMemory(
                    Array.Empty<float>(),
                    new long[] { B, NumKvHeads, 0, HeadDim });
                var v = OrtValue.CreateTensorValueFromMemory(
                    Array.Empty<float>(),
                    new long[] { B, NumKvHeads, 0, HeadDim });
                emptyPasts.Add(k);
                emptyPasts.Add(v);
                prefillInputs.Add(k);
                prefillInputs.Add(v);
            }

            var prefillOutputs = _decoder.Run(
                runOpts, decInputNames, prefillInputs, decOutputNames);

            // Argmax per row at the LAST position (S-1) — left-padding
            // means real last token sits there for every row.
            var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
            for (int b = 0; b < B; b++)
            {
                long tok = ArgmaxRowLastPosition(prefillLogitsSpan, b, sMax, VocabSize);
                nextTok[b] = tok;
                generated[b].Add(tok);
                if (tok == EosTokenId) { finished[b] = true; finishedCount++; }
            }
            prefillOutputs[0].Dispose();
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
            float[] dummyAudio = new float[B * 1 * (int)audioDim];

            int pastLen = sMax;
            for (int step = 0; step < maxNewTokens && finishedCount < B; step++)
            {
                int totalLen = pastLen + 1;

                for (int b = 0; b < B; b++)
                {
                    stepInputIds[b] = finished[b] ? EosTokenId : nextTok[b];
                    int rowOff = b * (sMax + maxNewTokens + 1);
                    stepAttnBuf[rowOff + pastLen] = 1L;  // new token position
                }
                cachePosStep[0] = pastLen;

                // Build a contiguous [B, totalLen] attn mask for this step:
                // ORT requires the OrtValue's backing memory to match the
                // declared shape, so we copy out the live prefix per row.
                var stepAttnFlat = new long[B * totalLen];
                for (int b = 0; b < B; b++)
                {
                    int srcOff = b * (sMax + maxNewTokens + 1);
                    int dstOff = b * totalLen;
                    Array.Copy(stepAttnBuf, srcOff, stepAttnFlat, dstOff, totalLen);
                }

                using var stepInputIdVal = OrtValue.CreateTensorValueFromMemory(
                    stepInputIds, new long[] { B, 1 });
                using var stepAudioVal = OrtValue.CreateTensorValueFromMemory(
                    dummyAudio, new long[] { B, 1, audioDim });
                using var stepAttnVal = OrtValue.CreateTensorValueFromMemory(
                    stepAttnFlat, new long[] { B, totalLen });
                using var cachePosVal = OrtValue.CreateTensorValueFromMemory(
                    cachePosStep, new long[] { 1 });

                var stepInputs = new List<OrtValue>(4 + 2 * NumDecoderLayers)
                { stepInputIdVal, stepAudioVal, stepAttnVal, cachePosVal };
                stepInputs.AddRange(pastKvs);

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
                // Step logits shape: [B, 1, V]
                for (int b = 0; b < B; b++)
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
                pastLen = totalLen;
                stepCountLocal++;
            }

            foreach (var ov in pastKvs) ov.Dispose();
            if (swStep != null) { swStep.Stop(); stepLocal = swStep.ElapsedMilliseconds; }
        }

        // Trim trailing EOS from every row, but ONLY trim the periodic loop
        // tail on rows where the runtime detector classified the row as a
        // runaway. Otherwise natural emphatic repetition (e.g. a speaker
        // saying "no, no, no!") would be silently truncated.
        var texts = new string[B];
        for (int b = 0; b < B; b++)
        {
            var toks = generated[b];
            int realCount = toks.Count;
            while (realCount > 0 && toks[realCount - 1] == EosTokenId) realCount--;
            int trimmed = loopDetected[b]
                ? TrimRepetitionTail(toks, realCount)
                : realCount;
            if (trimmed == toks.Count) texts[b] = DecodeTokens(toks);
            else texts[b] = DecodeTokens(toks.GetRange(0, trimmed));
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
        return texts;
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
        int offset = (b * seqLen + (seqLen - 1)) * vocab;
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
    /// Peak-VRAM cost model for batched Granite Speech decode. Compares
    /// three transient allocations and returns the worst case:
    ///   1. Decoder KV at end of decode  (B × 80 × 4 × seqLen × 128 × 4 bytes)
    ///   2. Prefill logits spike         (B × promptLen × VocabSize × 4 bytes)
    ///   3. Encoder full-attention bias  (num_heads × T_enc² × 4 bytes, B=1 because
    ///      mel/encoder/projector run serially)
    /// </summary>
    private sealed class GraniteBatchCostModel(int maxNewTokens) : IBatchCostModel
    {
        public long EstimatePeakBytes(int batchSize, double maxDurationSec)
        {
            const long bytesPerFloat = 4L;
            int promptLen = EstimatePromptLen(maxDurationSec);
            int seqLen    = promptLen + maxNewTokens;
            int encFrames = EstimateEncoderFrames(maxDurationSec);

            // KV cache: 2 (K+V) × 40 layers × B × 4 KV-heads × seqLen × 128 head dim
            long kvBytes = 2L * NumDecoderLayers * batchSize * NumKvHeads
                         * seqLen * HeadDim * bytesPerFloat;

            // Prefill logits: B × promptLen × VocabSize × float — peaks at start of decode
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
