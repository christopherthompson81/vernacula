using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Chatterbox.Base;

/// <summary>
/// Result of an LM rollout: the generated speech tokens (excluding the
/// leading START_SPEECH and trailing STOP_SPEECH that the LM uses as
/// loop sentinels) plus a step count for timing-aware callers.
/// When the caller passes <c>captureAlignment: true</c> to
/// <see cref="AcousticLM.Generate"/>, <see cref="Alignment"/> is
/// populated with the (speech_step, text_token) attention matrix
/// extracted from the alignment-bearing decoder layers.
/// </summary>
public sealed record AcousticLmResult(
    IReadOnlyList<long> RawGeneratedTokens,
    int Steps,
    AcousticLmAlignment? Alignment = null)
{
    /// <summary>
    /// Strip the LM sentinels (leading START_SPEECH, trailing STOP_SPEECH
    /// if present) and concatenate with the prompt's <c>audio_tokens</c>.
    /// This is the input the vocoder consumes.
    /// </summary>
    public long[] BuildSpeechTokens(long[] audioTokens)
    {
        int genStart = 1;
        int genEnd = RawGeneratedTokens[^1] == ChatterboxConstants.StopSpeechToken
            ? RawGeneratedTokens.Count - 1
            : RawGeneratedTokens.Count;
        int genCount = genEnd - genStart;
        var speechTokens = new long[audioTokens.Length + genCount];
        Array.Copy(audioTokens, speechTokens, audioTokens.Length);
        for (int i = 0; i < genCount; i++)
            speechTokens[audioTokens.Length + i] = RawGeneratedTokens[genStart + i];
        return speechTokens;
    }
}

/// <summary>
/// (NumSpeechRows × NumTextCols) row-major attention matrix from the LM's
/// alignment-bearing decoder layers. Each row corresponds to one LM
/// forward pass's "last-query" attention into the text-token range
/// (i.e. what text token the model attended to when generating that
/// speech token), mean-averaged across the three (layer, head) pairs
/// listed in <see cref="ChatterboxConstants.AlignmentLayerIndices"/> +
/// <see cref="ChatterboxConstants.AlignmentHeadIndices"/>.
///
/// Row 0 corresponds to the LM's prediction of the first generated
/// speech token (START_SPEECH sentinel); subsequent rows align with
/// the rest of <see cref="AcousticLmResult.RawGeneratedTokens"/>. The
/// alignment for the START sentinel is informational (it usually
/// attends to the conditioning prefix); callers typically drop it
/// alongside the sentinel when mapping to audio time.
///
/// Cols index the text tokens (BPE-encoded form, post WrapForLm —
/// includes the [EXAGGERATION, START, ...text..., STOP, START_SPEECH,
/// START_SPEECH] wrapper). The caller is responsible for mapping
/// columns back to the original input text via the tokenizer's
/// known token-to-text boundaries.
/// </summary>
public sealed class AcousticLmAlignment
{
    private readonly List<float[]> _rows = new();

    public int NumSpeechRows => _rows.Count;
    public int NumTextCols { get; }

    internal AcousticLmAlignment(int numTextCols) { NumTextCols = numTextCols; }

    /// <summary>Append the per-step row (length must equal NumTextCols).</summary>
    internal void AppendRow(float[] row)
    {
        if (row.Length != NumTextCols)
            throw new ArgumentException(
                $"Row length {row.Length} doesn't match NumTextCols {NumTextCols}.");
        _rows.Add(row);
    }

    /// <summary>
    /// Returns the alignment as a flat row-major float[] of length
    /// NumSpeechRows × NumTextCols. Allocates fresh on each call;
    /// cache the result in callers if accessed repeatedly.
    /// </summary>
    public float[] ToFlatRowMajor()
    {
        var data = new float[_rows.Count * NumTextCols];
        for (int r = 0; r < _rows.Count; r++)
            Array.Copy(_rows[r], 0, data, r * NumTextCols, NumTextCols);
        return data;
    }

    /// <summary>Row accessor. Returned span aliases internal storage —
    /// do not mutate.</summary>
    public ReadOnlySpan<float> Row(int speechStep) => _rows[speechStep];
}

/// <summary>
/// Wraps the two LM-side ONNX graphs: <c>embed_tokens.onnx</c> (text
/// token → embedding, with position + exaggeration conditioning) and
/// <c>language_model.onnx</c> (Llama backbone with KV-cache I/O).
///
/// Generation runs autoregressively up to
/// <see cref="ChatterboxConstants.DefaultMaxLmSteps"/> steps, with
/// upstream's repetition-penalty (only applied when the previous-token
/// logit is positive). The IoBinding path keeps KV tensors on the GPU
/// between steps via direct OrtValue chaining — adapted from
/// <c>WhisperTurbo.cs::TranscribeBatch</c>. ~3.5× faster than the basic
/// path on CUDA; both produce bit-identical token sequences.
///
/// Not thread-safe — matches the underlying ORT <see cref="InferenceSession"/>
/// semantics. Construct one instance per concurrent caller.
/// </summary>
public sealed class AcousticLM : IDisposable
{
    private readonly InferenceSession _lm;
    private readonly InferenceSession _embed;
    /// <summary>
    /// Whether the loaded sessions actually got the CUDA EP appended. Derived
    /// from <see cref="Vernacula.Base.Inference.OrtSessionBuilder.CreateCachedSession"/>'s
    /// effective-EP report rather than the requested <see cref="ExecutionProvider"/>,
    /// so an <c>Auto</c> request that silently fell back to DirectML correctly
    /// reports false (and won't try the CUDA-only IoBinding path).
    /// </summary>
    private readonly bool _effectiveCuda;
    /// <summary>
    /// True when the loaded LM graph expects fp16 <c>inputs_embeds</c> /
    /// <c>past_key_values.*</c> and emits fp16 <c>logits</c> / <c>present.*</c>.
    /// Detected from session input metadata at construction time. Drives
    /// the dtype-aware OrtValue/NamedOnnxValue construction in all four
    /// rollout paths (single+batched × IoBinding+Basic). Run 9 result:
    /// true-fp16 KV cache eliminates the boundary Casts that hurt Run 8a.
    ///
    /// Invariant: <c>embed_tokens.onnx</c> is assumed to stay fp32 (only
    /// the LM is quantized by quantize_lm.py). <see cref="EmbedOne"/> and
    /// <see cref="EmbedBatch"/> read its output via
    /// <c>.AsTensor&lt;float&gt;().ToArray()</c>; an fp16 embed graph would
    /// throw an unhelpful type error there. If we ever quantize embeds too,
    /// both helpers need the same dtype-aware treatment as the LM.
    /// </summary>
    private readonly bool _lmFp16;

    /// <summary>Load both graphs from disk.</summary>
    /// <param name="onLoad">Optional callback fired once per session (embed_tokens, then language_model).</param>
    public AcousticLM(string embedTokensPath, string languageModelPath, ExecutionProvider ep,
        SessionLoadObserver? onLoad = null)
    {
        _embed = SessionLoader.LoadAndReport(embedTokensPath, ep, onLoad);
        // Gate IoBinding on the LM session's effective EP (it's where
        // IoBinding actually fires); embed_tokens uses plain Run regardless.
        _lm = SessionLoader.LoadAndReport(languageModelPath, ep, onLoad, out var lmUsedCuda);
        _effectiveCuda = lmUsedCuda;
        _lmFp16 = _lm.InputMetadata.TryGetValue("inputs_embeds", out var iemd)
                  && iemd.ElementDataType == TensorElementType.Float16;
    }

    /// <summary>True when the LM expects fp16 floats (KV cache + inputs_embeds + logits).</summary>
    public bool LmIsFloat16 => _lmFp16;

    /// <summary>
    /// Run the LM autoregressively. Returns generated tokens including the
    /// leading START_SPEECH (and trailing STOP_SPEECH if the loop hit it
    /// before max steps).
    /// </summary>
    /// <param name="condEmb">The speaker conditioning from <see cref="SpeakerEmbedder"/>.</param>
    /// <param name="textTokenIds">Wrapped LM input ids — see <see cref="Tokenization.EnTokenizer.WrapForLm"/>.</param>
    /// <param name="useIoBinding">
    /// Null (default) → auto-detect from the EFFECTIVE EP (per
    /// <see cref="Vernacula.Base.Inference.OrtSessionBuilder.CreateCachedSession"/>'s
    /// <c>usedCuda</c> report): IoBinding when CUDA actually backs the session,
    /// basic Run otherwise. This correctly skips IoBinding when
    /// <see cref="ExecutionProvider.Auto"/> silently fell back to DirectML.
    /// True forces IoBinding; throws when the LM session is not CUDA-backed,
    /// since the IoBinding path hardcodes a CUDA <see cref="OrtMemoryInfo"/>.
    /// False forces the basic Run path (useful for A/B testing or non-CUDA EPs).
    /// </param>
    /// <param name="exaggeration">Conditioning scalar passed to embed_tokens; default 0.5.</param>
    /// <param name="maxSteps">Hard cap on rollout length; default 256.</param>
    /// <param name="repetitionPenalty">Divisor applied to positive logits of already-generated tokens; default 1.2.</param>
    /// <param name="diagDir">Optional path for step-0/step-1 diagnostic dumps.</param>
    public AcousticLmResult Generate(
        DenseTensor<float> condEmb,
        long[] textTokenIds,
        bool? useIoBinding = null,
        float exaggeration = ChatterboxConstants.DefaultExaggeration,
        int maxSteps = ChatterboxConstants.DefaultMaxLmSteps,
        float repetitionPenalty = ChatterboxConstants.DefaultRepetitionPenalty,
        string? diagDir = null,
        bool captureAlignment = false)
    {
        bool resolvedIoBinding = useIoBinding ?? _effectiveCuda;
        if (resolvedIoBinding && !_effectiveCuda)
            throw new InvalidOperationException(
                "useIoBinding=true requires the LM session to be CUDA-backed, " +
                "but it isn't (either the requested EP is CPU/DirectML, or Auto fell back). " +
                "Pass useIoBinding=false to use the basic Run path on this EP.");

        // Build text embeddings + position ids.
        int sText = textTokenIds.Length;
        var positionIds = new long[sText];
        for (int i = 0; i < sText; i++)
            positionIds[i] = textTokenIds[i] >= ChatterboxConstants.StartSpeechToken ? 0 : i - 1;
        var inputIdsT = new DenseTensor<long>(textTokenIds.ToArray(), [1, sText]);
        var posIdsT = new DenseTensor<long>(positionIds, [1, sText]);
        var exagT = new DenseTensor<float>(new float[] { exaggeration }, [1]);
        using var embOut = _embed.Run([
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsT),
            NamedOnnxValue.CreateFromTensor("position_ids", posIdsT),
            NamedOnnxValue.CreateFromTensor("exaggeration", exagT),
        ]);
        var textEmb = embOut.First().AsTensor<float>();

        // inputs_embeds = concat(cond_emb, text_emb) along sequence dim.
        int sCond = condEmb.Dimensions[1];
        int sTotal = sCond + sText;
        var inputsEmbeds = ConcatSeq(condEmb, textEmb, ChatterboxConstants.LlmHidden);

        // The alignment matrix's text-col axis covers all text tokens
        // (sCond..sCond+sText). The caller knows sCond from condEmb;
        // we record sText here for downstream slicing convenience.
        var alignment = captureAlignment ? new AcousticLmAlignment(sText) : null;
        int textColStart = sCond;
        int textColEnd = sCond + sText;
        return resolvedIoBinding
            ? RunLmLoopIoBinding(inputsEmbeds, sTotal, exaggeration, maxSteps, repetitionPenalty, diagDir, alignment, textColStart, textColEnd)
            : RunLmLoopBasic(inputsEmbeds, sTotal, exaggeration, maxSteps, repetitionPenalty, diagDir, alignment, textColStart, textColEnd);
    }

    /// <summary>
    /// Result of <see cref="GenerateBatch"/>: per-element rollouts in
    /// input order, plus the wall-clock steps actually run (the inner
    /// loop runs until all elements emit STOP or maxSteps is hit, so
    /// individual elements may have completed earlier).
    /// </summary>
    public sealed record BatchedAcousticLmResult(
        IReadOnlyList<AcousticLmResult> PerChunkResults,
        int Steps);

    /// <summary>
    /// Generate B chunks concurrently through a single batched LM
    /// rollout. Reaps the amortization benefit measured in Run 3 of
    /// <c>docs/chatterbox_perf_investigation.md</c> — at B=4 on fp32,
    /// per-batch-element step time drops from 14.0 ms (B=1) to 7.8 ms,
    /// translating to ~31% wall reduction per chunk in long-form synth.
    ///
    /// MVP constraints (Phase 1 — Phase 2 will relax both):
    ///   - All elements must share the same text-prompt length and
    ///     the same cond_emb sequence length. Caller pads in advance
    ///     if needed (chunk the text uniformly to keep this honest).
    ///   - Inner loop runs until ALL elements have emitted STOP_SPEECH
    ///     or maxSteps is hit; doesn't shrink the active batch as
    ///     elements complete early. Elements that finished early
    ///     waste their slot in subsequent steps. This is the simplest
    ///     correct implementation; per-element batch shrinking is a
    ///     follow-up if it bites.
    ///   - Always uses the basic Run path (no IoBinding) at B&gt;1 until
    ///     <paramref name="useIoBinding"/> is wired to the batched
    ///     IoBinding loop. See <c>RunBatchedLmLoopIoBinding</c>.
    /// </summary>
    /// <param name="useIoBinding">
    /// Null (default) → auto-detect from the LM session's effective EP,
    /// matching <see cref="Generate"/>'s rule (IoBinding when CUDA-backed).
    /// </param>
    public BatchedAcousticLmResult GenerateBatch(
        IReadOnlyList<DenseTensor<float>> condEmbs,
        IReadOnlyList<long[]> textTokenIdsPerChunk,
        bool? useIoBinding = null,
        float exaggeration = ChatterboxConstants.DefaultExaggeration,
        int maxSteps = ChatterboxConstants.DefaultMaxLmSteps,
        float repetitionPenalty = ChatterboxConstants.DefaultRepetitionPenalty)
    {
        bool resolvedIoBinding = useIoBinding ?? _effectiveCuda;
        if (resolvedIoBinding && !_effectiveCuda)
            throw new InvalidOperationException(
                "GenerateBatch(useIoBinding=true) requires the LM session to be CUDA-backed.");

        // Layer/head dims used only inside the dispatched loop bodies.
        const int LlmHidden = ChatterboxConstants.LlmHidden;

        int B = condEmbs.Count;
        if (B == 0 || textTokenIdsPerChunk.Count != B)
            throw new ArgumentException(
                $"condEmbs ({B}) and textTokenIdsPerChunk ({textTokenIdsPerChunk.Count}) must be same non-zero length.");

        int sText = textTokenIdsPerChunk[0].Length;
        int sCond = condEmbs[0].Dimensions[1];
        for (int b = 0; b < B; b++)
        {
            if (textTokenIdsPerChunk[b].Length != sText)
                throw new ArgumentException(
                    $"All chunks must have same text-prompt length (MVP); got {textTokenIdsPerChunk[b].Length} vs {sText} at index {b}.");
            if (condEmbs[b].Dimensions[1] != sCond)
                throw new ArgumentException(
                    $"All cond_embs must have same sequence dim (MVP); got {condEmbs[b].Dimensions[1]} vs {sCond} at index {b}.");
        }

        // ── B-batched text embeddings via a single embed_tokens call ──
        var idsB = new long[B * sText];
        var posB = new long[B * sText];
        for (int b = 0; b < B; b++)
        {
            Array.Copy(textTokenIdsPerChunk[b], 0, idsB, b * sText, sText);
            for (int i = 0; i < sText; i++)
                posB[b * sText + i] = textTokenIdsPerChunk[b][i] >= ChatterboxConstants.StartSpeechToken ? 0 : i - 1;
        }
        var exagB = new float[B];
        Array.Fill(exagB, exaggeration);
        using var embOut = _embed.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(idsB, new[] { B, sText })),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(posB, new[] { B, sText })),
            NamedOnnxValue.CreateFromTensor("exaggeration", new DenseTensor<float>(exagB, new[] { B })),
        });
        var textEmbB = embOut.First().AsTensor<float>().ToArray();  // [B, sText, LlmHidden]

        // ── Build inputs_embeds[B, sTotal, hidden] = concat(cond_emb[b], text_emb[b]) per row ──
        // Long-form audiobook case: all B chunks share the same speaker, so
        // condEmbs[0..B] reference the same tensor (cf. ChunkedSynthesizer.SynthesizeInGroups).
        // ToArray()-once when so, otherwise per-element.
        int sTotal = sCond + sText;
        var inputsEmbeds = new float[B * sTotal * LlmHidden];
        bool sharedCondEmb = true;
        for (int b = 1; b < B; b++)
            if (!ReferenceEquals(condEmbs[b], condEmbs[0])) { sharedCondEmb = false; break; }
        float[]? sharedCondArr = sharedCondEmb ? condEmbs[0].ToArray() : null;
        for (int b = 0; b < B; b++)
        {
            var condArr = sharedCondArr ?? condEmbs[b].ToArray();
            Array.Copy(condArr, 0, inputsEmbeds, b * sTotal * LlmHidden, sCond * LlmHidden);
            Array.Copy(textEmbB, b * sText * LlmHidden,
                       inputsEmbeds, b * sTotal * LlmHidden + sCond * LlmHidden,
                       sText * LlmHidden);
        }

        // ── State per element ──
        var perTokens = new List<long>[B];
        var doneAt = new int[B];
        for (int b = 0; b < B; b++)
        {
            perTokens[b] = new List<long> { ChatterboxConstants.StartSpeechToken };
            doneAt[b] = -1;
        }

        // ── Dispatch to the per-step rollout loop ─────────────────────
        int actualSteps = resolvedIoBinding
            ? RunBatchedLmLoopIoBinding(B, sTotal, inputsEmbeds, perTokens, doneAt, maxSteps, repetitionPenalty, exaggeration)
            : RunBatchedLmLoopBasic(B, sTotal, inputsEmbeds, perTokens, doneAt, maxSteps, repetitionPenalty, exaggeration);

        var perResults = new AcousticLmResult[B];
        for (int b = 0; b < B; b++)
            perResults[b] = new AcousticLmResult(perTokens[b], doneAt[b] >= 0 ? doneAt[b] + 1 : actualSteps);
        return new BatchedAcousticLmResult(perResults, actualSteps);
    }

    /// <summary>
    /// Batched LM rollout, basic Run path. Each step re-creates 2 + 60
    /// NamedOnnxValue inputs (embeds, mask, 30 layers × {key,value}) and
    /// host-roundtrips all KV outputs via <c>.ToArray()</c>. At full
    /// 174-step rollout with B=4, this is roughly 3.2× slower than the
    /// IoBinding sibling — see <c>docs/chatterbox_perf_investigation.md</c>
    /// Runs 4 and 5 for the side-by-side numbers.
    ///
    /// Kept for CPU EP and as the verbatim semantic anchor for the
    /// IoBinding path's correctness (both paths must produce
    /// bit-identical token streams; verified via parity probe).
    /// </summary>
    private int RunBatchedLmLoopBasic(int B, int sTotal, float[] inputsEmbeds,
        List<long>[] perTokens, int[] doneAt, int maxSteps,
        float repetitionPenalty, float exaggeration)
    {
        const int LlmLayers = ChatterboxConstants.LlmLayers;
        const int LlmKvHeads = ChatterboxConstants.LlmKvHeads;
        const int LlmHeadDim = ChatterboxConstants.LlmHeadDim;
        const int LlmHidden = ChatterboxConstants.LlmHidden;

        var attentionMask = new long[B * sTotal];
        Array.Fill(attentionMask, 1);
        var pastKv = new float[LlmLayers * 2][];
        var pastKvShape = new int[] { B, LlmKvHeads, 0, LlmHeadDim };
        for (int k = 0; k < pastKv.Length; k++) pastKv[k] = [];

        int actualSteps = 0;
        for (int step = 0; step < maxSteps; step++)
        {
            actualSteps = step + 1;
            int seqLen = step == 0 ? sTotal : 1;
            var attnMaskPerB = attentionMask.Length / B;
            var maskT = new DenseTensor<long>(attentionMask, new[] { B, attnMaskPerB });

            var inputs = new List<NamedOnnxValue>(2 + 2 * LlmLayers)
            {
                MakeEmbedsNamedValue(inputsEmbeds, new[] { B, seqLen, LlmHidden }),
                NamedOnnxValue.CreateFromTensor("attention_mask", maskT),
            };
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                inputs.Add(MakePastKvNamedValue($"past_key_values.{layer}.key",
                    pastKv[2 * layer], pastKvShape));
                inputs.Add(MakePastKvNamedValue($"past_key_values.{layer}.value",
                    pastKv[2 * layer + 1], pastKvShape));
            }
            using var output = _lm.Run(inputs);
            var outList = output.ToList();
            // Logits → fp32 array via dtype-aware reader.
            var logitsArr = PresentToFloatArray(outList[0]);
            var logitsDims = PresentDims(outList[0]);
            int vocab = logitsDims[2];
            int outSeqLen = logitsDims[1];

            var nextTokensFlat = ArgmaxBatchedLastRow(logitsArr, B, outSeqLen, vocab,
                perTokens, doneAt, repetitionPenalty, step);

            if (AllDone(doneAt, B)) break;

            inputsEmbeds = EmbedBatch(nextTokensFlat, step + 1, exaggeration);
            attentionMask = GrowBatchedMask(attentionMask, B, 1);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                pastKv[2 * layer] = PresentToFloatArray(outList[1 + 2 * layer]);
                pastKv[2 * layer + 1] = PresentToFloatArray(outList[1 + 2 * layer + 1]);
            }
            pastKvShape = PresentDims(outList[1]);
        }
        return actualSteps;
    }

    /// <summary>
    /// Batched LM rollout, IoBinding path. Keeps the per-layer KV-cache
    /// outputs GPU-resident between steps via direct <see cref="OrtValue"/>
    /// chaining instead of host-roundtripping ~250 MB per step (at B=4
    /// step 174). Same shape contract as the single-batch IoBinding
    /// path in <see cref="RunLmLoopIoBinding"/>, generalized on the
    /// leading batch dim.
    ///
    /// CUDA-only: hardcodes <c>OrtMemType.Default</c> on
    /// <c>OrtMemoryInfo("Cuda")</c> for the KV bindings, same as the
    /// single-batch path. Callers should auto-detect via the
    /// <c>_effectiveCuda</c> flag on the LM session.
    /// </summary>
    private int RunBatchedLmLoopIoBinding(int B, int sTotal, float[] inputsEmbeds,
        List<long>[] perTokens, int[] doneAt, int maxSteps,
        float repetitionPenalty, float exaggeration)
    {
        const int LlmLayers = ChatterboxConstants.LlmLayers;
        const int LlmKvHeads = ChatterboxConstants.LlmKvHeads;
        const int LlmHeadDim = ChatterboxConstants.LlmHeadDim;
        const int LlmHidden = ChatterboxConstants.LlmHidden;

        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cpuMemInfo = new OrtMemoryInfo("Cpu", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts = new RunOptions();

        var attentionMask = new long[B * sTotal];
        Array.Fill(attentionMask, 1);

        // Prefill: empty past_kv shaped [B, KvHeads, 0, HeadDim] (dtype-aware).
        var emptyPastValues = new List<OrtValue>(LlmLayers * 2);
        for (int i = 0; i < LlmLayers * 2; i++)
            emptyPastValues.Add(MakeEmptyPastKvOrtValue(new long[] { B, LlmKvHeads, 0L, LlmHeadDim }));

        IDisposableReadOnlyCollection<OrtValue> prefillOutputs;
        {
            using var prefillEmbeds = MakeEmbedsOrtValue(
                inputsEmbeds, new long[] { B, sTotal, LlmHidden }, out var prefillEmbedsScratch);
            using var prefillMask = OrtValue.CreateTensorValueFromMemory(
                attentionMask, new long[] { B, sTotal });
            using var binding = _lm.CreateIoBinding();
            binding.BindInput("inputs_embeds", prefillEmbeds);
            binding.BindInput("attention_mask", prefillMask);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindInput($"past_key_values.{layer}.key", emptyPastValues[2 * layer]);
                binding.BindInput($"past_key_values.{layer}.value", emptyPastValues[2 * layer + 1]);
            }
            binding.BindOutputToDevice("logits", cpuMemInfo);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindOutputToDevice($"present.{layer}.key", cudaMemInfo);
                binding.BindOutputToDevice($"present.{layer}.value", cudaMemInfo);
            }
            _lm.RunWithBinding(runOpts, binding);
            prefillOutputs = binding.GetOutputValues();
            GC.KeepAlive(prefillEmbedsScratch);
        }
        foreach (var v in emptyPastValues) v.Dispose();

        int vocab = (int)prefillOutputs[0].GetTensorTypeAndShape().Shape[2];
        int outSeqLenPrefill = (int)prefillOutputs[0].GetTensorTypeAndShape().Shape[1];  // == sTotal
        // Logits → fp32 array (dtype-aware). CPU-bound, full B*sTotal*vocab slab.
        var prefillLogitsArr = LogitsToFloatArray(prefillOutputs[0]);
        var nextTokensFlat = ArgmaxBatchedLastRow(prefillLogitsArr, B, outSeqLenPrefill, vocab,
            perTokens, doneAt, repetitionPenalty, 0);
        if (AllDone(doneAt, B))
        {
            prefillOutputs.Dispose();
            return 1;
        }

        // Step 1+: chain prev KV outputs as next KV inputs.
        inputsEmbeds = EmbedBatch(nextTokensFlat, 1, exaggeration);
        attentionMask = GrowBatchedMask(attentionMask, B, 1);

        IDisposableReadOnlyCollection<OrtValue> prevStep = prefillOutputs;
        int actualSteps = 1;
        for (int step = 1; step < maxSteps; step++)
        {
            using var stepEmbeds = MakeEmbedsOrtValue(
                inputsEmbeds, new long[] { B, 1L, LlmHidden }, out var stepEmbedsScratch);
            using var stepMask = OrtValue.CreateTensorValueFromMemory(
                attentionMask, new long[] { B, attentionMask.Length / B });
            using var binding = _lm.CreateIoBinding();
            binding.BindInput("inputs_embeds", stepEmbeds);
            binding.BindInput("attention_mask", stepMask);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindInput($"past_key_values.{layer}.key", prevStep[1 + 2 * layer]);
                binding.BindInput($"past_key_values.{layer}.value", prevStep[1 + 2 * layer + 1]);
            }
            binding.BindOutputToDevice("logits", cpuMemInfo);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindOutputToDevice($"present.{layer}.key", cudaMemInfo);
                binding.BindOutputToDevice($"present.{layer}.value", cudaMemInfo);
            }
            _lm.RunWithBinding(runOpts, binding);
            var curStep = binding.GetOutputValues();
            GC.KeepAlive(stepEmbedsScratch);

            // Safe to dispose prevStep here even though `binding` holds
            // BindInput references — RunWithBinding has completed and ORT
            // only reads input OrtValues during the Run call, not after.
            // Same pattern as RunLmLoopIoBinding.
            prevStep.Dispose();
            prevStep = curStep;

            // Step logits are [B, 1, vocab] (fp16 or fp32). LogitsToFloatArray
            // returns fp32 floats either way for downstream argmax math.
            var stepLogitsArr = LogitsToFloatArray(curStep[0]);
            nextTokensFlat = ArgmaxBatchedLastRow(stepLogitsArr, B, 1, vocab,
                perTokens, doneAt, repetitionPenalty, step);
            actualSteps = step + 1;

            if (AllDone(doneAt, B)) break;

            inputsEmbeds = EmbedBatch(nextTokensFlat, step + 1, exaggeration);
            attentionMask = GrowBatchedMask(attentionMask, B, 1);
        }
        prevStep.Dispose();
        return actualSteps;
    }

    /// <summary>
    /// Per-element argmax on the last-row logits, applying upstream's
    /// repetition penalty to already-generated tokens. Mutates
    /// <paramref name="perTokens"/> (appends each element's next token)
    /// and <paramref name="doneAt"/> (records the step that element
    /// emitted STOP_SPEECH, or leaves -1 if not yet done).
    /// </summary>
    private static long[] ArgmaxBatchedLastRow(
        float[] logitsArr, int B, int outSeqLen, int vocab,
        List<long>[] perTokens, int[] doneAt,
        float repetitionPenalty, int step)
    {
        var nextTokensFlat = new long[B];
        int lastRowOffsetInLogits = (outSeqLen - 1) * vocab;
        for (int b = 0; b < B; b++)
        {
            int rowStart = b * outSeqLen * vocab + lastRowOffsetInLogits;
            var rowLogits = new float[vocab];
            Array.Copy(logitsArr, rowStart, rowLogits, 0, vocab);
            if (doneAt[b] < 0)
            {
                foreach (var t in perTokens[b])
                    if (rowLogits[t] > 0) rowLogits[t] /= repetitionPenalty;
            }
            nextTokensFlat[b] = Argmax(rowLogits);
            if (doneAt[b] < 0)
            {
                perTokens[b].Add(nextTokensFlat[b]);
                if (nextTokensFlat[b] == ChatterboxConstants.StopSpeechToken) doneAt[b] = step;
            }
        }
        return nextTokensFlat;
    }

    private static bool AllDone(int[] doneAt, int B)
    {
        for (int b = 0; b < B; b++) if (doneAt[b] < 0) return false;
        return true;
    }

    /// <summary>Batched single-token embed for the autoregressive loop.</summary>
    private float[] EmbedBatch(long[] tokensB, int position, float exaggeration)
    {
        int B = tokensB.Length;
        var idsB = new long[B];
        Array.Copy(tokensB, idsB, B);
        var posB = new long[B];
        Array.Fill(posB, position);
        var exagB = new float[B];
        Array.Fill(exagB, exaggeration);
        using var o = _embed.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(idsB, new[] { B, 1 })),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(posB, new[] { B, 1 })),
            NamedOnnxValue.CreateFromTensor("exaggeration", new DenseTensor<float>(exagB, new[] { B })),
        });
        return o.First().AsTensor<float>().ToArray();
    }

    /// <summary>Grow each batch element's attention mask by `growBy` trailing 1s.</summary>
    private static long[] GrowBatchedMask(long[] mask, int batch, int growBy)
    {
        int oldPerB = mask.Length / batch;
        int newPerB = oldPerB + growBy;
        var grown = new long[batch * newPerB];
        for (int b = 0; b < batch; b++)
        {
            Array.Copy(mask, b * oldPerB, grown, b * newPerB, oldPerB);
            for (int i = oldPerB; i < newPerB; i++) grown[b * newPerB + i] = 1;
        }
        return grown;
    }

    /// <summary>
    /// LM autoregressive loop, IoBinding path. KV-cache outputs are kept
    /// CUDA-resident between steps via direct OrtValue chaining instead
    /// of host-roundtripping every layer's K/V each step. ~3.5× faster
    /// than the basic path on CUDA; bit-identical output. Pattern
    /// adapted from <c>WhisperTurbo.cs::TranscribeBatch</c>.
    /// </summary>
    private AcousticLmResult RunLmLoopIoBinding(
        float[] inputsEmbeds, int sTotal, float exaggeration,
        int maxSteps, float repetitionPenalty, string? diagDir,
        AcousticLmAlignment? alignment, int textColStart, int textColEnd)
    {
        const int LlmLayers = ChatterboxConstants.LlmLayers;
        const int LlmKvHeads = ChatterboxConstants.LlmKvHeads;
        const int LlmHeadDim = ChatterboxConstants.LlmHeadDim;
        const int LlmHidden = ChatterboxConstants.LlmHidden;

        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cpuMemInfo = new OrtMemoryInfo("Cpu", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts = new RunOptions();

        var attentionMask = new long[sTotal];
        Array.Fill(attentionMask, 1);
        var generateTokens = new List<long> { ChatterboxConstants.StartSpeechToken };

        // Prefill (step 0): empty past_kv (fp16 or fp32 depending on LM dtype).
        var emptyPastValues = new List<OrtValue>(LlmLayers * 2);
        for (int i = 0; i < LlmLayers * 2; i++)
            emptyPastValues.Add(MakeEmptyPastKvOrtValue([1L, LlmKvHeads, 0L, LlmHeadDim]));

        IDisposableReadOnlyCollection<OrtValue> prefillOutputs;
        {
            using var prefillEmbeds = MakeEmbedsOrtValue(
                inputsEmbeds, [1L, sTotal, LlmHidden], out var prefillEmbedsScratch);
            using var prefillMask = OrtValue.CreateTensorValueFromMemory(
                attentionMask, [1L, sTotal]);
            using var binding = _lm.CreateIoBinding();
            binding.BindInput("inputs_embeds", prefillEmbeds);
            binding.BindInput("attention_mask", prefillMask);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindInput($"past_key_values.{layer}.key", emptyPastValues[2 * layer]);
                binding.BindInput($"past_key_values.{layer}.value", emptyPastValues[2 * layer + 1]);
            }
            binding.BindOutputToDevice("logits", cpuMemInfo);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindOutputToDevice($"present.{layer}.key", cudaMemInfo);
                binding.BindOutputToDevice($"present.{layer}.value", cudaMemInfo);
            }
            // Attention outputs go to CPU — we read them right after Run
            // to slice into the alignment buffer. (Skipping the bind when
            // alignment is null would save a per-step GPU→CPU transfer,
            // but the graph still emits these tensors either way; ORT
            // just allocates default-EP storage if unbound. Binding to
            // CPU is the cheapest path when we DO want them.)
            if (alignment is not null)
            {
                foreach (int layer in ChatterboxConstants.AlignmentLayerIndices)
                    binding.BindOutputToDevice($"attentions.{layer}", cpuMemInfo);
            }
            _lm.RunWithBinding(runOpts, binding);
            prefillOutputs = binding.GetOutputValues();
            GC.KeepAlive(prefillEmbedsScratch);
        }
        foreach (var v in emptyPastValues) v.Dispose();

        // Read vocab from tensor metadata (robust against future batch>1).
        int vocab = (int)prefillOutputs[0].GetTensorTypeAndShape().Shape[2];
        // Logits → float[] via dtype-aware reader. Slice last row for argmax.
        var prefillLogitsAll = LogitsToFloatArray(prefillOutputs[0]);
        var lastLogits = new float[vocab];
        Array.Copy(prefillLogitsAll, (sTotal - 1) * vocab, lastLogits, 0, vocab);
        MaybeDumpStep(diagDir, 0, lastLogits, inputsEmbeds, attentionMask);
        // Prefill alignment row: last-query attention from each aligned
        // layer, sliced to text cols, mean across the three layers.
        if (alignment is not null)
            AppendAlignmentRowFromOrtValues(alignment, prefillOutputs,
                attnOutputBase: 1 + 2 * LlmLayers, textColStart, textColEnd);
        foreach (var t in generateTokens)
            if (lastLogits[t] > 0) lastLogits[t] /= repetitionPenalty;
        long nextToken = Argmax(lastLogits);
        generateTokens.Add(nextToken);
        if (nextToken == ChatterboxConstants.StopSpeechToken)
        {
            prefillOutputs.Dispose();
            return new AcousticLmResult(generateTokens, 1, alignment);
        }

        inputsEmbeds = EmbedOne(nextToken, 1, exaggeration);
        attentionMask = Grow(attentionMask, 1);

        IDisposableReadOnlyCollection<OrtValue> prevStep = prefillOutputs;
        int actualSteps = 1;
        for (int step = 1; step < maxSteps; step++)
        {
            // fp16 path: stepEmbedsScratch must stay alive until RunWithBinding
            // completes (OrtValue holds an unmanaged pointer into it).
            using var stepEmbeds = MakeEmbedsOrtValue(
                inputsEmbeds, [1L, 1L, LlmHidden], out var stepEmbedsScratch);
            using var stepMask = OrtValue.CreateTensorValueFromMemory(
                attentionMask, [1L, attentionMask.Length]);
            using var binding = _lm.CreateIoBinding();
            binding.BindInput("inputs_embeds", stepEmbeds);
            binding.BindInput("attention_mask", stepMask);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindInput($"past_key_values.{layer}.key", prevStep[1 + 2 * layer]);
                binding.BindInput($"past_key_values.{layer}.value", prevStep[1 + 2 * layer + 1]);
            }
            binding.BindOutputToDevice("logits", cpuMemInfo);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindOutputToDevice($"present.{layer}.key", cudaMemInfo);
                binding.BindOutputToDevice($"present.{layer}.value", cudaMemInfo);
            }
            if (alignment is not null)
            {
                foreach (int layer in ChatterboxConstants.AlignmentLayerIndices)
                    binding.BindOutputToDevice($"attentions.{layer}", cpuMemInfo);
            }
            _lm.RunWithBinding(runOpts, binding);
            var curStep = binding.GetOutputValues();
            // Keep the GC honest: scratch buffer must outlive Run. The using
            // on stepEmbeds disposes after this scope, by which point
            // stepEmbedsScratch is also unreachable — that's the correct
            // ordering. The explicit reference here forbids the JIT from
            // eliding it before RunWithBinding returns.
            GC.KeepAlive(stepEmbedsScratch);

            // Safe to dispose prevStep here even though `binding` (which holds
            // BindInput references to prevStep's OrtValues) is still alive:
            // RunWithBinding has completed and ORT only reads the input
            // OrtValues during the Run call, not afterward.
            prevStep.Dispose();
            prevStep = curStep;

            // Step logits = [1, 1, vocab]; vocab-length tail is the only row.
            var stepLogitsAll = LogitsToFloatArray(curStep[0]);
            lastLogits = stepLogitsAll[..vocab];
            MaybeDumpStep(diagDir, step, lastLogits, inputsEmbeds, attentionMask);
            if (alignment is not null)
                AppendAlignmentRowFromOrtValues(alignment, curStep,
                    attnOutputBase: 1 + 2 * LlmLayers, textColStart, textColEnd);
            foreach (var t in generateTokens)
                if (lastLogits[t] > 0) lastLogits[t] /= repetitionPenalty;
            nextToken = Argmax(lastLogits);
            generateTokens.Add(nextToken);
            if (nextToken == ChatterboxConstants.StopSpeechToken) { actualSteps = step + 1; break; }

            inputsEmbeds = EmbedOne(nextToken, step + 1, exaggeration);
            attentionMask = Grow(attentionMask, 1);
            actualSteps = step + 1;
        }
        prevStep.Dispose();
        return new AcousticLmResult(generateTokens, actualSteps, alignment);
    }

    /// <summary>
    /// IoBinding-path sibling of <see cref="AppendAlignmentRow"/>. Reads
    /// the three attention output tensors as fp32 spans, picks the
    /// per-layer alignment head, slices the text-token cols, and
    /// mean-averages into a single row appended to the alignment buffer.
    /// </summary>
    private static void AppendAlignmentRowFromOrtValues(
        AcousticLmAlignment alignment,
        IDisposableReadOnlyCollection<OrtValue> outputs,
        int attnOutputBase, int textColStart, int textColEnd)
    {
        int textLen = textColEnd - textColStart;
        var meanRow = new float[textLen];
        int numLayers = ChatterboxConstants.AlignmentLayerIndices.Length;
        for (int li = 0; li < numLayers; li++)
        {
            var ov = outputs[attnOutputBase + li];
            var shape = ov.GetTensorTypeAndShape().Shape;
            int H = (int)shape[1];   // num heads
            int sQ = (int)shape[2];  // current input length
            int sKv = (int)shape[3]; // total cached + current
            int head = ChatterboxConstants.AlignmentHeadIndices[li];
            // Flat index of [0, head, sQ-1, textColStart..)
            int rowBase = ((0 * H + head) * sQ + (sQ - 1)) * sKv + textColStart;
            var span = ov.GetTensorDataAsSpan<float>();
            for (int c = 0; c < textLen; c++)
                meanRow[c] += span[rowBase + c];
        }
        float invN = 1.0f / numLayers;
        for (int c = 0; c < textLen; c++) meanRow[c] *= invN;
        alignment.AppendRow(meanRow);
    }

    /// <summary>
    /// LM autoregressive loop, basic Run path. Each step builds 60
    /// NamedOnnxValue inputs (30 layers × {key,value}) from CPU-resident
    /// past_kv arrays, runs the session, copies all 60 KV outputs back to
    /// CPU. Slower than IoBinding but useful for A/B comparison and for
    /// platforms where IoBinding semantics don't apply (e.g. CPU EP, some
    /// DirectML configs).
    /// </summary>
    private AcousticLmResult RunLmLoopBasic(
        float[] inputsEmbeds, int sTotal, float exaggeration,
        int maxSteps, float repetitionPenalty, string? diagDir,
        AcousticLmAlignment? alignment, int textColStart, int textColEnd)
    {
        const int LlmLayers = ChatterboxConstants.LlmLayers;
        const int LlmKvHeads = ChatterboxConstants.LlmKvHeads;
        const int LlmHeadDim = ChatterboxConstants.LlmHeadDim;
        const int LlmHidden = ChatterboxConstants.LlmHidden;

        var attentionMask = new long[sTotal];
        Array.Fill(attentionMask, 1);
        var pastKv = new float[LlmLayers * 2][];
        var pastKvShape = new int[] { 1, LlmKvHeads, 0, LlmHeadDim };
        for (int k = 0; k < pastKv.Length; k++) pastKv[k] = [];

        var generateTokens = new List<long> { ChatterboxConstants.StartSpeechToken };
        int actualSteps = 0;
        for (int step = 0; step < maxSteps; step++)
        {
            int seqLen = step == 0 ? sTotal : 1;

            var inputs = new List<NamedOnnxValue>(2 + 2 * LlmLayers)
            {
                MakeEmbedsNamedValue(inputsEmbeds, [1, seqLen, LlmHidden]),
                NamedOnnxValue.CreateFromTensor("attention_mask",
                    new DenseTensor<long>(attentionMask.ToArray(), [1, attentionMask.Length])),
            };
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                inputs.Add(MakePastKvNamedValue($"past_key_values.{layer}.key",
                    pastKv[2 * layer], pastKvShape));
                inputs.Add(MakePastKvNamedValue($"past_key_values.{layer}.value",
                    pastKv[2 * layer + 1], pastKvShape));
            }
            using var output = _lm.Run(inputs);
            var outList = output.ToList();
            var logitsArr = PresentToFloatArray(outList[0]);
            var logitsDims = PresentDims(outList[0]);
            int vocab = logitsDims[2];
            var lastLogits = new float[vocab];
            int logitsOffset = (logitsDims[1] - 1) * vocab;
            Array.Copy(logitsArr, logitsOffset, lastLogits, 0, vocab);
            MaybeDumpStep(diagDir, step, lastLogits, inputsEmbeds, attentionMask, pastKv, pastKvShape);

            // Capture per-step alignment if requested. Attention outputs
            // are at indices [1 + 2*LlmLayers .. +3) — the three layers
            // listed in ChatterboxConstants.AlignmentLayerIndices.
            if (alignment is not null)
            {
                int attnBase = 1 + 2 * LlmLayers;
                AppendAlignmentRow(alignment, outList, attnBase, textColStart, textColEnd);
            }

            foreach (var t in generateTokens)
                if (lastLogits[t] > 0) lastLogits[t] /= repetitionPenalty;
            long nextToken = Argmax(lastLogits);
            generateTokens.Add(nextToken);
            if (nextToken == ChatterboxConstants.StopSpeechToken) { actualSteps = step + 1; break; }

            inputsEmbeds = EmbedOne(nextToken, step + 1, exaggeration);
            attentionMask = Grow(attentionMask, 1);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                pastKv[2 * layer] = PresentToFloatArray(outList[1 + 2 * layer]);
                pastKv[2 * layer + 1] = PresentToFloatArray(outList[1 + 2 * layer + 1]);
            }
            pastKvShape = PresentDims(outList[1]);
            actualSteps = step + 1;
        }
        return new AcousticLmResult(generateTokens, actualSteps, alignment);
    }

    /// <summary>
    /// Pull the last-query row from each of the three attention outputs
    /// (basic-Run path), pick the alignment head per layer, mean across
    /// layers, slice text-token cols, and append to the alignment buffer.
    /// Called per LM step.
    /// </summary>
    private static void AppendAlignmentRow(
        AcousticLmAlignment alignment, List<DisposableNamedOnnxValue> outList,
        int attnOutputBase, int textColStart, int textColEnd)
    {
        int textLen = textColEnd - textColStart;
        var meanRow = new float[textLen];
        int numLayers = ChatterboxConstants.AlignmentLayerIndices.Length;
        for (int li = 0; li < numLayers; li++)
        {
            var attnTensor = outList[attnOutputBase + li].AsTensor<float>();
            int H = attnTensor.Dimensions[1];   // num heads
            int sQ = attnTensor.Dimensions[2];  // current input length
            int sKv = attnTensor.Dimensions[3]; // total cached + current
            int head = ChatterboxConstants.AlignmentHeadIndices[li];
            // Flat index of [0, head, sQ-1, textColStart..textColEnd) row.
            int rowBase = ((0 * H + head) * sQ + (sQ - 1)) * sKv + textColStart;
            // .ToArray() materializes once; cheap relative to the rest
            // (~64 KB for a max-size attention output).
            var flat = attnTensor.ToArray();
            for (int c = 0; c < textLen; c++)
                meanRow[c] += flat[rowBase + c];
        }
        float invN = 1.0f / numLayers;
        for (int c = 0; c < textLen; c++) meanRow[c] *= invN;
        alignment.AppendRow(meanRow);
    }

    /// <summary>Run embed_tokens.onnx for a single token at the given position. CPU output.</summary>
    private float[] EmbedOne(long token, int position, float exaggeration)
    {
        var idT = new DenseTensor<long>(new long[] { token }, [1, 1]);
        var posT = new DenseTensor<long>(new long[] { position }, [1, 1]);
        var exT = new DenseTensor<float>(new float[] { exaggeration }, [1]);
        using var o = _embed.Run([
            NamedOnnxValue.CreateFromTensor("input_ids", idT),
            NamedOnnxValue.CreateFromTensor("position_ids", posT),
            NamedOnnxValue.CreateFromTensor("exaggeration", exT),
        ]);
        return o.First().AsTensor<float>().ToArray();
    }

    /// <summary>Concatenate two <c>[1, S_i, D]</c> tensors along the sequence dim, returning a flat float[].</summary>
    private static float[] ConcatSeq(Tensor<float> a, Tensor<float> b, int hiddenDim)
    {
        int sa = a.Dimensions[1];
        int sb = b.Dimensions[1];
        var aArr = a.ToArray();
        var bArr = b.ToArray();
        var outArr = new float[(sa + sb) * hiddenDim];
        Array.Copy(aArr, 0, outArr, 0, sa * hiddenDim);
        Array.Copy(bArr, 0, outArr, sa * hiddenDim, sb * hiddenDim);
        return outArr;
    }

    private static long[] Grow(long[] mask, int n)
    {
        var grown = new long[mask.Length + n];
        Array.Copy(mask, grown, mask.Length);
        for (int i = mask.Length; i < grown.Length; i++) grown[i] = 1;
        return grown;
    }

    private static long Argmax(float[] arr)
    {
        int best = 0;
        float bestVal = arr[0];
        for (int i = 1; i < arr.Length; i++)
            if (arr[i] > bestVal) { bestVal = arr[i]; best = i; }
        return best;
    }

    /// <summary>
    /// Diagnostic dump for LM step 0/1. Only fires when diagDir is non-null.
    /// The basic path passes past_kv (CPU arrays); the IoBinding path passes
    /// null since KV lives on CUDA and host extraction would defeat the
    /// purpose.
    /// </summary>
    private static void MaybeDumpStep(string? diagDir, int step,
        float[] lastLogits, float[] inputsEmbeds, long[] attentionMask,
        float[][]? pastKv = null, int[]? pastKvShape = null)
    {
        if (diagDir is null || step > 1) return;
        Console.WriteLine($"[step{step}] logits[-1, :10] (pre-penalty): "
            + string.Join(", ", lastLogits.Take(10).Select(x => x.ToString("F6"))));
        Console.WriteLine($"[step{step}] logits sum: {lastLogits.Sum():F4}, argmax(pre-penalty): {Argmax(lastLogits)}");
        File.WriteAllBytes(Path.Combine(diagDir, $"cs_step{step}_logits.bin"),
            MemoryMarshal.AsBytes<float>(lastLogits).ToArray());
        File.WriteAllBytes(Path.Combine(diagDir, $"cs_step{step}_inputs_embeds.bin"),
            MemoryMarshal.AsBytes<float>(inputsEmbeds).ToArray());
        File.WriteAllBytes(Path.Combine(diagDir, $"cs_step{step}_attention_mask.bin"),
            MemoryMarshal.AsBytes<long>(attentionMask).ToArray());
        if (step == 1 && pastKv is not null && pastKvShape is not null)
        {
            File.WriteAllBytes(Path.Combine(diagDir, "cs_step1_past_kv_l0_key.bin"),
                MemoryMarshal.AsBytes<float>(pastKv[0]).ToArray());
            File.WriteAllBytes(Path.Combine(diagDir, "cs_step1_past_kv_l0_value.bin"),
                MemoryMarshal.AsBytes<float>(pastKv[1]).ToArray());
        }
    }

    // ── Float / Float16 helpers for true-fp16 LM (Run 9) ───────────────────
    // When `_lmFp16` is true the LM expects fp16 inputs_embeds / past_kv and
    // emits fp16 logits / present.*. embed_tokens still produces fp32, so the
    // conversion happens at the BindInput/output boundary. attention_mask is
    // int64 in both worlds (untouched).

    /// <summary>Pack fp16-or-fp32 LM <c>inputs_embeds</c> from a CPU fp32 source.
    /// Caller must dispose the returned OrtValue AND keep the returned buffer
    /// alive until <c>RunWithBinding</c> completes (CreateTensorValueFromMemory
    /// doesn't copy). For fp16 we allocate a fresh Float16[] scratch and
    /// hand it back via <paramref name="fp16Scratch"/>; for fp32 the caller's
    /// <paramref name="src"/> buffer is used directly and <paramref name="fp16Scratch"/>
    /// is null.</summary>
    private OrtValue MakeEmbedsOrtValue(float[] src, long[] shape, out Float16[]? fp16Scratch)
    {
        if (_lmFp16)
        {
            var dst = new Float16[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = (Float16)src[i];
            fp16Scratch = dst;
            return OrtValue.CreateTensorValueFromMemory(dst, shape);
        }
        fp16Scratch = null;
        return OrtValue.CreateTensorValueFromMemory(src, shape);
    }

    /// <summary>Make an empty (zero-length sequence dim) past_kv tensor of
    /// the LM's expected dtype. Used only at prefill — subsequent steps
    /// chain real OrtValues from prior output.</summary>
    private OrtValue MakeEmptyPastKvOrtValue(long[] shape)
        => _lmFp16
            ? OrtValue.CreateTensorValueFromMemory(Array.Empty<Float16>(), shape)
            : OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(), shape);

    /// <summary>Read logits (CPU-bound output) as a fresh float[]. Always
    /// returns fp32 floats for the argmax + repetition-penalty math, even
    /// when the LM emitted fp16.</summary>
    private float[] LogitsToFloatArray(OrtValue logits)
    {
        if (_lmFp16)
        {
            var span = logits.GetTensorDataAsSpan<Float16>();
            var arr = new float[span.Length];
            for (int i = 0; i < span.Length; i++) arr[i] = (float)span[i];
            return arr;
        }
        return logits.GetTensorDataAsSpan<float>().ToArray();
    }

    /// <summary>Build a NamedOnnxValue carrying inputs_embeds at the LM's
    /// expected dtype. Used by the basic-Run paths (RunLmLoopBasic and
    /// RunBatchedLmLoopBasic).</summary>
    private NamedOnnxValue MakeEmbedsNamedValue(float[] src, int[] shape)
    {
        if (_lmFp16)
        {
            var dst = new Float16[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = (Float16)src[i];
            return NamedOnnxValue.CreateFromTensor("inputs_embeds",
                new DenseTensor<Float16>(dst, shape));
        }
        return NamedOnnxValue.CreateFromTensor("inputs_embeds",
            new DenseTensor<float>(src, shape));
    }

    /// <summary>Build a NamedOnnxValue carrying one past_kv slot at the
    /// LM's expected dtype, given a fp32 CPU buffer (the basic path's
    /// internal representation).</summary>
    private NamedOnnxValue MakePastKvNamedValue(string name, float[] src, int[] shape)
    {
        if (_lmFp16)
        {
            var dst = new Float16[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = (Float16)src[i];
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(dst, shape));
        }
        return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(src, shape));
    }

    /// <summary>Extract a present.* output to a CPU fp32 float[] for the
    /// basic-Run paths, regardless of fp16/fp32 LM. (IoBinding paths chain
    /// OrtValues directly and never visit this helper.)</summary>
    private float[] PresentToFloatArray(DisposableNamedOnnxValue v)
    {
        if (_lmFp16)
        {
            var t = v.AsTensor<Float16>().ToArray();
            var arr = new float[t.Length];
            for (int i = 0; i < t.Length; i++) arr[i] = (float)t[i];
            return arr;
        }
        return v.AsTensor<float>().ToArray();
    }

    /// <summary>Read the present.* output shape from a basic-Run output
    /// in a dtype-agnostic way (only the metadata is needed; values are
    /// already extracted via <see cref="PresentToFloatArray"/>).</summary>
    private int[] PresentDims(DisposableNamedOnnxValue v)
        => _lmFp16 ? v.AsTensor<Float16>().Dimensions.ToArray()
                   : v.AsTensor<float>().Dimensions.ToArray();

    public void Dispose()
    {
        _lm.Dispose();
        _embed.Dispose();
    }
}
