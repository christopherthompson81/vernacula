using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Chatterbox.Base;

/// <summary>
/// Result of an LM rollout: the generated speech tokens (excluding the
/// leading START_SPEECH and trailing STOP_SPEECH that the LM uses as
/// loop sentinels) plus a step count for timing-aware callers.
/// </summary>
public sealed record AcousticLmResult(IReadOnlyList<long> RawGeneratedTokens, int Steps)
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

    /// <summary>Load both graphs from disk.</summary>
    /// <param name="onLoad">Optional callback fired once per session (embed_tokens, then language_model).</param>
    public AcousticLM(string embedTokensPath, string languageModelPath, ExecutionProvider ep,
        SessionLoadObserver? onLoad = null)
    {
        _embed = LoadOne(embedTokensPath, ep, onLoad, out _);
        _lm = LoadOne(languageModelPath, ep, onLoad, out var lmUsedCuda);
        // We gate IoBinding on the LM session's effective EP (it's where
        // IoBinding actually fires); embed_tokens uses plain Run regardless.
        _effectiveCuda = lmUsedCuda;
    }

    private static InferenceSession LoadOne(string path, ExecutionProvider ep,
        SessionLoadObserver? onLoad, out bool usedCuda)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var s = OrtSessionBuilder.CreateCachedSession(path, ep, out var hit, out usedCuda);
        sw.Stop();
        onLoad?.Invoke(new SessionLoadEvent(
            Path.GetFileName(path), sw.ElapsedMilliseconds, hit, usedCuda,
            new FileInfo(path).Length));
        return s;
    }

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
        string? diagDir = null)
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

        return resolvedIoBinding
            ? RunLmLoopIoBinding(inputsEmbeds, sTotal, exaggeration, maxSteps, repetitionPenalty, diagDir)
            : RunLmLoopBasic(inputsEmbeds, sTotal, exaggeration, maxSteps, repetitionPenalty, diagDir);
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
        int maxSteps, float repetitionPenalty, string? diagDir)
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

        // Prefill (step 0): empty past_kv.
        var emptyPastValues = new List<OrtValue>(LlmLayers * 2);
        for (int i = 0; i < LlmLayers * 2; i++)
            emptyPastValues.Add(OrtValue.CreateTensorValueFromMemory(
                Array.Empty<float>(), [1L, LlmKvHeads, 0L, LlmHeadDim]));

        IDisposableReadOnlyCollection<OrtValue> prefillOutputs;
        {
            using var prefillEmbeds = OrtValue.CreateTensorValueFromMemory(
                inputsEmbeds, [1L, sTotal, LlmHidden]);
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
            _lm.RunWithBinding(runOpts, binding);
            prefillOutputs = binding.GetOutputValues();
        }
        foreach (var v in emptyPastValues) v.Dispose();

        // Read vocab from tensor metadata (robust against future batch>1).
        int vocab = (int)prefillOutputs[0].GetTensorTypeAndShape().Shape[2];
        var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
        var lastLogits = prefillLogitsSpan.Slice((sTotal - 1) * vocab, vocab).ToArray();
        MaybeDumpStep(diagDir, 0, lastLogits, inputsEmbeds, attentionMask);
        foreach (var t in generateTokens)
            if (lastLogits[t] > 0) lastLogits[t] /= repetitionPenalty;
        long nextToken = Argmax(lastLogits);
        generateTokens.Add(nextToken);
        if (nextToken == ChatterboxConstants.StopSpeechToken)
        {
            prefillOutputs.Dispose();
            return new AcousticLmResult(generateTokens, 1);
        }

        inputsEmbeds = EmbedOne(nextToken, 1, exaggeration);
        attentionMask = Grow(attentionMask, 1);

        IDisposableReadOnlyCollection<OrtValue> prevStep = prefillOutputs;
        int actualSteps = 1;
        for (int step = 1; step < maxSteps; step++)
        {
            using var stepEmbeds = OrtValue.CreateTensorValueFromMemory(
                inputsEmbeds, [1L, 1L, LlmHidden]);
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
            _lm.RunWithBinding(runOpts, binding);
            var curStep = binding.GetOutputValues();

            // Safe to dispose prevStep here even though `binding` (which holds
            // BindInput references to prevStep's OrtValues) is still alive:
            // RunWithBinding has completed and ORT only reads the input
            // OrtValues during the Run call, not afterward.
            prevStep.Dispose();
            prevStep = curStep;

            var stepLogitsSpan = curStep[0].GetTensorDataAsSpan<float>();
            lastLogits = stepLogitsSpan[..vocab].ToArray();
            MaybeDumpStep(diagDir, step, lastLogits, inputsEmbeds, attentionMask);
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
        return new AcousticLmResult(generateTokens, actualSteps);
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
        int maxSteps, float repetitionPenalty, string? diagDir)
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
            var embedT = new DenseTensor<float>(inputsEmbeds, [1, seqLen, LlmHidden]);

            var inputs = new List<NamedOnnxValue>(2 + 2 * LlmLayers)
            {
                NamedOnnxValue.CreateFromTensor("inputs_embeds", embedT),
                NamedOnnxValue.CreateFromTensor("attention_mask",
                    new DenseTensor<long>(attentionMask.ToArray(), [1, attentionMask.Length])),
            };
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{layer}.key",
                    new DenseTensor<float>(pastKv[2 * layer], pastKvShape)));
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{layer}.value",
                    new DenseTensor<float>(pastKv[2 * layer + 1], pastKvShape)));
            }
            using var output = _lm.Run(inputs);
            var outList = output.ToList();
            var logits = outList[0].AsTensor<float>();
            int vocab = logits.Dimensions[2];
            var lastLogits = new float[vocab];
            int logitsOffset = (logits.Dimensions[1] - 1) * vocab;
            var logitsArr = logits.ToArray();
            Array.Copy(logitsArr, logitsOffset, lastLogits, 0, vocab);
            MaybeDumpStep(diagDir, step, lastLogits, inputsEmbeds, attentionMask, pastKv, pastKvShape);

            foreach (var t in generateTokens)
                if (lastLogits[t] > 0) lastLogits[t] /= repetitionPenalty;
            long nextToken = Argmax(lastLogits);
            generateTokens.Add(nextToken);
            if (nextToken == ChatterboxConstants.StopSpeechToken) { actualSteps = step + 1; break; }

            inputsEmbeds = EmbedOne(nextToken, step + 1, exaggeration);
            attentionMask = Grow(attentionMask, 1);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                pastKv[2 * layer] = outList[1 + 2 * layer].AsTensor<float>().ToArray();
                pastKv[2 * layer + 1] = outList[1 + 2 * layer + 1].AsTensor<float>().ToArray();
            }
            pastKvShape = outList[1].AsTensor<float>().Dimensions.ToArray();
            actualSteps = step + 1;
        }
        return new AcousticLmResult(generateTokens, actualSteps);
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

    public void Dispose()
    {
        _lm.Dispose();
        _embed.Dispose();
    }
}
