using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Diagnostics;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// Qwen3-ASR 1.7B runtime for the split ONNX export produced by scripts/qwen3asr_export.
///
/// Pipeline:
///   1. Host-side Whisper-style log-mel frontend
///   2. encoder.onnx          mel [1,128,T] -> audio_features [1,N,2048]
///   3. decoder_init.onnx     prompt ids + audio features -> logits + KV cache
///   4. decoder_step.onnx     token embedding + KV cache -> logits + updated KV cache
///
/// The decoder token embedding matrix is kept in a memory-mapped file so we do not
/// need a 1.2 GB managed float[] on the LOH.
/// </summary>
public sealed class Qwen3Asr : IDisposable
{
    public const string EncoderFile = "encoder.onnx";
    public const string EncoderBatchedFile = "encoder_batched.onnx";
    public const string DecoderInitFile = "decoder_init.onnx";
    public const string DecoderInitBatchedFile = "decoder_init_batched.onnx";
    public const string DecoderStepFile = "decoder_step.onnx";
    public const string EmbedTokensFile = "embed_tokens.bin";
    public const string TokenizerFile = "tokenizer.json";
    public const string ConfigFile = "config.json";

    private const int SampleRate = 16_000;
    private const int NFft = 400;
    private const int HopLength = 160;
    private const int NMels = 128;
    private const float LogFloor = -10.0f;
    private const float LogClampSpan = 8.0f;
    private const float LogOffset = 4.0f;
    private const int ImStartTokenId = 151644;
    private const int ImEndTokenId = 151645;
    private const int AudioStartTokenId = 151669;
    private const int AudioEndTokenId = 151670;
    private const int AudioPadTokenId = 151676;
    private const int NewlineTokenId = 198;
    private const int SystemTokenId = 9125;
    private const int UserTokenId = 882;
    private const int AssistantTokenId = 77091;
    private const string LanguagePrefix = "language ";
    private const long BatchSizingReferenceFreeVramMb = 22_800;
    private const double BatchSizingReferenceTotalSecondsCeiling = 192.0;

    private static readonly float[,] MelFilterbank = CreateMelFilterbank();
    private static readonly double[] HannWindow = Window.HannPeriodic(NFft);
    private static readonly (double MaxSegmentSeconds, int ReferenceBatchCap)[] ExperimentalBatchFrontier =
    [
        (20.0, 10),
        (24.0, 9),
        (28.0, 8),
        (32.0, 7),
        (double.PositiveInfinity, 6),
    ];
    private static readonly string[] SpokenLanguageNames =
    [
        "Afrikaans",
        "Arabic",
        "Armenian",
        "Azerbaijani",
        "Belarusian",
        "Bosnian",
        "Bulgarian",
        "Catalan",
        "Chinese",
        "Croatian",
        "Czech",
        "Danish",
        "Dutch",
        "English",
        "Estonian",
        "Finnish",
        "French",
        "Galician",
        "German",
        "Greek",
        "Hebrew",
        "Hindi",
        "Hungarian",
        "Icelandic",
        "Indonesian",
        "Italian",
        "Japanese",
        "Kannada",
        "Kazakh",
        "Korean",
        "Latvian",
        "Lithuanian",
        "Macedonian",
        "Malay",
        "Marathi",
        "Nepali",
        "Norwegian",
        "Persian",
        "Polish",
        "Portuguese",
        "Romanian",
        "Russian",
        "Serbian",
        "Slovak",
        "Slovenian",
        "Spanish",
        "Swahili",
        "Swedish",
        "Tagalog",
        "Tamil",
        "Thai",
        "Turkish",
        "Ukrainian",
        "Urdu",
        "Vietnamese",
        "Welsh",
    ];

    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoderInit;
    private readonly InferenceSession _decoderStep;
    private readonly InferenceSession? _encoderBatched;
    private readonly InferenceSession? _decoderInitBatched;
    private readonly bool _useCudaIoBinding;
    private readonly bool _preferBatched;
    private readonly string _modelPath;
    private readonly ExecutionProvider _executionProvider;
    private readonly int _hiddenSize;
    private readonly int _vocabSize;
    private readonly int _baseVocabSize;
    private readonly int[] _eosTokenIds;
    private readonly HashSet<int> _eosTokenIdSet;
    private readonly MemoryMappedFile _embedMmf;
    private readonly MemoryMappedViewAccessor _embedAccessor;
    private readonly string?[] _idToToken;
    private readonly Dictionary<int, string> _addedTokenContent;
    private readonly Dictionary<char, byte> _byteLevelDecode;

    public Qwen3Asr(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto, bool preferBatched = false)
    {
        _modelPath = modelPath;
        _executionProvider = ep;
        string configJson = File.ReadAllText(Path.Combine(modelPath, ConfigFile));
        using var configDoc = JsonDocument.Parse(configJson);
        var root = configDoc.RootElement;
        var decoderConfig = root.GetProperty("decoder");
        var specialTokens = root.GetProperty("special_tokens");

        _hiddenSize = decoderConfig.GetProperty("hidden_size").GetInt32();
        _vocabSize = decoderConfig.GetProperty("vocab_size").GetInt32();
        _baseVocabSize = root.GetProperty("embed_tokens_shape")[0].GetInt32();
        _eosTokenIds = specialTokens.GetProperty("eos_token_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        _eosTokenIdSet = [.. _eosTokenIds];

        var encoderOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED, out bool encoderUsesCuda);
        var decoderOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED, out bool decoderUsesCuda);

        bool hasBatchedFiles = File.Exists(Path.Combine(modelPath, EncoderBatchedFile)) &&
                               File.Exists(Path.Combine(modelPath, DecoderInitBatchedFile));
        _preferBatched = preferBatched && hasBatchedFiles;

        if (_preferBatched)
        {
            // Load only batched encoder + decoder_init + shared decoder_step.
            // Skipping serial encoder/decoder_init avoids peak VRAM from 5 → 3 sessions.
            _encoder      = null!;
            _decoderInit  = null!;
            _encoderBatched      = new InferenceSession(Path.Combine(modelPath, EncoderBatchedFile),      encoderOpts);
            _decoderInitBatched  = new InferenceSession(Path.Combine(modelPath, DecoderInitBatchedFile),  decoderOpts);
            _useCudaIoBinding    = false;
        }
        else
        {
            _encoder     = new InferenceSession(Path.Combine(modelPath, EncoderFile),     encoderOpts);
            _decoderInit = new InferenceSession(Path.Combine(modelPath, DecoderInitFile), decoderOpts);
            _useCudaIoBinding = encoderUsesCuda && decoderUsesCuda;
        }

        _decoderStep = new InferenceSession(Path.Combine(modelPath, DecoderStepFile), decoderOpts);

        string embedPath = Path.Combine(modelPath, EmbedTokensFile);
        _embedMmf = MemoryMappedFile.CreateFromFile(embedPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _embedAccessor = _embedMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        (_idToToken, _addedTokenContent) = LoadTokenizerVocab(Path.Combine(modelPath, TokenizerFile));
        _byteLevelDecode = BuildByteLevelDecode();
    }

    /// <summary>
    /// Compute a conservative experimental GPU batching plan for the batched
    /// Qwen3-ASR encoder + decoder-prefill path. The current heuristic is based
    /// on CUDA sweep data from a 24 GB RTX 3090 with about 22.8 GB free VRAM
    /// after the model package is loaded.
    /// </summary>
    public static QwenBatchingPlan ComputeExperimentalBatchingPlan(
        IReadOnlyList<double> segmentDurationsSeconds,
        long freeGpuMemoryMb)
    {
        if (segmentDurationsSeconds.Count == 0)
            return new QwenBatchingPlan(0, 0, 0, 0, 0, freeGpuMemoryMb, 0);

        double maxSegmentSeconds = 0;
        foreach (double seconds in segmentDurationsSeconds)
            maxSegmentSeconds = Math.Max(maxSegmentSeconds, Math.Max(0, seconds));

        int batchCap = EstimateExperimentalBatchCap(maxSegmentSeconds, freeGpuMemoryMb);
        double totalSecondsCeiling = EstimateExperimentalTotalSecondsCeiling(freeGpuMemoryMb);
        double totalSeconds = 0;
        int batchCount = 0;

        foreach (double rawSeconds in segmentDurationsSeconds)
        {
            double seconds = Math.Max(0, rawSeconds);
            if (batchCount >= batchCap)
                break;

            double projectedTotal = totalSeconds + seconds;
            if (batchCount > 0 && projectedTotal > totalSecondsCeiling)
                break;

            batchCount++;
            totalSeconds = projectedTotal;
        }

        if (batchCount == 0)
        {
            batchCount = 1;
            totalSeconds = Math.Max(0, segmentDurationsSeconds[0]);
        }

        double scale = freeGpuMemoryMb > 0
            ? freeGpuMemoryMb / (double)BatchSizingReferenceFreeVramMb
            : 0.0;

        return new QwenBatchingPlan(
            batchCount,
            batchCap,
            totalSeconds,
            totalSecondsCeiling,
            maxSegmentSeconds,
            freeGpuMemoryMb,
            scale);
    }

    /// <summary>
    /// Estimate the experimental total audio-seconds ceiling for one GPU batch.
    /// </summary>
    public static double EstimateExperimentalTotalSecondsCeiling(long freeGpuMemoryMb)
    {
        if (freeGpuMemoryMb <= 0)
            return BatchSizingReferenceTotalSecondsCeiling;

        double scale = freeGpuMemoryMb / (double)BatchSizingReferenceFreeVramMb;
        return Math.Max(1.0, Math.Floor(BatchSizingReferenceTotalSecondsCeiling * scale));
    }

    /// <summary>
    /// Estimate the experimental batch cap from the longest segment in the batch
    /// and the current free GPU memory.
    /// </summary>
    public static int EstimateExperimentalBatchCap(double maxSegmentSeconds, long freeGpuMemoryMb)
    {
        int referenceCap = ExperimentalBatchFrontier[^1].ReferenceBatchCap;
        foreach (var entry in ExperimentalBatchFrontier)
        {
            if (maxSegmentSeconds <= entry.MaxSegmentSeconds)
            {
                referenceCap = entry.ReferenceBatchCap;
                break;
            }
        }

        if (freeGpuMemoryMb <= 0)
            return 1;

        double scale = freeGpuMemoryMb / (double)BatchSizingReferenceFreeVramMb;
        return Math.Max(1, (int)Math.Floor(referenceCap * scale));
    }

    public IEnumerable<(int segId, string text)> Recognize(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        foreach (var result in RecognizeDetailed(segs, audio, maxNewTokens))
            yield return (result.SegmentId, result.Text);
    }

    public IEnumerable<QwenRecognitionResult> RecognizeDetailed(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        if (_encoder is null)
            throw new InvalidOperationException(
                "Serial encoder sessions are not loaded. Construct with preferBatched:false or call RecognizeBatchedDetailed.");

        for (int segId = 0; segId < segs.Count; segId++)
        {
            var (start, end, _) = segs[segId];
            int startSample = Math.Clamp((int)Math.Round(start * SampleRate), 0, audio.Length);
            int endSample = Math.Clamp((int)Math.Round(end * SampleRate), 0, audio.Length);
            if (endSample <= startSample)
            {
                yield return new QwenRecognitionResult(segId, "", [], [], [], null, "");
                continue;
            }

            int length = endSample - startSample;
            float[] mel = ComputeLogMelSpectrogram(audio, startSample, length, out int melFrames);
            float[] audioFeatures = RunEncoder(mel, melFrames, out int audioTokenCount);

            List<int> promptIds = BuildPromptIds(audioTokenCount);
            int audioOffset = GetAudioPadStart(promptIds);

            var (rawTokens, rawLogprobs) = _useCudaIoBinding
                ? DecodeWithIoBinding(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens)
                : DecodeOnCpu(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens);

            var (textTokens, textLogprobs) = ExtractTextTokens(rawTokens, rawLogprobs);
            string rawText = DecodeTokens(textTokens);
            var parsed = ParseMetadataPrefix(rawText);
            yield return new QwenRecognitionResult(
                segId,
                parsed.Text,
                rawTokens,
                textTokens,
                textLogprobs,
                parsed.Language,
                rawText);
        }
    }

    private static SessionOptions MakeSessionOptions(
        ExecutionProvider ep,
        GraphOptimizationLevel optimizationLevel,
        out bool usesCuda)
    {
        var opts = new SessionOptions { GraphOptimizationLevel = optimizationLevel };
        usesCuda = false;

        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                {
                    try
                    {
                        opts.AppendExecutionProvider_CUDA(0);
                        usesCuda = true;
                        break;
                    }
                    catch { }
                }
                try { opts.AppendExecutionProvider_DML(0); } catch { }
                break;
            case ExecutionProvider.Cuda:
                opts.AppendExecutionProvider_CUDA(0);
                usesCuda = true;
                break;
            case ExecutionProvider.DirectML:
                opts.AppendExecutionProvider_DML(0);
                break;
            case ExecutionProvider.Cpu:
                break;
        }

        return opts;
    }

    private float[] RunEncoder(float[] mel, int melFrames, out int audioTokenCount)
    {
        using var results = _encoder.Run(
        [
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(mel, [1, NMels, melFrames])),
        ]);

        var features = results.First(r => r.Name == "audio_features").AsTensor<float>();
        audioTokenCount = features.Dimensions[1];
        return ExtractTensor(features);
    }

    private static float[] RunEncoderSession(
        InferenceSession encoder,
        float[] mel,
        int melFrames,
        out int audioTokenCount)
    {
        using var results = encoder.Run(
        [
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(mel, [1, NMels, melFrames])),
        ]);

        var features = results.First(r => r.Name == "audio_features").AsTensor<float>();
        audioTokenCount = features.Dimensions[1];
        return ExtractTensor(features);
    }

    private void RunDecoderPrefill(IReadOnlyList<int> promptIds, int audioOffset, float[] audioFeatures, int audioTokenCount)
    {
        long[] inputIds = promptIds.Select(i => (long)i).ToArray();
        long[] positionIds = Enumerable.Range(0, promptIds.Count).Select(i => (long)i).ToArray();
        long[] audioOffsetTensor = [audioOffset];

        using var initOutputs = _decoderInit.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, [1, inputIds.Length])),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(positionIds, [1, positionIds.Length])),
            NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures, [1, audioTokenCount, _hiddenSize])),
            NamedOnnxValue.CreateFromTensor("audio_offset", new DenseTensor<long>(audioOffsetTensor, [1])),
        ]);
    }

    private static void RunDecoderPrefillSession(
        InferenceSession decoderInit,
        int hiddenSize,
        IReadOnlyList<int> promptIds,
        int audioOffset,
        float[] audioFeatures,
        int audioTokenCount)
    {
        long[] inputIds = promptIds.Select(i => (long)i).ToArray();
        long[] positionIds = Enumerable.Range(0, promptIds.Count).Select(i => (long)i).ToArray();
        long[] audioOffsetTensor = [audioOffset];

        using var initOutputs = decoderInit.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, [1, inputIds.Length])),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(positionIds, [1, positionIds.Length])),
            NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures, [1, audioTokenCount, hiddenSize])),
            NamedOnnxValue.CreateFromTensor("audio_offset", new DenseTensor<long>(audioOffsetTensor, [1])),
        ]);
    }

    /// <summary>
    /// Batched encoder + decoder-prefill recognition. When batching artifacts are
    /// present this runs all segments through encoder_batched.onnx and
    /// decoder_init_batched.onnx in groups, then decodes each segment's token
    /// stream serially using decoder_step.onnx. Falls back to serial
    /// RecognizeDetailed when the batched sessions are not loaded.
    /// </summary>
    public IEnumerable<QwenRecognitionResult> RecognizeBatchedDetailed(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        if (_encoderBatched is null || _decoderInitBatched is null)
        {
            foreach (var r in RecognizeDetailed(segs, audio, maxNewTokens))
                yield return r;
            yield break;
        }

        // Prepare all segments: compute mel spectrograms (parallel inside ComputeLogMelSpectrogram)
        var prepared = new List<QwenBatchedItem>(segs.Count);
        for (int segId = 0; segId < segs.Count; segId++)
        {
            var (start, end, _) = segs[segId];
            int startSample = Math.Clamp((int)Math.Round(start * SampleRate), 0, audio.Length);
            int endSample   = Math.Clamp((int)Math.Round(end   * SampleRate), 0, audio.Length);
            if (endSample <= startSample)
            {
                prepared.Add(new QwenBatchedItem(segId, 0, null, 0));
                continue;
            }
            float[] mel = ComputeLogMelSpectrogram(audio, startSample, endSample - startSample, out int melFrames);
            prepared.Add(new QwenBatchedItem(segId, endSample - startSample, mel, melFrames));
        }

        foreach (var item in prepared.Where(it => it.Mel is null))
            yield return new QwenRecognitionResult(item.SegId, "", [], [], [], null, "");

        // Sort by duration descending for optimal GPU batching
        var validItems = prepared
            .Where(it => it.Mel is not null)
            .OrderByDescending(it => it.DurationSamples)
            .ToList();

        var (totalGpuMb, freeGpuMb) = HardwareInfo.GetGpuMemoryMb();
        // NVML may return 0 if the library is unavailable; fall back to a conservative 8 GB estimate
        // so batch sizing still works rather than clamping to 1.
        const long FallbackFreeVramMb = 8_000;
        long freeGpuMemoryMb = freeGpuMb > 0 ? freeGpuMb : FallbackFreeVramMb;
        Console.Error.WriteLine($"[Qwen3Asr] GPU memory: total={totalGpuMb} MB, free={freeGpuMb} MB (effective={freeGpuMemoryMb} MB)");
        int index = 0;

        while (index < validItems.Count)
        {
            var remainingDurations = validItems
                .Skip(index)
                .Select(it => it.DurationSamples / (double)SampleRate)
                .ToList();
            var plan = ComputeExperimentalBatchingPlan(remainingDurations, freeGpuMemoryMb);
            int take  = Math.Min(plan.BatchCount, validItems.Count - index);
            var batch = validItems.GetRange(index, take);
            index += take;

            // ── Batched encoder ──────────────────────────────────────────────
            int maxMelFrames = batch.Max(it => it.MelFrames);
            var melBatch     = new float[take * NMels * maxMelFrames];
            var inputLengths = new long[take];
            for (int b = 0; b < take; b++)
            {
                inputLengths[b] = batch[b].MelFrames;
                Array.Copy(batch[b].Mel!, 0, melBatch, b * NMels * maxMelFrames, batch[b].Mel!.Length);
            }

            using var encoderResults = _encoderBatched.Run(
            [
                NamedOnnxValue.CreateFromTensor("mel",          new DenseTensor<float>(melBatch,     [take, NMels, maxMelFrames])),
                NamedOnnxValue.CreateFromTensor("input_lengths", new DenseTensor<long>(inputLengths, [take])),
            ]);

            var audioFeaturesTensor = encoderResults.First(r => r.Name == "audio_features").AsTensor<float>();
            var audioLengthsTensor  = encoderResults.First(r => r.Name == "audio_feature_lengths").AsTensor<long>();
            int  maxAudioTokens     = (int)audioLengthsTensor.Max();
            long[] audioLengths     = [.. audioLengthsTensor];

            // Rearrange: [take, sourceTokens, hidden] → [take, maxAudioTokens, hidden]
            float[] audioFeaturesSource     = ExtractTensor(audioFeaturesTensor);
            int     sourceTokensPerBatch    = audioFeaturesTensor.Dimensions[1];
            float[] audioFeatures           = new float[take * maxAudioTokens * _hiddenSize];
            for (int b = 0; b < take; b++)
            for (int t = 0; t < maxAudioTokens; t++)
            {
                int srcBase = (b * sourceTokensPerBatch + t) * _hiddenSize;
                int dstBase = (b * maxAudioTokens      + t) * _hiddenSize;
                Array.Copy(audioFeaturesSource, srcBase, audioFeatures, dstBase, _hiddenSize);
            }

            // ── Batched decoder_init ─────────────────────────────────────────
            List<int> prompt   = BuildPromptIds(maxAudioTokens);
            int       audioOffset = GetAudioPadStart(prompt);
            int       seqLen   = prompt.Count;

            var inputIds    = new long[take * seqLen];
            var positionIds = new long[take * seqLen];
            for (int b = 0; b < take; b++)
            for (int t = 0; t < seqLen; t++)
            {
                inputIds   [b * seqLen + t] = prompt[t];
                positionIds[b * seqLen + t] = t;
            }

            using var initResults = _decoderInitBatched.Run(
            [
                NamedOnnxValue.CreateFromTensor("input_ids",      new DenseTensor<long> (inputIds,     [take, seqLen])),
                NamedOnnxValue.CreateFromTensor("position_ids",   new DenseTensor<long> (positionIds,  [take, seqLen])),
                NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures,[take, maxAudioTokens, _hiddenSize])),
                NamedOnnxValue.CreateFromTensor("audio_lengths",  new DenseTensor<long> (audioLengths, [take])),
                NamedOnnxValue.CreateFromTensor("audio_offset",   new DenseTensor<long> (new long[] { audioOffset }, [1])),
            ]);

            float[] batchedLogits = ExtractTensor(initResults.First(r => r.Name == "logits").AsTensor<float>());
            float[] batchedKeys   = ExtractTensor(initResults.First(r => r.Name == "present_keys").AsTensor<float>());
            float[] batchedValues = ExtractTensor(initResults.First(r => r.Name == "present_values").AsTensor<float>());

            int vocabSize      = batchedLogits.Length / (take * seqLen);
            const int NLayers  = 28;
            const int NKvHeads = 8;
            const int HdDim    = 128;

            // Extract first token per sequence from batched prefill logits
            var seqTokens    = new List<int>  [take];
            var seqLogprobs  = new List<float>[take];
            var seqDone      = new bool[take];
            var seqNextToken = new int[take];

            for (int b = 0; b < take; b++)
            {
                seqTokens[b]   = new List<int>  (maxNewTokens);
                seqLogprobs[b] = new List<float>(maxNewTokens);
                int logitOffset = b * seqLen * vocabSize + (seqLen - 1) * vocabSize;
                int firstToken  = ArgMaxSpan(batchedLogits.AsSpan(logitOffset, vocabSize), out float firstLogprob);
                seqTokens[b].Add(firstToken);
                seqLogprobs[b].Add(firstLogprob);
                seqNextToken[b] = firstToken;
                seqDone[b]      = IsEos(firstToken);
            }

            // ── Joint batched autoregressive decode ──────────────────────────
            // All sequences share the same pastSeqLen (same prefill prompt length),
            // so their KV caches stay in lockstep — no slicing needed.
            float[] pastKeys   = batchedKeys;
            float[] pastValues = batchedValues;
            int pastSeqLen     = seqLen;

            var batchEmbeds = new float[take * _hiddenSize];
            var stepPosArr  = new long[take];
            var tmpEmbed    = new float[_hiddenSize];

            for (int step = 0; step < maxNewTokens - 1; step++)
            {
                if (Array.TrueForAll(seqDone, d => d)) break;

                long pos = seqLen + step;
                for (int b = 0; b < take; b++)
                {
                    ReadTokenEmbedding(seqNextToken[b], tmpEmbed);
                    Array.Copy(tmpEmbed, 0, batchEmbeds, b * _hiddenSize, _hiddenSize);
                    stepPosArr[b] = pos;
                }

                using var stepOutputs = _decoderStep.Run(
                [
                    NamedOnnxValue.CreateFromTensor("input_embeds", new DenseTensor<float>(batchEmbeds, [take, 1, _hiddenSize])),
                    NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long> (stepPosArr,  [take, 1])),
                    NamedOnnxValue.CreateFromTensor("past_keys",    new DenseTensor<float>(pastKeys,    [NLayers, take, NKvHeads, pastSeqLen, HdDim])),
                    NamedOnnxValue.CreateFromTensor("past_values",  new DenseTensor<float>(pastValues,  [NLayers, take, NKvHeads, pastSeqLen, HdDim])),
                ]);

                float[] stepLogits = ExtractTensor(stepOutputs.First(r => r.Name == "logits").AsTensor<float>());
                pastKeys   = ExtractTensor(stepOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
                pastValues = ExtractTensor(stepOutputs.First(r => r.Name == "present_values").AsTensor<float>());
                pastSeqLen++;

                int stepVocabSize = stepLogits.Length / take; // [take, 1, vocab] flat
                for (int b = 0; b < take; b++)
                {
                    if (seqDone[b]) continue;
                    int nextToken = ArgMaxSpan(stepLogits.AsSpan(b * stepVocabSize, stepVocabSize), out float logprob);
                    seqNextToken[b] = nextToken;
                    seqTokens[b].Add(nextToken);
                    seqLogprobs[b].Add(logprob);
                    seqDone[b] = IsEos(nextToken) || seqTokens[b].Count >= maxNewTokens;
                }
            }

            // Yield results for this batch
            for (int b = 0; b < take; b++)
            {
                var (textTokens, textLogprobs2) = ExtractTextTokens(seqTokens[b], seqLogprobs[b]);
                string rawText = DecodeTokens(textTokens);
                var parsed = ParseMetadataPrefix(rawText);
                yield return new QwenRecognitionResult(
                    batch[b].SegId, parsed.Text, seqTokens[b], textTokens, textLogprobs2, parsed.Language, rawText);
            }
        }
    }

    private int ArgMaxSpan(Span<float> logits, out float logprob)
    {
        int   best    = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }

        double sumExp = 0;
        for (int i = 0; i < logits.Length; i++)
            sumExp += Math.Exp(logits[i] - bestVal);
        logprob = (float)(-Math.Log(sumExp));
        return best;
    }

    public bool HasExperimentalBatchingArtifacts()
        => _encoderBatched is not null && _decoderInitBatched is not null;

    public QwenExperimentalBatchBenchmark BenchmarkExperimentalBatching(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio)
        => BenchmarkExperimentalBatching(_modelPath, _executionProvider, segs, audio);

    public static QwenExperimentalBatchBenchmark BenchmarkExperimentalBatching(
        string modelPath,
        ExecutionProvider ep,
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio)
    {
        if (!File.Exists(Path.Combine(modelPath, EncoderBatchedFile)) ||
            !File.Exists(Path.Combine(modelPath, DecoderInitBatchedFile)))
            throw new FileNotFoundException("Experimental batching artifacts are missing.");

        string configJson = File.ReadAllText(Path.Combine(modelPath, ConfigFile));
        using var configDoc = JsonDocument.Parse(configJson);
        int hiddenSize = configDoc.RootElement.GetProperty("decoder").GetProperty("hidden_size").GetInt32();

        var prepared = new List<QwenPreparedSegment>(segs.Count);
        foreach (var (start, end, _) in segs)
        {
            int startSample = Math.Clamp((int)Math.Round(start * SampleRate), 0, audio.Length);
            int endSample = Math.Clamp((int)Math.Round(end * SampleRate), 0, audio.Length);
            if (endSample <= startSample)
                continue;

            int length = endSample - startSample;
            float[] mel = ComputeLogMelSpectrogram(audio, startSample, length, out int melFrames);
            prepared.Add(new QwenPreparedSegment(length / (double)SampleRate, mel, melFrames));
        }

        if (prepared.Count == 0)
            return new QwenExperimentalBatchBenchmark(0, 0, 0, 0, 0, [], 0, 0, 0);

        double serialEncoderMs = 0;
        double serialPrefillMs = 0;
        {
            var encoderOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED, out _);
            var decoderOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED, out _);
            using var encoder = new InferenceSession(Path.Combine(modelPath, EncoderFile), encoderOpts);
            using var decoderInit = new InferenceSession(Path.Combine(modelPath, DecoderInitFile), decoderOpts);
            foreach (var segment in prepared)
            {
                var sw = Stopwatch.StartNew();
                float[] audioFeatures = RunEncoderSession(encoder, segment.Mel, segment.MelFrames, out int audioTokenCount);
                sw.Stop();
                serialEncoderMs += sw.Elapsed.TotalMilliseconds;

                List<int> promptIds = BuildPromptIds(audioTokenCount);
                int audioOffset = GetAudioPadStart(promptIds);
                sw.Restart();
                RunDecoderPrefillSession(decoderInit, hiddenSize, promptIds, audioOffset, audioFeatures, audioTokenCount);
                sw.Stop();
                serialPrefillMs += sw.Elapsed.TotalMilliseconds;
            }
        }

        var sorted = prepared.OrderByDescending(p => p.DurationSeconds).ToList();
        var batchStats = new List<QwenExperimentalBatchBenchmarkBatch>();
        double batchedEncoderMs = 0;
        double batchedPrefillMs = 0;
        {
            var encoderOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED, out _);
            var decoderOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED, out _);
            using var encoderBatched = new InferenceSession(Path.Combine(modelPath, EncoderBatchedFile), encoderOpts);
            using var decoderInitBatched = new InferenceSession(Path.Combine(modelPath, DecoderInitBatchedFile), decoderOpts);
            long freeGpuMemoryMb = HardwareInfo.GetGpuMemoryMb().FreeMb;

            int index = 0;
            while (index < sorted.Count)
            {
                var remainingDurations = sorted.Skip(index).Select(s => s.DurationSeconds).ToArray();
                QwenBatchingPlan plan = ComputeExperimentalBatchingPlan(remainingDurations, freeGpuMemoryMb);
                int take = Math.Min(plan.BatchCount, sorted.Count - index);
                var batch = sorted.GetRange(index, take);
                index += take;

                int maxMelFrames = batch.Max(s => s.MelFrames);
                float[] mel = new float[take * NMels * maxMelFrames];
                long[] inputLengths = new long[take];
                for (int batchIndex = 0; batchIndex < take; batchIndex++)
                {
                    var segment = batch[batchIndex];
                    inputLengths[batchIndex] = segment.MelFrames;
                    Array.Copy(segment.Mel, 0, mel, batchIndex * NMels * maxMelFrames, segment.Mel.Length);
                }

                var sw = Stopwatch.StartNew();
                using var encoderResults = encoderBatched.Run(
                [
                    NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(mel, [take, NMels, maxMelFrames])),
                    NamedOnnxValue.CreateFromTensor("input_lengths", new DenseTensor<long>(inputLengths, [take])),
                ]);
                sw.Stop();
                double encoderMs = sw.Elapsed.TotalMilliseconds;
                batchedEncoderMs += encoderMs;

                var audioFeaturesTensor = encoderResults.First(r => r.Name == "audio_features").AsTensor<float>();
                var audioLengthsTensor = encoderResults.First(r => r.Name == "audio_feature_lengths").AsTensor<long>();
                int maxAudioTokens = (int)audioLengthsTensor.Max();
                int seqLen = BuildPromptIds(maxAudioTokens).Count;
                List<int> prompt = BuildPromptIds(maxAudioTokens);
                int audioOffset = GetAudioPadStart(prompt);

                long[] inputIds = new long[take * seqLen];
                long[] positionIds = new long[take * seqLen];
                for (int batchIndex = 0; batchIndex < take; batchIndex++)
                {
                    for (int tokenIndex = 0; tokenIndex < seqLen; tokenIndex++)
                    {
                        inputIds[batchIndex * seqLen + tokenIndex] = prompt[tokenIndex];
                        positionIds[batchIndex * seqLen + tokenIndex] = tokenIndex;
                    }
                }

                float[] audioFeaturesSource = ExtractTensor(audioFeaturesTensor);
                float[] audioFeatures = new float[take * maxAudioTokens * hiddenSize];
                int sourceTokensPerBatch = audioFeaturesTensor.Dimensions[1];
                for (int batchIndex = 0; batchIndex < take; batchIndex++)
                {
                    for (int tokenIndex = 0; tokenIndex < maxAudioTokens; tokenIndex++)
                    {
                        int srcBase = ((batchIndex * sourceTokensPerBatch) + tokenIndex) * hiddenSize;
                        int dstBase = ((batchIndex * maxAudioTokens) + tokenIndex) * hiddenSize;
                        Array.Copy(audioFeaturesSource, srcBase, audioFeatures, dstBase, hiddenSize);
                    }
                }

                long[] audioLengths = [.. audioLengthsTensor];
                long[] audioOffsetTensor = [audioOffset];

                sw.Restart();
                using var decoderResults = decoderInitBatched.Run(
                [
                    NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, [take, seqLen])),
                    NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(positionIds, [take, seqLen])),
                    NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures, [take, maxAudioTokens, hiddenSize])),
                    NamedOnnxValue.CreateFromTensor("audio_lengths", new DenseTensor<long>(audioLengths, [take])),
                    NamedOnnxValue.CreateFromTensor("audio_offset", new DenseTensor<long>(audioOffsetTensor, [1])),
                ]);
                sw.Stop();
                double prefillMs = sw.Elapsed.TotalMilliseconds;
                batchedPrefillMs += prefillMs;

                batchStats.Add(new QwenExperimentalBatchBenchmarkBatch(
                    take,
                    batch.Sum(s => s.DurationSeconds),
                    batch.Max(s => s.DurationSeconds),
                    plan.BatchCap,
                    plan.TotalSecondsCeiling,
                    encoderMs,
                    prefillMs));
            }

            return new QwenExperimentalBatchBenchmark(
                prepared.Count,
                serialEncoderMs,
                serialPrefillMs,
                batchedEncoderMs,
                batchedPrefillMs,
                batchStats,
                freeGpuMemoryMb,
                EstimateExperimentalTotalSecondsCeiling(freeGpuMemoryMb),
                batchStats.Count);
        }
    }

    private (List<int> rawTokens, List<float> rawLogprobs) DecodeOnCpu(
        IReadOnlyList<int> promptIds,
        int audioOffset,
        float[] audioFeatures,
        int audioTokenCount,
        int maxNewTokens)
    {
        long[] inputIds = promptIds.Select(i => (long)i).ToArray();
        long[] positionIds = Enumerable.Range(0, promptIds.Count).Select(i => (long)i).ToArray();
        long[] audioOffsetTensor = [audioOffset];

        using var initOutputs = _decoderInit.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, [1, inputIds.Length])),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(positionIds, [1, positionIds.Length])),
            NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures, [1, audioTokenCount, _hiddenSize])),
            NamedOnnxValue.CreateFromTensor("audio_offset", new DenseTensor<long>(audioOffsetTensor, [1])),
        ]);

        float[] logits = ExtractTensor(initOutputs.First(r => r.Name == "logits").AsTensor<float>());
        float[] pastKeys = ExtractTensor(initOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
        float[] pastValues = ExtractTensor(initOutputs.First(r => r.Name == "present_values").AsTensor<float>());
        int pastSeqLen = promptIds.Count;
        var tokenEmbed = new float[_hiddenSize];
        long[] stepPos = [0L];

        var rawTokens = new List<int>(maxNewTokens);
        var rawLogprobs = new List<float>(maxNewTokens);

        int nextToken = ArgMaxLastLogits(logits, promptIds.Count, out float nextLogprob);
        rawTokens.Add(nextToken);
        rawLogprobs.Add(nextLogprob);
        pastSeqLen = promptIds.Count;

        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tokenEmbed);
            stepPos[0] = promptIds.Count + rawTokens.Count - 1L;

            using var stepOutputs = _decoderStep.Run(
            [
                NamedOnnxValue.CreateFromTensor("input_embeds", new DenseTensor<float>(tokenEmbed, [1, 1, _hiddenSize])),
                NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(stepPos, [1, 1])),
                NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(pastKeys, [28, 1, 8, pastSeqLen, 128])),
                NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(pastValues, [28, 1, 8, pastSeqLen, 128])),
            ]);

            logits = ExtractTensor(stepOutputs.First(r => r.Name == "logits").AsTensor<float>());
            pastKeys = ExtractTensor(stepOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
            pastValues = ExtractTensor(stepOutputs.First(r => r.Name == "present_values").AsTensor<float>());
            pastSeqLen++;

            nextToken = ArgMaxLastLogits(logits, 1, out nextLogprob);
            rawTokens.Add(nextToken);
            rawLogprobs.Add(nextLogprob);
        }

        return (rawTokens, rawLogprobs);
    }

    private (List<int> rawTokens, List<float> rawLogprobs) DecodeWithIoBinding(
        IReadOnlyList<int> promptIds,
        int audioOffset,
        float[] audioFeatures,
        int audioTokenCount,
        int maxNewTokens)
    {
        using var cpuMemInfo = new OrtMemoryInfo("Cpu", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts = new RunOptions();

        long[] inputIds = promptIds.Select(i => (long)i).ToArray();
        long[] positionIds = Enumerable.Range(0, promptIds.Count).Select(i => (long)i).ToArray();
        long[] audioOffsetTensor = [audioOffset];

        using var inputIdsValue = OrtValue.CreateTensorValueFromMemory(inputIds, [1, inputIds.Length]);
        using var positionIdsValue = OrtValue.CreateTensorValueFromMemory(positionIds, [1, positionIds.Length]);
        using var audioFeaturesValue = OrtValue.CreateTensorValueFromMemory(audioFeatures, [1, audioTokenCount, _hiddenSize]);
        using var audioOffsetValue = OrtValue.CreateTensorValueFromMemory(audioOffsetTensor, [1]);

        using var initBinding = _decoderInit.CreateIoBinding();
        initBinding.BindInput("input_ids", inputIdsValue);
        initBinding.BindInput("position_ids", positionIdsValue);
        initBinding.BindInput("audio_features", audioFeaturesValue);
        initBinding.BindInput("audio_offset", audioOffsetValue);
        initBinding.BindOutputToDevice("logits", cpuMemInfo);
        initBinding.BindOutputToDevice("present_keys", cudaMemInfo);
        initBinding.BindOutputToDevice("present_values", cudaMemInfo);
        _decoderInit.RunWithBinding(runOpts, initBinding);

        var initOutputs = initBinding.GetOutputValues();
        var rawTokens = new List<int>(maxNewTokens);
        var rawLogprobs = new List<float>(maxNewTokens);
        var tokenEmbed = new float[_hiddenSize];
        long[] stepPos = [0L];

        int nextToken = ArgMaxLastLogits(initOutputs[0], promptIds.Count, out float nextLogprob);
        rawTokens.Add(nextToken);
        rawLogprobs.Add(nextLogprob);

        IDisposableReadOnlyCollection<OrtValue>? prevOutputs = initOutputs;
        int pastSeqLen = promptIds.Count;

        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tokenEmbed);
            stepPos[0] = promptIds.Count + rawTokens.Count - 1L;

            using var tokenEmbedValue = OrtValue.CreateTensorValueFromMemory(tokenEmbed, [1, 1, _hiddenSize]);
            using var stepPosValue = OrtValue.CreateTensorValueFromMemory(stepPos, [1, 1]);
            using var stepBinding = _decoderStep.CreateIoBinding();
            stepBinding.BindInput("input_embeds", tokenEmbedValue);
            stepBinding.BindInput("position_ids", stepPosValue);
            stepBinding.BindInput("past_keys", prevOutputs![1]);
            stepBinding.BindInput("past_values", prevOutputs[2]);
            stepBinding.BindOutputToDevice("logits", cpuMemInfo);
            stepBinding.BindOutputToDevice("present_keys", cudaMemInfo);
            stepBinding.BindOutputToDevice("present_values", cudaMemInfo);
            _decoderStep.RunWithBinding(runOpts, stepBinding);

            var curOutputs = stepBinding.GetOutputValues();
            nextToken = ArgMaxLastLogits(curOutputs[0], 1, out nextLogprob);
            rawTokens.Add(nextToken);
            rawLogprobs.Add(nextLogprob);
            pastSeqLen++;

            prevOutputs.Dispose();
            prevOutputs = curOutputs;
        }

        prevOutputs?.Dispose();
        return (rawTokens, rawLogprobs);
    }

    private int ArgMaxLastLogits(float[] logits, int seqLen, out float logprob)
    {
        int vocabSize = logits.Length / seqLen;
        int offset = (seqLen - 1) * vocabSize;
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < vocabSize; i++)
        {
            float value = logits[offset + i];
            if (value > bestVal)
            {
                bestVal = value;
                best = i;
            }
        }

        double sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
            sumExp += Math.Exp(logits[offset + i] - bestVal);

        logprob = (float)(-Math.Log(sumExp));
        return best;
    }

    private int ArgMaxLastLogits(OrtValue logits, int seqLen, out float logprob)
    {
        var span = logits.GetTensorDataAsSpan<float>();
        int vocabSize = span.Length / seqLen;
        int offset = (seqLen - 1) * vocabSize;
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < vocabSize; i++)
        {
            float value = span[offset + i];
            if (value > bestVal)
            {
                bestVal = value;
                best = i;
            }
        }

        double sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
            sumExp += Math.Exp(span[offset + i] - bestVal);

        logprob = (float)(-Math.Log(sumExp));
        return best;
    }

    private void ReadTokenEmbedding(int tokenId, float[] destination)
    {
        long byteOffset = (long)tokenId * _hiddenSize * sizeof(float);
        _embedAccessor.ReadArray(byteOffset, destination, 0, _hiddenSize);
    }

    private static (List<int> textTokens, List<float> textLogprobs) ExtractTextTokens(
        IReadOnlyList<int> rawTokens,
        IReadOnlyList<float> rawLogprobs)
    {
        var textTokens = new List<int>(rawTokens.Count);
        var textLogprobs = new List<float>(rawTokens.Count);
        for (int i = 0; i < rawTokens.Count; i++)
        {
            int token = rawTokens[i];
            if (token >= 0 && token < 151643)
            {
                textTokens.Add(token);
                textLogprobs.Add(i < rawLogprobs.Count ? rawLogprobs[i] : 0f);
            }
        }

        return (textTokens, textLogprobs);
    }

    public string DecodeTokens(IReadOnlyList<int> tokens)
    {
        var bytes = new List<byte>(tokens.Count * 4);
        foreach (int token in tokens)
        {
            if (token < 0 || token >= _idToToken.Length)
                continue;

            string? raw = _idToToken[token];
            if (raw is null)
                continue;

            foreach (char ch in raw)
                if (_byteLevelDecode.TryGetValue(ch, out byte value))
                    bytes.Add(value);
        }

        string text = Encoding.UTF8.GetString(bytes.ToArray());
        return text.Length > 0 && text[0] == ' ' ? text[1..] : text;
    }

    private bool IsEos(int token) => _eosTokenIdSet.Contains(token);

    private static QwenMetadataParseResult ParseMetadataPrefix(string text)
    {
        if (!text.StartsWith(LanguagePrefix, StringComparison.OrdinalIgnoreCase))
            return new QwenMetadataParseResult(text, null);

        string remainder = text[LanguagePrefix.Length..];
        foreach (string language in SpokenLanguageNames.OrderByDescending(name => name.Length))
        {
            if (!remainder.StartsWith(language, StringComparison.OrdinalIgnoreCase))
                continue;

            string content = remainder[language.Length..].TrimStart();
            return new QwenMetadataParseResult(content, language);
        }

        Match match = Regex.Match(remainder, @"^([A-Za-z][A-Za-z\-]+)\s*(.*)$");
        if (!match.Success)
            return new QwenMetadataParseResult(text, null);

        string detectedLanguage = match.Groups[1].Value;
        string strippedText = match.Groups[2].Value.TrimStart();
        return new QwenMetadataParseResult(strippedText, detectedLanguage);
    }

    private static List<int> BuildPromptIds(int audioTokenCount)
    {
        var ids = new List<int>(audioTokenCount + 16)
        {
            ImStartTokenId,
            SystemTokenId,
            NewlineTokenId,
            ImEndTokenId,
            NewlineTokenId,
            ImStartTokenId,
            UserTokenId,
            NewlineTokenId,
            AudioStartTokenId,
        };

        for (int i = 0; i < audioTokenCount; i++)
            ids.Add(AudioPadTokenId);

        ids.Add(AudioEndTokenId);
        ids.Add(ImEndTokenId);
        ids.Add(NewlineTokenId);
        ids.Add(ImStartTokenId);
        ids.Add(AssistantTokenId);
        ids.Add(NewlineTokenId);
        return ids;
    }

    private static int GetAudioPadStart(IReadOnlyList<int> promptIds)
    {
        for (int i = 0; i < promptIds.Count; i++)
            if (promptIds[i] == AudioPadTokenId)
                return i;
        throw new InvalidOperationException("Prompt does not contain <|audio_pad|> tokens.");
    }

    private static float[] ComputeLogMelSpectrogram(float[] signal, int start, int length, out int framesOut)
    {
        int pad = NFft / 2;
        float[] padded = ReflectPad(signal, start, length, pad);
        int frameCount = ((padded.Length - NFft) / HopLength) + 1;
        int keptFrames = Math.Max(frameCount - 1, 1);
        int freqBins = (NFft / 2) + 1;
        var mel = new float[NMels * keptFrames];
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
        {
            if (mel[i] > maxLog)
                maxLog = mel[i];
        }

        float floor = MathF.Max(maxLog - LogClampSpan, LogFloor);
        for (int i = 0; i < mel.Length; i++)
            mel[i] = (MathF.Max(mel[i], floor) + LogOffset) / LogOffset;

        framesOut = keptFrames;
        return mel;
    }

    private static float[] ReflectPad(float[] signal, int start, int length, int pad)
    {
        if (length == 0)
            return new float[pad * 2];

        var padded = new float[length + (pad * 2)];
        Array.Copy(signal, start, padded, pad, length);

        for (int i = 0; i < pad; i++)
        {
            int leftSrc = Math.Min(length - 1, pad - i);
            int rightSrc = Math.Max(0, length - 2 - i);
            padded[i] = signal[start + leftSrc];
            padded[pad + length + i] = signal[start + rightSrc];
        }

        return padded;
    }

    private static float[,] CreateMelFilterbank()
    {
        int freqBins = (NFft / 2) + 1;
        var fb = new float[NMels, freqBins];

        var fftFreqs = new double[freqBins];
        for (int k = 0; k < freqBins; k++)
            fftFreqs[k] = (double)k * SampleRate / NFft;

        double fminMel = AudioUtils.HzToMelSlaney(0.0);
        double fmaxMel = AudioUtils.HzToMelSlaney(SampleRate / 2.0);
        var melF = new double[NMels + 2];
        for (int i = 0; i <= NMels + 1; i++)
        {
            double m = fminMel + (fmaxMel - fminMel) * i / (NMels + 1);
            melF[i] = AudioUtils.MelToHzSlaney(m);
        }

        var fdiff = new double[NMels + 1];
        for (int i = 0; i <= NMels; i++)
            fdiff[i] = melF[i + 1] - melF[i];

        for (int i = 0; i < NMels; i++)
        {
            for (int k = 0; k < freqBins; k++)
            {
                double lower = (fftFreqs[k] - melF[i]) / fdiff[i];
                double upper = (melF[i + 2] - fftFreqs[k]) / fdiff[i + 1];
                fb[i, k] = (float)Math.Max(0.0, Math.Min(lower, upper));
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

    private static float[] ExtractTensor(Tensor<float> tensor)
    {
        if (tensor is DenseTensor<float> dense)
            return dense.Buffer.ToArray();

        var result = new float[tensor.Length];
        int i = 0;
        foreach (float value in tensor)
            result[i++] = value;
        return result;
    }

    private static (string?[] idToToken, Dictionary<int, string> addedTokenContent) LoadTokenizerVocab(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var vocab = root.GetProperty("model").GetProperty("vocab");
        int maxId = -1;
        foreach (var kv in vocab.EnumerateObject())
            maxId = Math.Max(maxId, kv.Value.GetInt32());

        var added = new Dictionary<int, string>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var token in addedTokens.EnumerateArray())
            {
                int id = token.GetProperty("id").GetInt32();
                string content = token.GetProperty("content").GetString() ?? "";
                added[id] = content;
                maxId = Math.Max(maxId, id);
            }
        }

        var idToToken = new string?[maxId + 1];
        foreach (var kv in vocab.EnumerateObject())
            idToToken[kv.Value.GetInt32()] = kv.Name;

        return (idToToken, added);
    }

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

    public void Dispose()
    {
        _embedAccessor.Dispose();
        _embedMmf.Dispose();
        _decoderStep.Dispose();
        _decoderInitBatched?.Dispose();
        _encoderBatched?.Dispose();
        _decoderInit?.Dispose();
        _encoder?.Dispose();
    }
}

public sealed record QwenRecognitionResult(
    int SegmentId,
    string Text,
    IReadOnlyList<int> RawTokens,
    IReadOnlyList<int> TextTokens,
    IReadOnlyList<float> TextLogprobs,
    string? Language,
    string RawText);

/// <summary>
/// Experimental GPU batching plan for the batched Qwen3-ASR encoder +
/// decoder-prefill path.
/// </summary>
public readonly record struct QwenBatchingPlan(
    int BatchCount,
    int BatchCap,
    double TotalSeconds,
    double TotalSecondsCeiling,
    double MaxSegmentSeconds,
    long FreeGpuMemoryMb,
    double Scale);

public sealed record QwenExperimentalBatchBenchmark(
    int SegmentCount,
    double SerialEncoderMilliseconds,
    double SerialPrefillMilliseconds,
    double BatchedEncoderMilliseconds,
    double BatchedPrefillMilliseconds,
    IReadOnlyList<QwenExperimentalBatchBenchmarkBatch> Batches,
    long FreeGpuMemoryMb,
    double TotalSecondsCeiling,
    int BatchRuns);

public sealed record QwenExperimentalBatchBenchmarkBatch(
    int SegmentCount,
    double TotalSeconds,
    double MaxSegmentSeconds,
    int BatchCap,
    double TotalSecondsCeiling,
    double EncoderMilliseconds,
    double PrefillMilliseconds);

internal sealed record QwenMetadataParseResult(
    string Text,
    string? Language);

internal sealed record QwenPreparedSegment(
    double DurationSeconds,
    float[] Mel,
    int MelFrames);

internal sealed record QwenBatchedItem(
    int SegId,
    int DurationSamples,
    float[]? Mel,
    int MelFrames);
