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
    public const string DecoderFile = "decoder.onnx";
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
    private const long BatchSizingReferenceFreeVramMb = 22_673;
    private const double BatchSizingReferenceTotalSecondsCeiling = 224.0;

    private static readonly float[,] MelFilterbank = CreateMelFilterbank();
    private static readonly double[] HannWindow = Window.HannPeriodic(NFft);
    // Frontier calibrated for decoder.onnx (unified) on RTX 3090 with 22,673 MB free.
    // Sweep: 2–32s segments × 1–16 batch. OOM boundaries: >14 at 16s, >10 at 24s, >8 at 32s.
    private static readonly (double MaxSegmentSeconds, int ReferenceBatchCap)[] ExperimentalBatchFrontier =
    [
        (16.0, 14),
        (24.0, 10),
        (32.0,  8),
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
    private readonly InferenceSession? _decoderStep;
    private readonly InferenceSession? _decoder;
    private readonly InferenceSession? _encoderBatched;
    private readonly InferenceSession? _decoderInitBatched;
    private readonly bool _useCudaIoBinding;
    private readonly bool _preferBatched;
    private readonly string _modelPath;
    private readonly ExecutionProvider _executionProvider;
    private readonly GraphOptimizationLevel _optimizationLevel;
    private readonly int _hiddenSize;
    private readonly int _nLayers;
    private readonly int _nKvHeads;
    private readonly int _headDim;
    private readonly int _vocabSize;
    private readonly int _baseVocabSize;
    private readonly int[] _eosTokenIds;
    private readonly HashSet<int> _eosTokenIdSet;
    private readonly MemoryMappedFile _embedMmf;
    private readonly MemoryMappedViewAccessor _embedAccessor;
    private readonly string?[] _idToToken;
    private readonly Dictionary<int, string> _addedTokenContent;
    private readonly Dictionary<char, byte> _byteLevelDecode;

    public Qwen3Asr(
        string modelPath,
        ExecutionProvider ep = ExecutionProvider.Auto,
        bool preferBatched = false,
        GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    {
        _modelPath = modelPath;
        _executionProvider = ep;
        _optimizationLevel = optimizationLevel;
        string configJson = File.ReadAllText(Path.Combine(modelPath, ConfigFile));
        using var configDoc = JsonDocument.Parse(configJson);
        var root = configDoc.RootElement;
        var decoderConfig = root.GetProperty("decoder");
        var specialTokens = root.GetProperty("special_tokens");

        _hiddenSize = decoderConfig.GetProperty("hidden_size").GetInt32();
        _nLayers    = decoderConfig.GetProperty("num_layers").GetInt32();
        _nKvHeads   = decoderConfig.GetProperty("num_key_value_heads").GetInt32();
        _headDim    = decoderConfig.GetProperty("head_dim").GetInt32();
        _vocabSize = decoderConfig.GetProperty("vocab_size").GetInt32();
        _baseVocabSize = root.GetProperty("embed_tokens_shape")[0].GetInt32();
        _eosTokenIds = specialTokens.GetProperty("eos_token_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        _eosTokenIdSet = [.. _eosTokenIds];

        var encoderOpts = MakeSessionOptions(ep, optimizationLevel, out bool encoderUsesCuda);
        var decoderOpts = MakeSessionOptions(ep, optimizationLevel, out bool decoderUsesCuda);

        bool hasUnified       = File.Exists(Path.Combine(modelPath, DecoderFile));
        bool hasBatchedEncoder = File.Exists(Path.Combine(modelPath, EncoderBatchedFile));
        bool hasBatchedInit   = File.Exists(Path.Combine(modelPath, DecoderInitBatchedFile));
        _preferBatched = preferBatched && hasBatchedEncoder && (hasBatchedInit || hasUnified);

        if (hasUnified)
        {
            _decoderInit      = null!;
            _useCudaIoBinding = false;

            if (_preferBatched)
            {
                // Unified batched path: encoder_batched + decoder.onnx only.
                // Do not load decoder_init_batched here; we want the runtime
                // contract to be unambiguously single-decoder.
                _decoder = new InferenceSession(Path.Combine(modelPath, DecoderFile), decoderOpts);
                _encoder        = null!;
                _encoderBatched = new InferenceSession(Path.Combine(modelPath, EncoderBatchedFile), encoderOpts);
                _decoderInitBatched = null;
                _decoderStep = null;
            }
            else
            {
                _decoder        = new InferenceSession(Path.Combine(modelPath, DecoderFile), decoderOpts);
                _encoder        = new InferenceSession(Path.Combine(modelPath, EncoderFile), encoderOpts);
                _encoderBatched = null;
                _decoderInitBatched = null;
                _decoderStep = null;
            }
        }
        else if (_preferBatched)
        {
            // Batched split path: encoder_batched + decoder_init_batched + decoder_step.
            _decoder             = null;
            _encoder             = null!;
            _decoderInit         = null!;
            _encoderBatched      = new InferenceSession(Path.Combine(modelPath, EncoderBatchedFile),     encoderOpts);
            _decoderInitBatched  = new InferenceSession(Path.Combine(modelPath, DecoderInitBatchedFile), decoderOpts);
            _decoderStep         = new InferenceSession(Path.Combine(modelPath, DecoderStepFile),        decoderOpts);
            _useCudaIoBinding    = false;
        }
        else
        {
            // Serial split path: encoder + decoder_init + decoder_step.
            _decoder         = null;
            _encoder         = new InferenceSession(Path.Combine(modelPath, EncoderFile),     encoderOpts);
            _decoderInit     = new InferenceSession(Path.Combine(modelPath, DecoderInitFile), decoderOpts);
            _decoderStep     = new InferenceSession(Path.Combine(modelPath, DecoderStepFile), decoderOpts);
            _encoderBatched     = null;
            _decoderInitBatched = null;
            _useCudaIoBinding   = encoderUsesCuda && decoderUsesCuda;
        }

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

            var (rawTokens, rawLogprobs) = _decoder is not null
                ? DecodeOnCpuUnified(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens)
                : _useCudaIoBinding
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
        if (_encoderBatched is null || (_decoderInitBatched is null && _decoder is null))
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
                for (int melBin = 0; melBin < NMels; melBin++)
                {
                    Array.Copy(
                        batch[b].Mel!,
                        melBin * batch[b].MelFrames,
                        melBatch,
                        (b * NMels + melBin) * maxMelFrames,
                        batch[b].MelFrames);
                }
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
            float[] audioFeatures           = new float[take * maxAudioTokens * _hiddenSize];
            for (int b = 0; b < take; b++)
            for (int t = 0; t < (int)audioLengths[b]; t++)
                for (int h = 0; h < _hiddenSize; h++)
                    audioFeatures[(b * maxAudioTokens + t) * _hiddenSize + h] = audioFeaturesTensor[b, t, h];

            // ── Batched prefill ──────────────────────────────────────────────
            if (_decoder is not null)
            {
                var (unifiedInputEmbeds, unifiedPositionIds, unifiedPrefillMask, unifiedSeqLengths) =
                    BuildUnifiedBatchPrefillInputs(audioFeatures, audioLengths, take, maxAudioTokens);

                float[] emptyKv = [];
                using var initResults = _decoder.Run(
                [
                    NamedOnnxValue.CreateFromTensor("input_embeds",   new DenseTensor<float>(unifiedInputEmbeds, [take, unifiedSeqLengths.Max(), _hiddenSize])),
                    NamedOnnxValue.CreateFromTensor("position_ids",   new DenseTensor<long>(unifiedPositionIds,  [take, unifiedSeqLengths.Max()])),
                    NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<float>(unifiedPrefillMask, [take, 1, unifiedSeqLengths.Max(), unifiedSeqLengths.Max()])),
                    NamedOnnxValue.CreateFromTensor("past_keys",      new DenseTensor<float>(emptyKv,     [_nLayers, take, _nKvHeads, 0, _headDim])),
                    NamedOnnxValue.CreateFromTensor("past_values",    new DenseTensor<float>(emptyKv,     [_nLayers, take, _nKvHeads, 0, _headDim])),
                ]);

                float[] unifiedBatchedLogits = ExtractTensor(initResults.First(r => r.Name == "logits").AsTensor<float>());
                float[] presentKeys = ExtractTensor(initResults.First(r => r.Name == "present_keys").AsTensor<float>());
                float[] presentValues = ExtractTensor(initResults.First(r => r.Name == "present_values").AsTensor<float>());
                float[] pastKeys = CompactPrefillKv(presentKeys, unifiedSeqLengths, take);
                float[] pastValues = CompactPrefillKv(presentValues, unifiedSeqLengths, take);
                int[] pastLengths = [.. unifiedSeqLengths];

                int unifiedVocabSize = unifiedBatchedLogits.Length / (take * unifiedSeqLengths.Max());
                var unifiedSeqTokens = new List<int>[take];
                var unifiedSeqLogprobs = new List<float>[take];
                var unifiedSeqDone = new bool[take];
                var nextTokens = new int[take];

                for (int b = 0; b < take; b++)
                {
                    unifiedSeqTokens[b] = new List<int>(maxNewTokens);
                    unifiedSeqLogprobs[b] = new List<float>(maxNewTokens);
                    int logitOffset = b * unifiedSeqLengths.Max() * unifiedVocabSize + (unifiedSeqLengths[b] - 1) * unifiedVocabSize;
                    int firstToken = ArgMaxSpan(unifiedBatchedLogits.AsSpan(logitOffset, unifiedVocabSize), out float firstLogprob);
                    nextTokens[b] = firstToken;
                    unifiedSeqTokens[b].Add(firstToken);
                    unifiedSeqLogprobs[b].Add(firstLogprob);
                    unifiedSeqDone[b] = IsEos(firstToken);
                }

                for (int step = 1; step < maxNewTokens && unifiedSeqDone.Any(done => !done); step++)
                {
                    float[] stepEmbeds = new float[take * _hiddenSize];
                    var tokenEmbed = new float[_hiddenSize];
                    for (int b = 0; b < take; b++)
                    {
                        ReadTokenEmbedding(nextTokens[b], tokenEmbed);
                        Array.Copy(tokenEmbed, 0, stepEmbeds, b * _hiddenSize, _hiddenSize);
                    }

                    long[] stepPos = new long[take];
                    for (int b = 0; b < take; b++)
                        stepPos[b] = pastLengths[b];

                    int maxPastLen = pastLengths.Max();
                    float[] stepMask = BuildBatchStepAttentionMask(pastLengths, maxPastLen);

                    using var stepOutputs = _decoder.Run(
                    [
                        NamedOnnxValue.CreateFromTensor("input_embeds",   new DenseTensor<float>(stepEmbeds, [take, 1, _hiddenSize])),
                        NamedOnnxValue.CreateFromTensor("position_ids",   new DenseTensor<long>(stepPos,     [take, 1])),
                        NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<float>(stepMask,   [take, 1, 1, maxPastLen + 1])),
                        NamedOnnxValue.CreateFromTensor("past_keys",      new DenseTensor<float>(pastKeys,   [_nLayers, take, _nKvHeads, maxPastLen, _headDim])),
                        NamedOnnxValue.CreateFromTensor("past_values",    new DenseTensor<float>(pastValues, [_nLayers, take, _nKvHeads, maxPastLen, _headDim])),
                    ]);

                    float[] stepLogits = ExtractTensor(stepOutputs.First(r => r.Name == "logits").AsTensor<float>());
                    float[] stepPresentKeys = ExtractTensor(stepOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
                    float[] stepPresentValues = ExtractTensor(stepOutputs.First(r => r.Name == "present_values").AsTensor<float>());
                    bool[] activeMask = new bool[take];
                    for (int b = 0; b < take; b++)
                        activeMask[b] = !unifiedSeqDone[b];

                    for (int b = 0; b < take; b++)
                    {
                        int token = ArgMaxSpan(stepLogits.AsSpan(b * unifiedVocabSize, unifiedVocabSize), out float logprob);
                        if (!activeMask[b])
                            continue;

                        nextTokens[b] = token;
                        unifiedSeqTokens[b].Add(token);
                        unifiedSeqLogprobs[b].Add(logprob);
                        if (IsEos(token))
                            unifiedSeqDone[b] = true;
                    }

                    CompactStepKv(stepPresentKeys, stepPresentValues, pastLengths, activeMask, take, out pastKeys, out pastValues, out pastLengths);
                }

                for (int b = 0; b < take; b++)
                {
                    var (textTokens, textLogprobs2) = ExtractTextTokens(unifiedSeqTokens[b], unifiedSeqLogprobs[b]);
                    string rawText = DecodeTokens(textTokens);
                    var parsed = ParseMetadataPrefix(rawText);
                    yield return new QwenRecognitionResult(
                        batch[b].SegId, parsed.Text, unifiedSeqTokens[b], textTokens, textLogprobs2, parsed.Language, rawText);
                }

                continue;
            }

            List<int> prompt    = BuildPromptIds(maxAudioTokens);
            int       audioOffset = GetAudioPadStart(prompt);
            int       seqLen    = prompt.Count;

            var positionIds = new long[take * seqLen];
            for (int b = 0; b < take; b++)
            for (int t = 0; t < seqLen; t++)
                positionIds[b * seqLen + t] = t;

            float[] batchedLogits;
            float[] batchedKeys;
            float[] batchedValues;

            if (_decoderInitBatched is not null)
            {
                var inputIds = new long[take * seqLen];
                for (int b = 0; b < take; b++)
                {
                    List<int> perItemPrompt = BuildPromptIds((int)audioLengths[b]);
                    for (int t = 0; t < perItemPrompt.Count; t++)
                        inputIds[b * seqLen + t] = perItemPrompt[t];
                }

                using var initResults = _decoderInitBatched.Run(
                [
                    NamedOnnxValue.CreateFromTensor("input_ids",      new DenseTensor<long> (inputIds,     [take, seqLen])),
                    NamedOnnxValue.CreateFromTensor("position_ids",   new DenseTensor<long> (positionIds,  [take, seqLen])),
                    NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures,[take, maxAudioTokens, _hiddenSize])),
                    NamedOnnxValue.CreateFromTensor("audio_lengths",  new DenseTensor<long> (audioLengths, [take])),
                    NamedOnnxValue.CreateFromTensor("audio_offset",   new DenseTensor<long> (new long[] { audioOffset }, [1])),
                ]);
                batchedLogits = ExtractTensor(initResults.First(r => r.Name == "logits").AsTensor<float>());
                batchedKeys   = ExtractTensor(initResults.First(r => r.Name == "present_keys").AsTensor<float>());
                batchedValues = ExtractTensor(initResults.First(r => r.Name == "present_values").AsTensor<float>());
            }
            else
            {
                var inputIds = new long[take * seqLen];
                for (int b = 0; b < take; b++)
                {
                    List<int> perItemPrompt = BuildPromptIds((int)audioLengths[b]);
                    for (int t = 0; t < perItemPrompt.Count; t++)
                        inputIds[b * seqLen + t] = perItemPrompt[t];
                }

                using var initResults = _decoderInitBatched!.Run(
                [
                    NamedOnnxValue.CreateFromTensor("input_ids",      new DenseTensor<long> (inputIds,     [take, seqLen])),
                    NamedOnnxValue.CreateFromTensor("position_ids",   new DenseTensor<long> (positionIds,  [take, seqLen])),
                    NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures,[take, maxAudioTokens, _hiddenSize])),
                    NamedOnnxValue.CreateFromTensor("audio_lengths",  new DenseTensor<long> (audioLengths, [take])),
                    NamedOnnxValue.CreateFromTensor("audio_offset",   new DenseTensor<long> (new long[] { audioOffset }, [1])),
                ]);
                batchedLogits = ExtractTensor(initResults.First(r => r.Name == "logits").AsTensor<float>());
                batchedKeys   = ExtractTensor(initResults.First(r => r.Name == "present_keys").AsTensor<float>());
                batchedValues = ExtractTensor(initResults.First(r => r.Name == "present_values").AsTensor<float>());
            }

            int vocabSize = batchedLogits.Length / (take * seqLen);

            // Extract first token per sequence from batched prefill logits
            var seqTokens    = new List<int>  [take];
            var seqLogprobs  = new List<float>[take];
            var seqDone      = new bool[take];
            var seqPromptLens = new int[take];

            for (int b = 0; b < take; b++)
            {
                int promptLen = BuildPromptIds((int)audioLengths[b]).Count;
                seqPromptLens[b] = promptLen;
                seqTokens[b]   = new List<int>  (maxNewTokens);
                seqLogprobs[b] = new List<float>(maxNewTokens);
                int logitOffset = b * seqLen * vocabSize + (promptLen - 1) * vocabSize;
                int firstToken  = ArgMaxSpan(batchedLogits.AsSpan(logitOffset, vocabSize), out float firstLogprob);
                seqTokens[b].Add(firstToken);
                seqLogprobs[b].Add(firstLogprob);
                seqDone[b]      = IsEos(firstToken);
            }

            // ── Serial autoregressive continuation from batched prefill ──────
            // The encoder + prefill passes batch cleanly and provide the useful
            // throughput win. Continue decoding each sequence independently from
            // its own sliced KV cache so the generated text exactly follows the
            // known-good serial decoder path.
            for (int b = 0; b < take; b++)
            {
                if (!seqDone[b] && seqTokens[b].Count < maxNewTokens)
                {
                    int promptLen = seqPromptLens[b];
                    float[] seqPastKeys = SliceBatchKv(
                        past: batchedKeys,
                        batchSize: take,
                        batchIndex: b,
                        sourceSeqLen: seqLen,
                        targetSeqLen: promptLen);
                    float[] seqPastValues = SliceBatchKv(
                        past: batchedValues,
                        batchSize: take,
                        batchIndex: b,
                        sourceSeqLen: seqLen,
                        targetSeqLen: promptLen);
                    ContinueDecodeFromPrefill(
                        seqTokens[b],
                        seqLogprobs[b],
                        seqTokens[b][^1],
                        seqPastKeys,
                        seqPastValues,
                        promptLen,
                        maxNewTokens);
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

    // Build a [1,1,seqLen,pastSeqLen+seqLen] causal mask (upper-tri -inf; past positions zero).
    private static float[] BuildCausalMask(int seqLen, int pastSeqLen)
    {
        int totalLen = pastSeqLen + seqLen;
        float[] mask = new float[seqLen * totalLen]; // broadcast batch dim from C# side
        for (int q = 0; q < seqLen; q++)
            for (int k = pastSeqLen + q + 1; k < totalLen; k++)
                mask[q * totalLen + k] = float.NegativeInfinity;
        return mask;
    }

    private (float[] inputEmbeds, long[] positionIds, float[] attentionMask, int[] seqLengths) BuildUnifiedBatchPrefillInputs(
        float[] audioFeatures,
        long[] audioLengths,
        int batchSize,
        int maxAudioTokens)
    {
        var prompts = new List<int>[batchSize];
        var seqLengths = new int[batchSize];
        int maxSeqLen = 0;

        for (int b = 0; b < batchSize; b++)
        {
            prompts[b] = BuildPromptIds((int)audioLengths[b]);
            seqLengths[b] = prompts[b].Count;
            maxSeqLen = Math.Max(maxSeqLen, seqLengths[b]);
        }

        float[] inputEmbeds = new float[batchSize * maxSeqLen * _hiddenSize];
        long[] positionIds = new long[batchSize * maxSeqLen];
        var tokenEmbed = new float[_hiddenSize];

        for (int b = 0; b < batchSize; b++)
        {
            List<int> prompt = prompts[b];
            int audioOffset = GetAudioPadStart(prompt);
            for (int t = 0; t < prompt.Count; t++)
            {
                ReadTokenEmbedding(prompt[t], tokenEmbed);
                Array.Copy(tokenEmbed, 0, inputEmbeds, (b * maxSeqLen + t) * _hiddenSize, _hiddenSize);
                positionIds[b * maxSeqLen + t] = t;
            }

            for (int t = 0; t < (int)audioLengths[b]; t++)
            {
                Array.Copy(
                    audioFeatures,
                    (b * maxAudioTokens + t) * _hiddenSize,
                    inputEmbeds,
                    (b * maxSeqLen + audioOffset + t) * _hiddenSize,
                    _hiddenSize);
            }
        }

        float[] attentionMask = BuildBatchPrefillAttentionMask(seqLengths, maxSeqLen);
        return (inputEmbeds, positionIds, attentionMask, seqLengths);
    }

    private static (float[] inputEmbeds, long[] positionIds, float[] attentionMask, int[] seqLengths) BuildUnifiedBatchPrefillInputsForBenchmark(
        string modelPath,
        float[] audioFeatures,
        long[] audioLengths,
        int batchSize,
        int maxAudioTokens,
        int hiddenSize)
    {
        using var mmf = MemoryMappedFile.CreateFromFile(Path.Combine(modelPath, EmbedTokensFile), FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        var prompts = new List<int>[batchSize];
        var seqLengths = new int[batchSize];
        int maxSeqLen = 0;

        for (int b = 0; b < batchSize; b++)
        {
            prompts[b] = BuildPromptIds((int)audioLengths[b]);
            seqLengths[b] = prompts[b].Count;
            maxSeqLen = Math.Max(maxSeqLen, seqLengths[b]);
        }

        float[] inputEmbeds = new float[batchSize * maxSeqLen * hiddenSize];
        long[] positionIds = new long[batchSize * maxSeqLen];
        var tokenEmbed = new float[hiddenSize];

        for (int b = 0; b < batchSize; b++)
        {
            List<int> prompt = prompts[b];
            int audioOffset = GetAudioPadStart(prompt);
            for (int t = 0; t < prompt.Count; t++)
            {
                long byteOffset = (long)prompt[t] * hiddenSize * sizeof(float);
                accessor.ReadArray(byteOffset, tokenEmbed, 0, hiddenSize);
                Array.Copy(tokenEmbed, 0, inputEmbeds, (b * maxSeqLen + t) * hiddenSize, hiddenSize);
                positionIds[b * maxSeqLen + t] = t;
            }

            for (int t = 0; t < (int)audioLengths[b]; t++)
            {
                Array.Copy(
                    audioFeatures,
                    (b * maxAudioTokens + t) * hiddenSize,
                    inputEmbeds,
                    (b * maxSeqLen + audioOffset + t) * hiddenSize,
                    hiddenSize);
            }
        }

        float[] attentionMask = BuildBatchPrefillAttentionMask(seqLengths, maxSeqLen);
        return (inputEmbeds, positionIds, attentionMask, seqLengths);
    }

    private static float[] BuildBatchPrefillAttentionMask(int[] seqLengths, int maxSeqLen)
    {
        float[] mask = new float[seqLengths.Length * maxSeqLen * maxSeqLen];
        Array.Fill(mask, float.MinValue);

        for (int b = 0; b < seqLengths.Length; b++)
        {
            int batchOffset = b * maxSeqLen * maxSeqLen;
            int validLen = seqLengths[b];
            for (int q = 0; q < validLen; q++)
                for (int k = 0; k <= q; k++)
                    mask[batchOffset + q * maxSeqLen + k] = 0f;

            for (int q = validLen; q < maxSeqLen; q++)
                mask[batchOffset + q * maxSeqLen] = 0f;
        }

        return mask;
    }

    private static float[] BuildBatchStepAttentionMask(int[] pastLengths, int maxPastLen)
    {
        float[] mask = new float[pastLengths.Length * (maxPastLen + 1)];
        Array.Fill(mask, float.MinValue);

        for (int b = 0; b < pastLengths.Length; b++)
        {
            int offset = b * (maxPastLen + 1);
            for (int k = 0; k < pastLengths[b]; k++)
                mask[offset + k] = 0f;
            mask[offset + maxPastLen] = 0f;
        }

        return mask;
    }

    private float[] CompactPrefillKv(float[] present, int[] seqLengths, int batchSize)
    {
        int maxSeqLen = seqLengths.Max();
        float[] compact = new float[_nLayers * batchSize * _nKvHeads * maxSeqLen * _headDim];

        for (int layer = 0; layer < _nLayers; layer++)
        for (int b = 0; b < batchSize; b++)
        for (int head = 0; head < _nKvHeads; head++)
        {
            int validLen = seqLengths[b];
            int sourceBase = (((layer * batchSize + b) * _nKvHeads + head) * maxSeqLen) * _headDim;
            int destBase = sourceBase;
            Array.Copy(present, sourceBase, compact, destBase, validLen * _headDim);
        }

        return compact;
    }

    private void CompactStepKv(
        float[] presentKeys,
        float[] presentValues,
        int[] pastLengths,
        bool[] activeMask,
        int batchSize,
        out float[] nextKeys,
        out float[] nextValues,
        out int[] nextLengths)
    {
        int appendIndex = pastLengths.Max();
        nextLengths = (int[])pastLengths.Clone();
        for (int b = 0; b < batchSize; b++)
            if (activeMask[b])
                nextLengths[b]++;

        int maxNextLen = nextLengths.Max();
        nextKeys = new float[_nLayers * batchSize * _nKvHeads * maxNextLen * _headDim];
        nextValues = new float[_nLayers * batchSize * _nKvHeads * maxNextLen * _headDim];

        for (int layer = 0; layer < _nLayers; layer++)
        for (int b = 0; b < batchSize; b++)
        for (int head = 0; head < _nKvHeads; head++)
        {
            int sourceStride = appendIndex + 1;
            int sourceBase = (((layer * batchSize + b) * _nKvHeads + head) * sourceStride) * _headDim;
            int destBase = (((layer * batchSize + b) * _nKvHeads + head) * maxNextLen) * _headDim;
            int validPastLen = pastLengths[b];

            if (validPastLen > 0)
            {
                Array.Copy(presentKeys, sourceBase, nextKeys, destBase, validPastLen * _headDim);
                Array.Copy(presentValues, sourceBase, nextValues, destBase, validPastLen * _headDim);
            }

            if (activeMask[b])
            {
                Array.Copy(
                    presentKeys,
                    sourceBase + appendIndex * _headDim,
                    nextKeys,
                    destBase + validPastLen * _headDim,
                    _headDim);
                Array.Copy(
                    presentValues,
                    sourceBase + appendIndex * _headDim,
                    nextValues,
                    destBase + validPastLen * _headDim,
                    _headDim);
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
        => _encoderBatched is not null && (_decoderInitBatched is not null || _decoder is not null);

    public QwenExperimentalBatchBenchmark BenchmarkExperimentalBatching(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio)
        => BenchmarkExperimentalBatching(
            _modelPath,
            _executionProvider,
            segs,
            audio,
            _optimizationLevel);

    public static QwenExperimentalBatchBenchmark BenchmarkExperimentalBatching(
        string modelPath,
        ExecutionProvider ep,
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    {
        bool hasEncoderBatched = File.Exists(Path.Combine(modelPath, EncoderBatchedFile));
        bool hasUnifiedDecoder = File.Exists(Path.Combine(modelPath, DecoderFile));
        bool hasDecoderInitBatched = File.Exists(Path.Combine(modelPath, DecoderInitBatchedFile));

        if (!hasEncoderBatched || (!hasUnifiedDecoder && !hasDecoderInitBatched))
            throw new FileNotFoundException("Experimental batching artifacts are missing.");

        string configJson = File.ReadAllText(Path.Combine(modelPath, ConfigFile));
        using var configDoc = JsonDocument.Parse(configJson);
        var decoderConfig = configDoc.RootElement.GetProperty("decoder");
        int hiddenSize = decoderConfig.GetProperty("hidden_size").GetInt32();
        int nLayers = decoderConfig.GetProperty("num_layers").GetInt32();
        int nKvHeads = decoderConfig.GetProperty("num_key_value_heads").GetInt32();
        int headDim = decoderConfig.GetProperty("head_dim").GetInt32();

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
            var encoderOpts = MakeSessionOptions(ep, optimizationLevel, out _);
            var decoderOpts = MakeSessionOptions(ep, optimizationLevel, out _);
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
            var encoderOpts = MakeSessionOptions(ep, optimizationLevel, out _);
            var decoderOpts = MakeSessionOptions(ep, optimizationLevel, out _);
            using var encoderBatched = new InferenceSession(Path.Combine(modelPath, EncoderBatchedFile), encoderOpts);
            using var decoder = hasUnifiedDecoder
                ? new InferenceSession(Path.Combine(modelPath, DecoderFile), decoderOpts)
                : null;
            using var decoderInitBatched = !hasUnifiedDecoder && hasDecoderInitBatched
                ? new InferenceSession(Path.Combine(modelPath, DecoderInitBatchedFile), decoderOpts)
                : null;
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
                    for (int melBin = 0; melBin < NMels; melBin++)
                    {
                        Array.Copy(
                            segment.Mel,
                            melBin * segment.MelFrames,
                            mel,
                            (batchIndex * NMels + melBin) * maxMelFrames,
                            segment.MelFrames);
                    }
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
                float[] audioFeatures = new float[take * maxAudioTokens * hiddenSize];
                for (int batchIndex = 0; batchIndex < take; batchIndex++)
                {
                    for (int tokenIndex = 0; tokenIndex < (int)audioLengthsTensor[batchIndex]; tokenIndex++)
                        for (int hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex++)
                            audioFeatures[(batchIndex * maxAudioTokens + tokenIndex) * hiddenSize + hiddenIndex]
                                = audioFeaturesTensor[batchIndex, tokenIndex, hiddenIndex];
                }

                long[] audioLengths = [.. audioLengthsTensor];

                sw.Restart();
                if (decoder is not null)
                {
                    var (inputEmbeds, positionIds, attentionMask, seqLengths) =
                        BuildUnifiedBatchPrefillInputsForBenchmark(modelPath, audioFeatures, audioLengths, take, maxAudioTokens, hiddenSize);
                    float[] emptyKv = [];
                    using var decoderResults = decoder.Run(
                    [
                        NamedOnnxValue.CreateFromTensor("input_embeds", new DenseTensor<float>(inputEmbeds, [take, seqLengths.Max(), hiddenSize])),
                        NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(positionIds, [take, seqLengths.Max()])),
                        NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<float>(attentionMask, [take, 1, seqLengths.Max(), seqLengths.Max()])),
                        NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(emptyKv, [nLayers, take, nKvHeads, 0, headDim])),
                        NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(emptyKv, [nLayers, take, nKvHeads, 0, headDim])),
                    ]);
                }
                else
                {
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

                    long[] audioOffsetTensor = [audioOffset];
                    using var decoderResults = decoderInitBatched!.Run(
                    [
                        NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, [take, seqLen])),
                        NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(positionIds, [take, seqLen])),
                        NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures, [take, maxAudioTokens, hiddenSize])),
                        NamedOnnxValue.CreateFromTensor("audio_lengths", new DenseTensor<long>(audioLengths, [take])),
                        NamedOnnxValue.CreateFromTensor("audio_offset", new DenseTensor<long>(audioOffsetTensor, [1])),
                    ]);
                }
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

            using var stepOutputs = _decoderStep!.Run(
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

    private (List<int> rawTokens, List<float> rawLogprobs) DecodeOnCpuUnified(
        IReadOnlyList<int> promptIds,
        int audioOffset,
        float[] audioFeatures,
        int audioTokenCount,
        int maxNewTokens)
    {
        int seqLen = promptIds.Count;

        // Build input_embeds: embedding lookup for all prompt tokens, then scatter audio features.
        float[] inputEmbeds = new float[seqLen * _hiddenSize];
        var tmpEmbed = new float[_hiddenSize];
        for (int t = 0; t < seqLen; t++)
        {
            ReadTokenEmbedding(promptIds[t], tmpEmbed);
            Array.Copy(tmpEmbed, 0, inputEmbeds, t * _hiddenSize, _hiddenSize);
        }
        Array.Copy(audioFeatures, 0, inputEmbeds, audioOffset * _hiddenSize, audioTokenCount * _hiddenSize);

        long[] positionIds = Enumerable.Range(0, seqLen).Select(i => (long)i).ToArray();
        float[] emptyKv = [];

        // Prefill causal mask: upper-triangular -inf [1, 1, seqLen, seqLen]
        float[] prefillMask = BuildCausalMask(seqLen, 0);

        using var initOutputs = _decoder!.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_embeds",   new DenseTensor<float>(inputEmbeds,  [1, seqLen, _hiddenSize])),
            NamedOnnxValue.CreateFromTensor("position_ids",   new DenseTensor<long>(positionIds,   [1, seqLen])),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<float>(prefillMask,  [1, 1, seqLen, seqLen])),
            NamedOnnxValue.CreateFromTensor("past_keys",      new DenseTensor<float>(emptyKv,      [_nLayers, 1, _nKvHeads, 0, _headDim])),
            NamedOnnxValue.CreateFromTensor("past_values",    new DenseTensor<float>(emptyKv,      [_nLayers, 1, _nKvHeads, 0, _headDim])),
        ]);

        float[] logits     = ExtractTensor(initOutputs.First(r => r.Name == "logits").AsTensor<float>());
        float[] pastKeys   = ExtractTensor(initOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
        float[] pastValues = ExtractTensor(initOutputs.First(r => r.Name == "present_values").AsTensor<float>());
        int pastSeqLen     = seqLen;

        var rawTokens   = new List<int>(maxNewTokens);
        var rawLogprobs = new List<float>(maxNewTokens);
        long[] stepPos  = [0L];

        int nextToken = ArgMaxLastLogits(logits, seqLen, out float nextLogprob);
        rawTokens.Add(nextToken);
        rawLogprobs.Add(nextLogprob);

        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tmpEmbed);
            stepPos[0] = seqLen + rawTokens.Count - 1L;
            // Step mask: all-zeros [1, 1, 1, pastSeqLen+1] — single query attends to all past
            float[] stepMask = new float[pastSeqLen + 1];

            using var stepOutputs = _decoder.Run(
            [
                NamedOnnxValue.CreateFromTensor("input_embeds",   new DenseTensor<float>(tmpEmbed,  [1, 1, _hiddenSize])),
                NamedOnnxValue.CreateFromTensor("position_ids",   new DenseTensor<long>(stepPos,    [1, 1])),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<float>(stepMask,  [1, 1, 1, pastSeqLen + 1])),
                NamedOnnxValue.CreateFromTensor("past_keys",      new DenseTensor<float>(pastKeys,  [_nLayers, 1, _nKvHeads, pastSeqLen, _headDim])),
                NamedOnnxValue.CreateFromTensor("past_values",    new DenseTensor<float>(pastValues,[_nLayers, 1, _nKvHeads, pastSeqLen, _headDim])),
            ]);

            logits     = ExtractTensor(stepOutputs.First(r => r.Name == "logits").AsTensor<float>());
            pastKeys   = ExtractTensor(stepOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
            pastValues = ExtractTensor(stepOutputs.First(r => r.Name == "present_values").AsTensor<float>());
            pastSeqLen++;

            nextToken = ArgMaxLastLogits(logits, 1, out nextLogprob);
            rawTokens.Add(nextToken);
            rawLogprobs.Add(nextLogprob);
        }

        return (rawTokens, rawLogprobs);
    }

    private void ContinueDecodeFromPrefill(
        List<int> rawTokens,
        List<float> rawLogprobs,
        int nextToken,
        float[] pastKeys,
        float[] pastValues,
        int promptSeqLen,
        int maxNewTokens)
    {
        if (rawTokens.Count == 0 || rawTokens.Count >= maxNewTokens || IsEos(nextToken))
            return;

        int pastSeqLen = promptSeqLen;
        var tokenEmbed = new float[_hiddenSize];
        long[] stepPos = [0L];

        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tokenEmbed);
            stepPos[0] = promptSeqLen + rawTokens.Count - 1L;

            if (_decoder is not null)
            {
                float[] stepMask = new float[pastSeqLen + 1];
                using var stepOutputs = _decoder.Run(
                [
                    NamedOnnxValue.CreateFromTensor("input_embeds",   new DenseTensor<float>(tokenEmbed,  [1, 1, _hiddenSize])),
                    NamedOnnxValue.CreateFromTensor("position_ids",   new DenseTensor<long>(stepPos,     [1, 1])),
                    NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<float>(stepMask,   [1, 1, 1, pastSeqLen + 1])),
                    NamedOnnxValue.CreateFromTensor("past_keys",      new DenseTensor<float>(pastKeys,   [_nLayers, 1, _nKvHeads, pastSeqLen, _headDim])),
                    NamedOnnxValue.CreateFromTensor("past_values",    new DenseTensor<float>(pastValues, [_nLayers, 1, _nKvHeads, pastSeqLen, _headDim])),
                ]);

                float[] logits = ExtractTensor(stepOutputs.First(r => r.Name == "logits").AsTensor<float>());
                pastKeys = ExtractTensor(stepOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
                pastValues = ExtractTensor(stepOutputs.First(r => r.Name == "present_values").AsTensor<float>());
                pastSeqLen++;

                nextToken = ArgMaxLastLogits(logits, 1, out float nextLogprob);
                rawTokens.Add(nextToken);
                rawLogprobs.Add(nextLogprob);
            }
            else
            {
                using var stepOutputs = _decoderStep!.Run(
                [
                    NamedOnnxValue.CreateFromTensor("input_embeds", new DenseTensor<float>(tokenEmbed, [1, 1, _hiddenSize])),
                    NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(stepPos, [1, 1])),
                    NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(pastKeys, [_nLayers, 1, _nKvHeads, pastSeqLen, _headDim])),
                    NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(pastValues, [_nLayers, 1, _nKvHeads, pastSeqLen, _headDim])),
                ]);

                float[] logits = ExtractTensor(stepOutputs.First(r => r.Name == "logits").AsTensor<float>());
                pastKeys = ExtractTensor(stepOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
                pastValues = ExtractTensor(stepOutputs.First(r => r.Name == "present_values").AsTensor<float>());
                pastSeqLen++;

                nextToken = ArgMaxLastLogits(logits, 1, out float nextLogprob);
                rawTokens.Add(nextToken);
                rawLogprobs.Add(nextLogprob);
            }
        }
    }

    private float[] SliceBatchKv(float[] past, int batchSize, int batchIndex, int sourceSeqLen, int targetSeqLen)
    {
        int sourcePerSequenceSize = _nKvHeads * sourceSeqLen * _headDim;
        int targetPerSequenceSize = _nKvHeads * targetSeqLen * _headDim;
        float[] sliced = new float[_nLayers * targetPerSequenceSize];
        int destinationOffset = 0;

        for (int layer = 0; layer < _nLayers; layer++)
        {
            int layerOffset = layer * batchSize * sourcePerSequenceSize;
            int sourceOffset = layerOffset + batchIndex * sourcePerSequenceSize;
            Array.Copy(past, sourceOffset, sliced, destinationOffset, targetPerSequenceSize);
            destinationOffset += targetPerSequenceSize;
        }

        return sliced;
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
            using var stepBinding = _decoderStep!.CreateIoBinding();
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
        _decoder?.Dispose();
        _decoderStep?.Dispose();
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
