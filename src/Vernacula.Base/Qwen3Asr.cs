using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Diagnostics;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
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
    // Maps ISO 639-1 codes → Qwen3-ASR language names (as they appear in model output).
    private static readonly Dictionary<string, string> IsoToLanguageName = new(StringComparer.OrdinalIgnoreCase)
    {
        ["af"] = "Afrikaans",  ["ar"] = "Arabic",     ["hy"] = "Armenian",   ["az"] = "Azerbaijani",
        ["be"] = "Belarusian", ["bs"] = "Bosnian",    ["bg"] = "Bulgarian",  ["ca"] = "Catalan",
        ["zh"] = "Chinese",    ["hr"] = "Croatian",   ["cs"] = "Czech",      ["da"] = "Danish",
        ["nl"] = "Dutch",      ["en"] = "English",    ["et"] = "Estonian",   ["fi"] = "Finnish",
        ["fr"] = "French",     ["gl"] = "Galician",   ["de"] = "German",     ["el"] = "Greek",
        ["he"] = "Hebrew",     ["hi"] = "Hindi",      ["hu"] = "Hungarian",  ["is"] = "Icelandic",
        ["id"] = "Indonesian", ["it"] = "Italian",    ["ja"] = "Japanese",   ["kn"] = "Kannada",
        ["kk"] = "Kazakh",     ["ko"] = "Korean",     ["lv"] = "Latvian",    ["lt"] = "Lithuanian",
        ["mk"] = "Macedonian", ["ms"] = "Malay",      ["mr"] = "Marathi",    ["ne"] = "Nepali",
        ["no"] = "Norwegian",  ["fa"] = "Persian",    ["pl"] = "Polish",     ["pt"] = "Portuguese",
        ["ro"] = "Romanian",   ["ru"] = "Russian",    ["sr"] = "Serbian",    ["sk"] = "Slovak",
        ["sl"] = "Slovenian",  ["es"] = "Spanish",    ["sw"] = "Swahili",    ["sv"] = "Swedish",
        ["tl"] = "Tagalog",    ["ta"] = "Tamil",      ["th"] = "Thai",       ["tr"] = "Turkish",
        ["uk"] = "Ukrainian",  ["ur"] = "Urdu",       ["vi"] = "Vietnamese", ["cy"] = "Welsh",
    };

    // Token ID sequences for "language <Name> " prefix, pre-computed from the Qwen3 BPE tokenizer.
    // First token is always 11528 ("language"); last is always 220 (" ").
    private static readonly Dictionary<string, int[]> LanguagePrefixTokens =
        new(StringComparer.OrdinalIgnoreCase)
    {
        ["Afrikaans"]  = [11528, 12907, 40454, 596,   220],
        ["Arabic"]     = [11528, 34117,               220],
        ["Armenian"]   = [11528, 66742,               220],
        ["Azerbaijani"]= [11528, 64323, 73,   5559,   220],
        ["Belarusian"] = [11528, 69506, 1103,         220],
        ["Bosnian"]    = [11528, 27971, 77,   1103,   220],
        ["Bulgarian"]  = [11528, 88624,               220],
        ["Catalan"]    = [11528, 80844,               220],
        ["Chinese"]    = [11528,  8453,               220],
        ["Croatian"]   = [11528, 99070,               220],
        ["Czech"]      = [11528, 33150,               220],
        ["Danish"]     = [11528, 43680,               220],
        ["Dutch"]      = [11528, 23234,               220],
        ["English"]    = [11528,  6364,               220],
        ["Estonian"]   = [11528, 53323, 1103,         220],
        ["Finnish"]    = [11528, 57853,               220],
        ["French"]     = [11528,  8585,               220],
        ["Galician"]   = [11528, 10620, 12452,        220],
        ["German"]     = [11528,  5938,               220],
        ["Greek"]      = [11528, 17860,               220],
        ["Hebrew"]     = [11528, 36266,               220],
        ["Hindi"]      = [11528, 43980,               220],
        ["Hungarian"]  = [11528, 56769,               220],
        ["Icelandic"]  = [11528, 99148,               220],
        ["Indonesian"] = [11528, 58829,               220],
        ["Italian"]    = [11528, 14811,               220],
        ["Japanese"]   = [11528, 10769,               220],
        ["Kannada"]    = [11528, 77211,  2584,        220],
        ["Kazakh"]     = [11528, 34974, 21758,        220],
        ["Korean"]     = [11528, 16134,               220],
        ["Latvian"]    = [11528,  9926,    85,  1103, 220],
        ["Lithuanian"] = [11528, 40578, 10386,  1103, 220],
        ["Macedonian"] = [11528, 56452, 75491,        220],
        ["Malay"]      = [11528, 79140,               220],
        ["Marathi"]    = [11528,  2876, 66531,        220],
        ["Nepali"]     = [11528, 36569,  7956,        220],
        ["Norwegian"]  = [11528, 44621,               220],
        ["Persian"]    = [11528, 49861,               220],
        ["Polish"]     = [11528, 31984,               220],
        ["Portuguese"] = [11528, 42188,               220],
        ["Romanian"]   = [11528, 73597,               220],
        ["Russian"]    = [11528,  8522,               220],
        ["Serbian"]    = [11528, 87164,               220],
        ["Slovak"]     = [11528, 61264,               220],
        ["Slovenian"]  = [11528, 59395,  1103,        220],
        ["Spanish"]    = [11528, 15154,               220],
        ["Swahili"]    = [11528,  4492,  1466,  3921, 220],
        ["Swedish"]    = [11528, 30109,               220],
        ["Tagalog"]    = [11528, 12353, 30951,        220],
        ["Tamil"]      = [11528, 43783,               220],
        ["Thai"]       = [11528, 26392,               220],
        ["Turkish"]    = [11528, 23734,               220],
        ["Ukrainian"]  = [11528, 33625,               220],
        ["Urdu"]       = [11528, 93335,               220],
        ["Vietnamese"] = [11528, 48477,               220],
        ["Welsh"]      = [11528, 45781,               220],
    };

    // Resolves an ISO 639-1 code or language name to a forced-prefix token sequence.
    // Returns [] when forceLanguage is null; throws ArgumentException for unrecognised codes.
    private static int[] ResolveForcedPrefix(string? forceLanguage)
    {
        if (forceLanguage is null) return [];
        string name = IsoToLanguageName.TryGetValue(forceLanguage, out string? mapped)
            ? mapped : forceLanguage;
        if (LanguagePrefixTokens.TryGetValue(name, out int[]? tokens))
        {
            // Omit the trailing 220 ("Ġ" space). The model naturally chooses whether to emit
            // a standalone space separator or jump straight to a Ġ-prefixed content token —
            // forcing a fixed separator gives KV states the model wasn't trained on.
            return tokens.Length > 0 && tokens[^1] == 220 ? tokens[..^1] : tokens;
        }
        throw new ArgumentException(
            $"Language '{forceLanguage}' not found. Use an ISO 639-1 code (e.g. 'en') or a language name (e.g. 'English').");
    }

    private static string? ResolveForcedLanguageName(string? forceLanguage)
    {
        if (forceLanguage is null) return null;
        return IsoToLanguageName.TryGetValue(forceLanguage, out string? name) ? name : forceLanguage;
    }

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

        var encoderOpts = OrtSessionBuilder.Create(ep, optimizationLevel, enableProfiling: false, out bool encoderUsesCuda);
        var decoderOpts = OrtSessionBuilder.Create(ep, optimizationLevel, enableProfiling: false, out bool decoderUsesCuda);

        bool hasUnified       = File.Exists(Path.Combine(modelPath, DecoderFile));
        bool hasBatchedEncoder = File.Exists(Path.Combine(modelPath, EncoderBatchedFile));
        bool hasBatchedInit   = File.Exists(Path.Combine(modelPath, DecoderInitBatchedFile));
        _preferBatched = preferBatched && hasBatchedEncoder && (hasBatchedInit || hasUnified);

        if (hasUnified)
        {
            _decoderInit = null!;

            if (_preferBatched)
            {
                // Unified batched path: encoder_batched + decoder.onnx only.
                // Do not load decoder_init_batched here; we want the runtime
                // contract to be unambiguously single-decoder. The batched
                // path drives its own KV handling via RecognizeUnifiedContinuousBatched
                // (CPU-side compaction) and does not use the serial IOBinding path.
                _decoder = new InferenceSession(Path.Combine(modelPath, DecoderFile), decoderOpts);
                _encoder        = null!;
                _encoderBatched = new InferenceSession(Path.Combine(modelPath, EncoderBatchedFile), encoderOpts);
                _decoderInitBatched = null;
                _decoderStep = null;
                _useCudaIoBinding   = false;
            }
            else
            {
                _decoder        = new InferenceSession(Path.Combine(modelPath, DecoderFile), decoderOpts);
                _encoder        = new InferenceSession(Path.Combine(modelPath, EncoderFile), encoderOpts);
                _encoderBatched = null;
                _decoderInitBatched = null;
                _decoderStep = null;
                // Serial unified path gets GPU-resident KV via DecodeOnGpuUnified when
                // both sessions actually landed on CUDA; falls back to DecodeOnCpuUnified
                // otherwise.
                _useCudaIoBinding = encoderUsesCuda && decoderUsesCuda;
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

    private List<QwenEncodedItem> EncodeBatchedItems(
        IReadOnlyList<QwenBatchedItem> validItems,
        long freeGpuMemoryMb)
    {
        var encodedItems = new List<QwenEncodedItem>(validItems.Count);
        int index = 0;

        while (index < validItems.Count)
        {
            var remainingDurations = validItems
                .Skip(index)
                .Select(it => it.DurationSamples / (double)SampleRate)
                .ToList();
            var plan = ComputeExperimentalBatchingPlan(remainingDurations, freeGpuMemoryMb);
            int take = Math.Min(plan.BatchCount, validItems.Count - index);
            var batch = validItems.Skip(index).Take(take).ToList();
            index += take;

            int maxMelFrames = batch.Max(it => it.MelFrames);
            var melBatch = new float[take * NMels * maxMelFrames];
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

            using var encoderResults = _encoderBatched!.Run(
            [
                NamedOnnxValue.CreateFromTensor("mel",           new DenseTensor<float>(melBatch, [take, NMels, maxMelFrames])),
                NamedOnnxValue.CreateFromTensor("input_lengths", new DenseTensor<long>(inputLengths, [take])),
            ]);

            var audioFeaturesTensor = encoderResults.First(r => r.Name == "audio_features").AsTensor<float>();
            var audioLengthsTensor = encoderResults.First(r => r.Name == "audio_feature_lengths").AsTensor<long>();

            for (int b = 0; b < take; b++)
            {
                int audioTokenCount = (int)audioLengthsTensor[b];
                float[] audioFeatures = new float[audioTokenCount * _hiddenSize];
                for (int t = 0; t < audioTokenCount; t++)
                for (int h = 0; h < _hiddenSize; h++)
                    audioFeatures[t * _hiddenSize + h] = audioFeaturesTensor[b, t, h];

                encodedItems.Add(new QwenEncodedItem(
                    batch[b].SegId,
                    batch[b].DurationSamples,
                    audioFeatures,
                    audioTokenCount));
            }
        }

        return encodedItems;
    }

    private IEnumerable<QwenRecognitionResult> RecognizeUnifiedContinuousBatched(
        IReadOnlyList<QwenBatchedItem> validItems,
        long freeGpuMemoryMb,
        int maxNewTokens,
        int[] forcedPrefix,
        string? forcedLanguageName = null)
    {
        forcedPrefix ??= [];
        var encodedItems = EncodeBatchedItems(validItems, freeGpuMemoryMb);
        if (encodedItems.Count == 0)
            yield break;

        // Capacity is bounded by the LARGER of two peak allocations:
        //   prefill:  [groupSize, maxPromptLen, vocabSize] logits  (lm_head output, temporary but large)
        //   decode KV: [nLayers, groupSize, nKvHeads, maxKvSeqLen, headDim] × 2 (persistent throughout decode)
        // Budget 25% of free VRAM; remainder for model weights, activations, ORT arena overhead.
        // The forced prefix is appended to each item's prompt, so add its length to the base prompt length.
        int  maxPromptSeqLen       = encodedItems.Max(it => BuildPromptIds(it.AudioTokenCount).Count)
                                     + forcedPrefix.Length;
        int  maxKvSeqLen           = maxPromptSeqLen + maxNewTokens;
        long prefillLogitsPerSlot  = (long)maxPromptSeqLen * _vocabSize * 4;
        long decodeKvPerSlot       = (long)_nLayers * _nKvHeads * maxKvSeqLen * _headDim * 4 * 2;
        long peakBytesPerSlot      = Math.Max(prefillLogitsPerSlot, decodeKvPerSlot);
        int  decodeCap             = (int)Math.Max(1, freeGpuMemoryMb * 1024L * 1024 / 4 / peakBytesPerSlot);
        int  slotCapacity          = Math.Min(encodedItems.Count, decodeCap);
        Console.Error.WriteLine($"[Qwen3Asr] static-batch decode: {encodedItems.Count} segments, slotCapacity={slotCapacity}");

        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts     = new RunOptions();

        float[] emptyKv    = [];
        float[] tokenEmbed = new float[_hiddenSize];

        int pendingIndex = 0;
        while (pendingIndex < encodedItems.Count)
        {
            int groupSize = Math.Min(slotCapacity, encodedItems.Count - pendingIndex);

            // --- Build batched prefill inputs for this group ---
            int maxAudioTokens = 0;
            for (int i = 0; i < groupSize; i++)
                maxAudioTokens = Math.Max(maxAudioTokens, encodedItems[pendingIndex + i].AudioTokenCount);

            float[] audioFeatures = new float[groupSize * maxAudioTokens * _hiddenSize];
            long[]  audioLengths  = new long[groupSize];
            for (int i = 0; i < groupSize; i++)
            {
                var item = encodedItems[pendingIndex + i];
                audioLengths[i] = item.AudioTokenCount;
                for (int t = 0; t < item.AudioTokenCount; t++)
                    Array.Copy(item.AudioFeatures, t * _hiddenSize,
                               audioFeatures, (i * maxAudioTokens + t) * _hiddenSize, _hiddenSize);
            }

            var (inputEmbeds, positionIds, prefillMask, seqLengths) =
                BuildUnifiedBatchPrefillInputs(audioFeatures, audioLengths, groupSize, maxAudioTokens, forcedPrefix);
            int maxPromptLen = seqLengths.Max();

            // --- Prefill: use Run() — avoids GetTensorDataAsSpan failing on _decoder with mixed CPU/CUDA IO binding ---
            using var prefillResults = _decoder!.Run([
                NamedOnnxValue.CreateFromTensor("input_embeds",
                    new DenseTensor<float>(inputEmbeds,  new[] { groupSize, maxPromptLen, _hiddenSize })),
                NamedOnnxValue.CreateFromTensor("position_ids",
                    new DenseTensor<long>(positionIds,   new[] { groupSize, maxPromptLen })),
                NamedOnnxValue.CreateFromTensor("attention_mask",
                    new DenseTensor<float>(prefillMask,  new[] { groupSize, 1, maxPromptLen, maxPromptLen })),
                NamedOnnxValue.CreateFromTensor("past_keys",
                    new DenseTensor<float>(Memory<float>.Empty, new[] { _nLayers, groupSize, _nKvHeads, 0, _headDim })),
                NamedOnnxValue.CreateFromTensor("past_values",
                    new DenseTensor<float>(Memory<float>.Empty, new[] { _nLayers, groupSize, _nKvHeads, 0, _headDim })),
            ]);

            // Extract last-position logits per slot (groupSize×vocabSize, ~8MB vs ~3.5GB full tensor)
            float[] lastPosLogits = new float[groupSize * _vocabSize];
            var prefillLogitTensor = prefillResults.First(r => r.Name == "logits").AsTensor<float>();
            if (prefillLogitTensor is DenseTensor<float> densePrefillLogits)
            {
                var span = densePrefillLogits.Buffer.Span;
                for (int b = 0; b < groupSize; b++)
                {
                    int srcOffset = (b * maxPromptLen + seqLengths[b] - 1) * _vocabSize;
                    span.Slice(srcOffset, _vocabSize).CopyTo(lastPosLogits.AsSpan(b * _vocabSize));
                }
            }

            // Extract KV for step-1 H2D bootstrap (one-time per group)
            float[] prefillKeysBuf   = ExtractTensor(prefillResults.First(r => r.Name == "present_keys").AsTensor<float>());
            float[] prefillValuesBuf = ExtractTensor(prefillResults.First(r => r.Name == "present_values").AsTensor<float>());

            // --- Init per-slot state from prefill logits ---
            int[]  nextTokens  = new int[groupSize];
            bool[] done        = new bool[groupSize];
            var    rawTokens   = new List<int>[groupSize];
            var    rawLogprobs = new List<float>[groupSize];

            for (int b = 0; b < groupSize; b++)
            {
                rawTokens[b]   = new List<int>(maxNewTokens);
                rawLogprobs[b] = new List<float>(maxNewTokens);
                int   firstToken  = ArgMaxSpan(lastPosLogits.AsSpan(b * _vocabSize, _vocabSize), out float firstLogprob);
                nextTokens[b] = firstToken;
                rawTokens[b].Add(firstToken);
                rawLogprobs[b].Add(firstLogprob);
                done[b] = IsEos(firstToken) || maxNewTokens <= 1;
            }

            // --- Decode: GPU KV chains step-to-step; only logits (~vocabSize×groupSize×4B) cross PCIe ---
            // Pre-allocate decode logits buffer; bind as output so ORT writes directly — no GetTensorDataAsSpan.
            float[] decodeLogBuf = new float[groupSize * _vocabSize];
            float[] stepEmbeds   = new float[groupSize * _hiddenSize];
            long[]  stepPos      = new long[groupSize];
            float[] stepMask     = new float[groupSize * (maxPromptLen + maxNewTokens)];

            bool firstStep = true;
            IDisposableReadOnlyCollection<OrtValue>? prevKv = null;
            int stepCount = 1;
            while (!Array.TrueForAll(done, d => d))
            {
                // totalPastLen = tokens already in KV: maxPromptLen positions from prefill + stepCount decode tokens
                int totalPastLen = maxPromptLen + stepCount;

                for (int b = 0; b < groupSize; b++)
                {
                    ReadTokenEmbedding(nextTokens[b], tokenEmbed);
                    Array.Copy(tokenEmbed, 0, stepEmbeds, b * _hiddenSize, _hiddenSize);
                    // actual sequence position for each slot (differs when prompt lengths differ)
                    stepPos[b] = seqLengths[b] + stepCount - 1;
                }

                // Attend to valid prefill [0, seqLengths[b]) and all decode tokens [maxPromptLen, totalPastLen).
                // Mask padding gap [seqLengths[b], maxPromptLen) which holds garbage from padded prefill.
                BuildStaticBatchStepMask(stepMask, seqLengths, groupSize, maxPromptLen, stepCount, totalPastLen);

                using var embVal       = OrtValue.CreateTensorValueFromMemory(stepEmbeds,  [groupSize, 1, _hiddenSize]);
                using var posVal       = OrtValue.CreateTensorValueFromMemory(stepPos,     [groupSize, 1]);
                using var maskVal      = OrtValue.CreateTensorValueFromMemory(stepMask,    [groupSize, 1, 1, totalPastLen]);
                using var logitsOutVal = OrtValue.CreateTensorValueFromMemory(decodeLogBuf, [groupSize, 1, _vocabSize]);

                using var stepBinding = _decoder!.CreateIoBinding();
                stepBinding.BindInput("input_embeds",   embVal);
                stepBinding.BindInput("position_ids",   posVal);
                stepBinding.BindInput("attention_mask", maskVal);
                if (firstStep)
                {
                    // H2D bootstrap: CPU prefill KV → GPU for step 1 (one-time per group)
                    using var pkv = OrtValue.CreateTensorValueFromMemory(prefillKeysBuf,   [_nLayers, groupSize, _nKvHeads, maxPromptLen, _headDim]);
                    using var pvv = OrtValue.CreateTensorValueFromMemory(prefillValuesBuf, [_nLayers, groupSize, _nKvHeads, maxPromptLen, _headDim]);
                    stepBinding.BindInput("past_keys",   pkv);
                    stepBinding.BindInput("past_values", pvv);
                    firstStep = false;
                }
                else
                {
                    stepBinding.BindInput("past_keys",   prevKv![1]); // GPU → GPU: no PCIe
                    stepBinding.BindInput("past_values", prevKv![2]); // GPU → GPU: no PCIe
                }
                stepBinding.BindOutput("logits",                   logitsOutVal); // writes to decodeLogBuf
                stepBinding.BindOutputToDevice("present_keys",   cudaMemInfo);
                stepBinding.BindOutputToDevice("present_values", cudaMemInfo);
                _decoder!.RunWithBinding(runOpts, stepBinding);

                var curKv = stepBinding.GetOutputValues();

                // Read logits from decodeLogBuf (float[]) — no GetTensorDataAsSpan needed
                for (int b = 0; b < groupSize; b++)
                {
                    if (done[b]) continue;
                    int token = ArgMaxSpan(decodeLogBuf.AsSpan(b * _vocabSize, _vocabSize), out float logprob);
                    nextTokens[b] = token;
                    rawTokens[b].Add(token);
                    rawLogprobs[b].Add(logprob);
                    if (IsEos(token) || rawTokens[b].Count >= maxNewTokens)
                        done[b] = true;
                }

                prevKv?.Dispose();
                prevKv = curKv;
                stepCount++;
            }

            prevKv?.Dispose();

            // --- Yield results for this group ---
            for (int i = 0; i < groupSize; i++)
            {
                var item = encodedItems[pendingIndex + i];
                var (textTokens, textLogprobs) = ExtractTextTokens(rawTokens[i], rawLogprobs[i]);
                string rawText = DecodeTokens(textTokens);
                var parsed = forcedLanguageName is null ? ParseMetadataPrefix(rawText) : null;
                string text      = parsed?.Text ?? rawText;
                string? language = forcedLanguageName ?? parsed?.Language;
                yield return new QwenRecognitionResult(
                    item.SegId, text, rawTokens[i], textTokens, textLogprobs, language, rawText);
            }

            pendingIndex += groupSize;
        }
    }

    // Attention mask for one decode step in a static-batch group.
    // Valid positions: prefill tokens [0, seqLengths[b]) and new tokens [maxPromptLen, maxPromptLen+stepCount).
    // Masked:          padding gap [seqLengths[b], maxPromptLen) — garbage KV from padded prefill.
    private static void BuildStaticBatchStepMask(
        float[] buffer, int[] seqLengths, int batchSize,
        int maxPromptLen, int stepCount, int totalPastLen)
    {
        Array.Fill(buffer, float.MinValue, 0, batchSize * totalPastLen);
        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * totalPastLen;
            for (int k = 0; k < seqLengths[b]; k++)
                buffer[offset + k] = 0f;
            for (int k = 0; k < stepCount; k++)
                buffer[offset + maxPromptLen + k] = 0f;
        }
    }

    public IEnumerable<(int segId, string text)> Recognize(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256,
        string? forceLanguage = null)
    {
        foreach (var result in RecognizeDetailed(segs, audio, maxNewTokens, forceLanguage))
            yield return (result.SegmentId, result.Text);
    }

    public IEnumerable<QwenRecognitionResult> RecognizeDetailed(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256,
        string? forceLanguage = null)
    {
        if (_encoder is null)
            throw new InvalidOperationException(
                "Serial encoder sessions are not loaded. Construct with preferBatched:false or call RecognizeBatchedDetailed.");

        int[]   forcedPrefix        = ResolveForcedPrefix(forceLanguage);
        string? forcedLanguageName  = ResolveForcedLanguageName(forceLanguage);

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
                ? (_useCudaIoBinding
                    ? DecodeOnGpuUnified(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens, forcedPrefix)
                    : DecodeOnCpuUnified(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens, forcedPrefix))
                : _useCudaIoBinding
                    ? DecodeWithIoBinding(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens, forcedPrefix)
                    : DecodeOnCpu(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens, forcedPrefix);

            var (textTokens, textLogprobs) = ExtractTextTokens(rawTokens, rawLogprobs);
            string rawText = DecodeTokens(textTokens);
            // When language is forced the prefix is in the prompt (not generated), so rawText IS
            // the content directly. For auto-detect the prefix was generated and stripped above.
            var parsed = forcedLanguageName is null ? ParseMetadataPrefix(rawText) : null;
            string text     = parsed?.Text ?? rawText;
            string? language = forcedLanguageName ?? parsed?.Language;
            yield return new QwenRecognitionResult(
                segId,
                text,
                rawTokens,
                textTokens,
                textLogprobs,
                language,
                rawText);
        }
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
        int maxNewTokens = 256,
        string? forceLanguage = null)
    {
        if (_encoderBatched is null || (_decoderInitBatched is null && _decoder is null))
        {
            foreach (var r in RecognizeDetailed(segs, audio, maxNewTokens, forceLanguage))
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

        // Unified decoder: hand off all segments to the continuous-batch scheduler,
        // which handles batched encoding + one decode scheduler over all segments.
        if (_decoder is not null)
        {
            int[]   forcedPrefix       = ResolveForcedPrefix(forceLanguage);
            string? forcedLanguageName = ResolveForcedLanguageName(forceLanguage);
            foreach (var r in RecognizeUnifiedContinuousBatched(validItems, freeGpuMemoryMb, maxNewTokens, forcedPrefix, forcedLanguageName))
                yield return r;
            yield break;
        }

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

            // ── Batched prefill (split decoder_init_batched path) ───────────

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
        int maxAudioTokens,
        int[]? forcedPrefix = null)
    {
        forcedPrefix ??= [];
        var prompts = new List<int>[batchSize];
        var seqLengths = new int[batchSize];
        int maxSeqLen = 0;

        for (int b = 0; b < batchSize; b++)
        {
            prompts[b] = BuildPromptIds((int)audioLengths[b]);
            seqLengths[b] = prompts[b].Count + forcedPrefix.Length;
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

            // Append forced language prefix tokens after the prompt
            for (int t = 0; t < forcedPrefix.Length; t++)
            {
                ReadTokenEmbedding(forcedPrefix[t], tokenEmbed);
                int pos = prompt.Count + t;
                Array.Copy(tokenEmbed, 0, inputEmbeds, (b * maxSeqLen + pos) * _hiddenSize, _hiddenSize);
                positionIds[b * maxSeqLen + pos] = pos;
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

    // Used by the fixed-batch RecognizeBatchedDetailed path to compact step KV after each decode step.
    private void CompactStepKvInto(
        float[] presentKeys,
        float[] presentValues,
        int[] pastLengths,
        bool[] activeMask,
        int batchSize,
        float[] nextKeys,
        float[] nextValues,
        int[] nextLengths)
    {
        int appendIndex = MaxPrefix(pastLengths, batchSize);
        Array.Copy(pastLengths, nextLengths, batchSize);
        for (int b = 0; b < batchSize; b++)
            if (activeMask[b])
                nextLengths[b]++;

        int maxNextLen = MaxPrefix(nextLengths, batchSize);
        Array.Clear(nextKeys,   0, _nLayers * batchSize * _nKvHeads * maxNextLen * _headDim);
        Array.Clear(nextValues, 0, _nLayers * batchSize * _nKvHeads * maxNextLen * _headDim);

        for (int layer = 0; layer < _nLayers; layer++)
        for (int b = 0; b < batchSize; b++)
        for (int head = 0; head < _nKvHeads; head++)
        {
            int sourceStride = appendIndex + 1;
            int sourceBase = (((layer * batchSize + b) * _nKvHeads + head) * sourceStride) * _headDim;
            int destBase   = (((layer * batchSize + b) * _nKvHeads + head) * maxNextLen)   * _headDim;
            int validPastLen = pastLengths[b];

            if (validPastLen > 0)
            {
                Array.Copy(presentKeys,   sourceBase, nextKeys,   destBase, validPastLen * _headDim);
                Array.Copy(presentValues, sourceBase, nextValues, destBase, validPastLen * _headDim);
            }

            if (activeMask[b])
            {
                Array.Copy(presentKeys,   sourceBase + appendIndex * _headDim, nextKeys,   destBase + validPastLen * _headDim, _headDim);
                Array.Copy(presentValues, sourceBase + appendIndex * _headDim, nextValues, destBase + validPastLen * _headDim, _headDim);
            }
        }
    }

    private static int MaxPrefix(int[] values, int count)
    {
        int max = 0;
        for (int i = 0; i < count; i++)
            max = Math.Max(max, values[i]);
        return max;
    }

    private int ArgMaxSpan(ReadOnlySpan<float> logits, out float logprob)
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
            var encoderOpts = OrtSessionBuilder.Create(ep, optimizationLevel, enableProfiling: false, out _);
            var decoderOpts = OrtSessionBuilder.Create(ep, optimizationLevel, enableProfiling: false, out _);
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
            var encoderOpts = OrtSessionBuilder.Create(ep, optimizationLevel, enableProfiling: false, out _);
            var decoderOpts = OrtSessionBuilder.Create(ep, optimizationLevel, enableProfiling: false, out _);
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
        int maxNewTokens,
        int[] forcedPrefix)
    {
        // Include forced language prefix in the prompt so the model sees it as established context
        // before generating content. This avoids the KV state distortion that occurs when token
        // selection is overridden mid-generation.
        IReadOnlyList<int> effectivePrompt = forcedPrefix.Length > 0
            ? [..promptIds, ..forcedPrefix]
            : promptIds;
        int seqLen = effectivePrompt.Count;

        long[] inputIds = effectivePrompt.Select(i => (long)i).ToArray();
        long[] positionIds = Enumerable.Range(0, seqLen).Select(i => (long)i).ToArray();
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
        int pastSeqLen = seqLen;
        var tokenEmbed = new float[_hiddenSize];
        long[] stepPos = [0L];

        var rawTokens   = new List<int>(maxNewTokens);
        var rawLogprobs = new List<float>(maxNewTokens);

        int   nextToken   = ArgMaxLastLogits(logits, seqLen, out float nextLogprob);
        rawTokens.Add(nextToken);
        rawLogprobs.Add(nextLogprob);

        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tokenEmbed);
            stepPos[0] = seqLen + rawTokens.Count - 1L;

            using var stepOutputs = _decoderStep!.Run(
            [
                NamedOnnxValue.CreateFromTensor("input_embeds", new DenseTensor<float>(tokenEmbed, [1, 1, _hiddenSize])),
                NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(stepPos, [1, 1])),
                NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(pastKeys, [_nLayers, 1, _nKvHeads, pastSeqLen, _headDim])),
                NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(pastValues, [_nLayers, 1, _nKvHeads, pastSeqLen, _headDim])),
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

    // GPU-resident KV variant of DecodeOnCpuUnified. Keeps present_keys / present_values
    // on the CUDA device between steps via OrtIoBinding + OrtValue handoff, eliminating
    // the device→host extract + host→device re-bind round trip that DecodeOnCpuUnified
    // pays on every token. Parallels DecodeWithIoBinding's pattern but uses the unified
    // decoder's I/O contract (input_embeds + attention_mask + packed KV tensors) rather
    // than the split decoder_init + decoder_step graphs.
    private (List<int> rawTokens, List<float> rawLogprobs) DecodeOnGpuUnified(
        IReadOnlyList<int> promptIds,
        int audioOffset,
        float[] audioFeatures,
        int audioTokenCount,
        int maxNewTokens,
        int[] forcedPrefix)
    {
        using var cpuMemInfo  = new OrtMemoryInfo("Cpu",  OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts = new RunOptions();

        int baseSeqLen = promptIds.Count;
        int seqLen     = baseSeqLen + forcedPrefix.Length;

        float[] inputEmbeds = new float[seqLen * _hiddenSize];
        var tmpEmbed = new float[_hiddenSize];
        for (int t = 0; t < baseSeqLen; t++)
        {
            ReadTokenEmbedding(promptIds[t], tmpEmbed);
            Array.Copy(tmpEmbed, 0, inputEmbeds, t * _hiddenSize, _hiddenSize);
        }
        Array.Copy(audioFeatures, 0, inputEmbeds, audioOffset * _hiddenSize, audioTokenCount * _hiddenSize);
        for (int t = 0; t < forcedPrefix.Length; t++)
        {
            ReadTokenEmbedding(forcedPrefix[t], tmpEmbed);
            Array.Copy(tmpEmbed, 0, inputEmbeds, (baseSeqLen + t) * _hiddenSize, _hiddenSize);
        }

        long[] positionIds  = Enumerable.Range(0, seqLen).Select(i => (long)i).ToArray();
        float[] prefillMask = BuildCausalMask(seqLen, 0);
        float[] emptyKv     = [];

        // ── Prefill ────────────────────────────────────────────────────────────
        using var inputEmbedsValue = OrtValue.CreateTensorValueFromMemory(inputEmbeds, [1, seqLen, _hiddenSize]);
        using var positionIdsValue = OrtValue.CreateTensorValueFromMemory(positionIds, [1, seqLen]);
        using var prefillMaskValue = OrtValue.CreateTensorValueFromMemory(prefillMask, [1, 1, seqLen, seqLen]);
        using var emptyKeysValue   = OrtValue.CreateTensorValueFromMemory(emptyKv,     [_nLayers, 1, _nKvHeads, 0, _headDim]);
        using var emptyValuesValue = OrtValue.CreateTensorValueFromMemory(emptyKv,     [_nLayers, 1, _nKvHeads, 0, _headDim]);

        using var initBinding = _decoder!.CreateIoBinding();
        initBinding.BindInput("input_embeds",   inputEmbedsValue);
        initBinding.BindInput("position_ids",   positionIdsValue);
        initBinding.BindInput("attention_mask", prefillMaskValue);
        initBinding.BindInput("past_keys",      emptyKeysValue);
        initBinding.BindInput("past_values",    emptyValuesValue);
        // Bind in order logits / present_keys / present_values so GetOutputValues()
        // returns them at indices [0]/[1]/[2]. Logits go to CPU for the argmax;
        // present KV stay on CUDA so the next step can feed them in place.
        initBinding.BindOutputToDevice("logits",         cpuMemInfo);
        initBinding.BindOutputToDevice("present_keys",   cudaMemInfo);
        initBinding.BindOutputToDevice("present_values", cudaMemInfo);
        _decoder.RunWithBinding(runOpts, initBinding);

        var initOutputs = initBinding.GetOutputValues();

        var rawTokens   = new List<int>(maxNewTokens);
        var rawLogprobs = new List<float>(maxNewTokens);
        long[] stepPos  = [0L];
        int pastSeqLen  = seqLen;

        int nextToken = ArgMaxLastLogits(initOutputs[0], seqLen, out float nextLogprob);
        rawTokens.Add(nextToken);
        rawLogprobs.Add(nextLogprob);

        IDisposableReadOnlyCollection<OrtValue>? prevOutputs = initOutputs;

        // ── Step loop ──────────────────────────────────────────────────────────
        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tmpEmbed);
            stepPos[0] = seqLen + rawTokens.Count - 1L;
            // All-zero mask → the single query attends to every past position.
            float[] stepMask = new float[pastSeqLen + 1];

            using var tokenEmbedValue = OrtValue.CreateTensorValueFromMemory(tmpEmbed, [1, 1, _hiddenSize]);
            using var stepPosValue    = OrtValue.CreateTensorValueFromMemory(stepPos,  [1, 1]);
            using var stepMaskValue   = OrtValue.CreateTensorValueFromMemory(stepMask, [1, 1, 1, pastSeqLen + 1]);

            using var stepBinding = _decoder.CreateIoBinding();
            stepBinding.BindInput("input_embeds",   tokenEmbedValue);
            stepBinding.BindInput("position_ids",   stepPosValue);
            stepBinding.BindInput("attention_mask", stepMaskValue);
            // prevOutputs[1] / [2] are the previous step's present_keys / present_values,
            // still resident on CUDA — no PCIe traffic.
            stepBinding.BindInput("past_keys",      prevOutputs![1]);
            stepBinding.BindInput("past_values",    prevOutputs[2]);
            stepBinding.BindOutputToDevice("logits",         cpuMemInfo);
            stepBinding.BindOutputToDevice("present_keys",   cudaMemInfo);
            stepBinding.BindOutputToDevice("present_values", cudaMemInfo);
            _decoder.RunWithBinding(runOpts, stepBinding);

            var curOutputs = stepBinding.GetOutputValues();
            nextToken = ArgMaxLastLogits(curOutputs[0], 1, out nextLogprob);
            rawTokens.Add(nextToken);
            rawLogprobs.Add(nextLogprob);

            // Hand off: dispose previous-step KV, take ownership of this step's.
            prevOutputs.Dispose();
            prevOutputs = curOutputs;
            pastSeqLen++;
        }

        prevOutputs?.Dispose();
        return (rawTokens, rawLogprobs);
    }

    private (List<int> rawTokens, List<float> rawLogprobs) DecodeOnCpuUnified(
        IReadOnlyList<int> promptIds,
        int audioOffset,
        float[] audioFeatures,
        int audioTokenCount,
        int maxNewTokens,
        int[] forcedPrefix)
    {
        int baseSeqLen = promptIds.Count;
        // Extend the embedded prompt with forced language prefix tokens so the model treats
        // them as established context rather than having its token selection overridden mid-gen.
        int seqLen = baseSeqLen + forcedPrefix.Length;

        // Build input_embeds: embedding lookup for all prompt + forced-prefix tokens, then scatter audio.
        float[] inputEmbeds = new float[seqLen * _hiddenSize];
        var tmpEmbed = new float[_hiddenSize];
        for (int t = 0; t < baseSeqLen; t++)
        {
            ReadTokenEmbedding(promptIds[t], tmpEmbed);
            Array.Copy(tmpEmbed, 0, inputEmbeds, t * _hiddenSize, _hiddenSize);
        }
        Array.Copy(audioFeatures, 0, inputEmbeds, audioOffset * _hiddenSize, audioTokenCount * _hiddenSize);
        for (int t = 0; t < forcedPrefix.Length; t++)
        {
            ReadTokenEmbedding(forcedPrefix[t], tmpEmbed);
            Array.Copy(tmpEmbed, 0, inputEmbeds, (baseSeqLen + t) * _hiddenSize, _hiddenSize);
        }

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

        int   nextToken   = ArgMaxLastLogits(logits, seqLen, out float nextLogprob);
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
        int maxNewTokens,
        int[] forcedPrefix)
    {
        using var cpuMemInfo = new OrtMemoryInfo("Cpu", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts = new RunOptions();

        // Extend prompt with forced language prefix so the model generates content conditioned on it.
        IReadOnlyList<int> effectivePrompt = forcedPrefix.Length > 0
            ? [..promptIds, ..forcedPrefix]
            : promptIds;
        int seqLen = effectivePrompt.Count;

        long[] inputIds = effectivePrompt.Select(i => (long)i).ToArray();
        long[] positionIds = Enumerable.Range(0, seqLen).Select(i => (long)i).ToArray();
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
        var rawTokens   = new List<int>(maxNewTokens);
        var rawLogprobs = new List<float>(maxNewTokens);
        var tokenEmbed  = new float[_hiddenSize];
        long[] stepPos  = [0L];

        int   nextToken   = ArgMaxLastLogits(initOutputs[0], seqLen, out float nextLogprob);
        rawTokens.Add(nextToken);
        rawLogprobs.Add(nextLogprob);

        IDisposableReadOnlyCollection<OrtValue>? prevOutputs = initOutputs;

        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tokenEmbed);
            stepPos[0] = seqLen + rawTokens.Count - 1L;

            using var tokenEmbedValue = OrtValue.CreateTensorValueFromMemory(tokenEmbed, [1, 1, _hiddenSize]);
            using var stepPosValue    = OrtValue.CreateTensorValueFromMemory(stepPos, [1, 1]);
            using var stepBinding     = _decoderStep!.CreateIoBinding();
            stepBinding.BindInput("input_embeds", tokenEmbedValue);
            stepBinding.BindInput("position_ids", stepPosValue);
            stepBinding.BindInput("past_keys",    prevOutputs![1]);
            stepBinding.BindInput("past_values",  prevOutputs[2]);
            stepBinding.BindOutputToDevice("logits",         cpuMemInfo);
            stepBinding.BindOutputToDevice("present_keys",   cudaMemInfo);
            stepBinding.BindOutputToDevice("present_values", cudaMemInfo);
            _decoderStep.RunWithBinding(runOpts, stepBinding);

            var curOutputs = stepBinding.GetOutputValues();
            nextToken = ArgMaxLastLogits(curOutputs[0], 1, out nextLogprob);
            rawTokens.Add(nextToken);
            rawLogprobs.Add(nextLogprob);

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

        // Strip the "language <Name>" prefix tokens so stored tokens match result.Text.
        // Known prefix entries end with 220 ("Ġ" space), but the model doesn't always emit that
        // separator — sometimes it jumps straight to a Ġ-prefixed content token.
        // Match the [11528, ...name tokens...] portion; optionally swallow a trailing 220.
        if (textTokens.Count > 0 && textTokens[0] == 11528)
        {
            foreach (var prefix in LanguagePrefixTokens.Values)
            {
                int nameLen = prefix[^1] == 220 ? prefix.Length - 1 : prefix.Length;
                if (textTokens.Count < nameLen) continue;
                bool match = true;
                for (int j = 0; j < nameLen && match; j++)
                    if (textTokens[j] != prefix[j]) match = false;
                if (!match) continue;

                int skip = nameLen;
                if (skip < textTokens.Count && textTokens[skip] == 220)
                    skip++;

                return (textTokens.GetRange(skip, textTokens.Count - skip),
                        textLogprobs.GetRange(skip, textLogprobs.Count - skip));
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

internal sealed record QwenEncodedItem(
    int SegId,
    int DurationSamples,
    float[] AudioFeatures,
    int AudioTokenCount);

internal sealed record QwenBatchedItem(
    int SegId,
    int DurationSamples,
    float[]? Mel,
    int MelFrames);
