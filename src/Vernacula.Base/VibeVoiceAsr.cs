using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Vernacula.Base;

/// <summary>
/// VibeVoice-ASR — whole-recording multimodal ASR with built-in diarization.
///
/// Uses two ONNX models in the model directory:
///   audio_encoder.onnx    waveform [1,T] + mask [1,T] → audio_embeddings [N,3584] (bf16)
///   decoder_single.onnx   prefix_input_ids [1,P] + audio_embeddings [N,3584] +
///                         suffix_input_ids [1,S] + past_key_0..27 [1,4,C,128] +
///                         past_value_0..27 [1,4,C,128]
///                         → logits [1,seq,152064] + present_key/value_0..27
///
/// Tokenizer metadata (prefix/suffix token arrays, digit→token map) is read from
/// export-report.json["tokenizer"] so no BPE tokenizer is needed at runtime.
/// Token-ID-to-text decoding uses the GPT-2 ByteLevel scheme read from tokenizer.json.
///
/// The decoder uses a split float32 KV cache (56 separate tensors) to avoid BFC
/// arena fragmentation at large cache lengths. Chunked prefill is supported to
/// keep peak VRAM bounded when the audio embedding sequence is long.
/// </summary>
public sealed class VibeVoiceAsr : IDisposable
{
    // ── File names ────────────────────────────────────────────────────────────

    public const string AudioEncoderFile  = "audio_encoder.onnx";
    public const string DecoderSingleFile = "decoder_single.onnx";
    public const string ExportReportFile  = "export-report.json";
    public const string TokenizerFile     = "tokenizer.json";

    // ── Audio ─────────────────────────────────────────────────────────────────

    /// <summary>Required sample rate for the audio encoder.</summary>
    public const int SampleRate = 24_000;

    /// <summary>
    /// Total compression ratio of the acoustic tokenizer.
    /// Audio length must be a multiple of this many samples.
    /// Product of downsampling_ratios [2,2,4,5,5,8] in the model config.
    /// </summary>
    public const int AudioStride = 3_200;

    // ── ORT sessions ─────────────────────────────────────────────────────────

    private readonly InferenceSession _audioEncoder;
    private readonly InferenceSession _decoder;

    // ── Model dimensions ──────────────────────────────────────────────────────

    private readonly int _numLayers;   // 28
    private readonly int _numKvHeads;  // 4
    private readonly int _headDim;     // 128
    private readonly int _hiddenSize;  // 3584

    // ── Tokenizer ─────────────────────────────────────────────────────────────

    /// <summary>Inverse vocabulary: token_id → raw ByteLevel token string.</summary>
    private readonly string?[] _idToToken;

    /// <summary>Special (added) tokens: token_id → decoded content string.</summary>
    private readonly Dictionary<int, string> _addedTokenContent;

    /// <summary>GPT-2 ByteLevel decode table: Unicode char in token string → byte value.</summary>
    private readonly Dictionary<char, byte> _byteLevelDecode;

    // ── Tokenizer extras from export-report.json ──────────────────────────────

    private readonly long[] _prefixTokenIds;
    private readonly long[] _suffixBeforeDuration;
    private readonly long[] _suffixAfterDuration;
    private readonly Dictionary<char, long> _digitCharToTokenId;
    private readonly long _eosTokenId;
    private readonly long _imEndTokenId;

    // ── Construction ──────────────────────────────────────────────────────────

    public VibeVoiceAsr(string modelDir, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        // Load export report
        string reportPath = Path.Combine(modelDir, ExportReportFile);
        using var reportDoc = JsonDocument.Parse(File.ReadAllText(reportPath));
        var report = reportDoc.RootElement;

        _numLayers  = report.GetProperty("num_layers").GetInt32();
        _numKvHeads = report.GetProperty("num_kv_heads").GetInt32();
        _headDim    = report.GetProperty("head_dim").GetInt32();
        _hiddenSize = report.GetProperty("hidden_size").GetInt32();

        var tok = report.GetProperty("tokenizer");
        _prefixTokenIds       = ReadLongArray(tok.GetProperty("prefix_token_ids"));
        _suffixBeforeDuration = ReadLongArray(tok.GetProperty("suffix_before_duration_token_ids"));
        _suffixAfterDuration  = ReadLongArray(tok.GetProperty("suffix_after_duration_token_ids"));
        _eosTokenId           = tok.GetProperty("eos_token_id").GetInt64();
        _imEndTokenId         = tok.GetProperty("im_end_token_id").GetInt64();

        var digitMap = tok.GetProperty("digit_char_to_token_id");
        _digitCharToTokenId = new Dictionary<char, long>(12);
        foreach (var kv in digitMap.EnumerateObject())
            _digitCharToTokenId[kv.Name[0]] = kv.Value.GetInt64();

        // Load tokenizer vocabulary for id→text decoding
        (_idToToken, _addedTokenContent) = LoadTokenizerVocab(Path.Combine(modelDir, TokenizerFile));
        _byteLevelDecode = BuildByteLevelDecode();

        // Create ORT sessions
        var cpuOpts = new SessionOptions();
        cpuOpts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        _audioEncoder = new InferenceSession(Path.Combine(modelDir, AudioEncoderFile), cpuOpts);

        var gpuOpts = MakeSessionOptions(ep);
        _decoder = new InferenceSession(Path.Combine(modelDir, DecoderSingleFile), gpuOpts);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Transcribe audio using VibeVoice-ASR's built-in diarization.
    ///
    /// The audio is resampled to 24 kHz mono, padded to a multiple of 3200 samples,
    /// run through the audio encoder, and then decoded autoregressively.  When
    /// prefillChunkTokens &gt; 0 the audio embedding sequence is processed in chunks
    /// to bound peak VRAM.
    /// </summary>
    /// <param name="rawAudio">Interleaved float samples in [-1, 1] at any sample rate.</param>
    /// <param name="sampleRate">Sample rate of <paramref name="rawAudio"/>.</param>
    /// <param name="channels">Channel count of <paramref name="rawAudio"/>.</param>
    /// <param name="prefillChunkTokens">
    ///   Audio tokens per prefill chunk (0 = no chunking).  Recommended: 512 for
    ///   recordings longer than ~5 minutes on a 24 GB GPU.
    /// </param>
    /// <param name="maxNewTokens">Maximum tokens to generate (default 8192).</param>
    /// <param name="ct">Cancellation token.</param>
    public IReadOnlyList<VibeVoiceSegment> Transcribe(
        float[] rawAudio,
        int sampleRate,
        int channels,
        int prefillChunkTokens = 512,
        int maxNewTokens = 8_192,
        CancellationToken ct = default)
    {
        // 1 — Resample to 24 kHz mono and pad to acoustic tokenizer stride
        float[] audio24k = AudioTo24kMono(rawAudio, sampleRate, channels);
        double durationSeconds = audio24k.Length / (double)SampleRate;
        audio24k = PadToStride(audio24k);

        // 2 — Build suffix token IDs (contains the audio duration)
        long[] suffixIds = BuildSuffixTokenIds(durationSeconds);

        // 3 — Run audio encoder → BF16 audio embeddings
        BFloat16[] audioEmbeddings = RunAudioEncoder(audio24k);
        int numAudioTokens = audioEmbeddings.Length / _hiddenSize;

        // 4 — Run decoder (chunked prefill + greedy decode) → raw JSON text
        string jsonText = RunDecoder(
            _prefixTokenIds,
            audioEmbeddings,
            numAudioTokens,
            suffixIds,
            prefillChunkTokens,
            maxNewTokens,
            ct);

        // 5 — Parse VibeVoice JSON output
        return ParseOutput(jsonText);
    }

    public void Dispose()
    {
        _audioEncoder.Dispose();
        _decoder.Dispose();
    }

    // ── Audio preprocessing ───────────────────────────────────────────────────

    private static float[] AudioTo24kMono(float[] audio, int sampleRate, int channels)
    {
        // Downmix
        float[] mono;
        if (channels == 1)
        {
            mono = audio;
        }
        else
        {
            mono = new float[audio.Length / channels];
            for (int i = 0; i < mono.Length; i++)
            {
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                    sum += audio[i * channels + c];
                mono[i] = sum / channels;
            }
        }

        if (sampleRate == SampleRate)
            return mono;

        var srcFormat = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        var provider  = new FloatArraySampleProvider(mono, srcFormat);
        var resampler = new WdlResamplingSampleProvider(provider, SampleRate);

        var outList   = new List<float>((int)((long)mono.Length * SampleRate / sampleRate + 1024));
        var outBuffer = new float[8192];
        int read;
        while ((read = resampler.Read(outBuffer, 0, outBuffer.Length)) > 0)
            for (int i = 0; i < read; i++) outList.Add(outBuffer[i]);

        return outList.ToArray();
    }

    private static float[] PadToStride(float[] audio)
    {
        int remainder = audio.Length % AudioStride;
        if (remainder == 0) return audio;
        int padded = audio.Length + (AudioStride - remainder);
        Array.Resize(ref audio, padded);  // zero-fills new elements
        return audio;
    }

    // ── Audio encoder ─────────────────────────────────────────────────────────

    /// <summary>
    /// Runs audio_encoder.onnx and returns the BF16 audio embeddings as a flat array
    /// in row-major order: [numTokens * hiddenSize].
    /// </summary>
    private BFloat16[] RunAudioEncoder(float[] audio24kPadded)
    {
        int n = audio24kPadded.Length;
        var inputValues = new DenseTensor<float>(audio24kPadded, new[] { 1, n });
        var paddingMask = new DenseTensor<float>(Enumerable.Repeat(1f, n).ToArray(), new[] { 1, n });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_values", inputValues),
            NamedOnnxValue.CreateFromTensor("padding_mask", paddingMask),
        };

        using var results = _audioEncoder.Run(inputs);
        var embTensor = results[0].AsTensor<BFloat16>();
        return embTensor.ToArray();
    }

    // ── Tokenizer helpers ─────────────────────────────────────────────────────

    private long[] BuildSuffixTokenIds(double durationSeconds)
    {
        // Format as "%.2f" (e.g. "90.00") and tokenize character by character
        string durStr = durationSeconds.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
        var durTokens = new long[durStr.Length];
        for (int i = 0; i < durStr.Length; i++)
        {
            if (!_digitCharToTokenId.TryGetValue(durStr[i], out long tid))
                throw new InvalidOperationException(
                    $"Duration character '{durStr[i]}' not in digit_char_to_token_id map.");
            durTokens[i] = tid;
        }

        return [.. _suffixBeforeDuration, .. durTokens, .. _suffixAfterDuration];
    }

    // ── Decoder ───────────────────────────────────────────────────────────────

    private string RunDecoder(
        long[] prefixIds,
        BFloat16[] audioEmbeddings,
        int numAudioTokens,
        long[] suffixIds,
        int prefillChunkTokens,
        int maxNewTokens,
        CancellationToken ct)
    {
        // Initialise 56 empty float32 KV arrays: [1, numKvHeads, 0, headDim]
        var pastKvs = new float[_numLayers * 2][];
        for (int i = 0; i < pastKvs.Length; i++)
            pastKvs[i] = [];

        // Prefill
        long firstToken;
        (firstToken, pastKvs) = RunPrefill(
            prefixIds, audioEmbeddings, numAudioTokens, suffixIds,
            pastKvs, prefillChunkTokens, ct);

        // Greedy decode
        var generated = new List<long>(256);
        long nextToken = firstToken;

        for (int step = 0; step < maxNewTokens; step++)
        {
            ct.ThrowIfCancellationRequested();
            if (nextToken == _eosTokenId || nextToken == _imEndTokenId)
                break;
            generated.Add(nextToken);
            (nextToken, pastKvs) = RunDecoderStep(nextToken, pastKvs);
        }

        return TokensToText(generated);
    }

    private (long firstToken, float[][] pastKvs) RunPrefill(
        long[] prefixIds,
        BFloat16[] audioEmbeddings,
        int numAudioTokens,
        long[] suffixIds,
        float[][] pastKvs,
        int prefillChunkTokens,
        CancellationToken ct)
    {
        if (prefillChunkTokens <= 0 || numAudioTokens <= prefillChunkTokens)
        {
            // Full prefill in one shot
            var audioTensor = MakeAudioTensor(audioEmbeddings, 0, numAudioTokens);
            return RunSinglePrefillChunk(prefixIds, audioTensor, suffixIds, pastKvs);
        }

        // Chunked prefill
        int numChunks = (numAudioTokens + prefillChunkTokens - 1) / prefillChunkTokens;
        long lastToken = 0;
        long[] emptyIds = [];

        for (int ci = 0; ci < numChunks; ci++)
        {
            ct.ThrowIfCancellationRequested();
            int start = ci * prefillChunkTokens;
            int end   = Math.Min(start + prefillChunkTokens, numAudioTokens);

            var pfx   = ci == 0             ? prefixIds : emptyIds;
            var sfx   = ci == numChunks - 1 ? suffixIds : emptyIds;
            var chunk = MakeAudioTensor(audioEmbeddings, start, end);

            (lastToken, pastKvs) = RunSinglePrefillChunk(pfx, chunk, sfx, pastKvs);
        }

        return (lastToken, pastKvs);
    }

    private (long firstToken, float[][] pastKvs) RunSinglePrefillChunk(
        long[] prefixIds,
        DenseTensor<BFloat16> audioChunk,
        long[] suffixIds,
        float[][] pastKvs)
    {
        var inputs = BuildDecoderInputs(
            prefixIds:  prefixIds,
            audioChunk: audioChunk,
            suffixIds:  suffixIds,
            stepToken:  null,
            pastKvs:    pastKvs);

        using var outputs = _decoder.Run(inputs);
        long firstToken = ArgmaxLastRow(outputs[0].AsTensor<BFloat16>());
        float[][] newKvs = ExtractKvs(outputs);
        return (firstToken, newKvs);
    }

    private (long nextToken, float[][] pastKvs) RunDecoderStep(long token, float[][] pastKvs)
    {
        // Pass zero-length audio and empty prefix/suffix for pure autoregressive step
        var emptyAudio = new DenseTensor<BFloat16>(new[] { 0, _hiddenSize });
        var inputs = BuildDecoderInputs(
            prefixIds:  [token],
            audioChunk: emptyAudio,
            suffixIds:  [],
            stepToken:  null,
            pastKvs:    pastKvs);

        using var outputs = _decoder.Run(inputs);
        long nextToken = ArgmaxLastRow(outputs[0].AsTensor<BFloat16>());
        float[][] newKvs = ExtractKvs(outputs);
        return (nextToken, newKvs);
    }

    // ── Decoder tensor helpers ────────────────────────────────────────────────

    private List<NamedOnnxValue> BuildDecoderInputs(
        long[]              prefixIds,
        DenseTensor<BFloat16> audioChunk,
        long[]              suffixIds,
        long?               stepToken,
        float[][]           pastKvs)
    {
        var inputs = new List<NamedOnnxValue>(_numLayers * 2 + 3);

        inputs.Add(NamedOnnxValue.CreateFromTensor(
            "prefix_input_ids",
            new DenseTensor<long>(prefixIds, new[] { 1, prefixIds.Length })));

        inputs.Add(NamedOnnxValue.CreateFromTensor("audio_embeddings", audioChunk));

        inputs.Add(NamedOnnxValue.CreateFromTensor(
            "suffix_input_ids",
            new DenseTensor<long>(suffixIds, new[] { 1, suffixIds.Length })));

        // 56 KV cache tensors
        int cacheLen = pastKvs.Length > 0 ? pastKvs[0].Length / (_numKvHeads * _headDim) : 0;
        for (int i = 0; i < _numLayers; i++)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                $"past_key_{i}",
                new DenseTensor<float>(pastKvs[i * 2], new[] { 1, _numKvHeads, cacheLen, _headDim })));
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                $"past_value_{i}",
                new DenseTensor<float>(pastKvs[i * 2 + 1], new[] { 1, _numKvHeads, cacheLen, _headDim })));
        }

        return inputs;
    }

    private float[][] ExtractKvs(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        // outputs[0] = logits, outputs[1..56] = present_key_0..27, present_value_0..27
        var kvs = new float[_numLayers * 2][];
        for (int i = 0; i < _numLayers * 2; i++)
            kvs[i] = outputs[i + 1].AsTensor<float>().ToArray();
        return kvs;
    }

    private DenseTensor<BFloat16> MakeAudioTensor(BFloat16[] embeddings, int startToken, int endToken)
    {
        int chunkLen = endToken - startToken;
        var data = new BFloat16[chunkLen * _hiddenSize];
        Array.Copy(embeddings, startToken * _hiddenSize, data, 0, data.Length);
        return new DenseTensor<BFloat16>(data, new[] { chunkLen, _hiddenSize });
    }

    private static long ArgmaxLastRow(Tensor<BFloat16> logits)
    {
        // logits shape: [1, seq_len, vocab_size]
        int seqLen   = logits.Dimensions[1];
        int vocabSize = logits.Dimensions[2];
        int rowOffset = (seqLen - 1) * vocabSize;

        long bestIdx = 0;
        float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocabSize; v++)
        {
            float val = (float)logits[0, seqLen - 1, v];
            if (val > bestVal) { bestVal = val; bestIdx = v; }
        }
        return bestIdx;
    }

    // ── Token decoding ────────────────────────────────────────────────────────

    private string TokensToText(List<long> tokenIds)
    {
        var bytes = new List<byte>(tokenIds.Count * 4);

        foreach (long id in tokenIds)
        {
            int iid = (int)id;

            // Added (special) tokens: use their content string as raw UTF-8
            if (_addedTokenContent.TryGetValue(iid, out string? special))
            {
                bytes.AddRange(Encoding.UTF8.GetBytes(special));
                continue;
            }

            // Regular BPE token: apply ByteLevel decode
            string? raw = iid >= 0 && iid < _idToToken.Length ? _idToToken[iid] : null;
            if (raw is null) continue;

            foreach (char ch in raw)
            {
                if (_byteLevelDecode.TryGetValue(ch, out byte b))
                    bytes.Add(b);
                // Unknown chars are silently dropped — shouldn't happen with a valid vocab
            }
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    // ── Output parsing ────────────────────────────────────────────────────────

    private static IReadOnlyList<VibeVoiceSegment> ParseOutput(string rawText)
    {
        // Generated text: "<|im_start|>assistant\n[...JSON...]\n"
        // Strip everything before the first '[' and after the last ']'
        int start = rawText.IndexOf('[');
        int end   = rawText.LastIndexOf(']');

        if (start < 0 || end <= start)
            return [];

        string json = rawText[start..(end + 1)];

        try
        {
            var items = JsonSerializer.Deserialize<List<VibeVoiceRawSegment>>(json);
            if (items is null) return [];

            return items
                .Where(s => s.Content is not null)
                .Select(s => new VibeVoiceSegment(s.Start, s.End, s.Speaker, s.Content!))
                .ToList();
        }
        catch (JsonException)
        {
            // Truncated or malformed JSON — return whatever we have via a lenient parse
            return ParsePartialJson(json);
        }
    }

    /// <summary>
    /// Best-effort parse of a truncated JSON array.  Extracts complete objects only.
    /// </summary>
    private static IReadOnlyList<VibeVoiceSegment> ParsePartialJson(string json)
    {
        var results = new List<VibeVoiceSegment>();

        // Find each complete "{...}" object
        int depth = 0;
        int objStart = -1;
        for (int i = 0; i < json.Length; i++)
        {
            if (json[i] == '{') { if (depth++ == 0) objStart = i; }
            else if (json[i] == '}' && depth > 0)
            {
                if (--depth == 0 && objStart >= 0)
                {
                    string obj = json[objStart..(i + 1)];
                    try
                    {
                        var seg = JsonSerializer.Deserialize<VibeVoiceRawSegment>(obj);
                        if (seg?.Content is not null)
                            results.Add(new VibeVoiceSegment(seg.Start, seg.End, seg.Speaker, seg.Content));
                    }
                    catch (JsonException) { /* skip malformed object */ }
                }
            }
        }

        return results;
    }

    // ── Tokenizer loading ─────────────────────────────────────────────────────

    private static (string?[] idToToken, Dictionary<int, string> addedContent)
        LoadTokenizerVocab(string path)
    {
        using var doc  = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;

        // Invert vocab dict (token_string → id) into id → token_string
        var vocab = root.GetProperty("model").GetProperty("vocab");
        int maxId = 0;
        foreach (var kv in vocab.EnumerateObject())
            if (kv.Value.GetInt32() > maxId) maxId = kv.Value.GetInt32();

        var idToToken = new string?[maxId + 1];
        foreach (var kv in vocab.EnumerateObject())
            idToToken[kv.Value.GetInt32()] = kv.Name;

        // Added tokens override the vocab (they use their "content" field directly
        // and bypass ByteLevel decoding)
        var addedContent = new Dictionary<int, string>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var at in addedTokens.EnumerateArray())
            {
                int    atId      = at.GetProperty("id").GetInt32();
                string atContent = at.GetProperty("content").GetString() ?? "";
                addedContent[atId] = atContent;
            }
        }

        return (idToToken, addedContent);
    }

    // ── GPT-2 ByteLevel decode table ──────────────────────────────────────────

    /// <summary>
    /// Builds a char→byte lookup for the GPT-2 ByteLevel encoding used by Qwen2.
    ///
    /// GPT-2 maps the 256 byte values to Unicode characters:
    ///   printable ASCII (33-126), ¡-¬ (161-172), ®-ÿ (174-255) → themselves
    ///   remaining 68 bytes (0-32, 127, 128-160, 173) → U+0100, U+0101, …
    ///
    /// This table is the reverse: Unicode char in a token string → byte value.
    /// </summary>
    private static Dictionary<char, byte> BuildByteLevelDecode()
    {
        // Bytes in the printable set map to themselves
        var printable = new HashSet<int>(
            Enumerable.Range(33, 94)           // 33-126  (! through ~)
            .Concat(Enumerable.Range(161, 12)) // 161-172 (¡ through ¬)
            .Concat(Enumerable.Range(174, 82)) // 174-255 (® through ÿ)
        );

        var dict = new Dictionary<char, byte>(280);

        foreach (int b in printable)
            dict[(char)b] = (byte)b;

        // Non-printable bytes get consecutive code points starting at U+0100
        int extra = 0;
        for (int b = 0; b < 256; b++)
            if (!printable.Contains(b))
                dict[(char)(0x100 + extra++)] = (byte)b;

        return dict;
    }

    // ── Session options ───────────────────────────────────────────────────────

    private static SessionOptions MakeSessionOptions(ExecutionProvider ep)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                    try { opts.AppendExecutionProvider_CUDA(0); } catch { }
#if DIRECTML
                try { opts.AppendExecutionProvider_DML(0); } catch { }
#endif
                break;
            case ExecutionProvider.Cuda:
                try { opts.AppendExecutionProvider_CUDA(0); }
                catch (Exception ex) { throw new InvalidOperationException("CUDA EP unavailable.", ex); }
                break;
            case ExecutionProvider.DirectML:
                try { opts.AppendExecutionProvider_DML(0); }
                catch (Exception ex) { throw new InvalidOperationException("DirectML EP unavailable.", ex); }
                break;
            case ExecutionProvider.Cpu:
                break;
        }

        return opts;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static long[] ReadLongArray(JsonElement el)
        => el.EnumerateArray().Select(e => e.GetInt64()).ToArray();
}

// ── Output types ─────────────────────────────────────────────────────────────

/// <summary>A single diarized, timestamped transcription segment from VibeVoice-ASR.</summary>
public sealed record VibeVoiceSegment(double Start, double End, int Speaker, string Content);

/// <summary>Internal DTO matching the JSON schema VibeVoice generates.</summary>
file sealed class VibeVoiceRawSegment
{
    [JsonPropertyName("Start")]   public double  Start   { get; set; }
    [JsonPropertyName("End")]     public double  End     { get; set; }
    [JsonPropertyName("Speaker")] public int     Speaker { get; set; }
    [JsonPropertyName("Content")] public string? Content { get; set; }
}
