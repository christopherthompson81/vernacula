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

    // The audio encoder is created on demand and disposed after each use so its
    // GPU arena (dual float32 Conv towers) does not compete with the decoder's
    // growing KV cache during the autoregressive decode loop.
    private readonly string           _audioEncoderPath;
    private readonly ExecutionProvider _ep;
    private readonly InferenceSession _decoder;

    // ── Model dimensions ──────────────────────────────────────────────────────

    private readonly int _numLayers;          // 28
    private readonly int _numKvHeads;         // 4
    private readonly int _headDim;            // 128
    private readonly int _hiddenSize;         // 3584
    private readonly int _encoderChunkSamples; // from export-report.json acoustic_tokenizer_chunk_size

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

        _numLayers           = report.GetProperty("num_layers").GetInt32();
        _numKvHeads          = report.GetProperty("num_kv_heads").GetInt32();
        _headDim             = report.GetProperty("head_dim").GetInt32();
        _hiddenSize          = report.GetProperty("hidden_size").GetInt32();
        _encoderChunkSamples = report.GetProperty("acoustic_tokenizer_chunk_size").GetInt32();

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
        // Both models need CUDA: audio_encoder.onnx contains BFloat16 MatMul nodes in
        // the multimodal projector that CPU EP does not implement.
        // The audio encoder is NOT loaded here — it is created and disposed within each
        // Transcribe call so its GPU arena is freed before the autoregressive decode loop.
        _audioEncoderPath = Path.Combine(modelDir, AudioEncoderFile);
        _ep               = ep;

        var gpuOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED);
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

        // 3 — Run audio encoder in chunks → BF16 audio embeddings.
        //     The encoder session is created here and disposed immediately after so its
        //     GPU arena (float32 Conv towers) is freed before the decoder KV cache grows.
        BFloat16[] audioEmbeddings;
        {
            using var audioEncoder = new InferenceSession(
                _audioEncoderPath, MakeSessionOptions(_ep));
            audioEmbeddings = RunAudioEncoderChunked(audio24k, audioEncoder);
        }
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
    /// Runs audio_encoder.onnx in chunks of _encoderChunkSamples and concatenates
    /// the BF16 audio embeddings: [numTokens * hiddenSize] in row-major order.
    /// </summary>
    private BFloat16[] RunAudioEncoderChunked(float[] audio24kPadded, InferenceSession audioEncoder)
    {
        var allEmbeddings = new List<BFloat16>();
        int total = audio24kPadded.Length;
        int chunkSize = _encoderChunkSamples; // must be a multiple of AudioStride

        for (int offset = 0; offset < total; offset += chunkSize)
        {
            int len = Math.Min(chunkSize, total - offset);
            // Pad the final chunk if shorter than chunkSize so the model sees a
            // consistent shape; the mask marks the valid region.
            int paddedLen = ((len + AudioStride - 1) / AudioStride) * AudioStride;

            var inputValuesBf16 = new BFloat16[paddedLen];
            for (int i = 0; i < len; i++)
                inputValuesBf16[i] = (BFloat16)audio24kPadded[offset + i];
            // remaining elements stay 0 (default BFloat16 zero)

            var maskBool = new bool[paddedLen];
            for (int i = 0; i < len; i++) maskBool[i] = true;
            // false for the padded tail

            var inputValues = new DenseTensor<BFloat16>(inputValuesBf16, new[] { 1, paddedLen });
            var paddingMask = new DenseTensor<bool>(maskBool,            new[] { 1, paddedLen });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_values", inputValues),
                NamedOnnxValue.CreateFromTensor("padding_mask", paddingMask),
            };

            using var results = audioEncoder.Run(inputs);
            var embTensor = results[0].AsTensor<BFloat16>();

            // If we padded the last chunk, trim the extra tokens
            int expectedTokens = len / AudioStride;
            int gotTokens      = embTensor.Dimensions[0]; // [numTokens, hiddenSize]
            int keepTokens     = Math.Min(gotTokens, expectedTokens);
            int keepElements   = keepTokens * _hiddenSize;
            var flat           = embTensor.ToArray();
            for (int i = 0; i < keepElements; i++)
                allEmbeddings.Add(flat[i]);
        }

        return [.. allEmbeddings];
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
        using var runOptions  = new RunOptions();
        using var binding     = _decoder.CreateIoBinding();
        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);

        // Empty (cache_len = 0) CPU OrtValues for the initial KV inputs
        OrtValue[] pastKvs = CreateEmptyKvOrtValues();
        try
        {
            // ── Chunked prefill ───────────────────────────────────────────────

            int    chunkSize = prefillChunkTokens > 0 ? prefillChunkTokens : numAudioTokens;
            int    numChunks = (numAudioTokens + chunkSize - 1) / Math.Max(1, chunkSize);
            long[] noIds     = [];
            long   lastToken = 0;

            for (int ci = 0; ci < numChunks; ci++)
            {
                ct.ThrowIfCancellationRequested();
                int start = ci * chunkSize;
                int count = Math.Min(chunkSize, numAudioTokens - start);

                long[] pfx = ci == 0             ? prefixIds : noIds;
                long[] sfx = ci == numChunks - 1 ? suffixIds : noIds;

                var prevKvs = pastKvs;
                (lastToken, pastKvs) = RunOnce(
                    pfx, audioEmbeddings, start, count, sfx, pastKvs, binding, runOptions, cudaMemInfo);
                foreach (var kv in prevKvs) kv.Dispose();
            }

            // ── Greedy decode ─────────────────────────────────────────────────

            var    generated  = new List<long>(maxNewTokens);
            long   nextToken  = lastToken;
            long[] tokenBuf   = new long[1]; // reused each step — avoids per-step allocation

            for (int step = 0; step < maxNewTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                if (nextToken == _eosTokenId || nextToken == _imEndTokenId) break;
                generated.Add(nextToken);

                tokenBuf[0] = nextToken;
                var prevKvs = pastKvs;
                (nextToken, pastKvs) = RunOnce(
                    tokenBuf, audioEmbeddings, 0, 0, noIds, pastKvs, binding, runOptions, cudaMemInfo);
                foreach (var kv in prevKvs) kv.Dispose();
            }

            return TokensToText(generated);
        }
        finally
        {
            foreach (var kv in pastKvs) kv.Dispose();
        }
    }

    /// <summary>
    /// Runs the decoder once (one prefill chunk or one autoregressive step) using IO binding.
    ///
    /// KV cache inputs are OrtValues that may live on CPU (initial empty cache) or on CUDA
    /// (every subsequent call). Because the outputs are also bound to CUDA, the KV data
    /// never crosses the PCIe bus after the first step.
    ///
    /// Ownership note: GetOutputValues() returns a DisposableList which has no finalizer.
    /// We extract the KV OrtValues by reference (they stay alive via their SafeHandles),
    /// dispose the logits OrtValue immediately, and let the bare DisposableList be GC-collected
    /// without calling Dispose — so the KV OrtValues it contained are not prematurely freed.
    /// </summary>
    private (long token, OrtValue[] presentKvs) RunOnce(
        long[]       prefixIds,
        BFloat16[]   audioData,
        int          audioStart,
        int          audioCount,
        long[]       suffixIds,
        OrtValue[]   pastKvs,
        OrtIoBinding binding,
        RunOptions   runOptions,
        OrtMemoryInfo cudaMemInfo)
    {
        binding.ClearBoundInputs();

        // Re-register outputs every call: the KV cache grows each step, so the shape
        // changes and ORT cannot reuse a previously allocated output buffer.
        binding.ClearBoundOutputs();
        binding.BindOutputToDevice("logits", OrtMemoryInfo.DefaultInstance);
        for (int i = 0; i < _numLayers; i++)
        {
            binding.BindOutputToDevice($"present_key_{i}",   cudaMemInfo);
            binding.BindOutputToDevice($"present_value_{i}", cudaMemInfo);
        }

        // prefix_input_ids [1, P]
        using var prefixOrtVal = OrtValue.CreateTensorValueFromMemory(
            prefixIds, new long[] { 1, prefixIds.Length });
        binding.BindInput("prefix_input_ids", prefixOrtVal);

        // audio_embeddings [N, hiddenSize] — zero-length during decode steps
        // Declared at method scope so it stays alive until after RunWithBinding.
        int audioElems = audioCount * _hiddenSize;
        using var audioOrtVal = audioElems > 0
            ? OrtValue.CreateTensorValueFromMemory(
                  OrtMemoryInfo.DefaultInstance,
                  new Memory<BFloat16>(audioData, audioStart * _hiddenSize, audioElems),
                  new long[] { audioCount, _hiddenSize })
            : OrtValue.CreateTensorValueFromMemory(
                  Array.Empty<BFloat16>(), new long[] { 0, _hiddenSize });
        binding.BindInput("audio_embeddings", audioOrtVal);

        // suffix_input_ids [1, S]
        using var suffixOrtVal = OrtValue.CreateTensorValueFromMemory(
            suffixIds, new long[] { 1, suffixIds.Length });
        binding.BindInput("suffix_input_ids", suffixOrtVal);

        // 56 KV cache inputs — GPU-resident after the first step, no copy
        for (int i = 0; i < _numLayers; i++)
        {
            binding.BindInput($"past_key_{i}",   pastKvs[i * 2]);
            binding.BindInput($"past_value_{i}", pastKvs[i * 2 + 1]);
        }

        _decoder.RunWithBinding(runOptions, binding);

        // GetOutputValues() → IDisposableReadOnlyCollection<OrtValue> (a DisposableList).
        // DisposableList has no finalizer, so skipping `using` is safe: GC collects the
        // list shell without touching the OrtValues inside it.
        var outputs = binding.GetOutputValues();

        // outputs[0] = logits (CPU) — read argmax, then free immediately
        int  seqLen = prefixIds.Length + audioCount + suffixIds.Length;
        long token  = ArgmaxOrtValue(outputs[0], seqLen);
        outputs[0].Dispose();

        // outputs[1..56] = present_key/value (CUDA) — take ownership
        var presentKvs = new OrtValue[_numLayers * 2];
        for (int i = 0; i < _numLayers * 2; i++)
            presentKvs[i] = outputs[i + 1];

        return (token, presentKvs);
    }

    // ── Decoder helpers ───────────────────────────────────────────────────────

    private OrtValue[] CreateEmptyKvOrtValues()
    {
        // shape [1, numKvHeads, 0, headDim] — zero total elements is valid
        var    kvs   = new OrtValue[_numLayers * 2];
        long[] shape = { 1, _numKvHeads, 0, _headDim };
        for (int i = 0; i < kvs.Length; i++)
            kvs[i] = OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(), shape);
        return kvs;
    }

    private static long ArgmaxOrtValue(OrtValue logits, int seqLen)
    {
        // logits: [1, seqLen, vocabSize] flattened row-major, CPU-resident
        var span      = logits.GetTensorDataAsSpan<BFloat16>();
        int vocabSize = span.Length / seqLen;      // batch = 1
        int offset    = (seqLen - 1) * vocabSize;  // last token's logit row

        long  best    = 0;
        float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocabSize; v++)
        {
            float val = (float)span[offset + v];
            if (val > bestVal) { bestVal = val; best = v; }
        }
        return best;
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

    // The decoder must use ORT_ENABLE_EXTENDED, not ORT_ENABLE_ALL.
    // ORT_ENABLE_ALL pattern-matches the Qwen2 BF16 softmax upcast (Cast→Softmax→Cast) and
    // replaces it with a fused CUDA kernel that computes softmax in BF16 — different numerics
    // from the original float32 upcast, causing content-word divergence after ~50 decode steps.
    // ORT_ENABLE_EXTENDED skips attention fusion while still applying GeLU/LayerNorm fusions.
    // The audio encoder keeps ORT_ENABLE_ALL because it needs memory-layout transforms to fit
    // the float32 Conv towers within 24 GB VRAM.
    private static SessionOptions MakeSessionOptions(
        ExecutionProvider ep,
        GraphOptimizationLevel optLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = optLevel;

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
