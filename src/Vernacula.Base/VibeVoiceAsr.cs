using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Encodings.Web;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
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
    private static readonly JsonSerializerOptions RelaxedJsonEscaping = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    // ── File names ────────────────────────────────────────────────────────────

    public const string AudioEncoderFile        = "audio_encoder.onnx";
    public const string DecoderSingleFile       = "decoder_single.onnx";
    public const string DecoderSingleStaticFile = "decoder_single_static.onnx";
    public const string ExportReportFile        = "export-report.json";
    public const string TokenizerFile           = "tokenizer.json";

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

    // The decoder session is kept for the lifetime of this object.
    //
    // The encoder session is optional: when persistEncoder = true (segmented mode)
    // it is loaded once in the constructor and reused across all Transcribe() calls,
    // avoiding 70+ session-load round-trips.  When persistEncoder = false
    // (whole-recording built-in path) the encoder is created and disposed inside
    // each Transcribe() call so its GPU arena (~1–3 GiB) is freed before the
    // long autoregressive decode loop starts, giving the KV cache more headroom.
    private InferenceSession?          _audioEncoder;   // null when not persisted
    private readonly string            _audioEncoderModelPath;
    private readonly bool              _persistEncoder;
    private readonly ExecutionProvider _ep;
    private readonly InferenceSession  _decoder;

    // ── Profiling ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Directory for ORT Chrome-trace JSON files, or null when profiling is off.
    /// Files are named encoder_&lt;timestamp&gt;.json and decoder_&lt;timestamp&gt;.json.
    /// </summary>
    private readonly string? _profileOutputDir;

    // ── Model dimensions ──────────────────────────────────────────────────────

    private readonly int  _numLayers;           // 28
    private readonly int  _numKvHeads;          // 4
    private readonly int  _headDim;             // 128
    private readonly int  _hiddenSize;          // 3584
    private readonly int  _encoderChunkSamples; // from export-report.json acoustic_tokenizer_chunk_size
    private readonly bool _kvCacheIsFloat32;           // true for --f32-kv-cache exports, false for BF16 KV
    private readonly bool _staticKvCache;              // true when decoder_single_static.onnx is in use
    private readonly int  _maxKvTokens;                // pre-allocated KV buffer length (static mode only)
    private readonly bool _audioEmbeddingIsFloat16;    // true when decoder expects float16 (static export artifact)

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

    public VibeVoiceAsr(
        string modelDir,
        ExecutionProvider ep = ExecutionProvider.Auto,
        bool persistEncoder = true,
        bool allowStaticKvCache = true,
        string? profileOutputDir = null)
    {
        if (profileOutputDir is not null)
            Directory.CreateDirectory(profileOutputDir);
        // Load export report
        string reportPath = Path.Combine(modelDir, ExportReportFile);
        using var reportDoc = JsonDocument.Parse(File.ReadAllText(reportPath));
        var report = reportDoc.RootElement;

        _numLayers           = report.GetProperty("num_layers").GetInt32();
        _numKvHeads          = report.GetProperty("num_kv_heads").GetInt32();
        _headDim             = report.GetProperty("head_dim").GetInt32();
        _hiddenSize          = report.GetProperty("hidden_size").GetInt32();
        _encoderChunkSamples = report.GetProperty("acoustic_tokenizer_chunk_size").GetInt32();
        _kvCacheIsFloat32    = report.TryGetProperty("f32_kv_cache", out var f32KvProp)
                               && f32KvProp.GetBoolean();
        _staticKvCache       = allowStaticKvCache
                               && report.TryGetProperty("static_kv_cache", out var staticKvProp)
                               && staticKvProp.GetBoolean();
        _maxKvTokens         = _staticKvCache && report.TryGetProperty("static_kv_max_tokens", out var maxTokProp)
                               ? maxTokProp.GetInt32()
                               : 0;

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

        // Create ORT sessions.
        // Both models need CUDA: audio_encoder.onnx contains BFloat16 MatMul nodes in
        // the multimodal projector that CPU EP does not implement.
        _ep                   = ep;
        _profileOutputDir     = profileOutputDir;
        _persistEncoder       = persistEncoder;
        _audioEncoderModelPath = Path.Combine(modelDir, AudioEncoderFile);

        if (persistEncoder)
            _audioEncoder = new InferenceSession(
                _audioEncoderModelPath,
                OrtSessionBuilder.Create(ep, enableProfiling: profileOutputDir is not null));

        var gpuOpts = OrtSessionBuilder.Create(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            enableProfiling: profileOutputDir is not null);
        string decoderFile = _staticKvCache ? DecoderSingleStaticFile : DecoderSingleFile;
        _decoder = new InferenceSession(Path.Combine(modelDir, decoderFile), gpuOpts);
        _audioEmbeddingIsFloat16 = _decoder.InputMetadata.TryGetValue("audio_embeddings", out var audioMeta)
                                   && audioMeta.ElementDataType == TensorElementType.Float16;
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
        Action<VibeVoiceSegment>? onSegment = null,
        CancellationToken ct = default)
    {
        // 1 — Resample to 24 kHz mono and pad to acoustic tokenizer stride
        float[] audio24k = AudioTo24kMono(rawAudio, sampleRate, channels);
        double durationSeconds = audio24k.Length / (double)SampleRate;
        audio24k = PadToStride(audio24k);

        // 2 — Build suffix token IDs (contains the audio duration)
        long[] suffixIds = BuildSuffixTokenIds(durationSeconds);

        // 3 — Run audio encoder in chunks → BF16 audio embeddings.
        // Note: SessionOptions.ProfileOutputPathPrefix is not wired to the native
        // OrtEnableProfiling call in ORT 1.24.2 managed bindings — ORT always writes
        // to cwd with the default onnxruntime_profile__{timestamp}.json filename.
        // We rename/move the file ourselves after EndProfiling().
        //
        // When the encoder is not persisted (whole-recording built-in path), create
        // a temporary session scoped to this call so its GPU arena is freed before
        // the long autoregressive decode loop begins.
        BFloat16[] audioEmbeddings;
        if (_audioEncoder is not null)
        {
            audioEmbeddings = RunAudioEncoderChunked(audio24k, _audioEncoder);
            if (_profileOutputDir is not null)
            {
                string rawPath = Path.GetFullPath(_audioEncoder.EndProfiling());
                string destPath = Path.Combine(_profileOutputDir, "encoder_" + Path.GetFileName(rawPath));
                File.Move(rawPath, destPath, overwrite: true);
                Console.Error.WriteLine($"[profile] encoder → {destPath}");
            }
        }
        else
        {
            using var tempEncoder = new InferenceSession(
                _audioEncoderModelPath,
                OrtSessionBuilder.Create(_ep, enableProfiling: _profileOutputDir is not null));
            audioEmbeddings = RunAudioEncoderChunked(audio24k, tempEncoder);
            if (_profileOutputDir is not null)
            {
                string rawPath = Path.GetFullPath(tempEncoder.EndProfiling());
                string destPath = Path.Combine(_profileOutputDir, "encoder_" + Path.GetFileName(rawPath));
                File.Move(rawPath, destPath, overwrite: true);
                Console.Error.WriteLine($"[profile] encoder → {destPath}");
            }
            // tempEncoder is disposed here → GPU arena freed before decoder loop
        }
        int numAudioTokens = audioEmbeddings.Length / _hiddenSize;

        // 4 — Run decoder (chunked prefill + greedy decode) → raw JSON text + per-segment token/word data
        var (jsonText, segTokenIds, segTokenLogprobs, segWordLogprobs) = RunDecoder(
            _prefixTokenIds,
            audioEmbeddings,
            numAudioTokens,
            suffixIds,
            prefillChunkTokens,
            maxNewTokens,
            onSegment,
            ct);

        if (_profileOutputDir is not null)
        {
            string rawPath = Path.GetFullPath(_decoder.EndProfiling());
            string destPath = Path.Combine(_profileOutputDir, "decoder_" + Path.GetFileName(rawPath));
            File.Move(rawPath, destPath, overwrite: true);
            Console.Error.WriteLine($"[profile] decoder → {destPath}");
        }

        // 5 — Parse VibeVoice JSON output and attach per-word logprobs
        var rawSegs = ParseOutput(jsonText);
        var result  = new VibeVoiceSegment[rawSegs.Count];
        for (int i = 0; i < rawSegs.Count; i++)
            result[i] = rawSegs[i] with
            {
                TokenIds      = i < segTokenIds.Length ? segTokenIds[i] : [],
                TokenLogprobs = i < segTokenLogprobs.Length ? segTokenLogprobs[i] : [],
                WordLogprobs  = i < segWordLogprobs.Length ? segWordLogprobs[i] : []
            };
        return result;
    }

    public void Dispose()
    {
        _audioEncoder?.Dispose();
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

    private (string json, int[][] segTokenIds, float[][] segTokenLogprobs, float[][] segWordLogprobs) RunDecoder(
        long[] prefixIds,
        BFloat16[] audioEmbeddings,
        int numAudioTokens,
        long[] suffixIds,
        int prefillChunkTokens,
        int maxNewTokens,
        Action<VibeVoiceSegment>? onSegment,
        CancellationToken ct)
    {
        using var runOptions  = new RunOptions();
        using var binding     = _decoder.CreateIoBinding();
        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);

        // Initial KV tensors:
        //   dynamic mode  → shape [1, kv_heads, 0, head_dim] (zero-length, grows via Concat)
        //   static mode   → shape [1, kv_heads, max_tokens, head_dim] (pre-allocated, zero-filled)
        OrtValue[] pastKvs = CreateInitialKvOrtValues();

        // kv_pos: current fill position in the KV buffers (static mode only).
        // For dynamic mode this is unused but still tracked so RunOnce has a uniform signature.
        long kvPos = 0;

        try
        {
            // ── Chunked prefill ───────────────────────────────────────────────

            int    chunkSize = prefillChunkTokens > 0 ? prefillChunkTokens : numAudioTokens;
            int    numChunks = (numAudioTokens + chunkSize - 1) / Math.Max(1, chunkSize);
            long[] noIds     = [];
            long   lastToken   = 0;
            float  lastLogprob = 0f;

            for (int ci = 0; ci < numChunks; ci++)
            {
                ct.ThrowIfCancellationRequested();
                int start = ci * chunkSize;
                int count = Math.Min(chunkSize, numAudioTokens - start);

                long[] pfx = ci == 0             ? prefixIds : noIds;
                long[] sfx = ci == numChunks - 1 ? suffixIds : noIds;

                int seqLen = pfx.Length + count + sfx.Length;
                var prevKvs = pastKvs;
                (lastToken, lastLogprob, pastKvs) = RunOnce(
                    kvPos, pfx, audioEmbeddings, start, count, sfx, pastKvs, binding, runOptions, cudaMemInfo);
                foreach (var kv in prevKvs) kv.Dispose();
                kvPos += seqLen;
            }

            // ── Greedy decode ─────────────────────────────────────────────────

            var    generated     = new List<long>(maxNewTokens);
            var    tokenLogprobs = new List<float>(maxNewTokens);
            long   nextToken     = lastToken;
            float  nextLogprob   = lastLogprob;  // logprob of nextToken, computed at previous step
            long[] tokenBuf      = new long[1];  // reused each step — avoids per-step allocation

            // Streaming: accumulate decoded bytes and emit complete segments via callback.
            List<byte>? streamBytes   = onSegment is not null ? new List<byte>(4096) : null;
            int         streamEmitted = 0;

            for (int step = 0; step < maxNewTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                if (nextToken == _eosTokenId || nextToken == _imEndTokenId) break;
                generated.Add(nextToken);
                tokenLogprobs.Add(nextLogprob);  // store logprob computed at previous RunOnce call

                // Streaming: decode this token's bytes and check for newly closed segments.
                if (streamBytes is not null)
                {
                    AppendTokenBytes(nextToken, streamBytes);

                    // Only parse when a '}' appears in the last few appended bytes.
                    bool mightClose = false;
                    for (int bi = Math.Max(0, streamBytes.Count - 8); bi < streamBytes.Count; bi++)
                        if (streamBytes[bi] == (byte)'}') { mightClose = true; break; }

                    if (mightClose)
                    {
                        string current    = Encoding.UTF8.GetString(streamBytes.ToArray());
                        int    arrayStart = current.IndexOf('[');
                        if (arrayStart >= 0)
                        {
                            var partial = ParsePartialJson(current[arrayStart..]);
                            while (streamEmitted < partial.Count)
                                onSegment!(partial[streamEmitted++]);
                        }
                    }
                }

                tokenBuf[0] = nextToken;
                var prevKvs = pastKvs;
                (nextToken, nextLogprob, pastKvs) = RunOnce(
                    kvPos, tokenBuf, audioEmbeddings, 0, 0, noIds, pastKvs, binding, runOptions, cudaMemInfo);
                foreach (var kv in prevKvs) kv.Dispose();
                kvPos += 1;  // decode step always produces exactly 1 token position
            }

            string jsonText = TokensToText(generated);
            var    segments = ParseOutput(jsonText);
            var    segData  = ComputeSegmentTokenData(generated, tokenLogprobs, jsonText, segments);
            return (jsonText, segData.TokenIds, segData.TokenLogprobs, segData.WordLogprobs);
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
    private (long token, float logprob, OrtValue[] presentKvs) RunOnce(
        long         kvPos,       // Current KV fill position (used only in static-KV mode)
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

        // Re-register outputs every call.
        // Dynamic mode: KV cache shape grows each step — must resize.
        // Static mode:  KV cache shape is fixed (max_tokens) — same size every step,
        //               so ORT's BFC arena reuses the same memory blocks efficiently.
        binding.ClearBoundOutputs();
        binding.BindOutputToDevice("logits", OrtMemoryInfo.DefaultInstance);
        for (int i = 0; i < _numLayers; i++)
        {
            binding.BindOutputToDevice($"present_key_{i}",   cudaMemInfo);
            binding.BindOutputToDevice($"present_value_{i}", cudaMemInfo);
        }

        // kv_pos (static mode only): int64 scalar telling the model where to scatter new K/V.
        using var kvPosOrtVal = _staticKvCache
            ? OrtValue.CreateTensorValueFromMemory(new long[] { kvPos }, new long[0])
            : null;
        if (kvPosOrtVal is not null)
            binding.BindInput("kv_pos", kvPosOrtVal);

        // prefix_input_ids [1, P]
        using var prefixOrtVal = OrtValue.CreateTensorValueFromMemory(
            prefixIds, new long[] { 1, prefixIds.Length });
        binding.BindInput("prefix_input_ids", prefixOrtVal);

        // audio_embeddings [N, hiddenSize] — zero-length during decode steps
        // Declared at method scope so it stays alive until after RunWithBinding.
        int audioElems = audioCount * _hiddenSize;
        OrtValue audioOrtVal;
        Float16[]? audioF16Scratch = null;   // kept alive until after RunWithBinding
        if (_audioEmbeddingIsFloat16)
        {
            if (audioElems > 0)
            {
                // Static model exports with float16 audio_embeddings; convert BF16 → F16.
                audioF16Scratch = new Float16[audioElems];
                int srcBase = audioStart * _hiddenSize;
                for (int j = 0; j < audioElems; j++)
                    audioF16Scratch[j] = (Float16)(float)audioData[srcBase + j];
                audioOrtVal = OrtValue.CreateTensorValueFromMemory(
                    audioF16Scratch, new long[] { audioCount, _hiddenSize });
            }
            else
            {
                audioOrtVal = OrtValue.CreateTensorValueFromMemory(
                    Array.Empty<Float16>(), new long[] { 0, _hiddenSize });
            }
        }
        else
        {
            audioOrtVal = audioElems > 0
                ? OrtValue.CreateTensorValueFromMemory(
                      OrtMemoryInfo.DefaultInstance,
                      new Memory<BFloat16>(audioData, audioStart * _hiddenSize, audioElems),
                      new long[] { audioCount, _hiddenSize })
                : OrtValue.CreateTensorValueFromMemory(
                      Array.Empty<BFloat16>(), new long[] { 0, _hiddenSize });
        }
        // audioOrtVal must stay alive until after RunWithBinding — dispose manually below.
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
        audioOrtVal.Dispose();

        // GetOutputValues() → IDisposableReadOnlyCollection<OrtValue> (a DisposableList).
        // DisposableList has no finalizer, so skipping `using` is safe: GC collects the
        // list shell without touching the OrtValues inside it.
        var outputs = binding.GetOutputValues();

        // outputs[0] = logits (CPU) — read argmax + logprob, then free immediately
        int   seqLen = prefixIds.Length + audioCount + suffixIds.Length;
        var   (token, logprob) = ArgmaxAndLogprobOrtValue(outputs[0], seqLen);
        outputs[0].Dispose();

        // outputs[1..56] = present_key/value (CUDA) — take ownership
        var presentKvs = new OrtValue[_numLayers * 2];
        for (int i = 0; i < _numLayers * 2; i++)
            presentKvs[i] = outputs[i + 1];

        return (token, logprob, presentKvs);
    }

    // ── Decoder helpers ───────────────────────────────────────────────────────

    private OrtValue[] CreateInitialKvOrtValues()
    {
        // Element type must match the KV cache dtype baked into the exported model.
        var kvs = new OrtValue[_numLayers * 2];

        if (_staticKvCache)
        {
            // Static mode: pre-allocate full-size zero-filled buffers.
            // Shape [1, numKvHeads, maxKvTokens, headDim] — fixed for all steps.
            // The CLR zero-initialises all arrays; ORT will copy them to CUDA on first use.
            long[] shape    = { 1, _numKvHeads, _maxKvTokens, _headDim };
            long   elements = shape[0] * shape[1] * shape[2] * shape[3];
            for (int i = 0; i < kvs.Length; i++)
                kvs[i] = _kvCacheIsFloat32
                    ? OrtValue.CreateTensorValueFromMemory(new float[elements],    shape)
                    : OrtValue.CreateTensorValueFromMemory(new BFloat16[elements], shape);
        }
        else
        {
            // Dynamic mode: zero-length initial tensors; shape grows via Concat each step.
            long[] shape = { 1, _numKvHeads, 0, _headDim };
            for (int i = 0; i < kvs.Length; i++)
                kvs[i] = _kvCacheIsFloat32
                    ? OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(),    shape)
                    : OrtValue.CreateTensorValueFromMemory(Array.Empty<BFloat16>(), shape);
        }

        return kvs;
    }

    /// <summary>
    /// Returns the argmax token id and its log-probability (log softmax of the last
    /// position's logits).  Mirrors the Parakeet approach: logprob = logit[best] - logsumexp.
    /// </summary>
    private (long token, float logprob) ArgmaxAndLogprobOrtValue(OrtValue logits, int seqLen)
    {
        // logits: [1, seqLen, vocabSize] flattened row-major, CPU-resident.
        // Static model outputs float16 logits; dynamic model outputs bfloat16.
        if (_audioEmbeddingIsFloat16)
        {
            var span      = logits.GetTensorDataAsSpan<Float16>();
            int vocabSize = span.Length / seqLen;
            int offset    = (seqLen - 1) * vocabSize;
            long  best    = 0;
            float bestVal = float.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                float val = (float)span[offset + v];
                if (val > bestVal) { bestVal = val; best = v; }
            }
            // logsumexp (stable): bestVal is already the max
            double sumExp = 0.0;
            for (int v = 0; v < vocabSize; v++)
                sumExp += Math.Exp((float)span[offset + v] - bestVal);
            return (best, (float)-Math.Log(sumExp));  // = bestVal - bestVal - log(sumExp)
        }
        else
        {
            var span      = logits.GetTensorDataAsSpan<BFloat16>();
            int vocabSize = span.Length / seqLen;
            int offset    = (seqLen - 1) * vocabSize;
            long  best    = 0;
            float bestVal = float.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                float val = (float)span[offset + v];
                if (val > bestVal) { bestVal = val; best = v; }
            }
            double sumExp = 0.0;
            for (int v = 0; v < vocabSize; v++)
                sumExp += Math.Exp((float)span[offset + v] - bestVal);
            return (best, (float)-Math.Log(sumExp));
        }
    }

    // ── Token decoding ────────────────────────────────────────────────────────

    private void AppendTokenBytes(long tokenId, List<byte> buffer)
    {
        int iid = (int)tokenId;
        if (_addedTokenContent.TryGetValue(iid, out string? special))
        {
            buffer.AddRange(Encoding.UTF8.GetBytes(special));
            return;
        }
        string? raw = iid >= 0 && iid < _idToToken.Length ? _idToToken[iid] : null;
        if (raw is null) return;
        foreach (char ch in raw)
            if (_byteLevelDecode.TryGetValue(ch, out byte b))
                buffer.Add(b);
    }

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

    // ── Per-segment word logprob extraction ──────────────────────────────────

    /// <summary>
    /// For each parsed segment, locate the generated decoder tokens that produced its
    /// content span, returning both token-level and word-level confidence arrays.
    ///
    /// Strategy:
    ///   1. Replay AppendTokenBytes to build a char-to-token-index map over fullJson.
    ///   2. For each segment, locate its "content" field value in fullJson (sequential
    ///      search — segments appear in generation order).
    ///   3. For each whitespace-delimited word in the content, average the logprobs of
    ///      every unique token that contributed at least one character to that word.
    /// </summary>
    private (int[][] TokenIds, float[][] TokenLogprobs, float[][] WordLogprobs) ComputeSegmentTokenData(
        List<long>                     generated,
        List<float>                    tokenLogprobs,
        string                         fullJson,
        IReadOnlyList<VibeVoiceSegment> segments)
    {
        var tokenIdsResult      = new int[segments.Count][];
        var tokenLogprobsResult = new float[segments.Count][];
        var wordLogprobsResult  = new float[segments.Count][];
        for (int i = 0; i < segments.Count; i++)
        {
            tokenIdsResult[i]      = [];
            tokenLogprobsResult[i] = [];
            wordLogprobsResult[i]  = [];
        }

        if (generated.Count == 0 || segments.Count == 0)
            return (tokenIdsResult, tokenLogprobsResult, wordLogprobsResult);

        // Build a char-to-token-index map: charToToken[c] = index of the token in `generated`
        // whose decoded output includes the character at position c in fullJson.
        var charToToken = Enumerable.Repeat(-1, fullJson.Length).ToArray();
        {
            int charPos = 0;
            var tempBuf = new List<byte>(16);
            for (int ti = 0; ti < generated.Count && charPos < fullJson.Length; ti++)
            {
                tempBuf.Clear();
                AppendTokenBytes(generated[ti], tempBuf);
                if (tempBuf.Count == 0) continue;
                string tokenStr = Encoding.UTF8.GetString(tempBuf.ToArray());
                for (int k = 0; k < tokenStr.Length && charPos < charToToken.Length; k++)
                    charToToken[charPos++] = ti;
            }
        }

        // Locate each segment's content value in the JSON and compute word-level logprobs.
        // Match against the serialized JSON string, not the decoded content length, so
        // escaped characters like \" and \\ stay aligned with the generated token span.
        // The JSON produced by VibeVoice has used PascalCase keys in practice, but we
        // accept either casing here because older comments/docs referred to lowercase.
        const string contentKeyPascal = "\"Content\":";
        const string contentKeyCamel  = "\"content\":";
        int searchFrom = 0;

        for (int si = 0; si < segments.Count; si++)
        {
            string content = segments[si].Content;
            if (string.IsNullOrEmpty(content)) continue;

            int keyIdxPascal = fullJson.IndexOf(contentKeyPascal, searchFrom, StringComparison.Ordinal);
            int keyIdxCamel  = fullJson.IndexOf(contentKeyCamel,  searchFrom, StringComparison.Ordinal);
            int keyIdx;
            int keyLength;

            if (keyIdxPascal >= 0 && (keyIdxCamel < 0 || keyIdxPascal < keyIdxCamel))
            {
                keyIdx    = keyIdxPascal;
                keyLength = contentKeyPascal.Length;
            }
            else
            {
                keyIdx    = keyIdxCamel;
                keyLength = contentKeyCamel.Length;
            }

            if (keyIdx < 0) break;

            string serializedContent = JsonSerializer.Serialize(content, RelaxedJsonEscaping);
            int serializedContentIdx = fullJson.IndexOf(
                serializedContent, keyIdx + keyLength, StringComparison.Ordinal);
            if (serializedContentIdx < 0)
            {
                searchFrom = keyIdx + keyLength;
                continue;
            }

            int contentCharStart = serializedContentIdx + 1; // skip opening quote
            int contentCharEnd   = serializedContentIdx + serializedContent.Length - 1; // exclude closing quote
            searchFrom = serializedContentIdx + serializedContent.Length;

            var (decodedCharStarts, decodedCharEnds) = BuildSerializedJsonCharMap(serializedContent);

            var tokenIndices = new List<int>();
            int prevTi = -1;
            for (int c = contentCharStart; c < contentCharEnd && c < charToToken.Length; c++)
            {
                int ti = charToToken[c];
                if (ti != prevTi && ti >= 0 && ti < generated.Count)
                {
                    tokenIndices.Add(ti);
                    prevTi = ti;
                }
            }

            tokenIdsResult[si] = tokenIndices.Select(ti => checked((int)generated[ti])).ToArray();
            tokenLogprobsResult[si] = tokenIndices
                .Where(ti => ti >= 0 && ti < tokenLogprobs.Count)
                .Select(ti => tokenLogprobs[ti])
                .ToArray();

            // Split into words and compute the mean logprob of tokens covering each word.
            string[] words       = content.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var      wordLogprobs = new float[words.Length];
            int      wordOffset  = 0;

            for (int wi = 0; wi < words.Length; wi++)
            {
                int wordPosInContent = content.IndexOf(words[wi], wordOffset, StringComparison.Ordinal);
                if (wordPosInContent < 0) { wordOffset += words[wi].Length; continue; }

                int wordCharEnd = wordPosInContent + words[wi].Length;
                if (wordPosInContent >= decodedCharStarts.Length || wordCharEnd > decodedCharEnds.Length)
                {
                    wordOffset = wordPosInContent + words[wi].Length;
                    continue;
                }

                int wcStart  = contentCharStart + decodedCharStarts[wordPosInContent];
                int wcEnd    = contentCharStart + decodedCharEnds[wordCharEnd - 1];
                wordOffset   = wordPosInContent + words[wi].Length;

                // Average logprobs of unique tokens covering chars [wcStart, wcEnd).
                float sum   = 0f;
                int   count = 0;
                prevTi = -1;
                for (int c = wcStart; c < wcEnd && c < charToToken.Length; c++)
                {
                    int ti = charToToken[c];
                    if (ti >= 0 && ti != prevTi && ti < tokenLogprobs.Count)
                    {
                        sum  += tokenLogprobs[ti];
                        count++;
                        prevTi = ti;
                    }
                }
                wordLogprobs[wi] = count > 0 ? sum / count : 0f;
            }

            wordLogprobsResult[si] = wordLogprobs;
        }

        return (tokenIdsResult, tokenLogprobsResult, wordLogprobsResult);
    }

    private static (int[] starts, int[] ends) BuildSerializedJsonCharMap(string serializedContent)
    {
        if (serializedContent.Length < 2)
            return ([], []);

        var starts = new List<int>(serializedContent.Length);
        var ends   = new List<int>(serializedContent.Length);

        for (int i = 1; i < serializedContent.Length - 1;)
        {
            starts.Add(i - 1); // offsets relative to the first content character

            if (serializedContent[i] == '\\' && i + 1 < serializedContent.Length - 1)
            {
                if (serializedContent[i + 1] == 'u' && i + 5 < serializedContent.Length)
                    i += 6;
                else
                    i += 2;
            }
            else
            {
                i += 1;
            }

            ends.Add(i - 1); // exclusive offset relative to the first content character
        }

        return ([.. starts], [.. ends]);
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

    // The decoder must use ORT_ENABLE_EXTENDED, not ORT_ENABLE_ALL.
    // ORT_ENABLE_ALL pattern-matches the Qwen2 BF16 softmax upcast (Cast→Softmax→Cast) and
    // replaces it with a fused CUDA kernel that computes softmax in BF16 — different numerics
    // from the original float32 upcast, causing content-word divergence after ~50 decode steps.
    // ORT_ENABLE_EXTENDED skips attention fusion while still applying GeLU/LayerNorm fusions.
    // The audio encoder keeps ORT_ENABLE_ALL because it needs memory-layout transforms to fit
    // the float32 Conv towers within 24 GB VRAM.

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static long[] ReadLongArray(JsonElement el)
        => el.EnumerateArray().Select(e => e.GetInt64()).ToArray();
}

// ── Output types ─────────────────────────────────────────────────────────────

/// <summary>A single diarized, timestamped transcription segment from VibeVoice-ASR.</summary>
public sealed record VibeVoiceSegment(double Start, double End, int Speaker, string Content)
{
    /// <summary>
    /// Raw VibeVoice decoder token ids that contributed characters to this segment's Content.
    /// </summary>
    public IReadOnlyList<int> TokenIds { get; init; } = [];

    /// <summary>
    /// Per-token log-probabilities aligned 1:1 with <see cref="TokenIds"/>.
    /// </summary>
    public IReadOnlyList<float> TokenLogprobs { get; init; } = [];

    /// <summary>
     /// Per-word log-probabilities (one entry per whitespace-delimited word in Content).
     /// Computed from the decoder logits at generation time using the same log-softmax
    /// formula as Parakeet.  Empty when not yet computed (e.g. streaming callbacks).
    /// </summary>
    public IReadOnlyList<float> WordLogprobs { get; init; } = [];
}

/// <summary>Internal DTO matching the JSON schema VibeVoice generates.</summary>
file sealed class VibeVoiceRawSegment
{
    [JsonPropertyName("Start")]   public double  Start   { get; set; }
    [JsonPropertyName("End")]     public double  End     { get; set; }
    [JsonPropertyName("Speaker")] public int     Speaker { get; set; }
    [JsonPropertyName("Content")] public string? Content { get; set; }
}
