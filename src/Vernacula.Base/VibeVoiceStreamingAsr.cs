using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// VibeVoice-ASR-Streaming: chunked speaker-attributed ASR, run the way upstream's
/// <c>streaming_generate</c> runs it (see docs/dev/vibevoice_asr_streaming_investigation.md).
///
/// Package (from scripts/vibevoice_streaming_export/export_gqa.py):
///   audio_encoder.onnx  input_values [1,T] (float16) → audio_embeddings [N,hidden] (float32),
///                       run once per fixed window (chunk + lookahead frames), cold each time.
///   decoder_gqa.onnx    prefix_input_ids [1,P] + audio_embeddings [N,hidden] (float16) +
///                       suffix_input_ids [1,S] + seqlens_k [1] + total_sequence_length [1]
///                       + past_key/value_i [1,KH,max,HD] (float16)
///                       → logits [1,seq,vocab] (float16) + present_key/value_i.
///
/// The decoder uses GroupQueryAttention with a shared cache buffer: past_key_i and
/// present_key_i are bound to the SAME device tensor, allocated once, and the kernel updates
/// it in place. Attention costs only the filled length, so VRAM is flat in recording length
/// and throughput does not decay (docs/dev/vibevoice_asr_streaming_investigation.md, Run 12).
/// The buffer ceiling bounds job length; exceeding it raises before any work is done.
///
/// Loop: prefill the prompt once; then for every window feed
/// [speech_start] + frames + [speech_end], greedy-decode until &lt;|text_chunk_end|&gt; or EOS,
/// then feed &lt;|text_chunk_end|&gt; itself so the cache always ends on it. The KV cache lives
/// on the device and grows for the whole recording; upstream never evicts.
/// </summary>
public sealed class VibeVoiceStreamingAsr : IDisposable
{
    public const string AudioEncoderFile  = "audio_encoder.onnx";
    public const string DecoderGqaFile    = "decoder_gqa.onnx";
    public const string ExportReportFile  = "export-report.json";
    public const string TokenizerFile     = "tokenizer.json";

    private readonly InferenceSession _audioEncoder;
    private readonly InferenceSession _decoder;

    private readonly int  _numLayers, _numKvHeads, _headDim, _hiddenSize;
    private readonly int  _maxKvTokens;          // cache ceiling baked into the export
    private readonly bool _encoderWantsFloat16;
    private bool _wantLogprobs;

    /// <summary>Longest recording this package can transcribe, from its cache ceiling.</summary>
    public double MaxAudioSeconds => _maxKvTokens / PositionsPerSecond;

    /// <summary>
    /// Cache positions consumed per second of audio: each hop contributes the speech markers,
    /// the window's frames and the text generated for it. Measured at ~16.0 (Run 14); the
    /// text share varies with speech density, so this is an estimate used for messages only.
    /// </summary>
    private const double PositionsPerSecond = 16.0;

    // Streaming constants from export-report.json["streaming"]
    public int SampleRate     { get; }
    public int WindowSamples  { get; }   // (chunk + lookahead) frames × samples/frame
    public int HopSamples     { get; }   // chunk frames × samples/frame
    public int FramesPerWindow { get; }

    // Tokens from export-report.json["tokenizer"]
    private readonly long[] _promptTokenIds;
    private readonly long[] _promptHotwordsHeadIds;
    private readonly long[] _promptTailIds;
    private readonly long   _speechStartId, _speechEndId, _textChunkEndId, _eosTokenId;

    private readonly string?[]               _idToToken;
    private readonly Dictionary<int, string> _addedTokenContent;
    private readonly Dictionary<char, byte>  _byteLevelDecode;

    public VibeVoiceStreamingAsr(string modelDir, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        using var reportDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelDir, ExportReportFile)));
        var report = reportDoc.RootElement;
        if (!report.TryGetProperty("streaming", out var st))
            throw new InvalidDataException($"{ExportReportFile} in {modelDir} has no \"streaming\" section; this is not a VibeVoice-ASR-Streaming package.");
        if (!report.TryGetProperty("attention", out var attn) || attn.GetString() != "GroupQueryAttention")
            throw new InvalidDataException(
                $"{modelDir} was exported without GroupQueryAttention. Re-export with " +
                "scripts/vibevoice_streaming_export/export_gqa.py.");

        _numLayers        = report.GetProperty("num_layers").GetInt32();
        _numKvHeads       = report.GetProperty("num_kv_heads").GetInt32();
        _headDim          = report.GetProperty("head_dim").GetInt32();
        _hiddenSize       = report.GetProperty("hidden_size").GetInt32();
        _maxKvTokens = report.GetProperty("static_kv_max_tokens").GetInt32();

        SampleRate      = st.GetProperty("sample_rate").GetInt32();
        WindowSamples   = st.GetProperty("window_samples").GetInt32();
        HopSamples      = st.GetProperty("hop_samples").GetInt32();
        FramesPerWindow = st.GetProperty("chunk_frames").GetInt32() + st.GetProperty("lookahead_frames").GetInt32();

        var tok = report.GetProperty("tokenizer");
        _promptTokenIds        = ReadLongArray(tok.GetProperty("prompt_token_ids"));
        _promptHotwordsHeadIds = ReadLongArray(tok.GetProperty("prompt_hotwords_head_token_ids"));
        _promptTailIds         = ReadLongArray(tok.GetProperty("prompt_tail_token_ids"));
        _speechStartId  = tok.GetProperty("speech_start_id").GetInt64();
        _speechEndId    = tok.GetProperty("speech_end_id").GetInt64();
        _textChunkEndId = tok.GetProperty("text_chunk_end_id").GetInt64();
        _eosTokenId     = tok.GetProperty("eos_token_id").GetInt64();

        (_idToToken, _addedTokenContent) = VibeVoiceAsr.LoadTokenizerVocab(Path.Combine(modelDir, TokenizerFile));
        _byteLevelDecode = VibeVoiceAsr.BuildByteLevelDecode();

        // EXTENDED, not ALL: the non-streaming port measured ORT_ENABLE_ALL replacing the
        // float32 softmax upcast with a lower-precision fused kernel and diverging earlier.
        _audioEncoder = new InferenceSession(Path.Combine(modelDir, AudioEncoderFile),
            OrtSessionBuilder.Create(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED));
        _decoder = new InferenceSession(Path.Combine(modelDir, DecoderGqaFile),
            OrtSessionBuilder.Create(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED));
        _encoderWantsFloat16 =
            _audioEncoder.InputMetadata.TryGetValue("input_values", out var encMeta)
            && encMeta.ElementDataType == TensorElementType.Float16;
    }

    /// <summary>
    /// Transcribe a recording chunk by chunk. Each chunk covers <see cref="HopSamples"/> of
    /// audio (2.93 s at the shipped configuration) and is reported through
    /// <paramref name="onChunk"/> as soon as the model ends it.
    /// </summary>
    /// <param name="hotwordTokenIds">
    ///   Optional token ids of a comma-separated hotword string (upstream's
    ///   <c>context_info</c>). The package carries no BPE encoder, so the caller tokenizes.
    /// </param>
    public IReadOnlyList<VibeVoiceStreamingChunk> Transcribe(
        float[] rawAudio, int sampleRate, int channels,
        long[]? hotwordTokenIds = null,
        int maxNewTokensPerChunk = 256,
        bool computeLogprobs = false,
        Action<VibeVoiceStreamingChunk>? onChunk = null,
        CancellationToken ct = default)
    {
        _wantLogprobs = computeLogprobs;
        float[] audio = VibeVoiceAsr.AudioTo24kMono(rawAudio, sampleRate, channels);
        int totalChunks = audio.Length == 0 ? 0 : (audio.Length + HopSamples - 1) / HopSamples;
        var chunks = new List<VibeVoiceStreamingChunk>(totalChunks);

        double estimated = audio.Length / (double)SampleRate * PositionsPerSecond;
        if (estimated > _maxKvTokens)
            throw new InvalidOperationException(
                $"Recording is about {audio.Length / (double)SampleRate / 60:F1} minutes, beyond " +
                $"this model's {MaxAudioSeconds / 60:F0}-minute cache ceiling " +
                $"({_maxKvTokens} positions). Re-export with a larger --max-tokens, or split the recording.");

        // The cache must live on the device. Allocating it from OrtAllocator.DefaultInstance
        // puts it in host memory, and every step then copies the whole buffer both ways --
        // measured at 9x slower than the Python harness before this was fixed.
        using var cudaMemInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        using var deviceAlloc = new OrtAllocator(_decoder, cudaMemInfo);
        using var binding    = _decoder.CreateIoBinding();
        using var runOptions = new RunOptions();
        var kvBuffers = CreateSharedKvBuffers(deviceAlloc);
        long kvPos    = 0;
        var window  = new float[WindowSamples];
        var emptyAudio = Array.Empty<Float16>();
        var emptyIds   = Array.Empty<long>();
        try
        {
            // 1 — prompt prefill
            long[] prompt = hotwordTokenIds is { Length: > 0 }
                ? [.. _promptHotwordsHeadIds, .. hotwordTokenIds, .. _promptTailIds]
                : _promptTokenIds;
            Step(prompt, emptyAudio, 0, emptyIds, kvBuffers, ref kvPos, binding, runOptions);

            // 2 — one window per hop, zero-padded at the end of the recording
            for (int ci = 0; ci < totalChunks; ci++)
            {
                ct.ThrowIfCancellationRequested();
                int start = ci * HopSamples;
                int n = Math.Min(WindowSamples, audio.Length - start);
                Array.Copy(audio, start, window, 0, n);
                Array.Clear(window, n, WindowSamples - n);
                Float16[] frames = EncodeWindow(window);
                int numFrames = frames.Length / _hiddenSize;

                long next = Step([_speechStartId], frames, numFrames, [_speechEndId],
                                 kvBuffers, ref kvPos, binding, runOptions).token;
                var ids = new List<long>();
                var logprobs = new List<float>();
                for (int t = 0; t < maxNewTokensPerChunk; t++)
                {
                    if (next == _textChunkEndId || next == _eosTokenId) break;
                    ids.Add(next);
                    var (tok, lp) = Step([next], emptyAudio, 0, emptyIds, kvBuffers, ref kvPos, binding, runOptions);
                    logprobs.Add(lp);
                    next = tok;
                }
                Step([_textChunkEndId], emptyAudio, 0, emptyIds, kvBuffers, ref kvPos, binding, runOptions);

                var chunk = new VibeVoiceStreamingChunk(
                    Index: ci,
                    Start: start / (double)SampleRate,
                    End:   Math.Min(start + HopSamples, audio.Length) / (double)SampleRate,
                    Text:  DecodeSkippingSpecial(ids),
                    TokenIds: ids.ToArray(),
                    TokenLogprobs: logprobs.ToArray());
                chunks.Add(chunk);
                onChunk?.Invoke(chunk);
            }
        }
        finally
        {
            foreach (var kv in kvBuffers) kv.Dispose();
        }
        return chunks;
    }

    /// <summary>
    /// Folds chunks into speaker turns as they arrive, so a caller can show the transcript
    /// while the recording is still being decoded — which is the whole point of this model.
    ///
    /// Upstream emits <c>\n Speaker N:</c> markers inline and a turn runs until the next
    /// marker, so the newest segment stays open and grows with each chunk. Callers should
    /// re-read <see cref="Segments"/> after every <see cref="Add"/>: entries beyond what they
    /// have already shown are new turns, and the last entry's text may have changed.
    /// </summary>
    public sealed class SegmentAssembler
    {
        private readonly List<VibeVoiceSegment> _closed = [];
        private readonly StringBuilder _open = new();
        private int    _speaker = -1;
        private double _start, _end;
        private bool   _started;

        /// <summary>Every turn so far. The last entry is still open and may grow.</summary>
        public IReadOnlyList<VibeVoiceSegment> Segments
        {
            get
            {
                string t = _open.ToString().Trim();
                if (t.Length == 0) return _closed;
                return [.. _closed, new VibeVoiceSegment(_start, Math.Max(_end, _start), _speaker, t)];
            }
        }

        public void Add(VibeVoiceStreamingChunk chunk)
        {
            if (!_started) { _start = chunk.Start; _started = true; }
            int pos = 0;
            foreach (Match m in SpeakerMarker.Matches(chunk.Text))
            {
                _open.Append(chunk.Text, pos, m.Index - pos);
                double at = At(chunk, m.Index);
                Close(at);
                _speaker = int.Parse(m.Groups[1].Value);
                _start   = at;
                pos      = m.Index + m.Length;
            }
            _open.Append(chunk.Text, pos, chunk.Text.Length - pos);
            _end = chunk.End;
        }

        /// <summary>Closes the final turn and returns every segment.</summary>
        public IReadOnlyList<VibeVoiceSegment> Finish()
        {
            Close(_end);
            return _closed;
        }

        private void Close(double end)
        {
            string t = _open.ToString().Trim();
            if (t.Length > 0) _closed.Add(new VibeVoiceSegment(_start, Math.Max(end, _start), _speaker, t));
            _open.Clear();
        }

        // The model gives no timestamps, so a turn boundary inside a chunk is placed
        // proportionally to where it falls in that chunk's text: accurate to within one hop.
        private static double At(VibeVoiceStreamingChunk c, int pos) =>
            c.Text.Length == 0 ? c.Start : c.Start + (c.End - c.Start) * pos / c.Text.Length;
    }

    /// <summary>Folds a completed chunk list into speaker turns. Batch form of <see cref="SegmentAssembler"/>.</summary>
    public static IReadOnlyList<VibeVoiceSegment> ToSegments(IReadOnlyList<VibeVoiceStreamingChunk> chunks)
    {
        var asm = new SegmentAssembler();
        foreach (var c in chunks) asm.Add(c);
        return asm.Finish();
    }

    private static readonly Regex SpeakerMarker = new(@"\s*Speaker (\d+):", RegexOptions.Compiled);

    public void Dispose()
    {
        _audioEncoder.Dispose();
        _decoder.Dispose();
    }

    // ── Audio encoder ─────────────────────────────────────────────────────────

    /// <summary>Encodes one window and returns [frames * hidden] float16, the dtype the decoder wants.</summary>
    private Float16[] EncodeWindow(float[] window)
    {
        OrtValue input;
        Float16[]? scratch = null;
        if (_encoderWantsFloat16)
        {
            scratch = new Float16[window.Length];
            for (int i = 0; i < window.Length; i++) scratch[i] = (Float16)window[i];
            input = OrtValue.CreateTensorValueFromMemory(scratch, [1, window.Length]);
        }
        else
        {
            input = OrtValue.CreateTensorValueFromMemory(window, [1, window.Length]);
        }
        using (input)
        using (var runOptions = new RunOptions())
        using (var outputs = _audioEncoder.Run(runOptions,
                   new Dictionary<string, OrtValue> { ["input_values"] = input }, ["audio_embeddings"]))
        {
            // The encoder emits float32 frames whichever precision its towers run in.
            var span = outputs[0].GetTensorDataAsSpan<float>();
            var f16 = new Float16[span.Length];
            for (int i = 0; i < span.Length; i++) f16[i] = (Float16)span[i];
            GC.KeepAlive(scratch);
            return f16;
        }
    }

    // ── Decoder ───────────────────────────────────────────────────────────────

    /// <summary>
    /// One device tensor per layer per side, allocated once at the export's ceiling and never
    /// resized. Each is bound as BOTH past_key_i and present_key_i so GroupQueryAttention
    /// writes the new keys in place instead of producing a larger tensor every step.
    /// </summary>
    private OrtValue[] CreateSharedKvBuffers(OrtAllocator allocator)
    {
        long[] shape = [1, _numKvHeads, _maxKvTokens, _headDim];
        var kvs = new OrtValue[_numLayers * 2];
        for (int i = 0; i < kvs.Length; i++)
            kvs[i] = OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float16, shape);
        return kvs;
    }

    /// <summary>
    /// One decoder call. <paramref name="kvPos"/> is the number of positions already in the
    /// cache; the model needs the resulting total both as <c>total_sequence_length</c> and,
    /// minus one, as <c>seqlens_k</c>.
    /// </summary>
    private (long token, float logprob) Step(
        long[] prefixIds, Float16[] audioData, int audioCount, long[] suffixIds,
        OrtValue[] kvBuffers, ref long kvPos, OrtIoBinding binding, RunOptions runOptions)
    {
        int seqLen = prefixIds.Length + audioCount + suffixIds.Length;
        long total = kvPos + seqLen;
        if (total > _maxKvTokens)
            throw new InvalidOperationException(
                $"KV cache full: {total} positions needed, ceiling is {_maxKvTokens}. " +
                $"This package handles about {MaxAudioSeconds / 60:F0} minutes of audio.");

        binding.ClearBoundInputs();
        binding.ClearBoundOutputs();

        // logits FIRST: GetOutputValues() returns values in binding order, not model order,
        // so binding it after the cache tensors would silently hand back present_key_0.
        binding.BindOutputToDevice("logits", OrtMemoryInfo.DefaultInstance);

        using var prefixVal = OrtValue.CreateTensorValueFromMemory(prefixIds, [1, prefixIds.Length]);
        binding.BindInput("prefix_input_ids", prefixVal);
        using var audioVal = OrtValue.CreateTensorValueFromMemory(
            OrtMemoryInfo.DefaultInstance, new Memory<Float16>(audioData, 0, audioCount * _hiddenSize),
            [audioCount, _hiddenSize]);
        binding.BindInput("audio_embeddings", audioVal);
        using var suffixVal = OrtValue.CreateTensorValueFromMemory(suffixIds, [1, suffixIds.Length]);
        binding.BindInput("suffix_input_ids", suffixVal);
        using var seqlensVal = OrtValue.CreateTensorValueFromMemory(new[] { (int)(total - 1) }, [1]);
        binding.BindInput("seqlens_k", seqlensVal);
        using var totalVal = OrtValue.CreateTensorValueFromMemory(new[] { (int)total }, [1]);
        binding.BindInput("total_sequence_length", totalVal);

        for (int i = 0; i < _numLayers; i++)
        {
            binding.BindInput($"past_key_{i}",    kvBuffers[i * 2]);
            binding.BindInput($"past_value_{i}",  kvBuffers[i * 2 + 1]);
            binding.BindOutput($"present_key_{i}",   kvBuffers[i * 2]);
            binding.BindOutput($"present_value_{i}", kvBuffers[i * 2 + 1]);
        }

        _decoder.RunWithBinding(runOptions, binding);
        var outputs = binding.GetOutputValues();
        var result = ArgmaxAndLogprob(outputs[0], seqLen, _wantLogprobs);
        outputs[0].Dispose();

        kvPos = total;
        return result;
    }

    /// <summary>
    /// Argmax over the last position's logits, and optionally its log-probability.
    ///
    /// The log-probability needs a second pass with a <c>Math.Exp</c> per vocabulary entry —
    /// 152k of them per token, which measured as the single largest cost in the C# loop
    /// (roughly 40% of decode time). It is therefore opt-in: callers that only want text pay
    /// one comparison pass instead.
    /// </summary>
    private static (long token, float logprob) ArgmaxAndLogprob(OrtValue logits, int seqLen, bool wantLogprob)
    {
        var span      = logits.GetTensorDataAsSpan<Float16>();
        int vocabSize = span.Length / seqLen;
        int offset    = (seqLen - 1) * vocabSize;
        long best = 0; float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocabSize; v++)
        {
            float val = (float)span[offset + v];
            if (val > bestVal) { bestVal = val; best = v; }
        }
        if (!wantLogprob) return (best, float.NaN);

        double sumExp = 0.0;
        for (int v = 0; v < vocabSize; v++)
            sumExp += Math.Exp((float)span[offset + v] - bestVal);
        return (best, (float)-Math.Log(sumExp));
    }

    // ── Token decoding (skip_special_tokens=True, as upstream decodes chunk text) ──

    private string DecodeSkippingSpecial(List<long> ids)
    {
        var bytes = new List<byte>(ids.Count * 4);
        foreach (long id in ids)
        {
            int iid = (int)id;
            if (_addedTokenContent.ContainsKey(iid)) continue;
            string? raw = iid >= 0 && iid < _idToToken.Length ? _idToToken[iid] : null;
            if (raw is null) continue;
            foreach (char ch in raw)
                if (_byteLevelDecode.TryGetValue(ch, out byte b)) bytes.Add(b);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private static long[] ReadLongArray(JsonElement el) => el.EnumerateArray().Select(e => e.GetInt64()).ToArray();
}

/// <summary>One streaming chunk: the text the model emitted for one hop of audio.</summary>
public sealed record VibeVoiceStreamingChunk(
    int Index, double Start, double End, string Text, long[] TokenIds, float[] TokenLogprobs);
