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
///
/// Loop: prefill the prompt; then for every window feed
/// [speech_start] + frames + [speech_end], greedy-decode until &lt;|text_chunk_end|&gt; or EOS,
/// then feed &lt;|text_chunk_end|&gt; itself so the cache always ends on it. The KV cache lives
/// on the device and grows for the whole pass; upstream never evicts.
///
/// A recording that needs more cache than the export or the card allows is decoded in several
/// such passes: the cache is reset and the prompt re-prefilled, which bounds VRAM by the pass
/// rather than by the recording. Nothing survives a reset, so the model renumbers its speakers
/// from scratch and <see cref="SegmentAssembler"/> keeps the passes' labels disjoint rather
/// than pretending they line up (issue #150).
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
    private readonly int  _vocabSize;
    private readonly bool _encoderWantsFloat16;

    /// <summary>
    /// The buffer length the decoder graph *requires*, or 0 when it declares the length
    /// symbolically and the runtime may choose. Packages exported before issue #150 baked the
    /// ceiling into the graph, and ORT rejects any other size outright, so they keep the old
    /// behaviour of paying for the whole ceiling on every run.
    /// </summary>
    private readonly int _fixedKvTokens;

    /// <summary>
    /// Longest stretch this package decodes as one pass, from its cache ceiling. Recordings
    /// longer than this are decoded in several passes rather than refused.
    /// </summary>
    public double MaxAudioSeconds => _maxKvTokens / PositionsPerSecond;

    /// <inheritdoc cref="VibeVoiceStreamingBudget.PositionsPerSecond"/>
    private const double PositionsPerSecond = VibeVoiceStreamingBudget.PositionsPerSecond;

    /// <summary>
    /// How much of the estimate to allocate. The estimate is an average and the buffer cannot
    /// grow once the run starts, so a recording denser than average would otherwise fail
    /// part-way through -- which costs the whole job. 2.5x the measured 16.0 clears the
    /// theoretical fixed floor of 9.9 with room for far more text than any measured file
    /// produced, and costs nothing on a card that has the memory anyway.
    /// </summary>
    private const double KvSafetyFactor = 2.5;

    /// <inheritdoc cref="VibeVoiceStreamingBudget.PromptPositionSlack"/>
    private const int PromptPositionSlack = VibeVoiceStreamingBudget.PromptPositionSlack;

    /// <summary>Never allocate a cache too small to hold the prompt and a few windows.</summary>
    private const int MinKvTokens = 2048;

    /// <inheritdoc cref="VibeVoiceStreamingBudget.KvBytesPerPosition"/>
    public long KvBytesPerPosition =>
        VibeVoiceStreamingBudget.KvBytesPerPosition(_numLayers, _numKvHeads, _headDim);

    /// <inheritdoc cref="VibeVoiceStreamingBudget.WorkingSetBytes"/>
    public long WorkingSetBytes => VibeVoiceStreamingBudget.WorkingSetBytes(_hiddenSize);

    /// <summary>
    /// Cache positions to allocate for a recording of <paramref name="audioSeconds"/>, and the
    /// reason if none will do.
    ///
    /// The cache used to be allocated at the export's ceiling on every run. That ceiling exists
    /// so a two-hour recording *can* be transcribed, and paying it up front costs 7.0 GiB on the
    /// 7B — enough on its own to put a 16 GB card over the line before a frame is encoded, for a
    /// one-minute file (issue #150). So the buffer is sized to the recording in hand and only
    /// then clamped to what the card has free.
    ///
    /// When the recording needs more than the clamp allows, the answer is the clamp rather than
    /// a refusal: <see cref="Transcribe"/> resets the cache and re-prefills the prompt when the
    /// next window would not fit, so a recording of any length runs in passes that do fit.
    ///
    /// The estimate is <see cref="PositionsPerSecond"/>, an average; <see cref="KvSafetyFactor"/>
    /// covers speech denser than average, and speech denser still only shortens a pass.
    /// </summary>
    private int ChooseKvTokens(double audioSeconds, bool onDevice)
    {
        long freeBytes = 0;
        if (onDevice)
        {
            var (_, freeMb) = HardwareInfo.GetGpuMemoryMb();
            if (freeMb > 0) freeBytes = freeMb * 1024L * 1024L;
        }
        return PlanKvTokens(audioSeconds, _maxKvTokens, _fixedKvTokens, KvBytesPerPosition,
                            WorkingSetBytes, freeBytes);
    }

    /// <summary>
    /// The arithmetic behind <see cref="ChooseKvTokens"/>, separated from the model so it can be
    /// driven directly by tests: every branch here is a refusal or a clamp that is otherwise only
    /// reachable with a particular card and a particular recording in front of you.
    /// </summary>
    /// <param name="freeBytes">Free device memory, or 0 when NVML could not say.</param>
    /// <param name="fixedKvTokens">
    ///   The length the graph insists on, or 0 when the runtime may choose.
    /// </param>
    internal static int PlanKvTokens(
        double audioSeconds, int maxKvTokens, int fixedKvTokens, long kvBytesPerPosition,
        long workingSetBytes, long freeBytes)
    {
        // Free device memory, less what a run needs beyond the cache. long.MaxValue when NVML
        // could not say: an unknown card is not a reason to allocate the ceiling, but it is no
        // reason to refuse either, so only the recording and the export decide.
        long budget     = freeBytes > 0 ? freeBytes - workingSetBytes : long.MaxValue;
        long affordable = freeBytes <= 0 ? long.MaxValue
                        : budget > 0     ? budget / kvBytesPerPosition
                        : 0;

        // A graph with a baked-in length has to have all of it, whatever the recording needs,
        // and there is nothing to plan: ORT rejects any other size.
        if (fixedKvTokens > 0)
        {
            if (affordable < fixedKvTokens)
                throw new InvalidOperationException(
                    $"Not enough free GPU memory for this package: its cache needs about " +
                    $"{fixedKvTokens * kvBytesPerPosition / (double)(1L << 30):F1} GiB and " +
                    $"{Math.Max(0, budget) / (double)(1L << 30):F1} GiB is free once the model and " +
                    "its working set are accounted for. This package was exported with a fixed " +
                    "cache length, so it pays for its whole context on every run; re-download it " +
                    "to have the cache sized to the recording instead.");
            return fixedKvTokens;
        }

        // The prompt, its hotwords, and the rounding on the last partial window all sit outside
        // the per-second estimate.
        long needed = (long)Math.Ceiling(audioSeconds * PositionsPerSecond) + PromptPositionSlack;
        long cap    = Math.Min(maxKvTokens, affordable);

        // Below the cap this is "size it to the recording". At the cap — a recording longer than
        // the export's context, or than this card can hold — it is "size it to one pass", and the
        // recording is transcribed in as many passes as it takes.
        //
        // A tenth is left unspent when the cap comes from memory rather than from the export.
        // `affordable` is what the memory model says is spare, and that model is an estimate
        // which has been wrong in this direction before (Run 34): spending all of it would put
        // the arena's own growth over the line partway through a long job — the failure this
        // issue opened with, except hours in rather than up front. The export ceiling is not an
        // estimate, so it is spent in full. A recording that fits is given what it needs either
        // way, so the length the settings window promises is still the length that is allocated.
        long usable = affordable < maxKvTokens ? cap - cap / 10 : cap;
        if (needed <= cap) usable = Math.Max(usable, needed);
        long want = Math.Min((long)(needed * KvSafetyFactor), usable);
        want = Math.Max(want, Math.Min(MinKvTokens, cap));

        // The floor is only a floor when the recording has to be split: a short file that fits
        // in less than the floor is one pass and costs what it costs. Below the floor there is
        // nothing worth splitting into, and passes that short would spend most of their cache
        // re-reading the prompt.
        if (want < Math.Min(MinKvTokens, needed))
            throw new InvalidOperationException(
                "Not enough free GPU memory to transcribe with this checkpoint: " +
                $"{Math.Max(0, budget) / (double)(1L << 30):F1} GiB is free once the model and its " +
                "working set are accounted for, which is not enough cache to be worth decoding a " +
                "pass into. Close other GPU applications, or use the 1.5B checkpoint.");

        return (int)want;
    }

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
        _vocabSize   = report.GetProperty("vocab_size").GetInt32();

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
        _fixedKvTokens =
            _decoder.InputMetadata.TryGetValue("past_key_0", out var kvMeta)
            && kvMeta.Dimensions.Length == 4 && kvMeta.Dimensions[2] > 0
                ? kvMeta.Dimensions[2]
                : 0;
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
        float[] audio = VibeVoiceAsr.AudioTo24kMono(rawAudio, sampleRate, channels);
        int totalChunks = audio.Length == 0 ? 0 : (audio.Length + HopSamples - 1) / HopSamples;
        var chunks = new List<VibeVoiceStreamingChunk>(totalChunks);

        // The cache belongs on the device when there is one. Allocating it from
        // OrtAllocator.DefaultInstance puts it in host memory, and a CUDA session then copies
        // the whole buffer both ways every step -- measured at 9x slower than the Python
        // harness before this was fixed. On a CPU session host memory is simply where it goes.
        using var cudaMemInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        OrtAllocator? cudaAlloc = null;
        try
        {
            cudaAlloc = new OrtAllocator(_decoder, cudaMemInfo);
        }
        catch (OnnxRuntimeException)
        {
            // "No requested allocator available" — the decoder session is not on CUDA, so there
            // is no device allocator to take the cache from. Not fatal: the same buffers work in
            // host memory when the session is on the CPU too, and the cost of that is speed
            // rather than correctness. Refusing here would have made a GPU the price of entry.
        }
        using var ownedAlloc = cudaAlloc;
        OrtAllocator deviceAlloc = cudaAlloc ?? OrtAllocator.DefaultInstance;

        // Sized after the allocator is settled, because the budget depends on where the cache
        // lands: VRAM is the constraint on the device, and on the host it is not a constraint
        // worth modelling -- the whole ceiling is 7.0 GiB against system memory measured in
        // tens of gigabytes.
        int kvTokens = ChooseKvTokens(audio.Length / (double)SampleRate, onDevice: cudaAlloc is not null);

        using var binding    = _decoder.CreateIoBinding();
        using var runOptions = new RunOptions();
        var kvBuffers = CreateSharedKvBuffers(deviceAlloc, kvTokens);
        long kvPos    = 0;
        var window  = new float[WindowSamples];
        var emptyAudio = Array.Empty<Float16>();
        var emptyIds   = Array.Empty<long>();
        // The most one window can cost: its two speech markers, its frames, every token the
        // decoder is allowed to generate for it, and the chunk-end token fed back afterwards.
        // A pass ends before a window that would not fit, so the cache never overflows and the
        // planner's average-based estimate only decides how long a pass tends to be.
        long worstCaseWindow = 2 + FramesPerWindow + maxNewTokensPerChunk + 1;

        try
        {
            long[] prompt = hotwordTokenIds is { Length: > 0 }
                ? [.. _promptHotwordsHeadIds, .. hotwordTokenIds, .. _promptTailIds]
                : _promptTokenIds;

            // One window per hop, zero-padded at the end of the recording, in as many passes as
            // the cache takes: the first window of a pass prefills the prompt at position 0, and
            // a pass ends as soon as the next window would not fit (issue #150).
            int pass = -1;
            for (int ci = 0; ci < totalChunks; ci++)
            {
                ct.ThrowIfCancellationRequested();
                if (pass < 0 || kvPos + worstCaseWindow > kvTokens)
                {
                    kvPos = 0;
                    pass++;
                    Step(prompt, emptyAudio, 0, emptyIds, kvBuffers, kvTokens, ref kvPos, binding, runOptions, false);
                    if (kvPos + worstCaseWindow > kvTokens)
                        throw new InvalidOperationException(
                            $"A {kvTokens}-position cache does not hold the prompt and one window " +
                            $"({kvPos} + {worstCaseWindow} positions). Free GPU memory, or use the " +
                            "1.5B checkpoint.");
                }
                int start = ci * HopSamples;
                int n = Math.Min(WindowSamples, audio.Length - start);
                Array.Copy(audio, start, window, 0, n);
                Array.Clear(window, n, WindowSamples - n);
                Float16[] frames = EncodeWindow(window);
                int numFrames = frames.Length / _hiddenSize;

                long next = Step([_speechStartId], frames, numFrames, [_speechEndId],
                                 kvBuffers, kvTokens, ref kvPos, binding, runOptions, false).token;
                var ids = new List<long>();
                // Only collected when asked for: the second pass over the vocabulary costs a
                // Math.Exp per entry. Left empty otherwise rather than filled with NaN, so a
                // consumer cannot mistake "not computed" for a confidence value.
                var logprobs = computeLogprobs ? new List<float>() : null;
                for (int t = 0; t < maxNewTokensPerChunk; t++)
                {
                    if (next == _textChunkEndId || next == _eosTokenId) break;
                    ids.Add(next);
                    var (tok, lp) = Step([next], emptyAudio, 0, emptyIds, kvBuffers, kvTokens, ref kvPos, binding, runOptions, computeLogprobs);
                    logprobs?.Add(lp);
                    next = tok;
                }
                Step([_textChunkEndId], emptyAudio, 0, emptyIds, kvBuffers, kvTokens, ref kvPos, binding, runOptions, false);

                var (chunkText, tokenCharEnds) = DecodeWithOffsets(ids);
                var chunk = new VibeVoiceStreamingChunk(
                    Index: ci,
                    Pass:  pass,
                    Start: start / (double)SampleRate,
                    End:   Math.Min(start + HopSamples, audio.Length) / (double)SampleRate,
                    Text:  chunkText,
                    TokenIds: ids.ToArray(),
                    TokenLogprobs: logprobs?.ToArray() ?? [],
                    TokenCharEnds: tokenCharEnds);
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
    ///
    /// A long recording is decoded in passes with an empty cache between them, so the model's
    /// numbering restarts and its "Speaker 1" after a boundary is not the one before it. The
    /// numbering is therefore shifted past everything already used: the passes get disjoint
    /// labels, and deciding that two of them are the same person is left to whoever can
    /// actually tell (issue #150).
    /// </summary>
    public sealed class SegmentAssembler
    {
        private readonly List<VibeVoiceSegment> _closed = [];
        private readonly StringBuilder _open = new();
        private readonly List<int>   _openTokens   = [];
        private readonly List<float> _openLogprobs = [];
        private int    _speaker = -1;
        private double _start, _end;
        private bool   _started;
        private int    _pass;
        /// <summary>
        /// Labels, in order of first appearance, keyed by the pass that produced them: the same
        /// "Speaker 1" either side of a cache reset is two different people and gets two labels.
        ///
        /// The model's own numbering cannot be handed out as-is. It restarts at every boundary,
        /// and consumers count on labels being dense and ascending — the results database keys
        /// speakers by row id and derives the tag back from it, so a gap makes the tag and the
        /// name disagree. Entries are created when a turn is first built, which is also when the
        /// <see cref="Segments"/> preview builds the open turn: the open turn is always the
        /// newest one, so previewing it assigns the label it was going to get anyway.
        /// </summary>
        private readonly Dictionary<(int Pass, int Speaker), int> _labels = [];

        /// <summary>Every turn so far. The last entry is still open and may grow.</summary>
        public IReadOnlyList<VibeVoiceSegment> Segments
        {
            get
            {
                string t = _open.ToString().Trim();
                if (t.Length == 0) return _closed;
                return [.. _closed, Build(t, Math.Max(_end, _start))];
            }
        }

        public void Add(VibeVoiceStreamingChunk chunk)
        {
            if (!_started) { _start = chunk.Start; _started = true; }
            if (chunk.Pass != _pass)
            {
                // The cache was reset here, so the turn in progress cannot continue across the
                // boundary and the numbering on the far side means something else.
                Close(chunk.Start);
                _pass    = chunk.Pass;
                _speaker = -1;
                _start   = chunk.Start;
            }
            int pos = 0;
            foreach (Match m in SpeakerMarker.Matches(chunk.Text))
            {
                Take(chunk, pos, m.Index);
                double at = At(chunk, m.Index);
                Close(at);
                _speaker = int.Parse(m.Groups[1].Value);
                _start   = at;
                // The marker's own tokens belong to no turn: they are structure, not speech.
                pos = m.Index + m.Length;
            }
            Take(chunk, pos, chunk.Text.Length);
            _end = chunk.End;
        }

        /// <summary>Closes the final turn and returns every segment.</summary>
        public IReadOnlyList<VibeVoiceSegment> Finish()
        {
            Close(_end);
            return _closed;
        }

        /// <summary>Appends one character range of a chunk, with the tokens that produced it.</summary>
        private void Take(VibeVoiceStreamingChunk chunk, int from, int to)
        {
            if (to <= from) return;
            _open.Append(chunk.Text, from, to - from);
            if (chunk.TokenCharEnds.Length != chunk.TokenIds.Length) return;   // offsets unavailable
            for (int i = 0; i < chunk.TokenIds.Length; i++)
            {
                int end   = chunk.TokenCharEnds[i];
                int begin = i == 0 ? 0 : chunk.TokenCharEnds[i - 1];
                // A token counts for this range when any of its characters fall inside it.
                if (end <= from || begin >= to) continue;
                _openTokens.Add((int)chunk.TokenIds[i]);
                if (i < chunk.TokenLogprobs.Length) _openLogprobs.Add(chunk.TokenLogprobs[i]);
            }
        }

        /// <summary>
        /// The label for the turn being built, allocating one if this speaker has not been seen
        /// in this pass before. Text the model attributed to nobody keeps the -1 callers already
        /// fold onto 0 in the first pass, but still consumes its label, so the first speaker of
        /// a later pass cannot land on top of it.
        /// </summary>
        private int Label()
        {
            var key = (_pass, _speaker);
            if (!_labels.TryGetValue(key, out int label))
            {
                label = _labels.Count;
                _labels[key] = label;
            }
            return _speaker < 0 && _pass == 0 ? -1 : label;
        }

        private VibeVoiceSegment Build(string text, double end) =>
            new(_start, end, Label(), text)
            {
                TokenIds      = _openTokens.ToArray(),
                TokenLogprobs = _openLogprobs.Count == _openTokens.Count ? _openLogprobs.ToArray() : [],
            };

        private void Close(double end)
        {
            string t = _open.ToString().Trim();
            if (t.Length > 0) _closed.Add(Build(t, Math.Max(end, _start)));
            _open.Clear();
            _openTokens.Clear();
            _openLogprobs.Clear();
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
    /// One device tensor per layer per side, allocated once at the size
    /// <see cref="ChooseKvTokens"/> settled on and never resized. Each is bound as BOTH
    /// past_key_i and present_key_i so GroupQueryAttention writes the new keys in place
    /// instead of producing a larger tensor every step.
    /// </summary>
    private OrtValue[] CreateSharedKvBuffers(OrtAllocator allocator, int kvTokens)
    {
        long[] shape = [1, _numKvHeads, kvTokens, _headDim];
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
        OrtValue[] kvBuffers, int kvCapacity, ref long kvPos, OrtIoBinding binding,
        RunOptions runOptions, bool wantLogprob)
    {
        int seqLen = prefixIds.Length + audioCount + suffixIds.Length;
        long total = kvPos + seqLen;
        // A backstop. Transcribe ends a pass before a window that would not fit, so reaching
        // this means a single window cost more than its worst case — not that the recording is
        // too long, which is no longer a way to fail.
        if (total > kvCapacity)
            throw new InvalidOperationException(
                $"KV cache full: {total} positions needed, {kvCapacity} allocated.");

        binding.ClearBoundInputs();
        binding.ClearBoundOutputs();

        // Bind a logits buffer we allocate and own, rather than letting the runtime allocate
        // and fetching everything back with GetOutputValues(). That call also returns wrappers
        // around the cache tensors, which are the same buffers we reuse on every later step —
        // handing their lifetime to the runtime while we still depend on them invites exactly
        // the kind of freed-pointer crash this loop must not have.
        using var logitsVal = OrtValue.CreateAllocatedTensorValue(
            OrtAllocator.DefaultInstance, TensorElementType.Float16,
            [1, seqLen, _vocabSize]);
        binding.BindOutput("logits", logitsVal);

        using var prefixVal = OrtValue.CreateTensorValueFromMemory(prefixIds, [1, prefixIds.Length]);
        binding.BindInput("prefix_input_ids", prefixVal);
        // A zero-length Memory<T> over an empty array has no pinnable storage, so the tensor
        // would carry a null data pointer into the runtime. Decode steps pass no audio, so this
        // is the common case, not an edge one; the plain-array overload is what the
        // non-streaming backend uses for it.
        int audioElems = audioCount * _hiddenSize;
        using var audioVal = audioElems > 0
            ? OrtValue.CreateTensorValueFromMemory(
                  OrtMemoryInfo.DefaultInstance, new Memory<Float16>(audioData, 0, audioElems),
                  [audioCount, _hiddenSize])
            : OrtValue.CreateTensorValueFromMemory(Array.Empty<Float16>(), [0, _hiddenSize]);
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
        var result = ArgmaxAndLogprob(logitsVal, seqLen, wantLogprob);

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

    /// <summary>
    /// Converts per-token byte offsets into character offsets.
    ///
    /// Byte offsets stop being character offsets the moment anything is non-ASCII, and a
    /// multi-byte sequence can straddle two tokens — this is a byte-level tokenizer, so a
    /// token boundary lands mid-character routinely for the CJK languages this model covers.
    /// Counting each token's bytes independently would therefore be wrong; a stateful decoder
    /// carries the partial sequence across the boundary, in one pass.
    /// </summary>
    internal static int[] CharEndsFromByteEnds(byte[] utf8, int[] byteEnds)
    {
        var ends = new int[byteEnds.Length];
        var decoder = Encoding.UTF8.GetDecoder();
        // GetChars, not GetCharCount: only the former advances the decoder's state, so only it
        // holds a partial sequence back until the token that completes it. GetCharCount would
        // report the same pending bytes on every call and count a split character repeatedly.
        char[] scratch = new char[utf8.Length + 1];
        int chars = 0, from = 0;
        for (int i = 0; i < byteEnds.Length; i++)
        {
            chars += decoder.GetChars(utf8, from, byteEnds[i] - from, scratch, 0, flush: false);
            from = byteEnds[i];
            ends[i] = chars;
        }
        return ends;
    }

    /// <summary>
    /// Decodes generated tokens, and reports for each one the character offset in the result at
    /// which its contribution ends. A token can produce no characters (a special token) or, in
    /// a multi-byte sequence, share a character with its neighbours, so offsets are
    /// non-decreasing rather than strictly increasing — which is exactly what a caller needs to
    /// ask "which tokens produced the text up to here".
    /// </summary>
    private (string text, int[] tokenCharEnds) DecodeWithOffsets(List<long> ids)
    {
        var bytes = new List<byte>(ids.Count * 4);
        var endsInBytes = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            int iid = (int)ids[i];
            if (!_addedTokenContent.ContainsKey(iid))
            {
                string? raw = iid >= 0 && iid < _idToToken.Length ? _idToToken[iid] : null;
                if (raw is not null)
                    foreach (char ch in raw)
                        if (_byteLevelDecode.TryGetValue(ch, out byte b)) bytes.Add(b);
            }
            endsInBytes[i] = bytes.Count;
        }

        byte[] all = [.. bytes];
        return (Encoding.UTF8.GetString(all), CharEndsFromByteEnds(all, endsInBytes));
    }

    private static long[] ReadLongArray(JsonElement el) => el.EnumerateArray().Select(e => e.GetInt64()).ToArray();
}

/// <summary>
/// One streaming chunk: the text the model emitted for one hop of audio.
/// <para><see cref="TokenLogprobs"/> is empty unless the caller asked for confidences, and is
/// not currently carried onto the segments <see cref="VibeVoiceStreamingAsr.ToSegments"/>
/// produces, so the editor shows no per-word confidence for this backend yet.</para>
/// </summary>
/// <param name="Pass">
///   Which decoding pass produced this chunk. A recording longer than the cache is decoded in
///   passes, each starting from an empty cache, so nothing — least of all speaker identity —
///   carries from one pass to the next. Zero for a recording that fits in one.
/// </param>
public sealed record VibeVoiceStreamingChunk(
    int Index, double Start, double End, string Text, long[] TokenIds, float[] TokenLogprobs,
    int[] TokenCharEnds, int Pass = 0);
