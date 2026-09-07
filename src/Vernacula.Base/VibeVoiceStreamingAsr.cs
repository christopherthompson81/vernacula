using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// VibeVoice-ASR-Streaming: chunked speaker-attributed ASR, run the way upstream's
/// <c>streaming_generate</c> runs it (see docs/dev/vibevoice_asr_streaming_investigation.md).
///
/// Package (from scripts/vibevoice_streaming_export):
///   audio_encoder.onnx    input_values [1,T] (float32) → audio_embeddings [N,hidden] (float32),
///                         run once per fixed window (chunk + lookahead frames), cold each time.
///   decoder_single.onnx   same contract as the non-streaming decoder: prefix_input_ids [1,P] +
///                         audio_embeddings [N,hidden] (bf16) + suffix_input_ids [1,S] +
///                         past_key/value_i → logits [1,seq,vocab] (bf16) + present_key/value_i.
///
/// Loop: prefill the prompt once; then for every window feed
/// [speech_start] + frames + [speech_end], greedy-decode until &lt;|text_chunk_end|&gt; or EOS,
/// then feed &lt;|text_chunk_end|&gt; itself so the cache always ends on it. The KV cache lives
/// on the device and grows for the whole recording; upstream never evicts.
/// </summary>
public sealed class VibeVoiceStreamingAsr : IDisposable
{
    public const string AudioEncoderFile  = "audio_encoder.onnx";
    public const string DecoderSingleFile = "decoder_single.onnx";
    public const string ExportReportFile  = "export-report.json";
    public const string TokenizerFile     = "tokenizer.json";

    private readonly InferenceSession _audioEncoder;
    private readonly InferenceSession _decoder;

    private readonly int  _numLayers, _numKvHeads, _headDim, _hiddenSize;
    private readonly bool _kvCacheIsFloat32;

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

        _numLayers        = report.GetProperty("num_layers").GetInt32();
        _numKvHeads       = report.GetProperty("num_kv_heads").GetInt32();
        _headDim          = report.GetProperty("head_dim").GetInt32();
        _hiddenSize       = report.GetProperty("hidden_size").GetInt32();
        _kvCacheIsFloat32 = report.TryGetProperty("f32_kv_cache", out var f32) && f32.GetBoolean();

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

        // The encoder is float32 convolution towers; the decoder keeps EXTENDED for the same
        // reason the non-streaming backend does (ORT_ENABLE_ALL fuses the BF16 softmax).
        _audioEncoder = new InferenceSession(Path.Combine(modelDir, AudioEncoderFile),
            OrtSessionBuilder.Create(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED));
        _decoder = new InferenceSession(Path.Combine(modelDir, DecoderSingleFile),
            OrtSessionBuilder.Create(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED));
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
        Action<VibeVoiceStreamingChunk>? onChunk = null,
        CancellationToken ct = default)
    {
        float[] audio = VibeVoiceAsr.AudioTo24kMono(rawAudio, sampleRate, channels);
        int totalChunks = audio.Length == 0 ? 0 : (audio.Length + HopSamples - 1) / HopSamples;
        var chunks = new List<VibeVoiceStreamingChunk>(totalChunks);

        using var cudaMemInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        using var binding    = _decoder.CreateIoBinding();
        using var runOptions = new RunOptions();
        var pastKvs = CreateInitialKvOrtValues();
        var window  = new float[WindowSamples];
        var emptyAudio = Array.Empty<BFloat16>();
        var emptyIds   = Array.Empty<long>();
        try
        {
            // 1 — prompt prefill
            long[] prompt = hotwordTokenIds is { Length: > 0 }
                ? [.. _promptHotwordsHeadIds, .. hotwordTokenIds, .. _promptTailIds]
                : _promptTokenIds;
            Step(prompt, emptyAudio, 0, emptyIds, ref pastKvs, binding, runOptions, cudaMemInfo);

            // 2 — one window per hop, zero-padded at the end of the recording
            for (int ci = 0; ci < totalChunks; ci++)
            {
                ct.ThrowIfCancellationRequested();
                int start = ci * HopSamples;
                int n = Math.Min(WindowSamples, audio.Length - start);
                Array.Copy(audio, start, window, 0, n);
                Array.Clear(window, n, WindowSamples - n);
                BFloat16[] frames = EncodeWindow(window);
                int numFrames = frames.Length / _hiddenSize;

                long next = Step([_speechStartId], frames, numFrames, [_speechEndId],
                                 ref pastKvs, binding, runOptions, cudaMemInfo).token;
                var ids = new List<long>();
                var logprobs = new List<float>();
                for (int t = 0; t < maxNewTokensPerChunk; t++)
                {
                    if (next == _textChunkEndId || next == _eosTokenId) break;
                    ids.Add(next);
                    var (tok, lp) = Step([next], emptyAudio, 0, emptyIds, ref pastKvs, binding, runOptions, cudaMemInfo);
                    logprobs.Add(lp);
                    next = tok;
                }
                Step([_textChunkEndId], emptyAudio, 0, emptyIds, ref pastKvs, binding, runOptions, cudaMemInfo);

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
            foreach (var kv in pastKvs) kv.Dispose();
        }
        return chunks;
    }

    /// <summary>
    /// Fold chunk texts into speaker turns. Upstream emits <c>\n Speaker N:</c> markers inline
    /// and a turn may span chunks. The model gives no timestamps, so a turn boundary that
    /// falls inside a chunk is placed proportionally to where it falls in that chunk's text;
    /// starts are therefore approximate to within a chunk (2.93 s) and never decrease.
    /// Text before the first marker (if any) is attributed to speaker -1.
    /// </summary>
    public static IReadOnlyList<VibeVoiceSegment> ToSegments(IReadOnlyList<VibeVoiceStreamingChunk> chunks)
    {
        var segs = new List<VibeVoiceSegment>();
        int    curSpk = -1;
        var    curText = new StringBuilder();
        double curStart = chunks.Count > 0 ? chunks[0].Start : 0;

        static double At(VibeVoiceStreamingChunk c, int pos) =>
            c.Text.Length == 0 ? c.Start : c.Start + (c.End - c.Start) * pos / c.Text.Length;

        void Flush(double end)
        {
            string t = curText.ToString().Trim();
            if (t.Length > 0) segs.Add(new VibeVoiceSegment(curStart, Math.Max(end, curStart), curSpk, t));
            curText.Clear();
        }

        foreach (var c in chunks)
        {
            int pos = 0;
            foreach (Match m in SpeakerMarker.Matches(c.Text))
            {
                curText.Append(c.Text, pos, m.Index - pos);
                Flush(At(c, m.Index));
                curSpk   = int.Parse(m.Groups[1].Value);
                curStart = At(c, m.Index);
                pos = m.Index + m.Length;
            }
            curText.Append(c.Text, pos, c.Text.Length - pos);
        }
        if (chunks.Count > 0) Flush(chunks[^1].End);
        return segs;
    }

    private static readonly Regex SpeakerMarker = new(@"\s*Speaker (\d+):", RegexOptions.Compiled);

    public void Dispose()
    {
        _audioEncoder.Dispose();
        _decoder.Dispose();
    }

    // ── Audio encoder ─────────────────────────────────────────────────────────

    private BFloat16[] EncodeWindow(float[] window)
    {
        using var input = OrtValue.CreateTensorValueFromMemory(window, [1, window.Length]);
        using var outputs = _audioEncoder.Run(new RunOptions(),
            new Dictionary<string, OrtValue> { ["input_values"] = input }, ["audio_embeddings"]);
        var span = outputs[0].GetTensorDataAsSpan<float>();
        var bf = new BFloat16[span.Length];
        for (int i = 0; i < span.Length; i++) bf[i] = (BFloat16)span[i];
        return bf;
    }

    // ── Decoder ───────────────────────────────────────────────────────────────

    private OrtValue[] CreateInitialKvOrtValues()
    {
        long[] shape = [1, _numKvHeads, 0, _headDim];
        var kvs = new OrtValue[_numLayers * 2];
        for (int i = 0; i < kvs.Length; i++)
            kvs[i] = _kvCacheIsFloat32
                ? OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(), shape)
                : OrtValue.CreateTensorValueFromMemory(Array.Empty<BFloat16>(), shape);
        return kvs;
    }

    /// <summary>One decoder call; the past KV values are replaced by the present ones (device-resident).</summary>
    private (long token, float logprob) Step(
        long[] prefixIds, BFloat16[] audioData, int audioCount, long[] suffixIds,
        ref OrtValue[] pastKvs, OrtIoBinding binding, RunOptions runOptions, OrtMemoryInfo cudaMemInfo)
    {
        binding.ClearBoundInputs();
        binding.ClearBoundOutputs();
        binding.BindOutputToDevice("logits", OrtMemoryInfo.DefaultInstance);
        for (int i = 0; i < _numLayers; i++)
        {
            binding.BindOutputToDevice($"present_key_{i}",   cudaMemInfo);
            binding.BindOutputToDevice($"present_value_{i}", cudaMemInfo);
        }

        using var prefixVal = OrtValue.CreateTensorValueFromMemory(prefixIds, [1, prefixIds.Length]);
        binding.BindInput("prefix_input_ids", prefixVal);
        using var audioVal = OrtValue.CreateTensorValueFromMemory(
            OrtMemoryInfo.DefaultInstance, new Memory<BFloat16>(audioData, 0, audioCount * _hiddenSize),
            [audioCount, _hiddenSize]);
        binding.BindInput("audio_embeddings", audioVal);
        using var suffixVal = OrtValue.CreateTensorValueFromMemory(suffixIds, [1, suffixIds.Length]);
        binding.BindInput("suffix_input_ids", suffixVal);
        for (int i = 0; i < _numLayers; i++)
        {
            binding.BindInput($"past_key_{i}",   pastKvs[i * 2]);
            binding.BindInput($"past_value_{i}", pastKvs[i * 2 + 1]);
        }

        _decoder.RunWithBinding(runOptions, binding);
        var outputs = binding.GetOutputValues();

        int seqLen = prefixIds.Length + audioCount + suffixIds.Length;
        var result = ArgmaxAndLogprob(outputs[0], seqLen);
        outputs[0].Dispose();

        foreach (var kv in pastKvs) kv.Dispose();
        for (int i = 0; i < _numLayers * 2; i++) pastKvs[i] = outputs[i + 1];
        return result;
    }

    private static (long token, float logprob) ArgmaxAndLogprob(OrtValue logits, int seqLen)
    {
        var span      = logits.GetTensorDataAsSpan<BFloat16>();
        int vocabSize = span.Length / seqLen;
        int offset    = (seqLen - 1) * vocabSize;
        long best = 0; float bestVal = float.NegativeInfinity;
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
