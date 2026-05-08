using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// IBM Granite Speech 4.1 — speech encoder + 1.84 B-param Granite LLM
/// decoder, with audio embedding fused into the prompt via a Q-Former
/// projector. Decoder-only attention; no encoder-decoder cross-KV.
///
/// Uses four ONNX models in the model directory:
///   mel.onnx           audio [1, samples] → input_features [1, T_stacked, 160]
///   encoder.onnx       input_features [B, T, 160] → encoder_hidden [B, T, 1024]
///   projector.onnx     encoder_hidden [B, T, 1024] → audio_embeds [B, A, 2048]
///   decoder.onnx       UNIFIED prefill + step graph:
///                       inputs:  input_ids, audio_embeds, attention_mask, cache_position,
///                                past_key_&lt;L&gt;/past_value_&lt;L&gt; for L in 0..39
///                       outputs: logits + present_key_&lt;L&gt;/present_value_&lt;L&gt; for L in 0..39
///                       Prefill is signalled by passing zero-length past_kv inputs;
///                       step is signalled by populated past_kv. Same graph either way.
///
/// The decode loop uses the **chained Run-with-OrtValue** pattern: each step's
/// output OrtValues become the next step's input OrtValues directly, so the KV
/// cache stays GPU-resident across steps without ever touching managed memory.
/// See docs/dev/granite_speech_perf_investigation.md Run 7 for why this beats
/// `OrtIoBinding.RunWithBinding` on this graph.
///
/// First C# integration: serial single-segment dispatch, no batching, fixed
/// English ASR prompt. Matches Cohere's <see cref="CohereTranscribe.Recognize"/>
/// public surface so <c>Vernacula.CLI</c> can dispatch the same way.
/// VRAM-budgeted batching via <see cref="BatchSizer"/> is a follow-up.
/// </summary>
public sealed class GraniteSpeech : IDisposable
{
    public const string MelFile       = "mel.onnx";
    public const string EncoderFile   = "encoder.onnx";
    public const string ProjectorFile = "projector.onnx";
    public const string DecoderFile   = "decoder.onnx";
    public const string TokenizerFile = "tokenizer.json";

    // ── Architecture constants ──────────────────────────────────────────
    // Mirror what the Python export wrote into config.json. Hardcoded here
    // (rather than read at runtime) because they're load-bearing for the
    // KV-cache shapes and don't change across model variants without a
    // re-export.
    private const int NumDecoderLayers = 40;
    private const int NumKvHeads       = 4;     // GQA — text_config.num_key_value_heads
    private const int HeadDim          = 128;   // text_config.hidden_size / num_attention_heads
    private const int VocabSize        = 100353;
    private const int AudioTokenId     = 100352;
    private const int EosTokenId       = 100257; // bos_token_id == eos_token_id
    private const int MaxContextLen    = 4096;   // text_config.max_position_embeddings

    // Mel frontend constants — driving the audio-token-count formula. Match
    // GraniteSpeechFeatureExtractor: hop_length=160, projector_window_size=15,
    // projector_downsample_rate=5 → effective_window = 3.
    private const int HopLength             = 160;
    private const int ProjectorWindowSize   = 15;
    private const int EffectiveWindowSize   = 3;  // = ProjectorWindowSize / projector_downsample_rate

    // ── Prompt template ─────────────────────────────────────────────────
    // Tokens for the standard ASR chat-template prompt:
    //   "USER: <|audio|>transcribe the speech with proper punctuation and capitalization.\n ASSISTANT:"
    // Generated once via the Granite tokenizer (see scripts/granite_export/
    // dump_inputs_for_csharp_smoke.py for the source command). We splice in N
    // copies of AudioTokenId between Prefix and Suffix where N is the
    // audio-token count derived from sample count.
    private static readonly long[] AsrPromptPrefix = { 6584, 25, 220 };
    private static readonly long[] AsrPromptSuffix =
    {
        1485, 3191, 279, 8982, 449, 6300, 62603, 323, 6864, 2065, 627,
        36660, 3931, 2891, 25,
    };

    // ── ONNX sessions + tokenizer ───────────────────────────────────────
    private readonly InferenceSession _mel;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _projector;
    private readonly InferenceSession _decoder;

    private readonly string?[] _idToToken;
    private readonly Dictionary<int, string> _addedTokens;
    private readonly Dictionary<char, byte> _byteLevelDecode;

    // ── Construction ────────────────────────────────────────────────────

    public GraniteSpeech(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = OrtSessionBuilder.Create(ep);
        _mel       = new InferenceSession(Path.Combine(modelPath, MelFile),       opts);
        _encoder   = new InferenceSession(Path.Combine(modelPath, EncoderFile),   opts);
        _projector = new InferenceSession(Path.Combine(modelPath, ProjectorFile), opts);
        _decoder   = new InferenceSession(Path.Combine(modelPath, DecoderFile),   opts);

        (_idToToken, _addedTokens) = LoadTokenizerVocab(Path.Combine(modelPath, TokenizerFile));
        _byteLevelDecode = BuildByteLevelDecodeTable();
    }

    public void Dispose()
    {
        _mel.Dispose();
        _encoder.Dispose();
        _projector.Dispose();
        _decoder.Dispose();
    }

    // ── Audio-token-count derivation ────────────────────────────────────

    /// <summary>
    /// Compute the number of audio tokens (projector output frames) that
    /// will be produced for an audio segment of the given sample count.
    /// Mirrors `GraniteSpeechFeatureExtractor._get_num_audio_features`:
    ///   mel_length     = samples / hop_length + 1
    ///   encoder_length = mel_length / 2          (frame stack)
    ///   nblocks        = ceil(encoder_length / projector_window_size)
    ///   audio_tokens   = nblocks * effective_window_size  (=3)
    /// </summary>
    public static int NumAudioTokens(int sampleCount)
    {
        int melLength     = sampleCount / HopLength + 1;
        int encoderLength = melLength / 2;
        int nblocks       = (encoderLength + ProjectorWindowSize - 1) / ProjectorWindowSize;
        return nblocks * EffectiveWindowSize;
    }

    private static long[] BuildPromptIds(int audioTokens)
    {
        var ids = new long[AsrPromptPrefix.Length + audioTokens + AsrPromptSuffix.Length];
        Array.Copy(AsrPromptPrefix, ids, AsrPromptPrefix.Length);
        for (int i = 0; i < audioTokens; i++)
            ids[AsrPromptPrefix.Length + i] = AudioTokenId;
        Array.Copy(AsrPromptSuffix, 0, ids, AsrPromptPrefix.Length + audioTokens,
                   AsrPromptSuffix.Length);
        return ids;
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// <summary>
    /// Transcribes each segment from <paramref name="segs"/> and yields
    /// <c>(segId, text)</c> in order as each segment completes. Speaker
    /// labels are passed through from the input segments — Granite Speech
    /// itself doesn't do speaker diarization.
    /// </summary>
    /// <param name="segs">Segment list as <c>(start_seconds, end_seconds, speaker_label)</c>.</param>
    /// <param name="audio">Mono 16 kHz waveform covering all segments.</param>
    /// <param name="maxNewTokens">Cap on generated tokens per segment.</param>
    public IEnumerable<(int segId, string text, string speaker)> Recognize(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        for (int segId = 0; segId < segs.Count; segId++)
        {
            var (start, end, spk) = segs[segId];
            float[] wave = ExtractSegment(audio, start, end);
            // Extreme-short skip: <100 ms produces no useful audio tokens.
            if (wave.Length < Config.SampleRate / 10)
            {
                yield return (segId, string.Empty, spk);
                continue;
            }
            string text = TranscribeOne(wave, maxNewTokens);
            yield return (segId, text, spk);
        }
    }

    /// <summary>Convenience entry point for transcribing a single waveform.</summary>
    public string Transcribe(float[] audio16kMono, int maxNewTokens = 256)
        => TranscribeOne(audio16kMono, maxNewTokens);

    // ── Per-segment pipeline ────────────────────────────────────────────

    private string TranscribeOne(float[] wave, int maxNewTokens)
    {
        // Capacity check — Granite Speech's decoder context is 4096 tokens.
        // At 10 Hz audio embedding rate that's ~6.7 minutes max audio,
        // minus prompt overhead. Caller is expected to pre-segment.
        int audioTokens = NumAudioTokens(wave.Length);
        int promptLen   = AsrPromptPrefix.Length + audioTokens + AsrPromptSuffix.Length;
        if (promptLen + maxNewTokens > MaxContextLen)
        {
            int allowedAudio = MaxContextLen - maxNewTokens
                             - AsrPromptPrefix.Length - AsrPromptSuffix.Length;
            int allowedSeconds = (int)(allowedAudio
                / (double)EffectiveWindowSize * ProjectorWindowSize * 2 * HopLength
                / Config.SampleRate);
            throw new InvalidOperationException(
                $"Segment too long: prompt+max_new_tokens ({promptLen + maxNewTokens}) " +
                $"exceeds Granite Speech context ({MaxContextLen}). " +
                $"Pre-segment audio to ≤{allowedSeconds} s.");
        }

        long[] inputIds = BuildPromptIds(audioTokens);

        // ── Mel ─────────────────────────────────────────────────────────
        var audioT = new DenseTensor<float>(wave, [1, wave.Length]);
        using var melResults = _mel.Run([NamedOnnxValue.CreateFromTensor("audio", audioT)]);
        var inputFeatures = (DenseTensor<float>)melResults[0].AsTensor<float>();
        var ifShape = inputFeatures.Dimensions.ToArray();
        var ifData = inputFeatures.ToArray();

        // ── Encoder ─────────────────────────────────────────────────────
        using var encResults = _encoder.Run([NamedOnnxValue.CreateFromTensor(
            "input_features", new DenseTensor<float>(ifData, ifShape))]);
        var encoderHidden = encResults[0].AsTensor<float>();
        var encShape = encoderHidden.Dimensions.ToArray();
        var encData = encoderHidden.ToArray();

        // ── Projector ───────────────────────────────────────────────────
        using var projResults = _projector.Run([NamedOnnxValue.CreateFromTensor(
            "encoder_hidden", new DenseTensor<float>(encData, encShape))]);
        var audioEmbeds = projResults[0].AsTensor<float>();
        var projShape = audioEmbeds.Dimensions.ToArray();
        var projData = audioEmbeds.ToArray();

        // Sanity: projector output length should match the formula. If they
        // disagree, the audio token count in input_ids would be off and the
        // model would hallucinate. Error out loudly rather than silently
        // produce garbage.
        if (projShape[1] != audioTokens)
            throw new InvalidOperationException(
                $"Audio token count mismatch: formula predicts {audioTokens}, "
              + $"projector produced {projShape[1]}. Input_ids/projector output disagree.");

        // ── Decoder: prefill + step loop via chained Run-with-OrtValue ──
        // Inputs for the unified decoder graph stay in this order; build the
        // names list once.
        var decInputNames = new List<string>(4 + 2 * NumDecoderLayers)
        { "input_ids", "audio_embeds", "attention_mask", "cache_position" };
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            decInputNames.Add($"past_key_{L}");
            decInputNames.Add($"past_value_{L}");
        }
        var decOutputNames = new List<string>(1 + 2 * NumDecoderLayers) { "logits" };
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_key_{L}");
        for (int L = 0; L < NumDecoderLayers; L++) decOutputNames.Add($"present_value_{L}");

        long audioDim = projShape[2];
        var generated = new List<long>(maxNewTokens);
        long nextToken;

        using (var runOpts = new RunOptions())
        {
            // ── Prefill ─────────────────────────────────────────────────
            using var inputIdsVal = OrtValue.CreateTensorValueFromMemory(
                inputIds, new long[] { 1, promptLen });
            using var audioVal = OrtValue.CreateTensorValueFromMemory(
                projData, new long[] { 1, projShape[1], audioDim });
            var attnArr = new long[promptLen];
            for (int i = 0; i < promptLen; i++) attnArr[i] = 1L;
            using var attnVal = OrtValue.CreateTensorValueFromMemory(
                attnArr, new long[] { 1, promptLen });
            var cpArr = new long[promptLen];
            for (int i = 0; i < promptLen; i++) cpArr[i] = i;
            using var cpVal = OrtValue.CreateTensorValueFromMemory(
                cpArr, new long[] { promptLen });

            var emptyPasts = new List<OrtValue>(2 * NumDecoderLayers);
            var prefillInputs = new List<OrtValue>(4 + 2 * NumDecoderLayers)
            { inputIdsVal, audioVal, attnVal, cpVal };
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                var k = OrtValue.CreateTensorValueFromMemory(
                    Array.Empty<float>(),
                    new long[] { 1, NumKvHeads, 0, HeadDim });
                var v = OrtValue.CreateTensorValueFromMemory(
                    Array.Empty<float>(),
                    new long[] { 1, NumKvHeads, 0, HeadDim });
                emptyPasts.Add(k);
                emptyPasts.Add(v);
                prefillInputs.Add(k);
                prefillInputs.Add(v);
            }

            var prefillOutputs = _decoder.Run(
                runOpts, decInputNames, prefillInputs, decOutputNames);

            // Argmax on the prefill last position; the first generated token.
            var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
            nextToken = ArgmaxLastPosition(prefillLogitsSpan, promptLen, VocabSize);
            prefillOutputs[0].Dispose();
            foreach (var p in emptyPasts) p.Dispose();

            // pastKvs = the prefill's GPU-resident output OrtValues. Becomes
            // the next step's past_kv inputs; chain forward without ever
            // copying to managed memory.
            var pastKvs = new OrtValue[2 * NumDecoderLayers];
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                pastKvs[2 * L]     = prefillOutputs[1 + L];
                pastKvs[2 * L + 1] = prefillOutputs[1 + NumDecoderLayers + L];
            }

            // ── Step loop ──────────────────────────────────────────────
            // Reusable scratch buffers; OrtValues wrap them with the right
            // shape for the current step.
            long[] stepInputIdArr = new long[1];
            long[] cachePosStepArr = new long[1];
            long[] attnScratch = new long[promptLen + maxNewTokens + 1];
            for (int i = 0; i < attnScratch.Length; i++) attnScratch[i] = 1L;
            float[] dummyAudio = new float[1 * 1 * (int)audioDim]; // step audio_embeds is a no-op

            int pastLen = promptLen;
            for (int step = 0; step < maxNewTokens; step++)
            {
                if (nextToken == EosTokenId) break;
                generated.Add(nextToken);
                int totalLen = pastLen + 1;

                stepInputIdArr[0] = nextToken;
                cachePosStepArr[0] = pastLen;
                using var stepInputId = OrtValue.CreateTensorValueFromMemory(
                    stepInputIdArr, new long[] { 1, 1 });
                using var stepAudio = OrtValue.CreateTensorValueFromMemory(
                    dummyAudio, new long[] { 1, 1, audioDim });
                using var stepAttn = OrtValue.CreateTensorValueFromMemory(
                    attnScratch, new long[] { 1, totalLen });
                using var cachePos = OrtValue.CreateTensorValueFromMemory(
                    cachePosStepArr, new long[] { 1 });

                var stepInputs = new List<OrtValue>(4 + 2 * NumDecoderLayers)
                { stepInputId, stepAudio, stepAttn, cachePos };
                stepInputs.AddRange(pastKvs);

                var outputs = _decoder.Run(
                    runOpts, decInputNames, stepInputs, decOutputNames);

                // Take ownership of the new GPU-resident KV; release the
                // previous step's OrtValues once we've swapped pastKvs.
                var oldPast = pastKvs;
                pastKvs = new OrtValue[2 * NumDecoderLayers];
                for (int L = 0; L < NumDecoderLayers; L++)
                {
                    pastKvs[2 * L]     = outputs[1 + L];
                    pastKvs[2 * L + 1] = outputs[1 + NumDecoderLayers + L];
                }

                var stepLogits = outputs[0];
                var stepLogitsSpan = stepLogits.GetTensorDataAsSpan<float>();
                nextToken = ArgmaxLastPosition(stepLogitsSpan, 1, VocabSize);
                stepLogits.Dispose();
                foreach (var ov in oldPast) ov.Dispose();

                pastLen = totalLen;
            }

            foreach (var ov in pastKvs) ov.Dispose();
        }

        return DecodeTokens(generated);
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    private static float[] ExtractSegment(float[] audio, double start, double end)
    {
        int s = Math.Max((int)(start * Config.SampleRate), 0);
        int e = Math.Min((int)(end   * Config.SampleRate), audio.Length);
        if (e <= s) return Array.Empty<float>();
        var seg = new float[e - s];
        Array.Copy(audio, s, seg, 0, e - s);
        return seg;
    }

    private static long ArgmaxLastPosition(ReadOnlySpan<float> logits, int seqLen, int vocab)
    {
        int offset = (seqLen - 1) * vocab;
        long best = 0;
        float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = logits[offset + v];
            if (val > bestVal) { bestVal = val; best = v; }
        }
        return best;
    }

    /// <summary>
    /// Decode a sequence of token IDs to a UTF-8 string using the GPT-2
    /// ByteLevel BPE scheme (the Granite tokenizer is GPT-2 family).
    /// Special / added tokens use their `content` field directly; regular
    /// BPE tokens go through the byte-level decode table.
    /// </summary>
    public string DecodeTokens(IReadOnlyList<long> tokenIds)
    {
        var bytes = new List<byte>(tokenIds.Count * 4);
        foreach (long id in tokenIds)
        {
            int iid = (int)id;
            if (_addedTokens.TryGetValue(iid, out string? special))
            {
                bytes.AddRange(Encoding.UTF8.GetBytes(special));
                continue;
            }
            string? raw = iid >= 0 && iid < _idToToken.Length ? _idToToken[iid] : null;
            if (raw is null) continue;
            foreach (char ch in raw)
                if (_byteLevelDecode.TryGetValue(ch, out byte b))
                    bytes.Add(b);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    // ── Tokenizer loading + ByteLevel decode (port from VibeVoice) ──────

    private static (string?[] idToToken, Dictionary<int, string> addedContent)
        LoadTokenizerVocab(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var vocab = root.GetProperty("model").GetProperty("vocab");

        int maxId = 0;
        foreach (var kv in vocab.EnumerateObject())
            if (kv.Value.GetInt32() > maxId) maxId = kv.Value.GetInt32();

        var idToToken = new string?[maxId + 1];
        foreach (var kv in vocab.EnumerateObject())
            idToToken[kv.Value.GetInt32()] = kv.Name;

        var addedContent = new Dictionary<int, string>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var at in addedTokens.EnumerateArray())
            {
                int atId = at.GetProperty("id").GetInt32();
                string atContent = at.GetProperty("content").GetString() ?? "";
                addedContent[atId] = atContent;
            }
        }

        return (idToToken, addedContent);
    }

    private static Dictionary<char, byte> BuildByteLevelDecodeTable()
    {
        var printable = new HashSet<int>();
        for (int b = 33; b <= 126; b++) printable.Add(b);
        for (int b = 161; b <= 172; b++) printable.Add(b);
        for (int b = 174; b <= 255; b++) printable.Add(b);

        var dict = new Dictionary<char, byte>(280);
        foreach (int b in printable) dict[(char)b] = (byte)b;
        int extra = 0;
        for (int b = 0; b < 256; b++)
            if (!printable.Contains(b))
                dict[(char)(0x100 + extra++)] = (byte)b;
        return dict;
    }
}
