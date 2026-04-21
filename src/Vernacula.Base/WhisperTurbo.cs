using System.Diagnostics;
using System.Text;
using System.Text.Json;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Base;

public readonly record struct WhisperTranscript(string Text, IReadOnlyList<int> Tokens);

/// <summary>
/// One segment's recognition result in the same shape TranscriptionService
/// expects from the other backends: a segment index back into the input
/// segment list, the transcript text, and the raw token ids (for DB
/// metadata / debugging).
/// </summary>
public sealed record WhisperRecognitionResult(
    int SegmentId,
    string Text,
    IReadOnlyList<int> Tokens);

/// <summary>
/// Per-phase timing breakdown, accumulated across one Recognize call.
/// All times in milliseconds. Useful for spotting bottlenecks — at a glance
/// tells you whether you're encoder-bound, decoder-step-bound, or
/// memory-copy-bound.
/// </summary>
public sealed record WhisperTimingBreakdown(
    long MelMs,            // CPU log-mel compute
    long EncoderMs,        // encoder forward (ORT + extract-to-float[])
    long DecoderInitMs,    // decoder-init forward (ORT + extract-to-float[])
    long DecoderStepOrtMs, // sum of _decoderStep.Run() wall time
    long DecoderStepExtractMs, // sum of per-step logits+KV extract-to-float[]
    long ArgmaxMs,         // sum of per-step argmax
    long TokenDecodeMs,    // sum of BPE decode-to-text
    long StepCalls,        // total number of decoder_step invocations
    long SegmentsProcessed);

/// <summary>
/// Whisper large-v3-turbo backend.  Phase 2a scope: Whisper-style log-mel
/// frontend and encoder-only inference.  Decoder pair, greedy loop, tokenizer,
/// and language-token handling land in Phase 2b.
///
/// ONNX layout (fp16 internal, fp32 graph boundaries — Optimum default export):
///   encoder_model_fp16.onnx
///     in  input_features [B, 128, 3000] float32   (30-second chunks, padded)
///     out last_hidden_state [B, 1500, 1280] float32
///
/// Files are downloaded from the onnx-community pre-export — see
/// <c>docs/whisper_turbo_investigation.md</c> for why we're not running our
/// own exporter.
/// </summary>
public sealed class WhisperTurbo : IDisposable
{
    // ── File layout (download manifest lives in ModelManagerService) ────────
    public const string MelFile                = "mel.onnx";
    public const string EncoderFile            = "encoder_model_fp16.onnx";
    // Merged decoder: single graph with `use_cache_branch` selecting prefill
    // vs step. Replaces the decoder_model + decoder_with_past_model pair —
    // same per-call compute, but ~318 MB less VRAM (one set of weights instead
    // of two) so we have more headroom for batching later.
    public const string DecoderFile            = "decoder_model_merged_fp16.onnx";
    public const string TokenizerFile          = "tokenizer.json";
    public const string ConfigFile             = "config.json";
    public const string GenerationConfigFile   = "generation_config.json";
    public const string PreprocessorConfigFile = "preprocessor_config.json";
    public const string ModelName              = "openai/whisper-large-v3-turbo";

    // ── DSP constants (match preprocessor_config.json) ──────────────────────
    private const int   SampleRate   = 16_000;
    private const int   NFft         = 400;
    private const int   HopLength    = 160;
    private const int   NMels        = 128;
    private const int   ChunkSeconds = 30;
    public  const int   ChunkSamples = ChunkSeconds * SampleRate;  // 480 000
    public  const int   ChunkFrames  = ChunkSamples / HopLength;   // 3 000
    private const float LogFloor     = -10.0f;
    private const float LogClampSpan = 8.0f;
    private const float LogOffset    = 4.0f;

    // ── Encoder output shape (fixed for 30 s chunks) ────────────────────────
    public const int EncoderOutFrames = ChunkFrames / 2;  // 1500, conv 2× downsampling
    public const int HiddenSize       = 1280;

    // ── Decoder geometry (large-v3-turbo) ───────────────────────────────────
    private const int NumDecoderLayers = 4;      // large-v3-turbo distillation
    private const int NumHeads         = 20;
    private const int HeadDim          = 64;     // 20 * 64 = 1280
    private const int MaxDecoderLength = 448;    // Whisper hard context limit
    private const int VocabSize        = 51_866; // large-v3-turbo vocab (fp16 export)

    // ── Whisper special tokens (from generation_config.json) ────────────────
    // Regular BPE vocab spans [0, 50256]. All special tokens are >= 50257.
    private const int SotToken           = 50258;  // <|startoftranscript|>
    private const int EotToken           = 50257;  // <|endoftext|> (also BOS/EOS/pad)
    private const int NoTimestampsToken  = 50364;
    private const int TranscribeToken    = 50360;
    private const int TranslateToken     = 50359;
    private const int SpecialTokenFloor  = 50257;  // everything >= this is a special token
    // Suppress these at the FIRST content step only: 220 = leading space, 50257 = immediate EOT.
    // Prevents the two degenerate argmax outcomes in the initial decode position.
    private static readonly int[] BeginSuppressTokens = [220, EotToken];

    // Language ISO → Whisper <|lang|> token id. Populated from
    // generation_config.json's lang_to_id dict on load.
    private readonly Dictionary<string, int> _langToId;

    // ── Static precomputed ──────────────────────────────────────────────────
    // Note: Qwen3Asr.cs has an identical Whisper-style mel frontend inline.
    // Dedup deferred — touching Qwen3 now would risk regressing a validated
    // backend for the sake of a backend that hasn't shipped yet. Revisit
    // after Phase 6 validation lands.
    private static readonly float[,] MelFilterbank = CreateMelFilterbank();
    private static readonly double[] HannWindow    = Window.HannPeriodic(NFft);

    private readonly InferenceSession _mel;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly string?[] _idToToken;
    private readonly Dictionary<char, byte> _byteLevelDecode;

    // ── Timing accumulators (milliseconds) ──────────────────────────────────
    // Reset at the start of every Recognize / RecognizeBatched call; read via
    // GetTimingsAndReset() at the end of the pipeline.  Zero allocation on
    // the hot path — just long increments.
    private long _msMel;
    private long _msEncoder;
    private long _msDecInit;
    private long _msDecStepOrt;
    private long _msDecStepExtract;
    private long _msArgmax;
    private long _msDecode;
    private long _stepCalls;
    private long _segments;

    public WhisperTurbo(string modelsDir, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = MakeSessionOptions(ep);
        _mel     = new InferenceSession(Path.Combine(modelsDir, MelFile),     opts);
        _encoder = new InferenceSession(Path.Combine(modelsDir, EncoderFile), opts);
        _decoder = new InferenceSession(Path.Combine(modelsDir, DecoderFile), opts);

        _idToToken       = LoadTokenizerVocab(Path.Combine(modelsDir, TokenizerFile));
        _byteLevelDecode = BuildByteLevelDecode();
        _langToId        = LoadLangToId(Path.Combine(modelsDir, GenerationConfigFile));
    }

    public void Dispose()
    {
        _decoder.Dispose();
        _encoder.Dispose();
        _mel.Dispose();
    }

    /// <summary>
    /// Snapshot the accumulated per-phase timings and reset the counters.
    /// Call after a Recognize / RecognizeBatched run to print a breakdown.
    /// </summary>
    public WhisperTimingBreakdown GetTimingsAndReset()
    {
        var r = new WhisperTimingBreakdown(
            MelMs: _msMel,
            EncoderMs: _msEncoder,
            DecoderInitMs: _msDecInit,
            DecoderStepOrtMs: _msDecStepOrt,
            DecoderStepExtractMs: _msDecStepExtract,
            ArgmaxMs: _msArgmax,
            TokenDecodeMs: _msDecode,
            StepCalls: _stepCalls,
            SegmentsProcessed: _segments);
        _msMel = _msEncoder = _msDecInit = 0;
        _msDecStepOrt = _msDecStepExtract = _msArgmax = _msDecode = 0;
        _stepCalls = _segments = 0;
        return r;
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// <summary>
    /// Prepare a chunk-sized log-mel spectrogram ready to feed the encoder,
    /// using the ONNX <c>mel.onnx</c> graph on whichever EP the sessions were
    /// loaded with. The waveform is zero-padded (or truncated) to 30 s first
    /// — matches the HuggingFace WhisperFeatureExtractor convention.
    ///
    /// Returns a flat float[<see cref="NMels"/> * <see cref="ChunkFrames"/>]
    /// in row-major <c>[mel, frame]</c> order.
    ///
    /// Moves a ~58 ms/segment CPU hotspot onto the same EP as the encoder.
    /// See <c>docs/whisper_turbo_investigation.md</c> Run 4 for the profile.
    /// </summary>
    public float[] PrepareChunkMel(float[] audio16k)
    {
        var chunk = new float[ChunkSamples];
        int copy  = Math.Min(audio16k.Length, ChunkSamples);
        Array.Copy(audio16k, chunk, copy);

        var input = new DenseTensor<float>(chunk, [1, ChunkSamples]);
        using var outputs = _mel.Run(
        [
            NamedOnnxValue.CreateFromTensor("audio", input),
        ]);
        return ExtractFloat(outputs.First(o => o.Name == "mel"));
    }

    /// <summary>
    /// Legacy CPU log-mel — the pre-mel-in-graph implementation. Kept as a
    /// reference and for `--whisper-check` smoke tests that want to isolate
    /// CPU numerics from EP-dependent ORT behaviour.
    /// </summary>
    public static float[] PrepareChunkMelCpu(float[] audio16k)
    {
        var chunk = new float[ChunkSamples];
        int copy  = Math.Min(audio16k.Length, ChunkSamples);
        Array.Copy(audio16k, chunk, copy);
        var (mel, frames) = ComputeLogMel(chunk, 0, ChunkSamples);
        if (frames != ChunkFrames)
            throw new InvalidOperationException(
                $"Expected {ChunkFrames} mel frames for {ChunkSamples}-sample input, got {frames}.");
        return mel;
    }

    /// <summary>
    /// Run the encoder on a padded chunk mel (<c>float[128 * 3000]</c>).
    /// Returns last_hidden_state as a flat float[1 * 1500 * 1280] in
    /// row-major <c>[batch, encoder_frame, hidden]</c> order.
    /// </summary>
    public float[] RunEncoder(float[] paddedMel)
    {
        if (paddedMel.Length != NMels * ChunkFrames)
            throw new ArgumentException(
                $"Expected mel length {NMels * ChunkFrames}, got {paddedMel.Length}.",
                nameof(paddedMel));

        var input = new DenseTensor<float>(paddedMel, [1, NMels, ChunkFrames]);
        using var outputs = _encoder.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_features", input),
        ]);
        return ExtractFloat(outputs.First(o => o.Name == "last_hidden_state"));
    }

    /// <summary>
    /// Segment-based transcription matching the convention other Vernacula
    /// backends use. Each VAD/diarization segment is transcribed
    /// independently; segments shorter than 30 s are zero-padded (cheap),
    /// segments longer than 30 s are split into contiguous 30-s sub-chunks
    /// and their transcripts concatenated.
    ///
    /// Sequential-only in this phase; batching + VRAM budgeting comes next
    /// (see <c>docs/whisper_turbo_investigation.md</c>).
    /// </summary>
    public IEnumerable<WhisperRecognitionResult> Recognize(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio16k,
        string? forceLanguage = null)
    {
        string lang = forceLanguage ?? "en";

        for (int segId = 0; segId < segs.Count; segId++)
        {
            var (start, end, _) = segs[segId];
            int startSample = Math.Max(0, (int)(start * SampleRate));
            int endSample   = Math.Min(audio16k.Length, (int)(end * SampleRate));
            int length      = Math.Max(0, endSample - startSample);

            if (length == 0)
            {
                yield return new WhisperRecognitionResult(segId, "", []);
                continue;
            }

            if (length <= ChunkSamples)
            {
                // Short segment — zero-pad to 30 s and transcribe in one call.
                var slice = new float[length];
                Array.Copy(audio16k, startSample, slice, 0, length);
                var t = Transcribe(slice, lang);
                yield return new WhisperRecognitionResult(segId, t.Text, t.Tokens);
            }
            else
            {
                // Long segment — split into contiguous 30-s sub-chunks.
                // No overlap (see investigation doc): 0.5-s overlap would let
                // us potentially fix boundary-word cases, but without real
                // timestamps we can't dedupe cleanly, so contiguous + plain
                // concatenation is the safer v1 choice. Real timestamps are
                // a separate deferred item; revisit together.
                var texts      = new List<string>();
                var allTokens  = new List<int>();
                int numChunks  = (length + ChunkSamples - 1) / ChunkSamples;
                for (int c = 0; c < numChunks; c++)
                {
                    int chunkStart = c * ChunkSamples;
                    int chunkLen   = Math.Min(ChunkSamples, length - chunkStart);
                    var chunk      = new float[chunkLen];
                    Array.Copy(audio16k, startSample + chunkStart, chunk, 0, chunkLen);

                    var t = Transcribe(chunk, lang);
                    if (!string.IsNullOrWhiteSpace(t.Text))
                        texts.Add(t.Text);
                    allTokens.AddRange(t.Tokens);
                }
                yield return new WhisperRecognitionResult(
                    segId, string.Join(" ", texts), allTokens);
            }
        }
    }

    /// <summary>
    /// Batched version of <see cref="Recognize"/>. Stacks multiple segments
    /// through a single encoder call and a single decoder-init call, then
    /// runs a padded decoder-step loop with per-item EOT tracking.
    ///
    /// Each input segment is split into 30-s work items (one per chunk);
    /// work items are length-sorted ascending so batches have a narrow
    /// spread in decode length, minimising straggler waste.
    ///
    /// <paramref name="initialBatchSize"/> is the starting batch size. On
    /// <c>OutOfMemory</c>, we halve and retry the same slice — this lets the
    /// scheduler auto-tune to whatever VRAM is actually free without needing
    /// to pre-compute a budget.
    /// </summary>
    public IEnumerable<WhisperRecognitionResult> RecognizeBatched(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio16k,
        string? forceLanguage = null,
        int initialBatchSize = 8)
    {
        string lang = forceLanguage ?? "en";
        var work = BuildWorkItems(segs, audio16k);

        // Length-sort ascending: segments of similar audio length tend to
        // have similar decode-token counts, which bounds the per-batch
        // straggler spread.
        work.Sort((a, b) => a.Audio.Length.CompareTo(b.Audio.Length));

        // Per-segment part accumulator; we emit the merged result the moment
        // a segment's last chunk completes (not in segId order — caller
        // should be order-agnostic, matching Cohere / Qwen3 semantics).
        var segParts  = new List<(int chunk, string text, List<int> tokens)>?[segs.Count];
        var remaining = new int[segs.Count];
        foreach (var w in work) remaining[w.SegId]++;

        int batchSize = Math.Max(1, initialBatchSize);
        int idx = 0;
        while (idx < work.Count)
        {
            int take = Math.Min(batchSize, work.Count - idx);
            var batch = new List<WorkItem>(take);
            for (int i = 0; i < take; i++) batch.Add(work[idx + i]);

            List<BatchOutput> batchResults;
            try
            {
                batchResults = TranscribeBatch(batch, lang);
            }
            catch (OnnxRuntimeException ex) when (IsLikelyOutOfMemory(ex) && take > 1)
            {
                batchSize = Math.Max(1, take / 2);
                continue;   // retry same idx with smaller batch
            }

            foreach (var r in batchResults)
            {
                (segParts[r.SegId] ??= new()).Add((r.ChunkIdx, r.Text, r.Tokens));
                if (--remaining[r.SegId] != 0) continue;

                var parts = segParts[r.SegId]!;
                parts.Sort((a, b) => a.chunk.CompareTo(b.chunk));
                var tokens = new List<int>();
                var texts  = new List<string>();
                foreach (var (_, text, tks) in parts)
                {
                    tokens.AddRange(tks);
                    if (!string.IsNullOrWhiteSpace(text)) texts.Add(text);
                }
                segParts[r.SegId] = null;   // release memory
                yield return new WhisperRecognitionResult(
                    r.SegId, string.Join(" ", texts), tokens);
            }

            idx += take;
        }
    }

    /// <summary>Per-chunk work unit used by the batched scheduler.</summary>
    private sealed record WorkItem(int SegId, int ChunkIdx, float[] Audio);

    /// <summary>Per-chunk output from a batch call.</summary>
    private sealed record BatchOutput(
        int SegId, int ChunkIdx, string Text, List<int> Tokens);

    private static List<WorkItem> BuildWorkItems(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio)
    {
        var items = new List<WorkItem>(segs.Count);
        for (int segId = 0; segId < segs.Count; segId++)
        {
            var (start, end, _) = segs[segId];
            int startSample = Math.Max(0, (int)(start * SampleRate));
            int endSample   = Math.Min(audio.Length, (int)(end * SampleRate));
            int length      = Math.Max(0, endSample - startSample);

            if (length == 0)
            {
                items.Add(new WorkItem(segId, 0, []));  // empty placeholder
                continue;
            }

            if (length <= ChunkSamples)
            {
                var slice = new float[length];
                Array.Copy(audio, startSample, slice, 0, length);
                items.Add(new WorkItem(segId, 0, slice));
            }
            else
            {
                int numChunks = (length + ChunkSamples - 1) / ChunkSamples;
                for (int c = 0; c < numChunks; c++)
                {
                    int chunkStart = c * ChunkSamples;
                    int chunkLen   = Math.Min(ChunkSamples, length - chunkStart);
                    var chunk      = new float[chunkLen];
                    Array.Copy(audio, startSample + chunkStart, chunk, 0, chunkLen);
                    items.Add(new WorkItem(segId, c, chunk));
                }
            }
        }
        return items;
    }

    /// <summary>
    /// Run a single batch through encoder + decoder-init + decoder-step loop.
    /// Items that emit EOT early are kept in the batch and fed EOT as a
    /// pad token — on GPU the extra flops are basically free (matmuls are
    /// batch-parallel), and keeping the tensor shape stable avoids
    /// re-allocating KV buffers.
    /// </summary>
    private List<BatchOutput> TranscribeBatch(List<WorkItem> batch, string lang)
    {
        int B = batch.Count;

        // ── Mels: compute per item, stack into [B, 128, 3000]. Empty work
        //     items (zero-length audio) produce an all-zero mel which the
        //     encoder will see as silence — we drop their outputs below. ──
        var melBuf = new float[B * NMels * ChunkFrames];
        for (int i = 0; i < B; i++)
        {
            if (batch[i].Audio.Length == 0) continue;
            var mel = PrepareChunkMel(batch[i].Audio);
            Array.Copy(mel, 0, melBuf, i * NMels * ChunkFrames, mel.Length);
        }

        // ── Batched encoder ─────────────────────────────────────────────────
        float[] hidden;
        {
            var encTensor = new DenseTensor<float>(melBuf, [B, NMels, ChunkFrames]);
            using var outs = _encoder.Run(
            [
                NamedOnnxValue.CreateFromTensor("input_features", encTensor),
            ]);
            hidden = ExtractFloat(outs.First(o => o.Name == "last_hidden_state"));
        }

        // ── Batched decoder-init (prefix is identical across batch) ─────────
        int langToken = ResolveLanguageToken(lang);
        int[] prefix  = [SotToken, langToken, TranscribeToken, NoTimestampsToken];
        int prefixLen = prefix.Length;

        var inputIds = new long[B * prefixLen];
        for (int b = 0; b < B; b++)
            for (int p = 0; p < prefixLen; p++)
                inputIds[b * prefixLen + p] = prefix[p];

        float[] initLogits;
        List<KvTensor> kv;
        {
            // Merged decoder needs all 16 past_key_values + use_cache_branch.
            var pastEmpty = EmptyPastKv(B);
            var initInputs = new List<NamedOnnxValue>(3 + pastEmpty.Count)
            {
                NamedOnnxValue.CreateFromTensor("input_ids",
                    new DenseTensor<long>(inputIds, [B, prefixLen])),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states",
                    new DenseTensor<float>(hidden, [B, EncoderOutFrames, HiddenSize])),
                NamedOnnxValue.CreateFromTensor("use_cache_branch",
                    new DenseTensor<bool>(new[] { false }, [1])),
            };
            foreach (var t in pastEmpty)
                initInputs.Add(NamedOnnxValue.CreateFromTensor(t.Name, new DenseTensor<float>(t.Data, t.Shape)));

            using var outs = _decoder.Run(initInputs);
            initLogits = ExtractFloat(outs.First(o => o.Name == "logits"));

            kv = new List<KvTensor>(NumDecoderLayers * 4);
            for (int layer = 0; layer < NumDecoderLayers; layer++)
            {
                foreach (bool encoder in new[] { false, true })
                {
                    foreach (bool isValue in new[] { false, true })
                    {
                        string presentName = $"present.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                        string pastName    = $"past_key_values.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                        var tensor = outs.First(o => o.Name == presentName).AsTensor<float>();
                        int seqLen = encoder ? EncoderOutFrames : prefixLen;
                        kv.Add(new KvTensor(pastName, ExtractFloatFromTensor(tensor),
                                            [B, NumHeads, seqLen, HeadDim]));
                    }
                }
            }
        }

        // ── First argmax per item at the last prefix position, with
        //     begin_suppress_tokens applied. ────────────────────────────────
        var outTokens = new List<int>[B];
        var nextTok   = new int[B];
        var done      = new bool[B];
        for (int b = 0; b < B; b++)
        {
            outTokens[b] = new List<int>(32);
            if (batch[b].Audio.Length == 0)
            {
                // Skip silent placeholders — treat as already finished.
                done[b] = true;
                nextTok[b] = EotToken;
                continue;
            }
            int baseIdx = b * prefixLen * VocabSize + (prefixLen - 1) * VocabSize;
            foreach (int id in BeginSuppressTokens)
                initLogits[baseIdx + id] = float.NegativeInfinity;
            int best = 0; float bestVal = initLogits[baseIdx];
            for (int v = 1; v < VocabSize; v++)
            {
                float x = initLogits[baseIdx + v];
                if (x > bestVal) { bestVal = x; best = v; }
            }
            nextTok[b] = best;
            if (best == EotToken) done[b] = true;
        }

        // ── Step loop. Runs until every item has emitted EOT or the context
        //     limit is reached. Finished items are fed EotToken as a no-op
        //     pad; their subsequent outputs are ignored. ─────────────────────
        int step = 0;
        int maxSteps = MaxDecoderLength - prefixLen;
        while (step < maxSteps && !AllTrue(done))
        {
            for (int b = 0; b < B; b++)
                if (!done[b]) outTokens[b].Add(nextTok[b]);

            var stepIds = new long[B];
            for (int b = 0; b < B; b++) stepIds[b] = done[b] ? EotToken : nextTok[b];

            var stepInputs = new List<NamedOnnxValue>(3 + kv.Count)
            {
                NamedOnnxValue.CreateFromTensor("input_ids",
                    new DenseTensor<long>(stepIds, [B, 1])),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states",
                    new DenseTensor<float>(Array.Empty<float>(), [B, 0, HiddenSize])),
                NamedOnnxValue.CreateFromTensor("use_cache_branch",
                    new DenseTensor<bool>(new[] { true }, [1])),
            };
            foreach (var t in kv)
                stepInputs.Add(NamedOnnxValue.CreateFromTensor(t.Name,
                    new DenseTensor<float>(t.Data, t.Shape)));

            using var stepOuts = _decoder.Run(stepInputs);
            float[] stepLogits = ExtractFloat(stepOuts.First(o => o.Name == "logits"));

            // Rebuild KV: decoder KVs come from step output with grown seq_len;
            // encoder KVs stay unchanged.
            var updated = new List<KvTensor>(NumDecoderLayers * 4);
            for (int layer = 0; layer < NumDecoderLayers; layer++)
            {
                foreach (bool encoder in new[] { false, true })
                {
                    foreach (bool isValue in new[] { false, true })
                    {
                        string pastName = $"past_key_values.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                        if (encoder)
                        {
                            updated.Add(kv.First(t => t.Name == pastName));
                        }
                        else
                        {
                            string presentName = $"present.{layer}.decoder.{(isValue ? "value" : "key")}";
                            var tensor = stepOuts.First(o => o.Name == presentName).AsTensor<float>();
                            var dims = tensor.Dimensions;
                            updated.Add(new KvTensor(pastName, ExtractFloatFromTensor(tensor),
                                                     [dims[0], dims[1], dims[2], dims[3]]));
                        }
                    }
                }
            }
            kv = updated;

            for (int b = 0; b < B; b++)
            {
                if (done[b]) continue;
                int baseIdx = b * VocabSize;
                int best = 0; float bestVal = stepLogits[baseIdx];
                for (int v = 1; v < VocabSize; v++)
                {
                    float x = stepLogits[baseIdx + v];
                    if (x > bestVal) { bestVal = x; best = v; }
                }
                nextTok[b] = best;
                if (best == EotToken) done[b] = true;
            }

            step++;
        }

        var results = new List<BatchOutput>(B);
        for (int b = 0; b < B; b++)
        {
            var content = outTokens[b].Where(t => t < SpecialTokenFloor).ToList();
            string text = DecodeTokens(content);
            results.Add(new BatchOutput(batch[b].SegId, batch[b].ChunkIdx, text, outTokens[b]));
        }
        return results;
    }

    private static bool AllTrue(bool[] xs)
    {
        for (int i = 0; i < xs.Length; i++) if (!xs[i]) return false;
        return true;
    }

    /// <summary>
    /// Match the substrings ORT surfaces when a CUDA allocation fails.
    /// Same heuristic as CohereTranscribe — mirrors the BFC arena +
    /// generic "out of memory" messages we see in practice.
    /// </summary>
    private static bool IsLikelyOutOfMemory(OnnxRuntimeException ex)
    {
        string message = ex.Message ?? string.Empty;
        return message.Contains("out of memory",          StringComparison.OrdinalIgnoreCase)
            || message.Contains("failed to allocate",     StringComparison.OrdinalIgnoreCase)
            || message.Contains("cuda out of memory",     StringComparison.OrdinalIgnoreCase)
            || message.Contains("bfc_arena",              StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Transcribe a single 30-second chunk (or less) of 16 kHz mono audio
    /// with greedy decode. Longer inputs are truncated to the first 30 s —
    /// use the segment-based <see cref="Recognize"/> overload for real files.
    /// <paramref name="languageIso"/> is an ISO 639-1 code; <c>"en"</c> by default.
    /// </summary>
    public WhisperTranscript Transcribe(float[] audio16k, string languageIso = "en")
    {
        var sw = new Stopwatch();

        sw.Restart();
        float[] mel = PrepareChunkMel(audio16k);
        sw.Stop(); _msMel += sw.ElapsedMilliseconds;

        sw.Restart();
        float[] hidden = RunEncoder(mel);
        sw.Stop(); _msEncoder += sw.ElapsedMilliseconds;

        int langToken = ResolveLanguageToken(languageIso);
        int[] prefix  = [SotToken, langToken, TranscribeToken, NoTimestampsToken];

        sw.Restart();
        var (initLogits, kv) = RunDecoderInit(prefix, hidden);
        sw.Stop(); _msDecInit += sw.ElapsedMilliseconds;

        sw.Restart();
        int nextToken = ArgmaxLastPosition(initLogits, prefix.Length, BeginSuppressTokens);
        sw.Stop(); _msArgmax += sw.ElapsedMilliseconds;

        var outTokens = new List<int>(64);
        while (nextToken != EotToken && outTokens.Count + prefix.Length < MaxDecoderLength)
        {
            outTokens.Add(nextToken);
            (float[] stepLogits, kv) = RunDecoderStep(nextToken, kv);
            _stepCalls++;

            sw.Restart();
            nextToken = ArgmaxLastPosition(stepLogits, 1, suppress: null);
            sw.Stop(); _msArgmax += sw.ElapsedMilliseconds;
        }

        sw.Restart();
        var contentTokens = outTokens.Where(t => t < SpecialTokenFloor).ToList();
        string text = DecodeTokens(contentTokens);
        sw.Stop(); _msDecode += sw.ElapsedMilliseconds;

        _segments++;
        return new WhisperTranscript(text, outTokens);
    }

    // ── Decoder helpers ─────────────────────────────────────────────────────

    /// <summary>
    /// One KV-cache tensor. Kept as flat float[] + explicit shape so we can
    /// round-trip through DenseTensor&lt;float&gt; each step without
    /// overcomplicating storage. 16 of these per decoder state (4 layers × 4
    /// tensors per layer).
    /// </summary>
    private sealed record KvTensor(string Name, float[] Data, int[] Shape);

    /// <summary>
    /// Build an empty past-KV list to feed into the merged decoder for the
    /// prefill call. The no-cache branch ignores these but the graph inputs
    /// are still required. Zero-sized decoder-seq-len; encoder KV shape must
    /// also be zero-seq because the no-cache branch computes it fresh.
    /// </summary>
    private static List<KvTensor> EmptyPastKv(int batch)
    {
        var list = new List<KvTensor>(NumDecoderLayers * 4);
        for (int layer = 0; layer < NumDecoderLayers; layer++)
            foreach (bool encoder in new[] { false, true })
                foreach (bool isValue in new[] { false, true })
                    list.Add(new KvTensor(
                        $"past_key_values.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}",
                        Array.Empty<float>(),
                        [batch, NumHeads, 0, HeadDim]));
        return list;
    }

    private (float[] logits, List<KvTensor> kv)
        RunDecoderInit(int[] prefix, float[] encoderHidden)
    {
        var inputIds = new long[prefix.Length];
        for (int i = 0; i < prefix.Length; i++) inputIds[i] = prefix[i];

        // Merged decoder needs all 16 past_key_values inputs even on prefill;
        // the use_cache_branch=false branch ignores them.
        var past = EmptyPastKv(1);
        var inputs = new List<NamedOnnxValue>(3 + past.Count)
        {
            NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<long>(inputIds, [1, prefix.Length])),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states",
                new DenseTensor<float>(encoderHidden, [1, EncoderOutFrames, HiddenSize])),
            NamedOnnxValue.CreateFromTensor("use_cache_branch",
                new DenseTensor<bool>(new[] { false }, [1])),
        };
        foreach (var t in past)
            inputs.Add(NamedOnnxValue.CreateFromTensor(t.Name, new DenseTensor<float>(t.Data, t.Shape)));

        using var outputs = _decoder.Run(inputs);

        float[] logits = ExtractFloat(outputs.First(o => o.Name == "logits"));

        // Merged decoder outputs all 16 present tensors (decoder + encoder) on
        // prefill — encoder KV was computed from encoder_hidden_states in the
        // no-cache branch, decoder KV populated from the full prefix.
        var kv = new List<KvTensor>(NumDecoderLayers * 4);
        for (int layer = 0; layer < NumDecoderLayers; layer++)
        {
            foreach (bool encoder in new[] { false, true })
            {
                foreach (bool isValue in new[] { false, true })
                {
                    string presentName = $"present.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                    string pastName    = $"past_key_values.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                    var tensor = outputs.First(o => o.Name == presentName).AsTensor<float>();
                    int seqLen = encoder ? EncoderOutFrames : prefix.Length;
                    kv.Add(new KvTensor(pastName, ExtractFloatFromTensor(tensor),
                                        [1, NumHeads, seqLen, HeadDim]));
                }
            }
        }
        return (logits, kv);
    }

    private (float[] logits, List<KvTensor> updatedKv)
        RunDecoderStep(int nextToken, List<KvTensor> pastKv)
    {
        var sw = new Stopwatch();

        // encoder_hidden_states input is still required on the step call —
        // the with-cache branch doesn't use it but graph inputs are mandatory.
        // Pass a zero-sized placeholder to skip the large [B, 1500, 1280] copy.
        var inputs = new List<NamedOnnxValue>(3 + pastKv.Count)
        {
            NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<long>(new long[] { nextToken }, [1, 1])),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states",
                new DenseTensor<float>(Array.Empty<float>(), [1, 0, HiddenSize])),
            NamedOnnxValue.CreateFromTensor("use_cache_branch",
                new DenseTensor<bool>(new[] { true }, [1])),
        };
        foreach (var t in pastKv)
            inputs.Add(NamedOnnxValue.CreateFromTensor(t.Name, new DenseTensor<float>(t.Data, t.Shape)));

        sw.Restart();
        using var outputs = _decoder.Run(inputs);
        sw.Stop(); _msDecStepOrt += sw.ElapsedMilliseconds;

        sw.Restart();
        float[] logits = ExtractFloat(outputs.First(o => o.Name == "logits"));

        // Merged decoder always emits all 16 present tensors; encoder ones
        // are identity-passthrough of the input (shape check confirmed on the
        // graph). Reuse the input encoder KV to avoid a wasted copy.
        var updatedKv = new List<KvTensor>(NumDecoderLayers * 4);
        for (int layer = 0; layer < NumDecoderLayers; layer++)
        {
            foreach (bool encoder in new[] { false, true })
            {
                foreach (bool isValue in new[] { false, true })
                {
                    string pastName = $"past_key_values.{layer}.{(encoder ? "encoder" : "decoder")}.{(isValue ? "value" : "key")}";
                    if (encoder)
                    {
                        updatedKv.Add(pastKv.First(t => t.Name == pastName));
                    }
                    else
                    {
                        string presentName = $"present.{layer}.decoder.{(isValue ? "value" : "key")}";
                        var tensor = outputs.First(o => o.Name == presentName).AsTensor<float>();
                        // Decoder seq grew by 1. Derive from tensor dims rather than
                        // tracking separately — avoids a second source of truth.
                        var dims = tensor.Dimensions;
                        updatedKv.Add(new KvTensor(pastName, ExtractFloatFromTensor(tensor),
                                                    [dims[0], dims[1], dims[2], dims[3]]));
                    }
                }
            }
        }
        sw.Stop(); _msDecStepExtract += sw.ElapsedMilliseconds;
        return (logits, updatedKv);
    }

    /// <summary>
    /// Argmax over the vocab at the last token position of a logits tensor
    /// shaped [1, seqLen, vocabSize]. Optional <paramref name="suppress"/>
    /// ids get -inf before argmax.
    /// </summary>
    private static int ArgmaxLastPosition(float[] logits, int seqLen, int[]? suppress)
    {
        int vocab = logits.Length / seqLen;
        int baseIdx = (seqLen - 1) * vocab;

        if (suppress is not null)
            foreach (int id in suppress)
                if (id >= 0 && id < vocab)
                    logits[baseIdx + id] = float.NegativeInfinity;

        int best = 0;
        float bestVal = logits[baseIdx];
        for (int i = 1; i < vocab; i++)
        {
            float v = logits[baseIdx + i];
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    private int ResolveLanguageToken(string languageIso)
    {
        string key = $"<|{languageIso.ToLowerInvariant()}|>";
        if (_langToId.TryGetValue(key, out int id)) return id;
        throw new ArgumentException(
            $"Unsupported Whisper language code '{languageIso}'. "
            + $"Expected one of the {_langToId.Count} languages in generation_config.json.");
    }

    // ── Tokenizer ───────────────────────────────────────────────────────────

    /// <summary>
    /// BPE-decode a token-id sequence to UTF-8 text. Regular vocab tokens
    /// (id &lt; 50257) get byte-level-BPE mapped; higher ids are Whisper
    /// specials and should be filtered by the caller before calling this.
    /// </summary>
    public string DecodeTokens(IReadOnlyList<int> tokens)
    {
        var bytes = new List<byte>(tokens.Count * 4);
        foreach (int token in tokens)
        {
            if (token < 0 || token >= _idToToken.Length) continue;
            string? raw = _idToToken[token];
            if (raw is null) continue;
            foreach (char ch in raw)
                if (_byteLevelDecode.TryGetValue(ch, out byte value))
                    bytes.Add(value);
        }
        string text = Encoding.UTF8.GetString(bytes.ToArray());
        return text.Length > 0 && text[0] == ' ' ? text[1..] : text;
    }

    /// <summary>
    /// Load tokenizer.json into a flat id → token string lookup. Handles both
    /// the regular BPE vocab under model.vocab and the added_tokens list
    /// (which is where all Whisper specials live).
    /// </summary>
    private static string?[] LoadTokenizerVocab(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var vocab = root.GetProperty("model").GetProperty("vocab");
        int maxId = -1;
        foreach (var kv in vocab.EnumerateObject())
            maxId = Math.Max(maxId, kv.Value.GetInt32());
        if (root.TryGetProperty("added_tokens", out var addedTokens))
            foreach (var tok in addedTokens.EnumerateArray())
                maxId = Math.Max(maxId, tok.GetProperty("id").GetInt32());

        var idToToken = new string?[maxId + 1];
        foreach (var kv in vocab.EnumerateObject())
            idToToken[kv.Value.GetInt32()] = kv.Name;
        if (root.TryGetProperty("added_tokens", out var added))
            foreach (var tok in added.EnumerateArray())
                idToToken[tok.GetProperty("id").GetInt32()] =
                    tok.GetProperty("content").GetString() ?? "";
        return idToToken;
    }

    /// <summary>
    /// GPT-2 byte-level BPE decode map: unicode surrogate char → underlying byte.
    /// Mirrors the canonical bytes_to_unicode table from OpenAI's GPT-2 repo,
    /// which Whisper inherits. Same as Qwen3Asr's copy; they use the same
    /// pre-tokenizer family.
    /// </summary>
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

    private static Dictionary<string, int> LoadLangToId(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var dict = new Dictionary<string, int>(StringComparer.Ordinal);
        if (root.TryGetProperty("lang_to_id", out var langToId))
            foreach (var kv in langToId.EnumerateObject())
                dict[kv.Name] = kv.Value.GetInt32();
        return dict;
    }

    // ── Tensor extraction helpers ───────────────────────────────────────────

    private static float[] ExtractFloat(DisposableNamedOnnxValue o)
        => ExtractFloatFromTensor(o.AsTensor<float>());

    private static float[] ExtractFloatFromTensor(Tensor<float> tensor)
    {
        if (tensor is DenseTensor<float> dense)
            return dense.Buffer.ToArray();
        var result = new float[tensor.Length];
        int i = 0;
        foreach (float v in tensor) result[i++] = v;
        return result;
    }

    // ── DSP internals ───────────────────────────────────────────────────────

    /// <summary>
    /// Whisper log-mel spectrogram (Hann STFT → power → Slaney-mel → log10 →
    /// clamp(max − 8) → (x + 4) / 4).  Output length: <c>keptFrames = max(1,
    /// ((length + pad·2 − n_fft) / hop_length))</c>.
    /// </summary>
    internal static (float[] mel, int frames) ComputeLogMel(float[] signal, int start, int length)
    {
        int pad       = NFft / 2;
        float[] padded = ReflectPad(signal, start, length, pad);
        int frameCount = ((padded.Length - NFft) / HopLength) + 1;
        int keptFrames = Math.Max(frameCount - 1, 1);
        int freqBins   = (NFft / 2) + 1;
        var mel        = new float[NMels * keptFrames];

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
            if (mel[i] > maxLog) maxLog = mel[i];

        float floor = MathF.Max(maxLog - LogClampSpan, LogFloor);
        for (int i = 0; i < mel.Length; i++)
            mel[i] = (MathF.Max(mel[i], floor) + LogOffset) / LogOffset;

        return (mel, keptFrames);
    }

    private static float[] ReflectPad(float[] signal, int start, int length, int pad)
    {
        if (length == 0) return new float[pad * 2];
        var padded = new float[length + pad * 2];
        Array.Copy(signal, start, padded, pad, length);
        for (int i = 0; i < pad; i++)
        {
            int leftSrc  = Math.Min(length - 1, pad - i);
            int rightSrc = Math.Max(0, length - 2 - i);
            padded[i]                     = signal[start + leftSrc];
            padded[pad + length + i]      = signal[start + rightSrc];
        }
        return padded;
    }

    private static float[,] CreateMelFilterbank()
    {
        int freqBins = (NFft / 2) + 1;
        var fb       = new float[NMels, freqBins];
        var fftFreqs = new double[freqBins];
        for (int k = 0; k < freqBins; k++)
            fftFreqs[k] = (double)k * SampleRate / NFft;

        double fminMel = AudioUtils.HzToMelSlaney(0.0);
        double fmaxMel = AudioUtils.HzToMelSlaney(SampleRate / 2.0);
        var melF = new double[NMels + 2];
        for (int i = 0; i <= NMels + 1; i++)
        {
            double m = fminMel + (fmaxMel - fminMel) * i / (NMels + 1);
            melF[i]  = AudioUtils.MelToHzSlaney(m);
        }

        var fdiff = new double[NMels + 1];
        for (int i = 0; i <= NMels; i++)
            fdiff[i] = melF[i + 1] - melF[i];

        for (int i = 0; i < NMels; i++)
        {
            for (int k = 0; k < freqBins; k++)
            {
                double lower = (fftFreqs[k] - melF[i])     / fdiff[i];
                double upper = (melF[i + 2] - fftFreqs[k]) / fdiff[i + 1];
                fb[i, k]     = (float)Math.Max(0.0, Math.Min(lower, upper));
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

    // ── Session options ─────────────────────────────────────────────────────

    private static SessionOptions MakeSessionOptions(ExecutionProvider ep)
    {
        var opts = new SessionOptions();
        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                    try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                try { opts.AppendExecutionProvider_DML(0); } catch { }
                break;
            case ExecutionProvider.Cuda:
                opts.AppendExecutionProvider_CUDA(0);
                break;
            case ExecutionProvider.DirectML:
                opts.AppendExecutionProvider_DML(0);
                break;
            case ExecutionProvider.Cpu:
                break;
        }
        return opts;
    }
}
