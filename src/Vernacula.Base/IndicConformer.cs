using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// AI4Bharat IndicConformer (Hybrid CTC-RNNT, CTC head only) — C# port of
/// the validate_indicconformer_package.py reference decoder, matching the
/// ONNX shipping package produced by scripts/indicconformer_export/ and
/// scripts/indicconformer_export/repackage_600m_indicconformer.py.
///
/// Mirrors <see cref="Parakeet"/>'s public surface so
/// <c>TranscriptionService</c> can swap backends without special-casing,
/// but the decoder runs a shared-encoder + per-language-span greedy CTC
/// collapse instead of an RNNT loop. Blank is implicit at id = vocab
/// size; the shared CTC dim is vocab_size + 1 (5633 for the published
/// 22-language checkpoint). Per-call <c>lang</c> picks which 256-token
/// span to decode into.
///
/// Helpers like <see cref="PadList"/>, <see cref="SegmentWaveform"/>, and
/// the VRAM-bounded batcher are duplicated verbatim from
/// <see cref="Parakeet"/> on this first pass; extracting them into a
/// shared <c>AsrUtils</c> is follow-up work.
/// </summary>
public sealed class IndicConformer : IDisposable
{
    private readonly string _modelPath;

    private readonly InferenceSession _preprocessor;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _ctcDecoder;

    /// <summary>Flat vocab, index-addressed (ids 0..vocab_size-1).</summary>
    private readonly string[] _vocab;

    /// <summary>Per-language [start, length] into the flat vocab.</summary>
    private readonly Dictionary<string, (int start, int length)> _langSpans;

    /// <summary>CTC blank token id — always equals <see cref="_vocab"/>.Length.</summary>
    private readonly int _blankIdx;

    // ── Construction ─────────────────────────────────────────────────────────

    public IndicConformer(string modelPath)
        : this(modelPath, Config.EncoderFile, Config.CtcDecoderFile) { }

    public IndicConformer(
        string modelPath,
        string encoderFile,
        string ctcDecoderFile,
        ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _modelPath = modelPath;

        var cpuOpts = new SessionOptions();
        _preprocessor = new InferenceSession(
            Path.Combine(modelPath, Config.PreprocessorFile), cpuOpts);

        var gpuOpts = MakeSessionOptions(ep);
        _encoder    = new InferenceSession(Path.Combine(modelPath, encoderFile),    gpuOpts);
        _ctcDecoder = new InferenceSession(Path.Combine(modelPath, ctcDecoderFile), gpuOpts);

        _vocab     = LoadFlatVocab(Path.Combine(modelPath, Config.VocabFile));
        _langSpans = LoadLanguageSpans(Path.Combine(modelPath, Config.IndicConformerLanguageSpansFile));
        _blankIdx  = _vocab.Length;
    }

    private static SessionOptions MakeSessionOptions(ExecutionProvider ep)
    {
        var opts = new SessionOptions();
        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                {
                    try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                }
                try { opts.AppendExecutionProvider_DML(0); } catch { }
                break;
            case ExecutionProvider.Cuda:
                try { opts.AppendExecutionProvider_CUDA(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("CUDA EP not available in current OnnxRuntime build."); }
                break;
            case ExecutionProvider.DirectML:
                try { opts.AppendExecutionProvider_DML(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("DirectML EP not available. Build with -p:UseDirectML=true."); }
                break;
            case ExecutionProvider.Cpu:
                break;
        }
        return opts;
    }

    // ── Vocab + language spans ───────────────────────────────────────────────

    private static string[] LoadFlatVocab(string path)
    {
        var raw = File.ReadAllLines(path);
        // Shipping vocab.txt is one token per line with id = line index, no
        // trailing blank row. A final empty line from newline-terminated
        // writes is tolerated by dropping blanks at the end only — mid-file
        // blanks would shift ids and are rejected.
        int end = raw.Length;
        while (end > 0 && string.IsNullOrEmpty(raw[end - 1])) end--;
        for (int i = 0; i < end; i++)
            if (string.IsNullOrEmpty(raw[i]))
                throw new InvalidDataException(
                    $"{path}: empty line at id {i}. Vocab format is one token per line, " +
                    "no gaps — blank lines would misalign the CTC output.");
        var clean = new string[end];
        Array.Copy(raw, clean, end);
        return clean;
    }

    private static Dictionary<string, (int start, int length)> LoadLanguageSpans(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        var spans = new Dictionary<string, (int, int)>(StringComparer.OrdinalIgnoreCase);
        foreach (var prop in root.GetProperty("languages").EnumerateObject())
        {
            int start  = prop.Value.GetProperty("start").GetInt32();
            int length = prop.Value.GetProperty("length").GetInt32();
            spans[prop.Name] = (start, length);
        }
        return spans;
    }

    // ── Array helpers (duplicated from Parakeet; candidate for AsrUtils) ─────

    public static (float[,] padded, long[] lens) PadList(IReadOnlyList<float[]> arrays)
    {
        if (arrays.Count == 0)
            return (new float[0, 0], Array.Empty<long>());

        var lens   = new long[arrays.Count];
        long maxLen = 0;
        for (int i = 0; i < arrays.Count; i++)
        {
            lens[i] = arrays[i].Length;
            if (lens[i] > maxLen) maxLen = lens[i];
        }

        var padded = new float[arrays.Count, maxLen];
        for (int i = 0; i < arrays.Count; i++)
        {
            int len = (int)Math.Min(lens[i], maxLen);
            for (int j = 0; j < len; j++)
                padded[i, j] = arrays[i][j];
        }
        return (padded, lens);
    }

    public static List<float[]> SegmentWaveform(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio)
    {
        var results = new List<float[]>(segs.Count);
        foreach (var (start, end, _) in segs)
        {
            int startFrame = Math.Max((int)(start * Config.SampleRate), 0);
            int endFrame   = Math.Min((int)(end   * Config.SampleRate), audio.Length);
            int len        = Math.Max(endFrame - startFrame, 0);
            var seg        = new float[len];
            if (len > 0)
                Array.Copy(audio, startFrame, seg, 0, len);
            results.Add(seg);
        }
        return results;
    }

    private static float[] Flatten2D(float[,] a, int d0, int d1)
    {
        var flat = new float[d0 * d1];
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                flat[i * d1 + j] = a[i, j];
        return flat;
    }

    private static float[] Flatten3D(float[,,] a, int d0, int d1, int d2)
    {
        var flat = new float[d0 * d1 * d2];
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    flat[i * d1 * d2 + j * d2 + k] = a[i, j, k];
        return flat;
    }

    // ── ONNX inference helpers ───────────────────────────────────────────────

    private (float[,,] features, long[] featLens) PreprocessFeatures(
        float[,] waveforms, long[] waveformsLens)
    {
        int B = waveforms.GetLength(0);
        int T = waveforms.GetLength(1);

        var waveT = new DenseTensor<float>(Flatten2D(waveforms, B, T), new[] { B, T });
        var waveL = new DenseTensor<long>(waveformsLens, new[] { B });

        using var results = _preprocessor.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("waveforms",      waveT),
            NamedOnnxValue.CreateFromTensor("waveforms_lens", waveL),
        });

        var featT  = results.First(r => r.Name == "features").AsTensor<float>();
        var featLT = results.First(r => r.Name == "features_lens").AsTensor<long>();

        int D2 = featT.Dimensions[1];
        int T2 = featT.Dimensions[2];
        var features = new float[B, D2, T2];
        for (int b = 0; b < B; b++)
            for (int d = 0; d < D2; d++)
                for (int t = 0; t < T2; t++)
                    features[b, d, t] = featT[b, d, t];

        var featLens = new long[B];
        for (int b = 0; b < B; b++)
            featLens[b] = featLT[b];

        return (features, featLens);
    }

    private (float[,,] encoded, long[] encodedLens) Encode(float[,,] features, long[] lens)
    {
        int B = features.GetLength(0);
        int D = features.GetLength(1);
        int T = features.GetLength(2);

        var featT = new DenseTensor<float>(Flatten3D(features, B, D, T), new[] { B, D, T });
        var lenT  = new DenseTensor<long>(lens, new[] { B });

        using var results = _encoder.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("features",      featT),
            NamedOnnxValue.CreateFromTensor("features_lens", lenT),
        });

        var outT  = results.First(r => r.Name == "encoded").AsTensor<float>();
        var outLT = results.First(r => r.Name == "encoded_lens").AsTensor<long>();

        // Encoder emits [B, D, T]; transpose to [B, T, D] so the CTC
        // session and downstream argmax operate over contiguous time.
        int D2 = outT.Dimensions[1];
        int T2 = outT.Dimensions[2];
        var encoded = new float[B, T2, D2];
        for (int b = 0; b < B; b++)
            for (int d = 0; d < D2; d++)
                for (int t = 0; t < T2; t++)
                    encoded[b, t, d] = outT[b, d, t];

        var encodedLens = new long[B];
        for (int b = 0; b < B; b++)
            encodedLens[b] = outLT[b];

        return (encoded, encodedLens);
    }

    private float[,,] RunCtcDecoder(float[,,] encoded)
    {
        int B = encoded.GetLength(0);
        int T = encoded.GetLength(1);
        int D = encoded.GetLength(2);

        // ctc_decoder.onnx expects [B, D, T] (ConvASRDecoder convention),
        // same layout the encoder emits natively. Re-transpose before
        // feeding.
        var encBDT = new float[B * D * T];
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
                for (int d = 0; d < D; d++)
                    encBDT[((b * D) + d) * T + t] = encoded[b, t, d];

        var encT = new DenseTensor<float>(encBDT, new[] { B, D, T });

        using var results = _ctcDecoder.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoded", encT),
        });

        var logitsT = results.First(r => r.Name == "logits").AsTensor<float>();
        // Graph returns [B, T, V]. Copy into a managed 3D array so callers
        // don't have to hold an InferenceSession result alive.
        int V = logitsT.Dimensions[2];
        var logits = new float[B, T, V];
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++)
                for (int v = 0; v < V; v++)
                    logits[b, t, v] = logitsT[b, t, v];
        return logits;
    }

    // ── CTC greedy decode ────────────────────────────────────────────────────

    /// <summary>
    /// Masked greedy CTC. For each frame: argmax over the (span-length + 1)
    /// positions that belong to the requested language plus the shared
    /// blank. Collapse runs of the same token, drop blanks, return emitted
    /// global token ids with per-token timestamps, durations, and
    /// log-softmax scores over the masked subspace.
    /// </summary>
    private (List<int> tokens, List<int> timestamps, List<int> durations, List<float> logprobs)
        GreedyCtcDecode(
            float[,,] logits, int batchIdx, int T,
            int spanStart, int spanLength)
    {
        int localBlankId = spanLength;       // masked-subspace index of blank
        int vocabDim     = logits.GetLength(2);

        var tokens     = new List<int>();
        var timestamps = new List<int>();
        var durations  = new List<int>();
        var logprobs   = new List<float>();

        // State across frames: currentLocalId is what we last emitted (or
        // localBlankId when in blank); startFrame is where the current run
        // began; bestLogprob tracks the tightest log-prob during the run.
        int currentLocalId = -1;
        int startFrame     = 0;
        float bestLogprob  = float.NegativeInfinity;

        void Emit(int endFrameExclusive)
        {
            if (currentLocalId < 0 || currentLocalId == localBlankId) return;
            int globalId = spanStart + currentLocalId;
            tokens.Add(globalId);
            timestamps.Add(startFrame);
            durations.Add(endFrameExclusive - startFrame);
            logprobs.Add(bestLogprob);
        }

        for (int t = 0; t < T; t++)
        {
            // Argmax in the masked subspace without materializing a copy.
            int   bestLocal  = -1;
            float bestLogit  = float.NegativeInfinity;
            for (int i = 0; i < spanLength; i++)
            {
                float v = logits[batchIdx, t, spanStart + i];
                if (v > bestLogit) { bestLogit = v; bestLocal = i; }
            }
            float blankLogit = logits[batchIdx, t, _blankIdx];
            if (blankLogit > bestLogit) { bestLogit = blankLogit; bestLocal = localBlankId; }

            if (bestLocal == currentLocalId)
            {
                // Extend the current run. Update bestLogprob with a fresh
                // log-softmax over the masked subspace at this frame.
                float lp = LogSoftmaxSelected(
                    logits, batchIdx, t, spanStart, spanLength, _blankIdx, bestLocal);
                if (lp > bestLogprob) bestLogprob = lp;
                continue;
            }

            // Run boundary: emit the outgoing run (if not blank), then start
            // a new one.
            Emit(endFrameExclusive: t);
            currentLocalId = bestLocal;
            startFrame     = t;
            bestLogprob    = LogSoftmaxSelected(
                logits, batchIdx, t, spanStart, spanLength, _blankIdx, bestLocal);
        }
        Emit(endFrameExclusive: T);

        _ = vocabDim; // silence analyser; keep the variable for future-prefix debugging
        return (tokens, timestamps, durations, logprobs);
    }

    /// <summary>
    /// log-softmax over the 257-dim subspace (span of <paramref name="spanLength"/>
    /// language tokens + the shared blank at <paramref name="blankIdx"/>),
    /// returning the value at <paramref name="localIdx"/>.
    /// </summary>
    private static float LogSoftmaxSelected(
        float[,,] logits, int b, int t,
        int spanStart, int spanLength, int blankIdx, int localIdx)
    {
        // Two-pass stable log-softmax: first find max for numerical stability,
        // then accumulate log-sum-exp.
        float max = float.NegativeInfinity;
        for (int i = 0; i < spanLength; i++)
        {
            float v = logits[b, t, spanStart + i];
            if (v > max) max = v;
        }
        float blankV = logits[b, t, blankIdx];
        if (blankV > max) max = blankV;

        double sum = 0.0;
        for (int i = 0; i < spanLength; i++)
            sum += Math.Exp(logits[b, t, spanStart + i] - max);
        sum += Math.Exp(blankV - max);

        float chosen = localIdx == spanLength ? blankV : logits[b, t, spanStart + localIdx];
        return (float)(chosen - max - Math.Log(sum));
    }

    // ── VRAM-bounded batching (duplicated from Parakeet) ─────────────────────

    private static long ComputeMaxFrames()
    {
        var (_, freeMb) = HardwareInfo.GetGpuMemoryMb();
        if (freeMb <= 0) return Config.FallbackMaxFrames;

        double available = freeMb - Config.VramSafetyMb;
        if (available <= 0) return Config.FallbackMaxFrames;

        long frames = (long)((available - Config.VramInterceptMb) / Config.VramSlopePerSample);
        return frames > 0 ? frames : Config.FallbackMaxFrames;
    }

    private static IEnumerable<List<float[]>> MakeBatches(
        List<float[]> sorted, int maxBatchSize, long maxFrames)
    {
        var current    = new List<float[]>();
        long currentMax = 0;

        foreach (var wave in sorted)
        {
            long newMax    = Math.Max(currentMax, wave.Length);
            long projected = (current.Count + 1) * newMax;
            bool sizeLimit  = current.Count >= maxBatchSize;
            bool frameLimit = current.Count > 0 && projected > maxFrames;

            if (sizeLimit || frameLimit)
            {
                yield return current;
                current    = new List<float[]>();
                currentMax = 0;
                newMax     = wave.Length;
            }

            current.Add(wave);
            currentMax = Math.Max(currentMax, wave.Length);
        }

        if (current.Count > 0)
            yield return current;
    }

    // ── Public recognition API ───────────────────────────────────────────────

    public IndicConformerPreparedBatch? PrepareBatch(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio)
    {
        var waveforms = SegmentWaveform(segs, audio);
        if (waveforms.Count == 0) return null;

        var sorted = waveforms
            .Select((w, i) => (w, origIdx: i))
            .OrderBy(x => x.w.Length)
            .ToList();

        long maxFrames   = ComputeMaxFrames();
        var sortedWaves  = sorted.Select(x => x.w).ToList();
        var batchedWaves = MakeBatches(sortedWaves, Config.MaxBatchSize, maxFrames).ToList();

        var batches = new List<(float[,,] features, long[] featLens, int segCount)>(batchedWaves.Count);
        foreach (var batch in batchedWaves)
        {
            var (padded, lens)       = PadList(batch);
            var (features, featLens) = PreprocessFeatures(padded, lens);
            batches.Add((features, featLens, batch.Count));
        }

        return new IndicConformerPreparedBatch(
            sorted.Select(x => x.origIdx).ToList(), batches);
    }

    /// <summary>
    /// GPU phase: encode and CTC-decode a batch already preprocessed by
    /// <see cref="PrepareBatch"/>. Results are yielded in original segment
    /// order. <paramref name="lang"/> selects which 256-token span the
    /// language mask will expose before argmax (BCP-47 primary subtag —
    /// e.g. "hi", "bn", "ta"; case-insensitive).
    /// </summary>
    public IEnumerable<(int segId, string text, List<int> tokens, List<int> timestamps, List<int> durations, List<float> logprobs)>
        RecognizePrepared(IndicConformerPreparedBatch prepared, string lang)
    {
        if (!_langSpans.TryGetValue(lang, out var span))
            throw new ArgumentException(
                $"Unknown language '{lang}'. Supported: " +
                string.Join(", ", _langSpans.Keys.OrderBy(k => k)),
                nameof(lang));

        int sortedPos = 0;
        foreach (var (features, featLens, segCount) in prepared.Batches)
        {
            var (encoded, encodedLens) = Encode(features, featLens);
            var logits = RunCtcDecoder(encoded);
            int B = logits.GetLength(0);

            for (int b = 0; b < B; b++)
            {
                int T = (int)encodedLens[b];
                var (tokens, timestamps, durations, logprobs) = GreedyCtcDecode(
                    logits, b, T, span.start, span.length);

                string text = Detokenize(tokens);
                int origIdx = prepared.OriginalIndices[sortedPos + b];
                yield return (origIdx, text, tokens, timestamps, durations, logprobs);
            }
            sortedPos += segCount;
        }
    }

    /// <summary>
    /// Convenience wrapper: CPU preprocess then GPU encode+decode in one call.
    /// </summary>
    public IEnumerable<(int segId, string text, List<int> tokens, List<int> timestamps, List<int> durations, List<float> logprobs)>
        Recognize(
            IReadOnlyList<(double start, double end, string spk)> segs,
            float[] audio,
            string lang)
    {
        var prepared = PrepareBatch(segs, audio);
        if (prepared == null) yield break;
        foreach (var result in RecognizePrepared(prepared, lang))
            yield return result;
    }

    /// <summary>
    /// SentencePiece-style detokenize: each piece with a leading `▁`
    /// (U+2581) marks a word boundary in the language-specific tokenizer.
    /// Trim edge whitespace; the first emission usually starts with the
    /// marker and would otherwise leave a leading space.
    /// </summary>
    private string Detokenize(List<int> tokenIds)
    {
        var sb = new System.Text.StringBuilder(tokenIds.Count * 2);
        foreach (int id in tokenIds)
        {
            if (id < 0 || id >= _vocab.Length) continue;
            string piece = _vocab[id];
            if (piece.Length == 0) continue;
            if (piece[0] == '\u2581') { sb.Append(' '); sb.Append(piece.AsSpan(1)); }
            else sb.Append(piece);
        }
        return sb.ToString().Trim();
    }

    // ── Supported languages ──────────────────────────────────────────────────

    /// <summary>BCP-47 primary subtags this package can decode (22 for the
    /// published IndicConformer).</summary>
    public IReadOnlyCollection<string> SupportedLanguages => _langSpans.Keys;

    // ── IDisposable ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        _preprocessor.Dispose();
        _encoder.Dispose();
        _ctcDecoder.Dispose();
    }
}

/// <summary>
/// CPU-preprocessed features ready for GPU encoding.
/// Separate type from <c>PreparedBatch</c> so the two backends can evolve
/// independently. TranscriptionService branches by backend already, so
/// sharing the wrapper type offers no reduction in branches.
/// </summary>
public sealed class IndicConformerPreparedBatch(
    IReadOnlyList<int> originalIndices,
    IReadOnlyList<(float[,,] features, long[] featLens, int segCount)> batches)
{
    public IReadOnlyList<int> OriginalIndices { get; } = originalIndices;
    public IReadOnlyList<(float[,,] features, long[] featLens, int segCount)> Batches { get; } = batches;
}
