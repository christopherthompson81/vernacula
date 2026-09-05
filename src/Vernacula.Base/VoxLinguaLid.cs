using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// VoxLingua107 language identifier (ECAPA-TDNN) using ONNX Runtime.
///
/// <para>
/// Input: raw 16 kHz mono audio as <c>float[]</c>. Preprocessing (FBANK
/// via Conv1D, per-utterance MVN) is folded into the graph.
/// </para>
///
/// <para>
/// Outputs a <see cref="LidResult"/> with a top-1 language, a top-K list,
/// an ambiguity flag (top-1 probability below
/// <see cref="Config.VoxLinguaAmbiguityThreshold"/>), and the 256-dim
/// pooled embedding. The embedding is kept on the surface for future
/// use (language-similarity clustering, mixed-language per-segment
/// analysis); consumers are free to ignore it.
/// </para>
///
/// <para>
/// Thread-safety: a single instance can be used from multiple threads
/// — ORT's InferenceSession supports concurrent Run() calls. The session
/// itself is not lightweight to construct (~100 ms on CPU, ~500 ms on
/// CUDA) so build once and share.
/// </para>
/// </summary>
public sealed class VoxLinguaLid : IDisposable
{
    public const int SampleRate = 16_000;
    public const int NumClasses = 107;
    public const int EmbeddingDim = 256;

    private readonly InferenceSession _session;
    private readonly IReadOnlyList<LanguageEntry> _languages;
    private readonly float _ambiguityThreshold;
    private bool _disposed;

    /// <summary>Companion to <c>lang_map.json</c>: index → (iso, name).</summary>
    private sealed record LanguageEntry(string Iso, string Name);

    public VoxLinguaLid(
        string modelDir,
        ExecutionProvider ep = ExecutionProvider.Auto,
        float ambiguityThreshold = Config.VoxLinguaAmbiguityThreshold)
    {
        string modelPath = Path.Combine(modelDir, Config.VoxLinguaModelFile);
        string langMapPath = Path.Combine(modelDir, Config.VoxLinguaLangMapFile);

        if (!File.Exists(modelPath))
            throw new FileNotFoundException(
                $"VoxLingua107 ONNX model not found at '{modelPath}'. " +
                $"Run the export script or download the assets before constructing the classifier.");
        if (!File.Exists(langMapPath))
            throw new FileNotFoundException(
                $"VoxLingua107 language map not found at '{langMapPath}'.");

        _languages = LoadLanguageMap(langMapPath);
        if (_languages.Count != NumClasses)
            throw new InvalidDataException(
                $"lang_map.json has {_languages.Count} entries; expected {NumClasses}.");

        _session = CreateSession(modelPath, ep);
        _ambiguityThreshold = ambiguityThreshold;
    }

    /// <summary>
    /// Run LID on a single clip. <paramref name="audio"/> must be 16 kHz mono
    /// float32 samples. The full clip is fed to the model — no internal
    /// chunking or averaging; the caller chooses the window (typically the
    /// longest VAD segment, clamped to ~15 s).
    /// </summary>
    public LidResult Classify(ReadOnlySpan<float> audio, int topK = 5)
    {
        if (topK <= 0)
            throw new ArgumentException($"topK must be > 0, got {topK}.", nameof(topK));
        if (audio.Length < SampleRate)
            throw new ArgumentException(
                $"audio is {audio.Length} samples (~{audio.Length / (float)SampleRate:F2} s); " +
                $"VoxLingua107 needs at least 1 s of audio to produce stable output.",
                nameof(audio));

        float[] inputBuf = audio.ToArray();
        var inputTensor = new DenseTensor<float>(inputBuf, [1, inputBuf.Length]);

        var feeds = new[]
        {
            NamedOnnxValue.CreateFromTensor("audio", inputTensor),
        };
        using var outputs = _session.Run(feeds);

        var logits    = outputs.First(r => r.Name == "logits").AsTensor<float>();
        var embedding = outputs.First(r => r.Name == "embedding").AsTensor<float>();

        if (logits.Length != NumClasses)
            throw new InvalidOperationException(
                $"logits shape mismatch: got {logits.Length}, expected {NumClasses}.");
        if (embedding.Length != EmbeddingDim)
            throw new InvalidOperationException(
                $"embedding shape mismatch: got {embedding.Length}, expected {EmbeddingDim}.");

        float[] logitsArr = logits.ToArray();
        float[] probs = Softmax(logitsArr);
        var top = SortTopK(probs, Math.Min(topK, NumClasses));
        var topCandidate = top[0];
        bool ambiguous = topCandidate.Probability < _ambiguityThreshold;

        float[] embeddingCopy = embedding.ToArray();
        return new LidResult(
            Top:                 topCandidate,
            TopK:                top,
            IsAmbiguous:         ambiguous,
            Embedding:           embeddingCopy,
            ClipDurationSeconds: audio.Length / (float)SampleRate);
    }

    /// <summary>
    /// Pick the longest VAD segment in <paramref name="audio"/>, clamp the
    /// classification window to
    /// <paramref name="defaultClipSeconds"/>, and run <see cref="Classify"/>.
    ///
    /// <para>
    /// If the first pass is flagged ambiguous and the longest VAD segment is
    /// at least <paramref name="escalationClipSeconds"/> long, re-runs on
    /// the longer window — per Phase 6 of the perf investigation the
    /// extra context helps resolve close-family confusions (ru/uk/be,
    /// de/lb/da) that short clips flap on.
    /// </para>
    ///
    /// <para>
    /// Returns <c>null</c> when no VAD segment is at least 1 second long
    /// (effectively silent audio). The classification window is centred
    /// inside the chosen VAD segment so we sample the middle rather than
    /// onset/offset artefacts.
    /// </para>
    /// </summary>
    public LidResult? ClassifyLongestSegment(
        float[] audio,
        IReadOnlyList<(double startSec, double endSec)> vadSegments,
        int defaultClipSeconds = Config.VoxLinguaDefaultClipSeconds,
        int escalationClipSeconds = Config.VoxLinguaEscalationClipSeconds,
        int topK = 5)
    {
        var (segStart, segEnd) = PickLongestSegment(vadSegments);
        double segDuration = segEnd - segStart;
        if (segDuration < 1.0) return null;

        int defaultSamples = defaultClipSeconds * SampleRate;
        float[] initialClip = SliceCentered(audio, segStart, segEnd, defaultSamples);
        if (initialClip.Length < SampleRate) return null;

        var first = Classify(initialClip, topK);

        if (first.IsAmbiguous && segDuration >= escalationClipSeconds)
        {
            int longSamples = escalationClipSeconds * SampleRate;
            float[] longClip = SliceCentered(audio, segStart, segEnd, longSamples);
            if (longClip.Length >= defaultSamples)
                return Classify(longClip, topK);
        }

        return first;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    /// <summary>Pick the longest VAD segment; empty input → (0, 0).</summary>
    private static (double start, double end) PickLongestSegment(
        IReadOnlyList<(double startSec, double endSec)> segs)
    {
        if (segs is null || segs.Count == 0) return (0, 0);
        double bestDur = -1;
        (double start, double end) best = (0, 0);
        foreach (var (s, e) in segs)
        {
            double dur = e - s;
            if (dur > bestDur) { bestDur = dur; best = (s, e); }
        }
        return best;
    }

    /// <summary>
    /// Extract up to <paramref name="targetSamples"/> samples from
    /// <paramref name="audio"/>, centred inside the given VAD segment.
    /// Falls back to whatever the segment contains if shorter.
    /// </summary>
    private static float[] SliceCentered(
        float[] audio, double segStart, double segEnd, int targetSamples)
    {
        int segStartSample = Math.Clamp((int)Math.Round(segStart * SampleRate), 0, audio.Length);
        int segEndSample   = Math.Clamp((int)Math.Round(segEnd   * SampleRate), 0, audio.Length);
        int segLen = segEndSample - segStartSample;
        if (segLen <= 0) return [];

        int take = Math.Min(targetSamples, segLen);
        int offset = segStartSample + Math.Max(0, (segLen - take) / 2);
        var slice = new float[take];
        Array.Copy(audio, offset, slice, 0, take);
        return slice;
    }

    private IReadOnlyList<LidCandidate> SortTopK(float[] probs, int k)
    {
        // Partial-sort: indices of the k largest entries.
        var indexed = new List<(int idx, float p)>(probs.Length);
        for (int i = 0; i < probs.Length; i++)
            indexed.Add((i, probs[i]));
        indexed.Sort((a, b) => b.p.CompareTo(a.p));

        var result = new List<LidCandidate>(k);
        for (int i = 0; i < k; i++)
        {
            var (idx, p) = indexed[i];
            var lang = _languages[idx];
            result.Add(new LidCandidate(idx, lang.Iso, lang.Name, p));
        }
        return result;
    }

    private static float[] Softmax(float[] logits)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        float[] exp = new float[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = MathF.Exp(logits[i] - max);
            exp[i] = v;
            sum += v;
        }

        float inv = (float)(1.0 / sum);
        for (int i = 0; i < exp.Length; i++)
            exp[i] *= inv;
        return exp;
    }

    private static IReadOnlyList<LanguageEntry> LoadLanguageMap(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);

        var root = doc.RootElement;
        var entries = new LanguageEntry?[NumClasses];
        foreach (var prop in root.EnumerateObject())
        {
            if (!int.TryParse(prop.Name, out int idx) || idx < 0 || idx >= NumClasses)
                throw new InvalidDataException(
                    $"lang_map.json has invalid index '{prop.Name}'; expected 0–{NumClasses - 1}.");
            string iso = prop.Value.GetProperty("iso").GetString()
                         ?? throw new InvalidDataException($"lang_map.json entry {idx} missing 'iso'.");
            string name = prop.Value.GetProperty("name").GetString()
                          ?? throw new InvalidDataException($"lang_map.json entry {idx} missing 'name'.");
            entries[idx] = new LanguageEntry(iso, name);
        }

        for (int i = 0; i < NumClasses; i++)
            if (entries[i] is null)
                throw new InvalidDataException(
                    $"lang_map.json is missing entry for index {i}.");

        return entries!;
    }

    private static InferenceSession CreateSession(string modelPath, ExecutionProvider ep)
    {
        var opts = new SessionOptions
        {
            // Phase 2 of the perf investigation showed this tier is as fast as
            // ORT_ENABLE_ALL on a well-optimised graph while being a little
            // cheaper to compile.
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            // Phase 2 sweet spot: at most 8 physical cores. Above 8 threads
            // SMT contention on the Conv1D hot path hurts more than it helps.
            IntraOpNumThreads = Math.Min(8, Math.Max(1, Environment.ProcessorCount)),
        };

        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                {
                    try { opts.AppendExecutionProvider_CUDA(0); }
                    catch { /* fall through to CPU */ }
                }
                try { opts.AppendExecutionProvider_DML(0); }
                catch { /* not available */ }
                break;
            case ExecutionProvider.Cuda:
                try { opts.AppendExecutionProvider_CUDA(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("CUDA EP not available."); }
                break;
            case ExecutionProvider.DirectML:
                try { opts.AppendExecutionProvider_DML(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("DirectML EP not available."); }
                break;
        }

        return new InferenceSession(modelPath, opts);
    }
}
