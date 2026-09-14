using AudioCpp;

namespace Vernacula.AudioCpp;

/// <summary>
/// One recognised segment, in the shape <c>TranscriptionService</c> stores.
/// </summary>
/// <param name="Words">
/// The recognition unit. audio.cpp reports words, not sub-word token ids, so
/// these are the runs the transcript editor colours — see
/// <c>VocabKind.AudioCpp</c>.
/// </param>
/// <param name="StartFrames">
/// Word start times in 10 ms frames, the unit the timestamps column already
/// uses. Measured, not spread evenly: the ABI reports real word boundaries.
/// </param>
/// <param name="Confidences">
/// Per-word confidence as reported. NOT a log probability — see the note on
/// <see cref="AudioCppAsr"/>.
/// </param>
public sealed record AudioCppRecognition(
    int SegmentId,
    string Text,
    IReadOnlyList<string> Words,
    IReadOnlyList<int> StartFrames,
    IReadOnlyList<float> Confidences,
    string? Language);

/// <summary>
/// Recognises Vernacula's segments through audio.cpp's C ABI.
/// </summary>
/// <remarks>
/// <para>
/// This is the one backend that does not run on ONNX Runtime. It exists to test
/// whether the published ABI is enough to drive a real pipeline, so it stays
/// deliberately thin: the segmentation, diarization and LID grouping around it
/// are Vernacula's own, and only recognition crosses the boundary.
/// </para>
/// <para>
/// One session is created and reused across every segment. Creating one per
/// segment would reload the weights each time; the ABI keeps the loaded model
/// behind the session, which is the whole reason to embed rather than shell out.
/// </para>
/// <para>
/// ⚠ Confidence is not a logprob. The editor colours runs by log probability and
/// audio.cpp reports a confidence in [0,1]. They are stored as-is rather than
/// converted: log(confidence) would put a confident word at 0 and an uncertain
/// one at minus infinity, which reads as "worse than any ONNX backend" on the
/// same audio. Whether the editor should scale them per-backend is an open
/// question in the investigation doc, not something to guess at here.
/// </para>
/// </remarks>
public sealed class AudioCppAsr : IDisposable
{
    // The timestamps column is in 10 ms frames across every other backend.
    private const int FramesPerSecond = 100;

    // Mirrors Vernacula.Base's Config.SampleRate rather than referencing it: this
    // project stays off the ONNX stack on purpose, so that what is under test is
    // the ABI and not a hybrid. The two must agree, and a mismatch would show as
    // segments sliced at the wrong offsets -- so the caller asserts it.
    public const int SampleRate = 16_000;

    private readonly AudioCppRegistry _registry;
    private readonly AudioCppModel    _model;
    private readonly AudioCppSession? _session;

    /// <param name="modelPath">A GGUF or a package directory, as the engine takes it.</param>
    /// <param name="familyHint">
    /// The engine's family name, e.g. <c>parakeet_tdt</c>. Passed rather than
    /// guessed: the registry can resolve a family from the weights, but a wrong
    /// guess surfaces as an unsupported-family failure at load rather than as a
    /// bad transcript later.
    /// </param>
    /// <summary>The backend the session actually opened on.</summary>
    public string Backend { get; } = "";

    /// <param name="backends">
    /// Backends to try, in order, taking the first that opens. More than one is
    /// how "auto" is expressed: whether the engine has CUDA registered is a
    /// property of how it was BUILT, which no caller can see, and the only
    /// reliable test is to ask it.
    /// </param>
    public AudioCppAsr(string modelPath, string familyHint,
                       IReadOnlyList<string> backends, int threads = 1)
    {
        ArgumentOutOfRangeException.ThrowIfZero(backends.Count);
        _registry = AudioCppRegistry.Create();
        try
        {
            _model = _registry.Load(modelPath, new ModelConfig(familyHint));

            AudioCppException? last = null;
            for (int i = 0; i < backends.Count; i++)
            {
                try
                {
                    _session = _model.CreateSession("asr", "offline",
                                                    new BackendConfig(backends[i], 0, threads));
                    Backend  = backends[i];
                    break;
                }
                catch (AudioCppException failure) when (i < backends.Count - 1)
                {
                    // A backend the engine was not built with fails at session
                    // create with a typed error naming what IS available. That
                    // is the signal to try the next one; anything else, and the
                    // last candidate's failure, propagates.
                    last = failure;
                }
            }
            if (_session is null)
                throw (Exception?)last
                      ?? new InvalidOperationException(
                          "no backend opened and none reported why");
        }
        catch
        {
            // Load and CreateSession both throw; without this a failure leaks the
            // registry and, on the second, the model as well.
            _model?.Dispose();
            _registry.Dispose();
            throw;
        }
    }

    /// <summary>Convenience for a caller that knows exactly which backend it wants.</summary>
    public AudioCppAsr(string modelPath, string familyHint, string backend = "cpu", int threads = 1)
        : this(modelPath, familyHint, [backend], threads) { }

    /// <summary>The language the engine reported for the last segment, if any.</summary>
    public IEnumerable<AudioCppRecognition> RecognizeDetailed(
        IReadOnlyList<(double start, double end, string spk)> segments,
        float[] audio,
        string? forceLanguage = null,
        CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(audio);

        for (int i = 0; i < segments.Count; i++)
        {
            cancel.ThrowIfCancellationRequested();

            var (start, end, _) = segments[i];
            var slice = Slice(audio, start, end);
            if (slice.Length == 0)
            {
                // A zero-length segment is a caller bug upstream, but returning
                // nothing for it would silently drop a row the database already
                // has. An empty result fills it instead.
                yield return new AudioCppRecognition(i, "", [], [], [], null);
                continue;
            }

            using var request = new AudioCppRequest();
            request.SetAudio(slice, SampleRate);
            // Blank means "detect", which is what the engine does by default; a
            // forced language has to be set explicitly or the detection wins.
            if (!string.IsNullOrWhiteSpace(forceLanguage))
                request.SetOption("language", forceLanguage);

            using var result = _session!.Run(request);

            var text  = result.Text?.Text ?? "";
            var lang  = result.Text?.Language;
            var words = result.Words;

            var wordTexts  = new List<string>(words.Count);
            var startFrames = new List<int>(words.Count);
            var confidences = new List<float>(words.Count);
            foreach (var word in words)
            {
                wordTexts.Add(word.Word);
                // Word times are absolute within the slice that was submitted,
                // so they are already segment-relative. The editor wants frames.
                startFrames.Add((int)(word.StartSample * FramesPerSecond / SampleRate));
                confidences.Add(word.Confidence);
            }

            // A model that reports no words still reports text. Falling back to
            // whitespace splitting keeps the editor's runs aligned with the text
            // it renders, which is the property the runs actually need.
            if (wordTexts.Count == 0 && text.Length > 0)
            {
                var split = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                double span = Math.Max(end - start, 0);
                for (int w = 0; w < split.Length; w++)
                {
                    wordTexts.Add(split[w]);
                    startFrames.Add((int)(span * w / Math.Max(split.Length, 1) * FramesPerSecond));
                    confidences.Add(0f);
                }
            }

            yield return new AudioCppRecognition(i, text, wordTexts, startFrames, confidences, lang);
        }
    }

    /// <summary>
    /// Finds the Parakeet weights inside a models root laid out by audio.cpp's
    /// own model manager, which installs each package into its own directory.
    /// </summary>
    /// <remarks>
    /// A search rather than a fixed path: the package directory and the file
    /// inside it are named by the engine's catalogue, and a quantisation the
    /// user chose (q8_0, f16, ...) changes the file name. Returns null rather
    /// than throwing so the caller can say what it was looking for and where.
    /// </remarks>
    public static string? ResolveParakeet(string modelsRoot)
    {
        if (!Directory.Exists(modelsRoot)) return null;
        return Directory
            .EnumerateFiles(modelsRoot, "parakeet-tdt*.gguf", SearchOption.AllDirectories)
            .OrderBy(path => path, StringComparer.Ordinal)
            .FirstOrDefault();
    }

    private static float[] Slice(float[] audio, double start, double end)
    {
        int s   = Math.Clamp((int)(start * SampleRate), 0, audio.Length);
        int e   = Math.Clamp((int)(end   * SampleRate), 0, audio.Length);
        int len = Math.Max(e - s, 0);
        var slice = new float[len];
        if (len > 0) Array.Copy(audio, s, slice, 0, len);
        return slice;
    }

    public void Dispose()
    {
        _session?.Dispose();
        _model.Dispose();
        _registry.Dispose();
    }
}
