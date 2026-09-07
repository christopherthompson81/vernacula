using Vernacula.App.Models;
using Vernacula.Phonemizer;
using Vernacula.Tts.Base;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Everything the app needs to know about one text-to-speech engine, in one place: what it is
/// called, what it needs on disk, how to build it, how a job's choices become a request, how
/// its output is phonemized, and which per-job controls it wants shown.
///
/// <para>
/// ⚠ THIS IS THE ONLY PLACE AN ENGINE IS ENUMERATED. Before it existed the Kokoro / OmniVoice /
/// Chatterbox branch was repeated in about a dozen switch expressions across the runner,
/// prerequisites, export, reader and two view models, each with a <c>_ =&gt;</c> arm that
/// silently treated an unknown engine as Chatterbox — so adding one was a scavenger hunt with
/// no compiler help. Adding an engine now means: a subclass here, an entry in
/// <see cref="TtsEngines.All"/>, a <see cref="TtsBackendKind"/> member, and its model set in
/// ModelManagerService.
/// </para>
/// </summary>
internal abstract class TtsEngine
{
    public abstract TtsBackendKind Kind { get; }
    /// <summary>Short name, shown in pickers and in the reader's job line.</summary>
    public abstract string DisplayName { get; }
    /// <summary>One or two sentences for the Settings radio: what it is good at, what it costs.</summary>
    public abstract string Description { get; }
    /// <summary>Output sample rate (Hz) — the reader needs it before any audio exists.</summary>
    public abstract int SampleRate { get; }
    /// <summary>The model sets that must be complete on disk before a job can run.</summary>
    public abstract TtsModelSet[] RequiredSets { get; }

    /// <summary>How the export's phoneme column was produced, named in the CSV.</summary>
    public virtual string PhonemeScheme => "ipa";

    // ── Which per-job controls the dialog shows ──────────────────────────────
    // Capabilities, not identity: a new engine with named voices gets the voice drop-down by
    // saying so, without any view or view model learning its name.

    /// <summary>Clones a reference clip picked per job (Chatterbox).</summary>
    public virtual bool UsesReferenceClip => false;
    /// <summary>Has a fixed list of named voices to choose from (Kokoro).</summary>
    public virtual bool UsesVoiceList => false;
    /// <summary>Reads any phonemizer language, with the voice chosen per language (OmniVoice).</summary>
    public virtual bool UsesLanguage => false;
    /// <summary>Has a speech-rate multiplier.</summary>
    public virtual bool UsesSpeed => false;
    /// <summary>Has a diffusion step count.</summary>
    public virtual bool UsesDiffusionSteps => false;

    // ── Building and running ─────────────────────────────────────────────────

    /// <summary>
    /// Identity of a loaded backend: everything that would make the cached one wrong. The job
    /// runner rebuilds when this changes, so a model location edited in Settings takes effect
    /// on the next job without an app restart.
    /// </summary>
    public abstract string CacheKey(SettingsService s);

    public abstract ITtsBackend CreateBackend(SettingsService s);

    /// <summary>The job's choices as a request for this engine — it decides which fields mean anything.</summary>
    public virtual TtsRequest BuildRequest(string text, string wavPath, string segmentsDir, TtsJobSettings job) =>
        new(text, wavPath, job.Voice, SegmentsDir: segmentsDir);

    /// <summary>
    /// What is wrong with this job's own choices (an unpicked voice, a language the catalogue
    /// does not know), or null. Whether the MODEL FILES are present is not asked here — that is
    /// <see cref="TtsPrerequisites.Describe"/>'s job, via <see cref="RequiredSets"/>, so there
    /// is one on-disk truth for the runner, the dialog and the Settings rows.
    /// </summary>
    public virtual string? DescribeJobIssue(SettingsService s, TtsJobSettings job) => null;

    /// <summary>Text → the phonemes this engine would be given, for the export's CSV. Blocking.</summary>
    public abstract Func<string, string> CreatePhonemizer(SettingsService s, TtsJobSettings job);

    // ── Describing a finished job ────────────────────────────────────────────

    /// <summary>The language the reader's IPA annotation is read in for this job.</summary>
    public virtual string AnnotationLanguage(JobRecord job) => "en";

    /// <summary>The parts after the engine name in the reader's job line ("af_heart", "1.00×").</summary>
    public virtual IEnumerable<string> DescribeJob(JobRecord job)
    {
        if (!string.IsNullOrWhiteSpace(job.TtsVoice)) yield return job.TtsVoice;
    }

    // ── Per-engine defaults in AppSettings ───────────────────────────────────
    // The voice lives in a different field per engine (a path, a name, a library id), so the
    // engine reads and writes its own.

    public abstract string ReadStoredVoice(AppSettings s);
    public abstract void WriteStoredVoice(AppSettings s, string voice);

    /// <summary>The named voices on disk, for an engine with a fixed list (<see cref="UsesVoiceList"/>).</summary>
    public virtual IReadOnlyList<string> AvailableVoices(SettingsService s) => [];
}

/// <summary>The engines this build ships, in the order they are offered.</summary>
internal static class TtsEngines
{
    public static IReadOnlyList<TtsEngine> All { get; } =
        [new KokoroEngine(), new OmniVoiceEngine(), new ChatterboxEngine()];

    /// <summary>The default for a new job, and the fallback for anything unrecognised.</summary>
    public static TtsEngine Default => All[0];

    public static TtsEngine For(TtsBackendKind kind) =>
        All.FirstOrDefault(e => e.Kind == kind) ?? Default;

    /// <summary>
    /// By persisted name ("Kokoro"), or null when the name is empty or names no engine this
    /// build has. Callers that must produce audio use <see cref="For(string?)"/>; callers that
    /// merely DESCRIBE a stored job use this, so a job saved by another build is reported as
    /// what it says it is rather than relabelled as the fallback.
    /// </summary>
    public static TtsEngine? TryFor(string? name) =>
        Enum.TryParse<TtsBackendKind>(name, ignoreCase: true, out var kind)
            ? All.FirstOrDefault(e => e.Kind == kind)
            : null;

    /// <summary>By persisted name; an unknown name falls back to <see cref="Default"/>.</summary>
    public static TtsEngine For(string? name) => TryFor(name) ?? Default;

    /// <summary>The engine a job was rendered with (the fallback applies to an unknown name).</summary>
    public static TtsEngine For(JobRecord job) => For(job.TtsBackend);
}

// ── Kokoro ───────────────────────────────────────────────────────────────────

internal sealed class KokoroEngine : TtsEngine
{
    public override TtsBackendKind Kind => TtsBackendKind.Kokoro;
    public override string DisplayName => "Kokoro-82M";
    public override string Description =>
        "Small and fast. Named English voices (American and British), adjustable speed, word timing from the model's own durations.";
    public override int SampleRate => Kokoro.SampleRate;
    public override TtsModelSet[] RequiredSets =>
        [TtsModelSets.Kokoro, TtsModelSets.PhonemizerData];
    public override string PhonemeScheme => "kokoro";   // its own vocabulary, exactly what the model consumed

    public override bool UsesVoiceList => true;
    public override bool UsesSpeed => true;

    /// <summary>The bf_/bm_ voices are the British ones, which is also what picks en-GB phonemization.</summary>
    private static bool IsBritish(string voice) =>
        voice.StartsWith("bf_", StringComparison.Ordinal) || voice.StartsWith("bm_", StringComparison.Ordinal);

    public override string CacheKey(SettingsService s) =>
        $"kokoro|{s.GetKokoroModelsDir()}|{s.GetPhonemizerDataDir()}";

    public override ITtsBackend CreateBackend(SettingsService s) =>
        new KokoroSynthesisService(s.GetKokoroModelsDir(), s.GetPhonemizerDataDir());

    public override TtsRequest BuildRequest(string text, string wavPath, string segmentsDir, TtsJobSettings job) =>
        new(text, wavPath, job.Voice, job.Speed, SegmentsDir: segmentsDir);

    public override string? DescribeJobIssue(SettingsService s, TtsJobSettings job)
    {
        if (string.IsNullOrWhiteSpace(job.Voice)) return "No Kokoro voice selected.";
        string path = Path.Combine(s.GetKokoroModelsDir(), "voices", job.Voice + ".bin");
        return File.Exists(path) ? null : $"Kokoro voice not found: {path}";
    }

    public override Func<string, string> CreatePhonemizer(SettingsService s, TtsJobSettings job)
    {
        var g2p = new KokoroPhonemizer(s.GetPhonemizerDataDir());
        bool british = IsBritish(job.Voice);
        return text => g2p.ToPhonemes(text, british);
    }

    public override string AnnotationLanguage(JobRecord job) => IsBritish(job.TtsVoice) ? "en-GB" : "en";

    public override IEnumerable<string> DescribeJob(JobRecord job)
    {
        if (!string.IsNullOrWhiteSpace(job.TtsVoice)) yield return job.TtsVoice;
        yield return $"{job.TtsSpeed:F2}×";
    }

    /// <summary>The voice packs on disk, by name (scripts/kokoro_export/export_voices.py writes them).</summary>
    public override IReadOnlyList<string> AvailableVoices(SettingsService s)
    {
        var dir = Path.Combine(s.GetKokoroModelsDir(), "voices");
        if (!Directory.Exists(dir)) return [];
        return [.. Directory.EnumerateFiles(dir, "*.bin").Select(Path.GetFileNameWithoutExtension).OrderBy(v => v)!];
    }

    public override string ReadStoredVoice(AppSettings s) => s.KokoroVoice;
    public override void WriteStoredVoice(AppSettings s, string voice) => s.KokoroVoice = voice;
}

// ── OmniVoice-IPA ────────────────────────────────────────────────────────────

internal sealed class OmniVoiceEngine : TtsEngine
{
    public override TtsBackendKind Kind => TtsBackendKind.OmniVoice;
    public override string DisplayName => "OmniVoice-IPA";
    public override string Description =>
        "Any of the phonemizer's 190+ languages through the IPA fine-tune, in a stored voice from the voice library. Large model; word timing is estimated.";
    public override int SampleRate => OmniVoiceIpaTts.SampleRate;
    public override TtsModelSet[] RequiredSets =>
    [
        TtsModelSets.OmniVoice,
        TtsModelSets.OmniVoiceVoices,
        TtsModelSets.PhonemizerData,
    ];

    public override bool UsesLanguage => true;
    public override bool UsesDiffusionSteps => true;

    private static string LangOf(string? code) => string.IsNullOrWhiteSpace(code) ? "en" : code.Trim();

    public override string CacheKey(SettingsService s) =>
        $"omnivoice|{s.GetOmniVoiceModelsDir()}|{s.Current.OmniVoiceTokenizerJson}"
        + $"|{s.GetPhonemizerDataDir()}|{s.GetOmniVoiceVoiceLibDir()}";

    public override ITtsBackend CreateBackend(SettingsService s) =>
        new OmniVoiceSynthesisService(
            s.GetOmniVoiceModelsDir(),
            File.Exists(s.Current.OmniVoiceTokenizerJson) ? s.Current.OmniVoiceTokenizerJson : null,
            s.GetPhonemizerDataDir(),
            s.GetOmniVoiceVoiceLibDir());

    public override TtsRequest BuildRequest(string text, string wavPath, string segmentsDir, TtsJobSettings job) =>
        new(text, wavPath, job.Voice, Lang: LangOf(job.Language), NumStep: job.NumStep, SegmentsDir: segmentsDir);

    public override string? DescribeJobIssue(SettingsService s, TtsJobSettings job)
    {
        if (string.IsNullOrWhiteSpace(job.Voice)) return "No OmniVoice voice selected.";
        if (LanguageCatalog.ByCode(job.Language) is null)
            return $"Unknown language \"{job.Language}\" — pick one from the list.";
        return null;
    }

    public override Func<string, string> CreatePhonemizer(SettingsService s, TtsJobSettings job)
    {
        if (PhonemizerData.Resolve(s.GetPhonemizerDataDir()) is null)
            throw new DirectoryNotFoundException(PhonemizerData.NotFoundMessage());
        Registry.EnsureLanguages();
        string lang = LangOf(job.Language);
        return text => OmniVoiceIpaTts.Phonemize(text, lang);
    }

    public override string AnnotationLanguage(JobRecord job) => LangOf(job.TtsLanguage);

    public override IEnumerable<string> DescribeJob(JobRecord job)
    {
        if (!string.IsNullOrWhiteSpace(job.TtsLanguage))
            yield return LanguageCatalog.ByCode(job.TtsLanguage)?.Name ?? job.TtsLanguage;
        if (!string.IsNullOrWhiteSpace(job.TtsVoice)) yield return job.TtsVoice;
        yield return $"{job.TtsNumStep} steps";
    }

    public override string ReadStoredVoice(AppSettings s) => s.OmniVoiceVoice;
    public override void WriteStoredVoice(AppSettings s, string voice) => s.OmniVoiceVoice = voice;
}

// ── Chatterbox ───────────────────────────────────────────────────────────────

internal sealed class ChatterboxEngine : TtsEngine
{
    public override TtsBackendKind Kind => TtsBackendKind.Chatterbox;
    public override string DisplayName => "Chatterbox";
    public override string Description =>
        "English voice cloning from a short reference clip. Word timing from the model's cross-attention.";
    public override int SampleRate => ChatterboxConstants.S3GenSr;
    public override TtsModelSet[] RequiredSets => [TtsModelSets.Chatterbox];

    public override bool UsesReferenceClip => true;

    public override string CacheKey(SettingsService s) => $"chatterbox|{s.GetChatterboxModelsDir()}";

    public override ITtsBackend CreateBackend(SettingsService s)
    {
        string dir = s.GetChatterboxModelsDir();
        string tokenizer = Path.Combine(dir, "tokenizer.json");
        return new ChatterboxSynthesisService(dir, File.Exists(tokenizer) ? tokenizer : null);
    }

    public override string? DescribeJobIssue(SettingsService s, TtsJobSettings job) =>
        File.Exists(job.Voice)
            ? null
            : $"Reference voice clip not found: {(string.IsNullOrWhiteSpace(job.Voice) ? "(not set)" : job.Voice)}";

    /// <summary>
    /// Chatterbox is fed text, not phonemes, so the export's column is the phonemizer's reading
    /// of that text — informative, not what the model consumed. The scheme column says "ipa" so
    /// the distinction is on the page rather than implied.
    /// </summary>
    public override Func<string, string> CreatePhonemizer(SettingsService s, TtsJobSettings job)
    {
        if (PhonemizerData.Resolve(s.GetPhonemizerDataDir()) is null)
            throw new DirectoryNotFoundException(PhonemizerData.NotFoundMessage());
        Registry.EnsureLanguages();
        return text => OmniVoiceIpaTts.Phonemize(text, "en");
    }

    public override IEnumerable<string> DescribeJob(JobRecord job)
    {
        // A path, so the file name is the useful half.
        if (!string.IsNullOrWhiteSpace(job.TtsVoice)) yield return Path.GetFileName(job.TtsVoice);
    }

    public override string ReadStoredVoice(AppSettings s) => s.ChatterboxVoicePath;
    public override void WriteStoredVoice(AppSettings s, string voice) => s.ChatterboxVoicePath = voice;
}
