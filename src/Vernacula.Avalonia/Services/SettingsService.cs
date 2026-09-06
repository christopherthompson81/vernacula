using System.Text.Json;
using Vernacula.Base;
using Vernacula.App.Models;

namespace Vernacula.App.Services;

internal class SettingsService
{
    public const string DiariZenGatedModelId = "diarizen";

    private static readonly string SettingsPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Vernacula", "settings.json");

    private static readonly string DefaultModelsDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Vernacula", "models");

    public AppSettings Current { get; private set; } = new();

    public void Load()
    {
        try
        {
            if (!File.Exists(SettingsPath)) return;
            var json = File.ReadAllText(SettingsPath);
            Current = JsonSerializer.Deserialize<AppSettings>(json) ?? new();
            MigrateLegacySettings();
        }
        catch
        {
            Current = new();
        }
    }

    public void Save()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(SettingsPath)!);
        File.WriteAllText(SettingsPath,
            JsonSerializer.Serialize(Current, new JsonSerializerOptions { WriteIndented = true }));
    }

    private void MigrateLegacySettings()
    {
        Current.AcceptedGatedModels ??= [];

        if (Current.DiariZenNoticeAccepted &&
            !Current.AcceptedGatedModels.Contains(DiariZenGatedModelId, StringComparer.OrdinalIgnoreCase))
        {
            Current.AcceptedGatedModels.Add(DiariZenGatedModelId);
        }

        MigrateLegacyModelLayout();
        MigrateLegacyReaderSettings();
    }

    /// <summary>
    /// The standalone TTS reader (Vernacula.Tts.Avalonia) kept its own settings.json under
    /// VernaculaTtsReader/. On the first run after it was folded into this app, carry its
    /// model locations and last-used choices over — only into fields that are still empty, so
    /// anything the user has since set here wins. The old file is left in place.
    /// </summary>
    private void MigrateLegacyReaderSettings()
    {
        try
        {
            string legacyPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "VernaculaTtsReader", "settings.json");
            if (!File.Exists(legacyPath)) return;
            var legacy = JsonSerializer.Deserialize<LegacyReaderSettings>(File.ReadAllText(legacyPath));
            if (legacy is null) return;

            bool changed = false;
            void Take(Func<string> get, Action<string> set, string? value)
            {
                if (string.IsNullOrWhiteSpace(get()) && !string.IsNullOrWhiteSpace(value)) { set(value); changed = true; }
            }
            var c = Current;
            Take(() => c.ChatterboxBundleDir, v => c.ChatterboxBundleDir = v, legacy.OnnxBundleDir);
            Take(() => c.ChatterboxVoicePath,     v => c.ChatterboxVoicePath = v, legacy.VoicePath);
            Take(() => c.KokoroModelDir,          v => c.KokoroModelDir = v,      legacy.KokoroModelDir);
            Take(() => c.KokoroVoice,             v => c.KokoroVoice = v,         legacy.KokoroVoice);
            Take(() => c.PhonemizerDataDir,       v => c.PhonemizerDataDir = v,
                 !string.IsNullOrWhiteSpace(legacy.PhonemizerDataDir) ? legacy.PhonemizerDataDir : legacy.KokoroDataDir);
            Take(() => c.OmniVoiceOnnxDir,        v => c.OmniVoiceOnnxDir = v,    legacy.OmniVoiceOnnxDir);
            Take(() => c.OmniVoiceTokenizerJson,  v => c.OmniVoiceTokenizerJson = v, legacy.OmniVoiceTokenizerJson);
            Take(() => c.OmniVoiceVoiceLib,       v => c.OmniVoiceVoiceLib = v,   legacy.OmniVoiceVoiceLib);
            Take(() => c.OmniVoiceVoice,          v => c.OmniVoiceVoice = v,      legacy.OmniVoiceVoice);
            if (c.TtsSettingsMigrated) return;   // choices below have defaults, so only take them once
            if (!string.IsNullOrWhiteSpace(legacy.TtsBackend))   { c.TtsBackend = legacy.TtsBackend; changed = true; }
            if (!string.IsNullOrWhiteSpace(legacy.OmniVoiceLang)) { c.OmniVoiceLang = legacy.OmniVoiceLang; changed = true; }
            if (legacy.KokoroSpeed > 0)                            { c.KokoroSpeed = legacy.KokoroSpeed; changed = true; }
            if (legacy.OmniVoiceNumStep is > 0 and <= 64)          { c.OmniVoiceNumStep = legacy.OmniVoiceNumStep; changed = true; }
            c.TtsRenderMarkdown    = legacy.RenderMarkdown;
            c.TtsShowIpaAnnotation = legacy.ShowIpaAnnotation;
            c.TtsSettingsMigrated  = true;
            changed = true;
            if (changed) Save();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[Settings] legacy reader settings not migrated: {ex.Message}");
        }
    }

    /// <summary>Shape of the standalone reader's settings.json (Vernacula.Tts.App.Services.ReaderSettings).</summary>
    private sealed class LegacyReaderSettings
    {
        public string VoicePath { get; set; } = "";
        public string OnnxBundleDir { get; set; } = "";
        public bool RenderMarkdown { get; set; }
        public bool ShowIpaAnnotation { get; set; }
        public string TtsBackend { get; set; } = "";
        public string PhonemizerDataDir { get; set; } = "";
        public string KokoroDataDir { get; set; } = "";
        public string KokoroModelDir { get; set; } = "";
        public string KokoroVoice { get; set; } = "";
        public float KokoroSpeed { get; set; }
        public string OmniVoiceOnnxDir { get; set; } = "";
        public string OmniVoiceTokenizerJson { get; set; } = "";
        public string OmniVoiceVoiceLib { get; set; } = "";
        public string OmniVoiceLang { get; set; } = "";
        public string OmniVoiceVoice { get; set; } = "";
        public int OmniVoiceNumStep { get; set; }
    }

    public bool IsGatedModelAccepted(string modelId) =>
        Current.AcceptedGatedModels.Any(id => string.Equals(id, modelId, StringComparison.OrdinalIgnoreCase));

    public bool AcceptGatedModel(string modelId)
    {
        Current.AcceptedGatedModels ??= [];

        if (IsGatedModelAccepted(modelId))
            return false;

        Current.AcceptedGatedModels.Add(modelId);

        if (string.Equals(modelId, DiariZenGatedModelId, StringComparison.OrdinalIgnoreCase))
            Current.DiariZenNoticeAccepted = true;

        Save();
        return true;
    }

    public string GetModelsDir() =>
        string.IsNullOrEmpty(Current.ModelsDir) ? DefaultModelsDir : Current.ModelsDir;

    public string GetDiariZenModelsDir() =>
        string.IsNullOrWhiteSpace(Current.DiariZenModelsDir)
            ? Path.Combine(GetModelsDir(), "diarizen")
            : Current.DiariZenModelsDir;

    public string GetParakeetModelsDir() =>
        Path.Combine(GetModelsDir(), Config.ParakeetSubDir);

    public string GetSortformerModelsDir() =>
        Path.Combine(GetModelsDir(), Config.SortformerSubDir);

    public string GetSileroModelsDir() =>
        Path.Combine(GetModelsDir(), Config.VadSubDir);

    public string GetCohereModelsDir() =>
        Path.Combine(GetModelsDir(), "cohere_transcribe");

    public string GetQwen3AsrModelsDir() =>
        Path.Combine(GetModelsDir(), Config.Qwen3AsrSubDir);

    public string GetVibeVoiceModelsDir() =>
        Path.Combine(GetModelsDir(), "vibevoice_asr");

    public string GetIndicConformerModelsDir() =>
        Path.Combine(GetModelsDir(), Config.IndicConformerSubDir);

    public string GetWhisperTurboModelsDir() =>
        Path.Combine(GetModelsDir(), Config.WhisperTurboSubDir);

    /// <summary>
    /// Resolves the active Granite Speech 4.1 bundle directory: prefers
    /// the BF16 mixed-precision bundle when present on disk and the host
    /// has CUDA EP with Ampere+ tensor cores; otherwise falls back to the
    /// FP32 bundle. Mirrors the CLI's selection logic so editor and CLI
    /// pick the same bundle on the same machine.
    ///
    /// Implicit invariant with <c>ModelManagerService.ActiveRepos</c>:
    /// both consult <see cref="HardwareInfo.SupportsBf16Acceleration"/> to
    /// decide which bundle to fetch (manager) vs load (this). The result
    /// is cached process-wide on the first call, so the two call sites
    /// always agree within a single run.
    /// </summary>
    public string GetGraniteSpeechModelsDir()
    {
        string bf16Dir = Path.Combine(GetModelsDir(), Config.GraniteSpeechBf16SubDir);
        string fp32Dir = Path.Combine(GetModelsDir(), Config.GraniteSpeechSubDir);
        if (Directory.Exists(bf16Dir) && HardwareInfo.SupportsBf16Acceleration())
            return bf16Dir;
        return fp32Dir;
    }

    /// <summary>
    /// Returns the path to a Granite Speech tokenizer.json from whichever
    /// sibling bundle is actually installed (BF16 preferred, FP32 fallback),
    /// or null when neither bundle is present. The two bundles ship
    /// identical tokenizer.json content, so file-presence — not hardware
    /// capability — is the right gate here: the editor's vocab loader and
    /// any other consumer that needs the tokenizer should be able to find
    /// it regardless of whether the user is on the "right" hardware for
    /// the bundle they happened to download.
    /// </summary>
    public string? TryGetGraniteSpeechTokenizerPath()
    {
        string bf16Path = Path.Combine(GetModelsDir(), Config.GraniteSpeechBf16SubDir, GraniteSpeech.TokenizerFile);
        if (File.Exists(bf16Path)) return bf16Path;
        string fp32Path = Path.Combine(GetModelsDir(), Config.GraniteSpeechSubDir, GraniteSpeech.TokenizerFile);
        if (File.Exists(fp32Path)) return fp32Path;
        return null;
    }

    public string GetVoxLinguaModelsDir() =>
        string.IsNullOrWhiteSpace(Current.VoxLinguaModelsDir)
            ? Path.Combine(GetModelsDir(), Config.VoxLinguaSubDir)
            : Current.VoxLinguaModelsDir;

    public string GetKenLmParakeetDir() =>
        Path.Combine(GetModelsDir(), Config.KenLmParakeetSubDir);

    // ── Text-to-speech locations ─────────────────────────────────────────
    // Each is the user's pick when set, else a subfolder of the models dir so a
    // fresh install has one place to put (or download) everything.

    public string GetChatterboxModelsDir() =>
        string.IsNullOrWhiteSpace(Current.ChatterboxBundleDir)
            ? Path.Combine(GetModelsDir(), Config.ChatterboxSubDir)
            : Current.ChatterboxBundleDir;

    public string GetKokoroModelsDir() =>
        string.IsNullOrWhiteSpace(Current.KokoroModelDir)
            ? Path.Combine(GetModelsDir(), Config.KokoroSubDir)
            : Current.KokoroModelDir;

    public string GetOmniVoiceModelsDir() =>
        string.IsNullOrWhiteSpace(Current.OmniVoiceOnnxDir)
            ? (Environment.GetEnvironmentVariable("OMNIVOICE_ONNX_DIR")
               ?? Path.Combine(GetModelsDir(), Config.OmniVoiceSubDir))
            : Current.OmniVoiceOnnxDir;

    /// <summary>
    /// The vernacula-phonemizer data/ root: the user's pick, else whatever
    /// <see cref="Vernacula.Tts.Base.PhonemizerData.Resolve"/> finds (VERNACULA_DATA_DIR, then
    /// the submodule beside a source build), else the models-dir default. May not exist.
    /// </summary>
    public string GetPhonemizerDataDir()
    {
        if (Vernacula.Tts.Base.PhonemizerData.IsDataRoot(Current.PhonemizerDataDir))
            return Current.PhonemizerDataDir;
        return Vernacula.Tts.Base.PhonemizerData.Resolve(null)
               ?? Path.Combine(GetModelsDir(), Config.PhonemizerDataSubDir);
    }

    /// <summary>
    /// The OmniVoice stored-voice library: the user's pick, else the web demo's library beside a
    /// source build, else the models-dir default.
    /// </summary>
    public string GetOmniVoiceVoiceLibDir()
    {
        if (Vernacula.Tts.Base.StoredVoice.IsLibrary(Current.OmniVoiceVoiceLib))
            return Current.OmniVoiceVoiceLib;
        return Vernacula.Tts.Base.StoredVoice.ResolveDefaultLibrary()
               ?? Path.Combine(GetModelsDir(), Config.OmniVoiceVoiceLibSubDir);
    }

    public string GetJobsDir()
    {
        string dir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Vernacula", "jobs");
        Directory.CreateDirectory(dir);
        return dir;
    }

    public string GetControlDbPath() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Vernacula", "vernacula.db");

    private void MigrateLegacyModelLayout()
    {
        string rootDir = GetModelsDir();
        if (!Directory.Exists(rootDir))
            return;

        MoveLegacyModelFiles(rootDir, GetParakeetModelsDir(),
        [
            Config.PreprocessorFile,
            Config.EncoderFile,
            $"{Config.EncoderFile}.data",
            Config.DecoderJointFile,
            Config.VocabFile,
            Config.AsrConfigFile,
        ]);

        MoveLegacyModelFiles(rootDir, GetSortformerModelsDir(),
        [
            Config.SortformerFile,
        ]);

        MoveLegacyModelFiles(rootDir, GetSileroModelsDir(),
        [
            Config.VadFile,
        ]);
    }

    private static void MoveLegacyModelFiles(string sourceRoot, string targetDir, IEnumerable<string> fileNames)
    {
        foreach (string fileName in fileNames)
        {
            string sourcePath = Path.Combine(sourceRoot, fileName);
            if (!File.Exists(sourcePath))
                continue;

            string destPath = Path.Combine(targetDir, fileName);
            if (File.Exists(destPath))
                continue;

            Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);
            File.Move(sourcePath, destPath);
        }
    }
}
