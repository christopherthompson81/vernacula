using System;
using System.IO;
using System.Text.Json;

namespace Vernacula.Tts.App.Services;

/// <summary>
/// Persistent picker state for the reader. Mirrors
/// Vernacula.Avalonia.SettingsService's JSON-in-LocalApplicationData
/// pattern. Settings live at
/// <c>%LocalApplicationData%/VernaculaTtsReader/settings.json</c>
/// (Linux: <c>~/.local/share/VernaculaTtsReader/settings.json</c>). The app was
/// called Chatterbox Reader before it grew a second and third backend; a settings
/// file left at the old <c>ChatterboxReader/</c> path is read when the new one
/// does not exist yet, and the next save writes it to the new location.
///
/// The file is tiny (a handful of paths) so we serialize the whole
/// thing on every save — no diffing. Read failures fall back to a
/// fresh default; write failures are logged to stderr and swallowed
/// (preferring a working UI over crashing the user out on disk-full
/// or permission errors).
/// </summary>
public sealed class SettingsService
{
    private static readonly string SettingsPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "VernaculaTtsReader", "settings.json");

    /// <summary>Pre-rename location, read only when <see cref="SettingsPath"/> is absent.</summary>
    private static readonly string LegacySettingsPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "ChatterboxReader", "settings.json");

    public ReaderSettings Current { get; private set; } = new();

    public void Load()
    {
        try
        {
            var path = File.Exists(SettingsPath) ? SettingsPath
                     : File.Exists(LegacySettingsPath) ? LegacySettingsPath : null;
            if (path is null) return;
            var json = File.ReadAllText(path);
            Current = JsonSerializer.Deserialize<ReaderSettings>(json) ?? new ReaderSettings();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[SettingsService] load failed, using defaults: {ex.Message}");
            Current = new ReaderSettings();
        }
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(SettingsPath)!);
            File.WriteAllText(SettingsPath,
                JsonSerializer.Serialize(Current, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[SettingsService] save failed: {ex.Message}");
        }
    }
}

public sealed class ReaderSettings
{
    public string VoicePath { get; set; } = "";
    public string OnnxBundleDir { get; set; } = "";
    public bool RenderMarkdown { get; set; } = false;

    // TTS backend selection. TtsBackend stores the TtsBackendKind enum name
    // ("Chatterbox" / "Kokoro" / "OmniVoice").
    public string TtsBackend { get; set; } = "Chatterbox";
    // vernacula-phonemizer data/ root, shared by Kokoro and OmniVoice. (Was KokoroDataDir
    // when Kokoro was the only phonemizer user; that field is still read as a fallback.)
    public string PhonemizerDataDir { get; set; } = "";
    public string KokoroDataDir { get; set; } = "";
    public string KokoroModelDir { get; set; } = "";
    public string KokoroVoice { get; set; } = "";
    public float KokoroSpeed { get; set; } = 1.0f;
    public string OmniVoiceOnnxDir { get; set; } = "";
    public string OmniVoiceTokenizerJson { get; set; } = "";
    public string OmniVoiceVoiceLib { get; set; } = "";
    public string OmniVoiceLang { get; set; } = "en";
    public string OmniVoiceVoice { get; set; } = "";
    public int OmniVoiceNumStep { get; set; } = 32;
    // (Older settings.json files may contain a NfaBundleDir field —
    // System.Text.Json silently ignores unknown JSON properties, so the
    // old field is dropped harmlessly on load. New saves omit it.)
}
