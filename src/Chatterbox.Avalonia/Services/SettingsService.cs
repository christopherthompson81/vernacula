using System;
using System.IO;
using System.Text.Json;

namespace Chatterbox.App.Services;

/// <summary>
/// Persistent picker state for the reader. Mirrors
/// Vernacula.Avalonia.SettingsService's JSON-in-LocalApplicationData
/// pattern. Settings live at
/// <c>%LocalApplicationData%/ChatterboxReader/settings.json</c>
/// (Linux: <c>~/.local/share/ChatterboxReader/settings.json</c>).
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
        "ChatterboxReader", "settings.json");

    public ReaderSettings Current { get; private set; } = new();

    public void Load()
    {
        try
        {
            if (!File.Exists(SettingsPath)) return;
            var json = File.ReadAllText(SettingsPath);
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
    // (Older settings.json files may contain a NfaBundleDir field —
    // System.Text.Json silently ignores unknown JSON properties, so the
    // old field is dropped harmlessly on load. New saves omit it.)
}
