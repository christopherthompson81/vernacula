using System.Text.Json;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Services;

internal class SettingsService
{
    private static readonly string SettingsPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Parakeet", "settings.json");

    private static readonly string DefaultModelsDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Parakeet", "models");

    public AppSettings Current { get; private set; } = new();

    public void Load()
    {
        try
        {
            if (!File.Exists(SettingsPath)) return;
            var json = File.ReadAllText(SettingsPath);
            Current = JsonSerializer.Deserialize<AppSettings>(json) ?? new();
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

    public string GetModelsDir() =>
        string.IsNullOrEmpty(Current.ModelsDir) ? DefaultModelsDir : Current.ModelsDir;

    public string GetDiariZenModelsDir() =>
        string.IsNullOrWhiteSpace(Current.DiariZenModelsDir)
            ? Path.Combine(GetModelsDir(), "diarizen")
            : Current.DiariZenModelsDir;

    public string GetJobsDir()
    {
        string dir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Parakeet", "jobs");
        Directory.CreateDirectory(dir);
        return dir;
    }

    public string GetControlDbPath() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Parakeet", "parakeet.db");
}
