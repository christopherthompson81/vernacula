using System.Text.Json;
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

    public string GetDenoiserModelsDir() =>
        Path.Combine(GetModelsDir(), "deepfilternet3");

    public string GetCohereModelsDir() =>
        Path.Combine(GetModelsDir(), "cohere_transcribe");

    public string GetVibeVoiceModelsDir() =>
        Path.Combine(GetModelsDir(), "vibevoice_asr");

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
}
