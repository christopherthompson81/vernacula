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
        Path.Combine(GetModelsDir(), Config.Dfn3SubDir);

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

    public string GetVoxLinguaModelsDir() =>
        string.IsNullOrWhiteSpace(Current.VoxLinguaModelsDir)
            ? Path.Combine(GetModelsDir(), Config.VoxLinguaSubDir)
            : Current.VoxLinguaModelsDir;

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
