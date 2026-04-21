using System.Net.Http;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Vernacula.Base;
using Vernacula.Base.Models;
using Vernacula.App.Models;

namespace Vernacula.App.Services;

internal class ModelManagerService
{
    private readonly record struct ModelAsset(string LocalRelativePath, string RemoteRelativePath);
    private readonly record struct AssetRepo(string RepoBase, string ManifestUrl, ModelAsset[] Assets);
    private readonly record struct RepoAsset(string RepoBase, string LocalRelativePath, string RemoteRelativePath);

    private const string CoreRepoBase =
        "https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx/resolve/main";
    private const string CoreManifestUrl =
        "https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx/resolve/main/manifest.json";

    private const string DiariZenRepoBase =
        "https://huggingface.co/christopherthompson81/diarizen_onnx/resolve/main";

    private const string CohereRepoBase =
        "https://huggingface.co/christopherthompson81/cohere-transcribe-03-2026-onnx/resolve/main";
    private const string CohereManifestUrl =
        "https://huggingface.co/christopherthompson81/cohere-transcribe-03-2026-onnx/resolve/main/manifest.json";

    private const string VibeVoiceRepoBase =
        "https://huggingface.co/christopherthompson81/vibevoice-asr-onnx/resolve/main";
    private const string VibeVoiceManifestUrl =
        "https://huggingface.co/christopherthompson81/vibevoice-asr-onnx/resolve/main/manifest.json";

    private const string VoxLinguaRepoBase =
        "https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx/resolve/main";
    private const string VoxLinguaManifestUrl =
        "https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx/resolve/main/manifest.json";

    private const string IndicConformerRepoBase =
        "https://huggingface.co/christopherthompson81/indicconformer-600m-onnx/resolve/main";
    private const string IndicConformerManifestUrl =
        "https://huggingface.co/christopherthompson81/indicconformer-600m-onnx/resolve/main/manifest.json";

    // Whisper turbo pulls directly from the onnx-community pre-exported repo
    // (no re-hosting). No manifest: onnx-community doesn't ship our format, so
    // the MD5 update-check pass silently skips this repo (per
    // GetOutdatedFilesAsync, which gates on non-empty ManifestUrl).
    private const string WhisperTurboRepoBase =
        "https://huggingface.co/onnx-community/whisper-large-v3-turbo/resolve/main";

    private static readonly ModelAsset[] CoreDiarizationFiles =
        [
            new(Path.Combine(Config.SortformerSubDir, Config.SortformerFile), Config.SortformerFile),
            new(Path.Combine(Config.VadSubDir, Config.VadFile), Config.VadFile),
        ];

    private static readonly ModelAsset[] DiariZenFiles =
        [
            new(Path.Combine("diarizen", Config.DiariZenFile), Config.DiariZenFile),
            new(Path.Combine("diarizen", $"{Config.DiariZenFile}.data"), $"{Config.DiariZenFile}.data"),
            new(Path.Combine("diarizen", Config.DiariZenEmbedderFile), Config.DiariZenEmbedderFile),
            new(Path.Combine("diarizen", Config.DiariZenLdaDir, "lda.bin"), $"{Config.DiariZenLdaDir}/lda.bin"),
            new(Path.Combine("diarizen", Config.DiariZenLdaDir, "mean1.bin"), $"{Config.DiariZenLdaDir}/mean1.bin"),
            new(Path.Combine("diarizen", Config.DiariZenLdaDir, "mean2.bin"), $"{Config.DiariZenLdaDir}/mean2.bin"),
            new(Path.Combine("diarizen", Config.DiariZenLdaDir, "plda_mu.bin"), $"{Config.DiariZenLdaDir}/plda_mu.bin"),
            new(Path.Combine("diarizen", Config.DiariZenLdaDir, "plda_psi.bin"), $"{Config.DiariZenLdaDir}/plda_psi.bin"),
            new(Path.Combine("diarizen", Config.DiariZenLdaDir, "plda_tr.bin"), $"{Config.DiariZenLdaDir}/plda_tr.bin"),
        ];

    private static readonly ModelAsset[] AsrFilesFp32 =
        [
            new(Path.Combine(Config.ParakeetSubDir, Config.PreprocessorFile), Config.PreprocessorFile),
            new(Path.Combine(Config.ParakeetSubDir, Config.EncoderFile), Config.EncoderFile),
            new(Path.Combine(Config.ParakeetSubDir, $"{Config.EncoderFile}.data"), $"{Config.EncoderFile}.data"),
            new(Path.Combine(Config.ParakeetSubDir, Config.DecoderJointFile), Config.DecoderJointFile),
            new(Path.Combine(Config.ParakeetSubDir, Config.VocabFile), Config.VocabFile),
            new(Path.Combine(Config.ParakeetSubDir, Config.AsrConfigFile), Config.AsrConfigFile)
        ];

    private static readonly ModelAsset[] CohereFiles =
        [
            // mel/decoder weight footprints fit comfortably below the 2 GiB
            // protobuf single-file limit and are shipped in-graph with no
            // .data sidecar. Only the ~7 GiB encoder needs an external sidecar.
            new(Path.Combine("cohere_transcribe", CohereTranscribe.MelFile), CohereTranscribe.MelFile),
            new(Path.Combine("cohere_transcribe", CohereTranscribe.EncoderFile), CohereTranscribe.EncoderFile),
            new(Path.Combine("cohere_transcribe", $"{CohereTranscribe.EncoderFile}.data"), $"{CohereTranscribe.EncoderFile}.data"),
            new(Path.Combine("cohere_transcribe", CohereTranscribe.DecoderInitFile), CohereTranscribe.DecoderInitFile),
            new(Path.Combine("cohere_transcribe", CohereTranscribe.DecoderStepFile), CohereTranscribe.DecoderStepFile),
            new(Path.Combine("cohere_transcribe", CohereTranscribe.VocabFile), CohereTranscribe.VocabFile),
            new(Path.Combine("cohere_transcribe", CohereTranscribe.ConfigFile), CohereTranscribe.ConfigFile),
        ];

    // Unified decoder (decoder.onnx + decoder.onnx.data) is the current required shape.
    // The earlier split decoder (decoder_init + decoder_step) is no longer required at
    // runtime — when decoder.onnx is present, the constructor prefers it and never loads
    // the split files (see Qwen3Asr.cs constructor branch on hasUnified). Batched
    // artifacts (encoder_batched.onnx + .data) remain intentionally optional: the
    // runtime lights up batched transcription opportunistically when they are present,
    // so listing them as required would reject valid single-stream installs.
    private static readonly ModelAsset[] Qwen3AsrFiles =
        [
            new(Path.Combine(Config.Qwen3AsrSubDir, Qwen3Asr.EncoderFile), Qwen3Asr.EncoderFile),
            new(Path.Combine(Config.Qwen3AsrSubDir, Qwen3Asr.DecoderFile), Qwen3Asr.DecoderFile),
            new(Path.Combine(Config.Qwen3AsrSubDir, $"{Qwen3Asr.DecoderFile}.data"), $"{Qwen3Asr.DecoderFile}.data"),
            new(Path.Combine(Config.Qwen3AsrSubDir, Qwen3Asr.EmbedTokensFile), Qwen3Asr.EmbedTokensFile),
            new(Path.Combine(Config.Qwen3AsrSubDir, Qwen3Asr.TokenizerFile), Qwen3Asr.TokenizerFile),
            new(Path.Combine(Config.Qwen3AsrSubDir, "tokenizer_config.json"), "tokenizer_config.json"),
            new(Path.Combine(Config.Qwen3AsrSubDir, Qwen3Asr.ConfigFile), Qwen3Asr.ConfigFile),
            new(Path.Combine(Config.Qwen3AsrSubDir, "preprocessor_config.json"), "preprocessor_config.json"),
        ];

    // Files ship under onnx/ in the upstream repo for the ONNX graphs; root
    // for tokenizer/config assets. Remote paths reflect that split, local
    // paths collapse them into one flat whisper_turbo/ directory alongside
    // our other backends.
    private static readonly ModelAsset[] WhisperTurboFiles =
        [
            new(Path.Combine(Config.WhisperTurboSubDir, WhisperTurbo.EncoderFile),           $"onnx/{WhisperTurbo.EncoderFile}"),
            new(Path.Combine(Config.WhisperTurboSubDir, WhisperTurbo.DecoderFile),           $"onnx/{WhisperTurbo.DecoderFile}"),
            new(Path.Combine(Config.WhisperTurboSubDir, WhisperTurbo.TokenizerFile),         WhisperTurbo.TokenizerFile),
            new(Path.Combine(Config.WhisperTurboSubDir, WhisperTurbo.ConfigFile),            WhisperTurbo.ConfigFile),
            new(Path.Combine(Config.WhisperTurboSubDir, WhisperTurbo.GenerationConfigFile),  WhisperTurbo.GenerationConfigFile),
            new(Path.Combine(Config.WhisperTurboSubDir, WhisperTurbo.PreprocessorConfigFile),WhisperTurbo.PreprocessorConfigFile),
        ];

    // Whisper auxiliary files that don't live in the onnx-community repo —
    // primarily mel.onnx, a small Whisper-style log-mel graph we export
    // ourselves (see scripts/whisper_export/). Required for the WhisperTurbo
    // backend but not auto-downloaded for now; users run the export script
    // locally or copy the file in manually. Distribution path is TBD — the
    // empty RepoBase below mirrors the Qwen3-ASR pattern for the same case.
    private static readonly ModelAsset[] WhisperTurboAuxFiles =
        [
            new(Path.Combine(Config.WhisperTurboSubDir, WhisperTurbo.MelFile), WhisperTurbo.MelFile),
        ];

    private static readonly ModelAsset[] VoxLinguaFiles =
        [
            new(Path.Combine(Config.VoxLinguaSubDir, Config.VoxLinguaModelFile),   Config.VoxLinguaModelFile),
            new(Path.Combine(Config.VoxLinguaSubDir, Config.VoxLinguaLangMapFile), Config.VoxLinguaLangMapFile),
        ];

    private static readonly ModelAsset[] IndicConformerFiles =
        [
            new(Path.Combine(Config.IndicConformerSubDir, Config.PreprocessorFile),        Config.PreprocessorFile),
            new(Path.Combine(Config.IndicConformerSubDir, Config.EncoderFile),             Config.EncoderFile),
            new(Path.Combine(Config.IndicConformerSubDir, $"{Config.EncoderFile}.data"),   $"{Config.EncoderFile}.data"),
            new(Path.Combine(Config.IndicConformerSubDir, Config.CtcDecoderFile),          Config.CtcDecoderFile),
            new(Path.Combine(Config.IndicConformerSubDir, Config.VocabFile),               Config.VocabFile),
            new(Path.Combine(Config.IndicConformerSubDir, Config.IndicConformerLanguageSpansFile), Config.IndicConformerLanguageSpansFile),
            new(Path.Combine(Config.IndicConformerSubDir, Config.AsrConfigFile),           Config.AsrConfigFile),
        ];

    private static readonly ModelAsset[] VibeVoiceFiles =
        [
            new(Path.Combine(Config.VibeVoiceSubDir, VibeVoiceAsr.AudioEncoderFile),                       VibeVoiceAsr.AudioEncoderFile),
            new(Path.Combine(Config.VibeVoiceSubDir, $"{VibeVoiceAsr.AudioEncoderFile}.data"),            $"{VibeVoiceAsr.AudioEncoderFile}.data"),
            new(Path.Combine(Config.VibeVoiceSubDir, VibeVoiceAsr.DecoderSingleFile),                     VibeVoiceAsr.DecoderSingleFile),
            new(Path.Combine(Config.VibeVoiceSubDir, $"{VibeVoiceAsr.DecoderSingleFile}.data"),            $"{VibeVoiceAsr.DecoderSingleFile}.data"),
            new(Path.Combine(Config.VibeVoiceSubDir, "config.json"),                                      "config.json"),
            new(Path.Combine(Config.VibeVoiceSubDir, "processor_config.json"),                            "processor_config.json"),
            new(Path.Combine(Config.VibeVoiceSubDir, "tokenizer_config.json"),                            "tokenizer_config.json"),
            new(Path.Combine(Config.VibeVoiceSubDir, "chat_template.jinja"),                              "chat_template.jinja"),
            new(Path.Combine(Config.VibeVoiceSubDir, VibeVoiceAsr.ExportReportFile),                       VibeVoiceAsr.ExportReportFile),
            new(Path.Combine(Config.VibeVoiceSubDir, VibeVoiceAsr.TokenizerFile),                          VibeVoiceAsr.TokenizerFile),
        ];

    private readonly SettingsService _settings;
    private readonly HttpClient _http = new(new HttpClientHandler { AllowAutoRedirect = true });

    public ModelManagerService(SettingsService settings) => _settings = settings;

    private AssetRepo[] ActiveRepos()
    {
        if (_settings.Current.AsrBackend == AsrBackend.VibeVoice ||
            _settings.Current.Segmentation == SegmentationMode.VibeVoiceBuiltin)
        {
            return [new AssetRepo(VibeVoiceRepoBase, VibeVoiceManifestUrl, VibeVoiceFiles)];
        }

        return _settings.Current.AsrBackend switch
        {
            AsrBackend.Cohere =>
            [
                new AssetRepo(CoreRepoBase, CoreManifestUrl, CoreDiarizationFiles),
                new AssetRepo(CohereRepoBase, CohereManifestUrl, CohereFiles),
            ],
            AsrBackend.Qwen3Asr =>
            [
                new AssetRepo(CoreRepoBase, CoreManifestUrl, CoreDiarizationFiles),
                new AssetRepo("", "", Qwen3AsrFiles),
            ],
            AsrBackend.IndicConformer =>
            [
                new AssetRepo(CoreRepoBase, CoreManifestUrl, CoreDiarizationFiles),
                new AssetRepo(IndicConformerRepoBase, IndicConformerManifestUrl, IndicConformerFiles),
            ],
            AsrBackend.WhisperTurbo =>
            [
                new AssetRepo(CoreRepoBase, CoreManifestUrl, CoreDiarizationFiles),
                new AssetRepo(WhisperTurboRepoBase, "", WhisperTurboFiles),
                new AssetRepo("", "", WhisperTurboAuxFiles),
            ],
            _ =>
            [
                new AssetRepo(CoreRepoBase, CoreManifestUrl, [.. CoreDiarizationFiles, .. AsrFilesFp32]),
            ],
        };
    }

    private ModelAsset[] RequiredFiles() =>
        [.. ActiveRepos().SelectMany(repo => repo.Assets)];

    private RepoAsset[] DownloadableFiles() =>
        [.. ActiveRepos()
            .Where(repo => !string.IsNullOrEmpty(repo.RepoBase))
            .SelectMany(repo => repo.Assets.Select(asset =>
                new RepoAsset(repo.RepoBase, asset.LocalRelativePath, asset.RemoteRelativePath)))];

    public IReadOnlyList<string> GetMissingFiles()
    {
        string dir = _settings.GetModelsDir();
        return RequiredFiles()
            .Where(asset => !File.Exists(Path.Combine(dir, asset.LocalRelativePath)))
            .Select(asset => asset.LocalRelativePath)
            .ToList();
    }

    public IReadOnlyList<string> GetPresentFiles()
    {
        string dir = _settings.GetModelsDir();
        return RequiredFiles()
            .Where(asset => File.Exists(Path.Combine(dir, asset.LocalRelativePath)))
            .Select(asset => asset.LocalRelativePath)
            .ToList();
    }

    public IReadOnlyList<string> GetMissingDownloadableFiles()
    {
        string dir = _settings.GetModelsDir();
        return DownloadableFiles()
            .Where(asset => !File.Exists(Path.Combine(dir, asset.LocalRelativePath)))
            .Select(asset => asset.LocalRelativePath)
            .ToList();
    }

    public IReadOnlyList<string> GetMissingDiariZenFiles(string? diarizenDir = null)
    {
        string dir = diarizenDir ?? _settings.GetDiariZenModelsDir();
        return DiariZenFiles
            .Where(asset => !File.Exists(Path.Combine(dir, Path.GetRelativePath("diarizen", asset.LocalRelativePath))))
            .Select(asset => Path.GetRelativePath("diarizen", asset.LocalRelativePath))
            .ToList();
    }

    public IReadOnlyList<string> GetPresentDiariZenFiles(string? diarizenDir = null)
    {
        string dir = diarizenDir ?? _settings.GetDiariZenModelsDir();
        return DiariZenFiles
            .Where(asset => File.Exists(Path.Combine(dir, Path.GetRelativePath("diarizen", asset.LocalRelativePath))))
            .Select(asset => Path.GetRelativePath("diarizen", asset.LocalRelativePath))
            .ToList();
    }

    public bool AreDiariZenModelsPresent(string? diarizenDir = null) =>
        GetMissingDiariZenFiles(diarizenDir).Count == 0;

    public IReadOnlyList<string> GetMissingVoxLinguaFiles(string? voxLinguaDir = null)
    {
        string dir = voxLinguaDir ?? _settings.GetVoxLinguaModelsDir();
        return VoxLinguaFiles
            .Where(asset => !File.Exists(Path.Combine(dir, Path.GetRelativePath(Config.VoxLinguaSubDir, asset.LocalRelativePath))))
            .Select(asset => Path.GetRelativePath(Config.VoxLinguaSubDir, asset.LocalRelativePath))
            .ToList();
    }

    public IReadOnlyList<string> GetPresentVoxLinguaFiles(string? voxLinguaDir = null)
    {
        string dir = voxLinguaDir ?? _settings.GetVoxLinguaModelsDir();
        return VoxLinguaFiles
            .Where(asset => File.Exists(Path.Combine(dir, Path.GetRelativePath(Config.VoxLinguaSubDir, asset.LocalRelativePath))))
            .Select(asset => Path.GetRelativePath(Config.VoxLinguaSubDir, asset.LocalRelativePath))
            .ToList();
    }

    public bool AreVoxLinguaModelsPresent(string? voxLinguaDir = null) =>
        GetMissingVoxLinguaFiles(voxLinguaDir).Count == 0;

    public bool AllModelsPresent() => GetMissingFiles().Count == 0;

    /// <summary>
    /// Fetches the remote manifest and returns the names of any local files whose
    /// MD5 hash doesn't match the manifest (i.e. legacy / outdated / corrupt weights).
    /// Returns null if the manifest cannot be fetched (offline, server error, etc.)
    /// so callers can silently skip the check.
    /// An optional progress callback receives (fileName, fileIndex, totalFiles) as each
    /// file is hashed so the UI can show progress on large downloads.
    /// </summary>
    public async Task<IReadOnlyList<string>?> GetOutdatedFilesAsync(
        IProgress<(string fileName, int index, int total)>? progress = null,
        CancellationToken ct = default)
    {
        string dir     = _settings.GetModelsDir();
        var outdated = new List<string>();
        var toCheck = new List<(string localRelativePath, string expectedHash)>();

        foreach (var repo in ActiveRepos())
        {
            if (string.IsNullOrWhiteSpace(repo.ManifestUrl))
                continue;

            string json;
            try
            {
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                cts.CancelAfter(TimeSpan.FromSeconds(10));
                json = await _http.GetStringAsync(repo.ManifestUrl, cts.Token);
            }
            catch { return null; }

            Dictionary<string, string> manifest;
            try   { manifest = ParseManifestHashes(json); }
            catch { return null; }

            foreach (var asset in repo.Assets)
            {
                // Skip Sortformer .onnx.data files — ONNX Runtime regenerates
                // these as external weights at runtime, so their hashes will
                // never match the manifest.  The main .onnx model file is
                // checked normally below.
                if (asset.LocalRelativePath.EndsWith(Config.SortformerDataFile, StringComparison.OrdinalIgnoreCase))
                    continue;

                string path = Path.Combine(dir, asset.LocalRelativePath);
                if (manifest.TryGetValue(asset.RemoteRelativePath, out var expectedHash) &&
                    expectedHash is not null &&
                    File.Exists(path))
                {
                    toCheck.Add((asset.LocalRelativePath, expectedHash));
                }
            }
        }

        int total = toCheck.Count;

        for (int i = 0; i < total; i++)
        {
            ct.ThrowIfCancellationRequested();
            var asset = toCheck[i];
            progress?.Report((asset.localRelativePath, i, total));

            string path       = Path.Combine(dir, asset.localRelativePath);
            string actualHash = await Task.Run(() => ComputeMd5(path), ct);

            if (!string.Equals(actualHash, asset.expectedHash, StringComparison.OrdinalIgnoreCase))
                outdated.Add(asset.localRelativePath);
        }
        return outdated;
    }

    private static Dictionary<string, string> ParseManifestHashes(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var files = doc.RootElement.GetProperty("files");
        var dict  = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var prop in files.EnumerateObject())
            dict[prop.Name] = prop.Value.GetProperty("md5").GetString() ?? "";
        return dict;
    }

    private static string ComputeMd5(string filePath)
    {
        using var md5    = MD5.Create();
        using var stream = File.OpenRead(filePath);
        return Convert.ToHexString(md5.ComputeHash(stream)).ToLowerInvariant();
    }

    /// <summary>
    /// Checks whether the Sortformer model file on disk matches the expected
    /// version (by MD5 hash).  Returns a tuple of (isCorrectVersion, message):
    ///  - (true, _)  = the model file hash matches the manifest
    ///  - (false, _) = the model is outdated, corrupted, or missing
    /// Returns (null, _) if the check cannot be performed (offline, no file, etc.)
    /// so callers can silently skip it.
    /// </summary>
    public async Task<(bool? isCorrect, string message)?> CheckSortformerModelAsync(
        CancellationToken ct = default)
    {
        string modelPath = Path.Combine(_settings.GetModelsDir(),
            Path.Combine(Config.SortformerSubDir, Config.SortformerFile));

        if (!File.Exists(modelPath))
            return (false, $"{Config.SortformerFile} not found.");

        // Fetch the manifest to get the expected hash
        string json;
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(10));
            json = await _http.GetStringAsync(CoreManifestUrl, cts.Token);
        }
        catch { return null; }

        Dictionary<string, string> manifest;
        try { manifest = ParseManifestHashes(json); }
        catch { return null; }

        if (!manifest.TryGetValue(Config.SortformerFile, out var expectedHash) ||
            string.IsNullOrEmpty(expectedHash))
            return null;

        string actualHash = await Task.Run(() => ComputeMd5(modelPath), ct);

        if (string.Equals(actualHash, expectedHash, StringComparison.OrdinalIgnoreCase))
            return (true, $"{Config.SortformerFile} is up to date.");

        return (false, $"{Config.SortformerFile} is outdated (expected {expectedHash[..8]}…, got {actualHash[..8]}…).");
    }

    /// <summary>
    /// Deletes the specified files from the models directory so they will be
    /// treated as missing and re-downloaded by DownloadMissingModelsAsync.
    /// </summary>
    public void PrepareRedownload(IEnumerable<string> files)
    {
        string dir = _settings.GetModelsDir();
        foreach (var file in files)
        {
            var path = Path.Combine(dir, file);
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AddDllDirectory(string newDirectory);

    /// <summary>
    /// MSIX packaged apps use a restricted DLL search that excludes the system PATH.
    /// Explicitly register CUDA Toolkit directories so onnxruntime_providers_cuda.dll
    /// can find cudart, cublas, cudnn, etc. at runtime.
    /// </summary>
    internal static void AddCudaToSearchPath()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        var dirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // CUDA Toolkit installer sets CUDA_PATH (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8)
        string? cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (!string.IsNullOrEmpty(cudaPath))
            dirs.Add(Path.Combine(cudaPath, "bin"));

        // Also scan PATH for any CUDA/cuDNN entries (handles cuDNN installed separately)
        string? path = Environment.GetEnvironmentVariable("PATH");
        if (path != null)
            foreach (var entry in path.Split(';'))
                if (!string.IsNullOrWhiteSpace(entry) &&
                    (entry.Contains("CUDA",  StringComparison.OrdinalIgnoreCase) ||
                     entry.Contains("cuDNN", StringComparison.OrdinalIgnoreCase)))
                    dirs.Add(entry.Trim());

        foreach (var dir in dirs)
            if (Directory.Exists(dir))
                AddDllDirectory(dir);
    }

    public (bool Available, string Message) CheckCuda()
    {
        string logPath  = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Vernacula", "cuda_debug.txt");
        Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);

        string modelFile = Path.Combine(_settings.GetParakeetModelsDir(), Config.PreprocessorFile);
        if (!File.Exists(modelFile))
        {
            string msg = $"{Config.PreprocessorFile} not found at: {modelFile}";
            File.WriteAllText(logPath, msg);
            return (false, msg);
        }
        try
        {
            if (OperatingSystem.IsMacOS())
            {
                const string macOsMessage = "CUDA execution is not supported on macOS.";
                File.WriteAllText(logPath, macOsMessage);
                return (false, macOsMessage);
            }

            if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            {
                string unsupportedMessage = $"CUDA execution is not supported on {RuntimeInformation.OSDescription}.";
                File.WriteAllText(logPath, unsupportedMessage);
                return (false, unsupportedMessage);
            }

            if (!HardwareInfo.CanProbeCudaExecutionProvider())
            {
                const string unavailableMessage = "CUDA execution is unavailable on this machine.";
                File.WriteAllText(logPath, unavailableMessage);
                return (false, unavailableMessage);
            }

            AddCudaToSearchPath();
            var opts = new SessionOptions();
            opts.AppendExecutionProvider_CUDA(0);
            using var session = new InferenceSession(modelFile, opts);
            string msg = $"CUDA OK on {GetPlatformName()}";
            File.WriteAllText(logPath, msg);
            return (true, msg);
        }
        catch (Exception ex)
        {
            string msg = $"CUDA check failed on {GetPlatformName()}.\nException: {ex.GetType().Name}\n{ex.Message}\n\nInner: {ex.InnerException?.Message}\n\nStack:\n{ex.StackTrace}";
            File.WriteAllText(logPath, msg);
            return (false, msg);
        }
    }

    public bool IsCudaAvailable() => CheckCuda().Available;

    private static string GetPlatformName()
    {
        if (OperatingSystem.IsWindows()) return "Windows";
        if (OperatingSystem.IsLinux()) return "Linux";
        if (OperatingSystem.IsMacOS()) return "macOS";
        return RuntimeInformation.OSDescription;
    }

    public async Task DownloadMissingModelsAsync(
        IProgress<DownloadProgress> progress,
        CancellationToken ct = default)
    {
        string dir = _settings.GetModelsDir();
        Directory.CreateDirectory(dir);

        var missing = DownloadableFiles()
            .Where(asset => !File.Exists(Path.Combine(dir, asset.LocalRelativePath)))
            .ToList();
        await DownloadMissingAssetsAsync(dir, missing, progress, ct);
    }

    public async Task DownloadMissingDiariZenModelsAsync(
        IProgress<DownloadProgress> progress,
        string? diarizenDir = null,
        CancellationToken ct = default)
    {
        string dir = diarizenDir ?? _settings.GetDiariZenModelsDir();
        Directory.CreateDirectory(dir);

        var missing = DiariZenFiles
            .Where(asset => !File.Exists(Path.Combine(dir, Path.GetRelativePath("diarizen", asset.LocalRelativePath))))
            .Select(asset => new RepoAsset(
                DiariZenRepoBase,
                Path.GetRelativePath("diarizen", asset.LocalRelativePath),
                asset.RemoteRelativePath))
            .ToList();

        await DownloadMissingAssetsAsync(dir, missing, progress, ct);
    }

    // ── KenLM for Parakeet (per-language/domain files under a single repo) ──

    /// <summary>
    /// Full local path the given built-in LM option would live at. Returns
    /// <c>null</c> for the <c>none</c> and <c>custom</c> slots, which have
    /// no remote file.
    /// </summary>
    public string? GetKenLmLocalPath(KenLmOption option)
    {
        if (option.RemoteFileName is null) return null;
        return Path.Combine(_settings.GetKenLmParakeetDir(), option.RemoteFileName);
    }

    /// <summary>True when the option's file is already on disk.</summary>
    public bool IsKenLmReady(KenLmOption option)
    {
        string? path = GetKenLmLocalPath(option);
        return path is not null && File.Exists(path);
    }

    /// <summary>
    /// Downloads a single KenLM file if missing. Reports byte-precise
    /// progress through the shared pipeline. No-op if the file already
    /// exists, or the option has no remote file (none/custom).
    /// </summary>
    public async Task DownloadKenLmAsync(
        KenLmOption option,
        IProgress<DownloadProgress> progress,
        CancellationToken ct = default)
    {
        if (option.RemoteFileName is null) return;
        string dir = _settings.GetKenLmParakeetDir();
        Directory.CreateDirectory(dir);

        string localPath = Path.Combine(dir, option.RemoteFileName);
        if (File.Exists(localPath)) return;

        var asset = new RepoAsset(
            Config.KenLmParakeetRepoBase,
            option.RemoteFileName,
            option.RemoteFileName);
        await DownloadMissingAssetsAsync(dir, [asset], progress, ct);
    }

    public async Task DownloadMissingVoxLinguaModelsAsync(
        IProgress<DownloadProgress> progress,
        string? voxLinguaDir = null,
        CancellationToken ct = default)
    {
        string dir = voxLinguaDir ?? _settings.GetVoxLinguaModelsDir();
        Directory.CreateDirectory(dir);

        var missing = VoxLinguaFiles
            .Where(asset => !File.Exists(Path.Combine(dir, Path.GetRelativePath(Config.VoxLinguaSubDir, asset.LocalRelativePath))))
            .Select(asset => new RepoAsset(
                VoxLinguaRepoBase,
                Path.GetRelativePath(Config.VoxLinguaSubDir, asset.LocalRelativePath),
                asset.RemoteRelativePath))
            .ToList();

        await DownloadMissingAssetsAsync(dir, missing, progress, ct);
    }

    private async Task DownloadMissingAssetsAsync(
        string dir,
        IReadOnlyList<RepoAsset> missing,
        IProgress<DownloadProgress> progress,
        CancellationToken ct)
    {
        int total = missing.Count;
        if (total == 0) return;

        // Phase 1: HEAD each file to get its size so we can compute a byte-weighted grand total.
        var fileSizes = new long[total];
        for (int i = 0; i < total; i++)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                using var req  = new HttpRequestMessage(HttpMethod.Head, $"{missing[i].RepoBase}/{missing[i].RemoteRelativePath}");
                using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
                fileSizes[i] = resp.Content.Headers.ContentLength ?? 0;
            }
            catch { fileSizes[i] = 0; }
        }
        long grandTotal = fileSizes.Sum();

        // Phase 2: Stream-download each file, reporting byte-precise progress.
        long prevBytes = 0;
        for (int i = 0; i < total; i++)
        {
            ct.ThrowIfCancellationRequested();
            var asset = missing[i];
            string fileName = asset.LocalRelativePath;
            string url      = $"{asset.RepoBase}/{asset.RemoteRelativePath}";
            string destPath = Path.Combine(dir, asset.LocalRelativePath);
            string tmpPath  = destPath + ".download";
            Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);

            // Check for a partial download to resume
            long resumeFrom = File.Exists(tmpPath) ? new FileInfo(tmpPath).Length : 0;

            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, url);
                if (resumeFrom > 0)
                    req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(resumeFrom, null);

                using var response = await _http.SendAsync(
                    req, HttpCompletionOption.ResponseHeadersRead, ct);

                // If server ignored Range and returned 200, restart from the beginning
                bool appending = resumeFrom > 0 && response.StatusCode == System.Net.HttpStatusCode.PartialContent;
                if (!appending) resumeFrom = 0;
                response.EnsureSuccessStatusCode();

                long totalBytes = fileSizes[i];
                long downloaded = resumeFrom;

                await using var src  = await response.Content.ReadAsStreamAsync(ct);
                await using var dest = appending
                    ? new FileStream(tmpPath, FileMode.Append, FileAccess.Write)
                    : File.Create(tmpPath);

                var buffer = new byte[81920];
                int read;
                while ((read = await src.ReadAsync(buffer, ct)) > 0)
                {
                    await dest.WriteAsync(buffer.AsMemory(0, read), ct);
                    downloaded += read;
                    progress.Report(new DownloadProgress(
                        fileName, i, total, downloaded, totalBytes, grandTotal, prevBytes));
                }
            }
            catch
            {
                // Preserve the partial file so the download can be resumed next time
                throw;
            }

            if (File.Exists(destPath)) File.Delete(destPath);
            File.Move(tmpPath, destPath);
            prevBytes += fileSizes[i];
        }
    }
}
