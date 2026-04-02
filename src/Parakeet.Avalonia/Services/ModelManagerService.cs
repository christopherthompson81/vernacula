using System.Net.Http;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Parakeet.Base;
using Parakeet.Base.Models;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Services;

internal class ModelManagerService
{
    private const string RepoBase =
        "https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx/resolve/main";

    private static readonly string[] DiarizationFiles =
        ["diar_streaming_sortformer_4spk-v2.1.onnx", Config.VadFile];

    private static readonly string[] AsrFilesInt8 =
        ["nemo128.onnx", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx",
         "vocab.txt", "config.json"];

    private static readonly string[] AsrFilesFp32 =
        ["nemo128.onnx", "encoder-model.onnx", "encoder-model.onnx.data",
         "decoder_joint-model.onnx", "vocab.txt", "config.json"];

    private readonly SettingsService _settings;
    private readonly HttpClient _http = new(new HttpClientHandler { AllowAutoRedirect = true });

    public ModelManagerService(SettingsService settings) => _settings = settings;

    private string[] AllFiles() =>
        [.. DiarizationFiles,
         .. (_settings.Current.Precision == ModelPrecision.Int8 ? AsrFilesInt8 : AsrFilesFp32)];

    public IReadOnlyList<string> GetMissingFiles()
    {
        string dir = _settings.GetModelsDir();
        return AllFiles().Where(f => !File.Exists(Path.Combine(dir, f))).ToList();
    }

    public IReadOnlyList<string> GetPresentFiles()
    {
        string dir = _settings.GetModelsDir();
        return AllFiles().Where(f => File.Exists(Path.Combine(dir, f))).ToList();
    }

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
        string json;
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(10));
            json = await _http.GetStringAsync(Config.ManifestUrl, cts.Token);
        }
        catch { return null; }

        Dictionary<string, string> manifest;
        try   { manifest = ParseManifestHashes(json); }
        catch { return null; }

        string dir     = _settings.GetModelsDir();
        var outdated   = new List<string>();
        var toCheck    = manifest.Where(kv => File.Exists(Path.Combine(dir, kv.Key))).ToList();
        int total      = toCheck.Count;

        for (int i = 0; i < total; i++)
        {
            ct.ThrowIfCancellationRequested();
            var (fileName, expectedHash) = toCheck[i];
            progress?.Report((fileName, i, total));

            string path       = Path.Combine(dir, fileName);
            string actualHash = await Task.Run(() => ComputeMd5(path), ct);

            if (!string.Equals(actualHash, expectedHash, StringComparison.OrdinalIgnoreCase))
                outdated.Add(fileName);
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
            "Parakeet", "cuda_debug.txt");
        Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);

        string modelFile = Path.Combine(_settings.GetModelsDir(), "nemo128.onnx");
        if (!File.Exists(modelFile))
        {
            string msg = $"nemo128.onnx not found at: {modelFile}";
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

        var missing = GetMissingFiles();
        int total   = missing.Count;
        if (total == 0) return;

        // Phase 1: HEAD each file to get its size so we can compute a byte-weighted grand total.
        var fileSizes = new long[total];
        for (int i = 0; i < total; i++)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                using var req  = new HttpRequestMessage(HttpMethod.Head, $"{RepoBase}/{missing[i]}");
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
            string fileName = missing[i];
            string url      = $"{RepoBase}/{fileName}";
            string destPath = Path.Combine(dir, fileName);
            string tmpPath  = destPath + ".download";

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
