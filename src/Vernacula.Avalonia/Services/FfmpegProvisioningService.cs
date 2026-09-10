using System;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using Vernacula.App.Models;
using Vernacula.Base;

namespace Vernacula.App.Services;

/// <summary>
/// Downloads FFmpeg on demand, the way the app already downloads models.
///
/// <para>
/// ⚠ WHY THIS EXISTS AT ALL. Vernacula decodes WAV, MP3, AIFF, Ogg Vorbis and Ogg Opus
/// in-process (see <see cref="ManagedAudioDecoders"/>), which covers most of what an ASR
/// user arrives with. FLAC, M4A/AAC, WMA and every video container still need FFmpeg, and
/// on Windows — unlike Linux and macOS, which have package managers — asking for it means
/// asking a user to find a build, unzip it, and edit PATH before the app will open an iPhone
/// voice memo. The app already fetches large platform-specific binaries on demand
/// (ModelManagerService, including the CoreML Sortformer variant), so it can fetch this one.
/// </para>
///
/// <para>
/// ⚠ THE EXECUTABLES, NOT THE SHARED LIBRARIES, AND THAT IS THE WHOLE DESIGN. FFmpeg.AutoGen
/// binds a fixed soname — avformat-60, i.e. FFmpeg 6.x — while current FFmpeg releases ship
/// 61, 62 and 63, so a downloaded shared build would have to be version-matched to the
/// binding forever, and would still leave Vernacula.CLI (which shells out) unfixed. The
/// executables have no such coupling: any recent ffmpeg/ffprobe works, both surfaces resolve
/// them through <see cref="FfmpegBinaries"/>, and the in-process AutoGen path stays as an
/// optimisation for whoever happens to have matching libraries installed.
/// </para>
///
/// <para>
/// ⚠ PINNED BY VERSION AND HASH, FROM AN ARCHIVE THAT DOES NOT MOVE. GyanD/codexffmpeg keeps
/// every tagged release (back to 2023), so the URL below stays valid and the SHA-256 stays
/// meaningful. This deliberately rules out the "latest" endpoints that other build hosts
/// offer: BtbN's assets are re-uploaded in place and its autobuild tags are pruned after
/// about a month, so neither a URL nor a hash pinned against them survives.
/// </para>
///
/// <para>
/// ⚠ LICENSING. This is a GPL FFmpeg build, and Vernacula neither redistributes nor links to
/// it: the user's machine fetches it, and it is invoked as a separate program over a process
/// boundary. That is the same arrangement Audacity and many other permissively-licensed
/// tools use. Nothing here is bundled into a Vernacula build, and the archive's own LICENSE
/// is written out beside the binaries. If this project ever wants to ship FFmpeg in the
/// installer instead, that is a different question with a different answer (an LGPL build),
/// and <see cref="Source"/> is the single place to change.
/// </para>
/// </summary>
internal sealed class FfmpegProvisioningService
{
    /// <summary>Where a build comes from, and how to know it arrived intact.</summary>
    private sealed record Build(string Version, string Url, string Sha256, string ArchiveBinDirSuffix);

    /// <summary>
    /// ⚠ CHANGING THIS MEANS CHANGING THE HASH. The hash is the only thing standing between a
    /// user and an executable served by whoever controls the network in between, so it is
    /// verified before a single byte is unpacked. Recompute with:
    ///   <c>sha256sum ffmpeg-&lt;version&gt;-essentials_build.zip</c>
    /// </summary>
    private static readonly Build Source = new(
        Version: "9.0.1",
        Url: "https://github.com/GyanD/codexffmpeg/releases/download/9.0.1/ffmpeg-9.0.1-essentials_build.zip",
        Sha256: "fec81ae03971d9dd4be3ebe02e263bd2ec1d789483f931bdba5f5715e65da2e9",
        ArchiveBinDirSuffix: "ffmpeg-9.0.1-essentials_build/bin/");

    /// <summary>Roughly the download size, for a caller that wants to ask before starting.</summary>
    public const long ApproximateDownloadBytes = 111_253_802;

    private readonly HttpClient _http = new(new HttpClientHandler { AllowAutoRedirect = true })
    {
        Timeout = TimeSpan.FromMinutes(30),
    };

    /// <summary>
    /// ⚠ ONE DOWNLOAD AT A TIME. Adding a folder of M4As calls TryEnsureAsync once per file,
    /// and the UI can start an enqueue while another is in flight. Without this, two callers
    /// that both saw "not installed" would each fetch 111 MB and then race to File.Move the
    /// same two executables into place.
    /// </summary>
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>
    /// Set once FFmpeg has been seen working. Probing costs two process spawns, and the gate
    /// above serialises callers, so a bulk add of fifty files would otherwise spend a hundred
    /// spawns re-answering a question whose answer cannot go from true to false while the app
    /// runs — short of someone deleting FFmpeg mid-session, which the decode error covers.
    /// </summary>
    private static volatile bool _known;

    /// <summary>True when ffmpeg AND ffprobe can both be launched. Both are needed.</summary>
    /// <remarks>
    /// ⚠ BOTH, NOT JUST ffmpeg. The desktop app probes multi-stream files with ffprobe, and a
    /// build carrying only ffmpeg (some minimal packages, and pip's imageio-ffmpeg) leaves
    /// video files failing at enqueue time while audio decodes fine — a confusing half-working
    /// state that reads as a Vernacula bug.
    /// </remarks>
    public static bool IsInstalled
    {
        get
        {
            if (_known) return true;
            bool present = FfmpegBinaries.IsAvailable("ffmpeg") && FfmpegBinaries.IsAvailable("ffprobe");
            if (present) _known = true;
            return present;
        }
    }

    /// <summary>
    /// True where a download is offered. Windows only, deliberately: Linux and macOS have
    /// package managers that install FFmpeg correctly in one command, and docs/installation.md
    /// gives that command. Second-guessing them would mean maintaining a build matrix to
    /// replace something the platform already does better.
    /// </summary>
    public static bool CanDownload =>
        OperatingSystem.IsWindows() && System.Runtime.InteropServices.RuntimeInformation.OSArchitecture
            is System.Runtime.InteropServices.Architecture.X64;

    /// <summary>True when <paramref name="path"/> is a format that cannot be read without FFmpeg.</summary>
    public static bool NeedsFfmpeg(string path) =>
        !ManagedAudioDecoders.Handles(Path.GetExtension(path));

    /// <summary>
    /// Make FFmpeg available if it is not already, and report whether it now is.
    /// Never throws for the ordinary reasons — no network, unsupported platform — so a caller
    /// can treat it as "best effort" and let the decode produce the actionable error.
    /// </summary>
    public async Task<bool> TryEnsureAsync(
        IProgress<DownloadProgress>? progress = null, CancellationToken ct = default)
    {
        if (IsInstalled) return true;
        if (!CanDownload) return false;

        await _gate.WaitAsync(ct);
        try
        {
            // Re-check inside the gate: while this caller waited, the one ahead of it may
            // have finished the download, and this would otherwise fetch it a second time.
            if (IsInstalled) return true;

            await DownloadAsync(progress, ct);
            return IsInstalled;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[FFmpeg] provisioning failed: {ex}");
            return false;
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task DownloadAsync(IProgress<DownloadProgress>? progress, CancellationToken ct)
    {
        string target = FfmpegBinaries.ManagedDirectory;
        Directory.CreateDirectory(target);

        // ⚠ STAGE IN A TEMP FILE, NOT IN THE TARGET. An interrupted download that left a
        // half-written ffmpeg.exe in place would be found by FfmpegBinaries on the next run
        // and spawned, and the failure would look like a corrupt media file rather than a
        // corrupt install.
        string staging = Path.Combine(Path.GetTempPath(), $"vernacula-ffmpeg-{Guid.NewGuid():N}.zip");

        try
        {
            using (var response = await _http.GetAsync(Source.Url, HttpCompletionOption.ResponseHeadersRead, ct))
            {
                response.EnsureSuccessStatusCode();
                long total = response.Content.Headers.ContentLength ?? ApproximateDownloadBytes;

                using var http = await response.Content.ReadAsStreamAsync(ct);
                using var file = File.Create(staging);

                var buffer = new byte[1 << 20];
                long done = 0;
                int read;
                while ((read = await http.ReadAsync(buffer, ct)) > 0)
                {
                    await file.WriteAsync(buffer.AsMemory(0, read), ct);
                    done += read;
                    progress?.Report(new DownloadProgress(
                        $"ffmpeg {Source.Version}", 0, 1, done, total, total, 0));
                }
            }

            string actual = await Sha256Async(staging, ct);
            if (!string.Equals(actual, Source.Sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException(
                    $"Downloaded FFmpeg archive has SHA-256 {actual}, expected {Source.Sha256}. "
                    + "Refusing to unpack it.");

            Extract(staging, target);
        }
        finally
        {
            try { if (File.Exists(staging)) File.Delete(staging); } catch { /* best-effort */ }
        }
    }

    /// <summary>
    /// Unpack just the two executables and the licence. ffplay is another 100 MB of video
    /// player this app never invokes, and the rest of the archive is documentation and presets.
    /// </summary>
    private static void Extract(string archivePath, string target)
    {
        using var zip = ZipFile.OpenRead(archivePath);

        foreach (string wanted in new[] { "ffmpeg.exe", "ffprobe.exe" })
        {
            ZipArchiveEntry entry = zip.GetEntry(Source.ArchiveBinDirSuffix + wanted)
                ?? throw new InvalidOperationException(
                    $"FFmpeg archive did not contain {Source.ArchiveBinDirSuffix}{wanted}.");

            // Extract beside the destination and move into place, so a reader that catches
            // the directory mid-unpack never sees a partial executable under the real name.
            string final   = Path.Combine(target, wanted);
            string partial = final + ".partial";
            entry.ExtractToFile(partial, overwrite: true);
            File.Move(partial, final, overwrite: true);
        }

        // The archive's own licence text travels with the binaries, since that is what the
        // user now has a copy of.
        zip.GetEntry(Source.ArchiveBinDirSuffix.Replace("bin/", "LICENSE"))
           ?.ExtractToFile(Path.Combine(target, "LICENSE"), overwrite: true);

        File.WriteAllText(
            Path.Combine(target, "SOURCE.txt"),
            $"FFmpeg {Source.Version}, downloaded by Vernacula from:{Environment.NewLine}"
            + $"{Source.Url}{Environment.NewLine}"
            + $"SHA-256 {Source.Sha256}{Environment.NewLine}{Environment.NewLine}"
            + $"Delete this directory to remove it. Vernacula prefers an ffmpeg on PATH over "
            + $"this copy, so a system install takes over automatically.{Environment.NewLine}");
    }

    private static async Task<string> Sha256Async(string path, CancellationToken ct)
    {
        using var sha = SHA256.Create();
        await using var stream = File.OpenRead(path);
        byte[] hash = await sha.ComputeHashAsync(stream, ct);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
