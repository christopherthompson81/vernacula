using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.Base.Models;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// Selecting a backend must download that backend's model, and only that model.
///
/// This exists because it once did not. VibeVoice-ASR Streaming forces
/// <see cref="SegmentationMode.VibeVoiceBuiltin"/>, and the repo selector had an early return
/// keyed on that mode which pre-dated the streaming backend — so choosing Streaming queued the
/// *non-streaming* package: a different model, and 17 GB of it. Nothing else caught that,
/// because every other check only asks whether a mapping exists, not whether it is the right one.
/// </summary>
public class ModelRepoSelectionTests
{
    private static SettingsService SettingsIn(string dir, Action<AppSettings> configure)
    {
        var svc = new SettingsService();
        svc.Current.ModelsDir = dir;
        configure(svc.Current);
        return svc;
    }

    /// <summary>Local paths of every file the manager would require for the current settings.</summary>
    private static IReadOnlyList<string> RequiredPaths(SettingsService settings)
    {
        var mgr = new ModelManagerService(settings);
        var method = typeof(ModelManagerService).GetMethod(
            "RequiredFiles", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(method);
        var assets = (Array)method!.Invoke(mgr, null)!;
        return [.. assets.Cast<object>().Select(a =>
            (string)a.GetType().GetProperty("LocalRelativePath")!.GetValue(a)!)];
    }

    [Theory]
    [InlineData(VibeVoiceStreamingSize.Small1_5B, "vibevoice_asr_streaming_1_5b")]
    [InlineData(VibeVoiceStreamingSize.Large7B,   "vibevoice_asr_streaming_7b")]
    public void StreamingBackend_RequiresItsOwnPackage_AndNotTheNonStreamingOne(
        VibeVoiceStreamingSize size, string expectedDir)
    {
        using var tmp = new TempDir();
        var settings = SettingsIn(tmp.Path, s =>
        {
            s.AsrBackend = AsrBackend.VibeVoiceStreaming;
            s.VibeVoiceStreamingSize = size;
            // The app forces this mode for both VibeVoice backends; the selector must still
            // tell them apart.
            s.Segmentation = SegmentationMode.VibeVoiceBuiltin;
        });

        var paths = RequiredPaths(settings);
        Assert.NotEmpty(paths);
        Assert.All(paths, p => Assert.StartsWith(expectedDir, p));
        Assert.Contains(paths, p => p.EndsWith("decoder_gqa.onnx"));
        // The non-streaming package is a different model entirely.
        Assert.DoesNotContain(paths, p => p.EndsWith("decoder_single.onnx"));
    }

    [Fact]
    public void NonStreamingVibeVoice_IsUnaffected()
    {
        using var tmp = new TempDir();
        var settings = SettingsIn(tmp.Path, s =>
        {
            s.AsrBackend = AsrBackend.VibeVoice;
            s.Segmentation = SegmentationMode.VibeVoiceBuiltin;
        });

        var paths = RequiredPaths(settings);
        Assert.Contains(paths, p => p.EndsWith("decoder_single.onnx"));
        Assert.DoesNotContain(paths, p => p.StartsWith("vibevoice_asr_streaming"));
    }

    [Fact]
    public void BuiltinSegmentationAlone_StillMeansTheNonStreamingPackage()
    {
        // A non-VibeVoice ASR backend can still use VibeVoice for segmentation; that path must
        // keep pulling the non-streaming package.
        using var tmp = new TempDir();
        var settings = SettingsIn(tmp.Path, s =>
        {
            s.AsrBackend = AsrBackend.Parakeet;
            s.Segmentation = SegmentationMode.VibeVoiceBuiltin;
        });

        var paths = RequiredPaths(settings);
        Assert.Contains(paths, p => p.EndsWith("decoder_single.onnx"));
    }

    [Fact]
    public void EveryBackend_RequiresSomething_AndNoForeignVibeVoicePackage()
    {
        foreach (var backend in Enum.GetValues<AsrBackend>())
        {
            using var tmp = new TempDir();
            var settings = SettingsIn(tmp.Path, s =>
            {
                s.AsrBackend = backend;
                s.Segmentation = SegmentationMode.SileroVad;
            });

            var paths = RequiredPaths(settings);
            Assert.True(paths.Count > 0, $"{backend} requires no files at all");

            // Match the package folder, not the word: Sortformer's own file is named
            // diar_streaming_sortformer_4spk-v2.1.onnx and is nothing to do with this backend.
            bool streaming = backend == AsrBackend.VibeVoiceStreaming;
            Assert.False(paths.Any(p => p.StartsWith("vibevoice_asr_streaming")) && !streaming,
                $"{backend} pulls the streaming package");
            Assert.False(paths.Any(p => p.EndsWith("decoder_single.onnx")) && streaming,
                $"{backend} pulls the non-streaming VibeVoice package");
        }
    }

    private sealed class TempDir : IDisposable
    {
        public string Path { get; } =
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), "vernacula-repo-tests", Guid.NewGuid().ToString("N"));
        public TempDir() => Directory.CreateDirectory(Path);
        public void Dispose() { try { Directory.Delete(Path, true); } catch { } }
    }
}
