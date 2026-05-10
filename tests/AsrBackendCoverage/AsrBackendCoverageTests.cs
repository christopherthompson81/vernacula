using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.ViewModels;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// Coverage test for ASR backend dispatch fan-out (issue #37).
///
/// <para>
/// Iterates every value of <see cref="AsrBackend"/> and asserts that each
/// dispatch site has a non-default response. A new backend that lands in
/// the enum but is missed at any of the sites below will fail loudly here
/// instead of silently falling through to a Parakeet code path at user-
/// visible time. The Granite Speech integration (issue #33 / PR #35) was
/// the precipitating example: <see cref="VocabService"/> was missed despite
/// a code review and per-PR touch-list, and the bug only surfaced when an
/// editor session rendered raw GPT-2 ByteLevel BPE chars instead of
/// decoded text.
/// </para>
///
/// <para>
/// See <c>docs/dev/asr_backend_dispatch.md</c> for the full list of
/// dispatch sites and which ones are covered here vs documented as
/// manual-review.
/// </para>
/// </summary>
[CollectionDefinition(nameof(AsrBackendCoverageTests), DisableParallelization = true)]
public class AsrBackendCoverageCollection { }

[Collection(nameof(AsrBackendCoverageTests))]
public class AsrBackendCoverageTests
{
    public static IEnumerable<object[]> AllBackends =>
        Enum.GetValues<AsrBackend>().Select(b => new object[] { b });

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void ModelName_DoesNotThrow_AndIsNonEmpty(AsrBackend backend)
    {
        string name = AsrLanguageSupport.ModelName(backend);
        Assert.False(string.IsNullOrWhiteSpace(name),
            $"AsrLanguageSupport.ModelName({backend}) returned empty.");
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void DisplayName_DoesNotThrow_AndIsNonEmpty(AsrBackend backend)
    {
        // DisplayName throws on an unhandled backend (see commit history /
        // docs/dev/asr_backend_dispatch.md). The act of returning at all is
        // the assertion; we additionally require non-empty to guard against
        // a future "" fallback. We deliberately do NOT compare against
        // backend.ToString() — for Parakeet and IndicConformer, the
        // marketing name happens to match the enum name.
        string display = AsrLanguageSupport.DisplayName(backend);
        Assert.False(string.IsNullOrWhiteSpace(display));
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void LanguageSet_IsNonEmpty(AsrBackend backend)
    {
        var langs = AsrLanguageSupport.Get(backend);
        Assert.NotEmpty(langs);
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void BackendOf_ModelName_RoundTrip(AsrBackend backend)
    {
        string name = AsrLanguageSupport.ModelName(backend);
        AsrBackend? roundTripped = AsrLanguageSupport.BackendOf(name);
        Assert.Equal(backend, roundTripped);
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void LanguageOptions_AreNonEmpty(AsrBackend backend)
    {
        var opts = AsrLanguageSupport.LanguageOptions(backend);
        Assert.NotEmpty(opts);
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void VocabService_KindOfBackend_IsDefined(AsrBackend backend)
    {
        // KindOfBackend throws on an unmapped backend, so reaching this assert
        // means the dispatch is wired. The IsDefined check guards against the
        // VocabKind enum having a stale member that no AsrBackend maps to.
        var kind = VocabService.KindOfBackend(backend);
        Assert.True(Enum.IsDefined(kind),
            $"VocabService.KindOfBackend({backend}) returned undefined VocabKind {kind}.");
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void VocabService_Constructor_DoesNotWarnOnRecognizedModel(AsrBackend backend)
    {
        // VocabService's per-backend asrModel branches must recognize every
        // backend's ModelName(). The constructor logs a stderr warning on the
        // unrecognized-non-null fallback (see commit a68e416). Capture stderr
        // and assert no warning fired. Vocab-loaders tolerate missing files
        // so an empty temp dir without any installed models is sufficient;
        // the Guid-named subdir guarantees no stray /tmp file (e.g. an
        // unrelated tokenizer.json from another tool) accidentally satisfies
        // a File.Exists probe and changes loader behaviour.
        //
        // Console.SetError is process-global, so this whole class lives in
        // [Collection(nameof(AsrBackendCoverageTests))] with parallelization
        // disabled to keep the swap from racing with a parallel test.
        string modelName = AsrLanguageSupport.ModelName(backend);
        string tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));

        var origErr = Console.Error;
        using var stderr = new StringWriter();
        Console.SetError(stderr);
        try
        {
            _ = new VocabService(tempDir, modelName);
        }
        finally
        {
            Console.SetError(origErr);
        }

        string err = stderr.ToString();
        Assert.False(
            err.Contains("WARNING", StringComparison.OrdinalIgnoreCase),
            $"VocabService constructor logged a fallback warning for backend {backend} " +
            $"(model name '{modelName}'). Add a per-backend branch in VocabService.\n" +
            $"Captured stderr: {err}");
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void SettingsService_HasModelsDirGetter(AsrBackend backend)
    {
        // Convention: SettingsService.Get<EnumName>ModelsDir() per backend.
        string methodName = $"Get{backend}ModelsDir";
        var method = typeof(SettingsService).GetMethod(
            methodName,
            BindingFlags.Public | BindingFlags.Instance,
            types: Type.EmptyTypes);
        Assert.NotNull(method);
        Assert.Equal(typeof(string), method!.ReturnType);
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void SettingsViewModel_HasIsAsrProperty(AsrBackend backend)
    {
        // Convention: SettingsViewModel.IsAsr<EnumName> per backend, used by
        // the SettingsWindow.axaml RadioButton block to drive selection.
        string propName = $"IsAsr{backend}";
        var prop = typeof(SettingsViewModel).GetProperty(
            propName,
            BindingFlags.Public | BindingFlags.Instance);
        Assert.NotNull(prop);
        Assert.Equal(typeof(bool), prop!.PropertyType);
    }
}
