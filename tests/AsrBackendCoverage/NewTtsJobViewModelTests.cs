using System;
using System.IO;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Vernacula.App.ViewModels;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The New TTS Job dialog holds one set of fields for every engine and shows the subset the
/// selected engine uses. These pin the part that is easy to get wrong: moving between engines
/// must not quietly rewrite a choice, because Start persists whatever the fields hold.
/// </summary>
public class NewTtsJobViewModelTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));

    public NewTtsJobViewModelTests()
    {
        // A Kokoro voices folder with a known list, so the dialog has a voice list to disturb.
        Directory.CreateDirectory(Path.Combine(_dir, "kokoro", "voices"));
        foreach (var v in new[] { "af_alloy", "af_heart", "bf_emma" })
            File.WriteAllBytes(Path.Combine(_dir, "kokoro", "voices", v + ".bin"), new byte[16]);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private NewTtsJobViewModel MakeViewModel()
    {
        var settings = new SettingsService();
        settings.Current.KokoroModelDir = Path.Combine(_dir, "kokoro");
        settings.Current.TtsBackend = "Kokoro";
        settings.Current.KokoroVoice = "af_heart";
        return new NewTtsJobViewModel(settings);
    }

    [Fact]
    public void SwitchingEngineAndBackKeepsTheChosenVoice()
    {
        var vm = MakeViewModel();
        Assert.Equal("af_heart", vm.KokoroVoice);

        // The dialog's ComboBox binds SelectedItem two-way to KokoroVoice over this collection,
        // so emptying the collection clears the selection — the half of the bug that only shows
        // up with a control attached. Stand in for it here, or the round trip passes either way.
        vm.KokoroVoices.CollectionChanged += (_, _) =>
        {
            if (vm.KokoroVoices.Count == 0) vm.KokoroVoice = "";
        };

        vm.SelectedEngine = TtsEngines.For(TtsBackendKind.OmniVoice);
        vm.SelectedEngine = TtsEngines.For(TtsBackendKind.Kokoro);

        Assert.Equal("af_heart", vm.KokoroVoice);
        Assert.Equal("af_heart", vm.CurrentSettings().Voice);
    }

    [Fact]
    public void AnEngineWithoutAVoiceListLeavesTheListAlone()
    {
        var vm = MakeViewModel();
        var before = vm.KokoroVoices.ToArray();

        vm.SelectedEngine = TtsEngines.For(TtsBackendKind.Chatterbox);

        Assert.Equal(before, vm.KokoroVoices);
        Assert.False(vm.ShowVoiceList);
        Assert.True(vm.ShowReferenceClip);
    }

    [Fact]
    public void CurrentSettingsCarriesOnlyWhatTheEngineUses()
    {
        var vm = MakeViewModel();
        vm.KokoroSpeed = 1.4f;
        vm.OmniVoiceNumStep = 16;

        var kokoro = vm.CurrentSettings();
        Assert.Equal("Kokoro", kokoro.Backend);
        Assert.Equal(1.4f, kokoro.Speed);
        Assert.Equal("", kokoro.Language);          // Kokoro has no language picker
        Assert.Equal(32, kokoro.NumStep);           // nor diffusion steps: the record's default

        vm.SelectedEngine = TtsEngines.For(TtsBackendKind.Chatterbox);
        vm.ChatterboxVoicePath = "/clips/ref.wav";
        var chatterbox = vm.CurrentSettings();
        Assert.Equal("/clips/ref.wav", chatterbox.Voice);
        Assert.Equal(1.0f, chatterbox.Speed);
    }
}
