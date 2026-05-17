using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using Chatterbox.App.Models;
using Chatterbox.App.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Chatterbox.App.ViewModels;

/// <summary>
/// Single VM for the reader MVP. State lives here:
/// pickers (voice / Chatterbox bundle / NFA bundle), text, last
/// synthesis result, playback position, current highlighted word
/// index into <see cref="Words"/>. Bindings are compiled at AXAML
/// load (csproj's AvaloniaUseCompiledBindingsByDefault=true).
/// </summary>
public sealed partial class MainViewModel : ObservableObject, IDisposable
{
    // Pickers — paths the user fills in via file/folder dialogs.
    [ObservableProperty] private string _voicePath = "";
    [ObservableProperty] private string _onnxBundleDir = "";
    [ObservableProperty] private string _nfaBundleDir = "";

    // Text input — either typed/pasted in the UI or loaded from a .md file.
    [ObservableProperty] private string _text = "";

    // Status line below the synthesize button.
    [ObservableProperty] private string _statusMessage = "Ready.";

    // True while a synthesis task is running. Disables most UI controls.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    private bool _isBusy;

    // True after a successful synthesis — enables Play and the highlight loop.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    private bool _hasSynthesizedAudio;

    // The aligned words from the last synthesis as per-item VMs so the
    // AXAML can class-trigger the highlight on IsCurrent without a
    // multi-binding converter.
    public ObservableCollection<WordItemViewModel> Words { get; } = new();

    // Index into Words[] of the word currently being spoken (or -1 when
    // not playing or before the first word). Internal — UI doesn't bind
    // to this directly, the per-word IsCurrent does the work.
    private int _currentWordIndex = -1;

    // Read-only audio position label for the UI footer.
    [ObservableProperty] private string _positionLabel = "0.00 / 0.00 s";

    private AlignmentSidecar? _lastAlignment;
    private SynthesisService? _synthService;
    private readonly PlaybackService _playback = new();
    private CancellationTokenSource? _synthCts;

    public MainViewModel()
    {
        _playback.PositionChanged += OnPlaybackPositionChanged;
        _playback.PlaybackStopped += _ => Dispatcher.UIThread.Post(() =>
        {
            if (_currentWordIndex >= 0 && _currentWordIndex < Words.Count)
                Words[_currentWordIndex].IsCurrent = false;
            _currentWordIndex = -1;
            StatusMessage = "Playback stopped.";
        });

        if (!_playback.CanPlayOnThisPlatform)
            StatusMessage = _playback.UnavailableReason!;
    }

    [RelayCommand]
    private async Task PickVoiceAsync()
    {
        var path = await PickFileAsync("Pick a voice reference WAV",
            new FilePickerFileType("Audio") { Patterns = new[] { "*.wav", "*.flac", "*.mp3" } });
        if (path is not null) VoicePath = path;
    }

    [RelayCommand]
    private async Task PickOnnxBundleAsync()
    {
        var path = await PickFolderAsync("Pick the Chatterbox ONNX bundle directory");
        if (path is not null) OnnxBundleDir = path;
    }

    [RelayCommand]
    private async Task PickNfaBundleAsync()
    {
        var path = await PickFolderAsync(
            "Pick the NFA ONNX bundle directory (optional — without it, no word highlights)");
        if (path is not null) NfaBundleDir = path;
    }

    [RelayCommand]
    private async Task PickTextFileAsync()
    {
        var path = await PickFileAsync("Pick a markdown or text file",
            new FilePickerFileType("Text") { Patterns = new[] { "*.md", "*.markdown", "*.txt" } });
        if (path is null) return;
        try { Text = await File.ReadAllTextAsync(path); }
        catch (Exception ex) { StatusMessage = $"Read failed: {ex.Message}"; }
    }

    [RelayCommand(CanExecute = nameof(CanSynthesize))]
    private async Task SynthesizeAsync()
    {
        IsBusy = true;
        StatusMessage = _synthService is null ? "Loading models (one-time)..." : "Synthesizing...";
        Words.Clear();
        _currentWordIndex = -1;
        HasSynthesizedAudio = false;
        _synthCts?.Dispose();
        _synthCts = new CancellationTokenSource();
        var token = _synthCts.Token;

        try
        {
            // Lazy-construct the synthesis service so the heavy ORT
            // session loads happen on the first synthesize click rather
            // than at startup (faster cold UI).
            _synthService ??= new SynthesisService(
                OnnxBundleDir,
                nfaBundleDir: string.IsNullOrWhiteSpace(NfaBundleDir) ? null : NfaBundleDir);

            string outWav = Path.Combine(Path.GetTempPath(),
                $"chatterbox_app_{DateTime.UtcNow:yyyyMMddHHmmss}.wav");

            var result = await Task.Run(
                () => _synthService.Synthesize(VoicePath, Text, outWav, token),
                token);

            _lastAlignment = result.Alignment;
            HasSynthesizedAudio = true;
            int idx = 0;
            foreach (var w in result.Alignment.Words)
                Words.Add(new WordItemViewModel(w, idx++));
            StatusMessage = $"Synthesized {result.Alignment.AudioDurationSeconds:F2}s, "
                + $"{result.Alignment.Words.Count} words. Ready to play.";
        }
        catch (OperationCanceledException)
        {
            StatusMessage = "Synthesis cancelled.";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Synthesis failed: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool CanSynthesize()
        => !IsBusy
        && !string.IsNullOrWhiteSpace(VoicePath)
        && !string.IsNullOrWhiteSpace(OnnxBundleDir)
        && !string.IsNullOrWhiteSpace(Text);

    [RelayCommand(CanExecute = nameof(CanPlay))]
    private void Play()
    {
        if (_lastAlignment is null) return;
        try
        {
            _playback.Play(_lastAlignment.AudioPath, _lastAlignment.AudioDurationSeconds);
            StatusMessage = "Playing.";
        }
        catch (Exception ex) { StatusMessage = $"Play failed: {ex.Message}"; }
    }

    private bool CanPlay() => HasSynthesizedAudio && !IsBusy && _playback.CanPlayOnThisPlatform;

    [RelayCommand]
    private void Stop() => _playback.Stop();

    private void OnPlaybackPositionChanged(double posSec)
        => Dispatcher.UIThread.Post(() =>
        {
            // Binary-search the words list for the word whose interval
            // contains posSec. Words are sorted by StartSeconds per the
            // sidecar contract, so a linear scan from the current index
            // forward would also work (amortized O(N) over a playback) —
            // binary search is just as cheap and handles backwards seeks
            // when scrubbing lands.
            int idx = FindWordAt(posSec);
            if (idx != _currentWordIndex)
            {
                if (_currentWordIndex >= 0 && _currentWordIndex < Words.Count)
                    Words[_currentWordIndex].IsCurrent = false;
                _currentWordIndex = idx;
                if (idx >= 0 && idx < Words.Count) Words[idx].IsCurrent = true;
            }
            double dur = _lastAlignment?.AudioDurationSeconds ?? 0;
            PositionLabel = $"{posSec:F2} / {dur:F2} s";
        });

    private int FindWordAt(double posSec)
    {
        // Standard binary search over Words[i].StartSeconds. Returns the
        // last word whose StartSeconds <= posSec, or -1 when posSec is
        // before the first word. The "current word" is whatever started
        // most recently; we don't gate on EndSeconds so brief inter-word
        // gaps still show the trailing word highlighted (no flicker).
        if (Words.Count == 0) return -1;
        int lo = 0, hi = Words.Count - 1, best = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (Words[mid].StartSeconds <= posSec)
            {
                best = mid;
                lo = mid + 1;
            }
            else { hi = mid - 1; }
        }
        return best;
    }

    private static async Task<string?> PickFileAsync(string title, FilePickerFileType filter)
    {
        var top = TopLevel();
        if (top is null) return null;
        var picks = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = new[] { filter },
        });
        return picks.Count > 0 ? picks[0].TryGetLocalPath() : null;
    }

    private static async Task<string?> PickFolderAsync(string title)
    {
        var top = TopLevel();
        if (top is null) return null;
        var picks = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
        });
        return picks.Count > 0 ? picks[0].TryGetLocalPath() : null;
    }

    private static TopLevel? TopLevel()
    {
        if (Avalonia.Application.Current?.ApplicationLifetime
            is IClassicDesktopStyleApplicationLifetime desktop
            && desktop.MainWindow is not null)
        {
            return Avalonia.Controls.TopLevel.GetTopLevel(desktop.MainWindow);
        }
        return null;
    }

    public void Dispose()
    {
        _synthCts?.Cancel();
        _synthCts?.Dispose();
        _playback.Dispose();
        _synthService?.Dispose();
    }
}
