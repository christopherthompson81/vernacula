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
using Chatterbox.Base;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Chatterbox.App.ViewModels;

/// <summary>
/// Reader VM. Holds picker state, persists it across runs via
/// <see cref="SettingsService"/>, drives streaming synthesis +
/// playback (chunks start playing as soon as the first one is ready),
/// and toggles between the flat word-by-word view and a Markdown
/// rendering of the source.
/// </summary>
public sealed partial class MainViewModel : ObservableObject, IDisposable
{
    // Pickers. NotifyCanExecuteChangedFor is what re-queries the
    // SynthesizeCommand's CanExecute when the picker fills.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _voicePath = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _onnxBundleDir = "";

    // NFA bundle is OPTIONAL — synthesis runs without it (just no word
    // highlight). Doesn't gate Synthesize, so no notify here.
    [ObservableProperty] private string _nfaBundleDir = "";

    // Text input — either typed/pasted in the UI or loaded from a .md file.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _text = "";

    // Markdown vs plain word-grid rendering of the source pane. Persisted
    // across runs.
    [ObservableProperty] private bool _renderMarkdown;

    // Status line below the synthesize button.
    [ObservableProperty] private string _statusMessage = "Ready.";

    // Active while synthesis is running. Disables most controls.
    // Note: gates SynthesizeCommand (don't allow re-start mid-run),
    // PlayCommand (no replay while still generating), and the new
    // CancelSynthesisCommand (only enabled when there's something to
    // cancel). Stop is independent — it operates on playback only.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(CancelSynthesisCommand))]
    private bool _isBusy;

    // True after a synthesis finishes successfully. Enables replay.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    private bool _hasSynthesizedAudio;

    // Mirrors PlaybackService.IsPlaying so the AXAML CanExecute can
    // re-query. PlaybackService raises IsPlayingChanged; the handler
    // updates this field. Gates Play (can't start when already playing)
    // and Stop (nothing to stop when not playing).
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    private bool _isPlayingBack;

    // Chunk progress drives the determinate progress bar. ChunksDone is
    // bound to ProgressBar.Value; TotalChunks to .Maximum. Both zero
    // when idle (the bar's IsVisible binding to IsBusy hides it then).
    [ObservableProperty] private int _chunksDone;
    [ObservableProperty] private int _totalChunks;

    // Aligned words from the running/last synthesis. Streamed in as
    // chunks complete.
    public ObservableCollection<WordItemViewModel> Words { get; } = new();

    // -1 = nothing currently highlighted.
    private int _currentWordIndex = -1;

    [ObservableProperty] private string _positionLabel = "0.00 / 0.00 s";

    private AlignmentSidecar? _lastAlignment;
    private string? _lastAudioPath;
    private SynthesisService? _synthService;
    private readonly PlaybackService _playback = new();
    private readonly SettingsService _settings = new();
    private CancellationTokenSource? _synthCts;

    // Gates the autosave during the initial settings load so we don't
    // write N times as each property hydrates.
    private bool _loadingSettings;

    public MainViewModel()
    {
        _playback.PositionChanged += OnPlaybackPositionChanged;
        _playback.PlaybackStopped += _ =>
        {
            if (_currentWordIndex >= 0 && _currentWordIndex < Words.Count)
                Words[_currentWordIndex].IsCurrent = false;
            _currentWordIndex = -1;
            // Don't overwrite synthesis-in-progress status with "stopped".
            if (!IsBusy) StatusMessage = "Playback stopped.";
        };
        _playback.IsPlayingChanged += playing => Dispatcher.UIThread.Post(() => IsPlayingBack = playing);
        // Refresh the position label whenever the total grows — without
        // this it would only update on tick-timer ticks, so after Stop
        // (which kills the timer) any chunks still landing would update
        // the total internally but the label would stay frozen.
        _playback.TotalChanged += total => Dispatcher.UIThread.Post(() =>
            PositionLabel = $"{_playback.PositionSeconds:F2} / {total:F2} s");

        LoadSettings();

        if (!_playback.CanPlayOnThisPlatform)
            StatusMessage = _playback.UnavailableReason!;
    }

    // ── Settings persistence ─────────────────────────────────────────

    private void LoadSettings()
    {
        _loadingSettings = true;
        try
        {
            _settings.Load();
            VoicePath = _settings.Current.VoicePath;
            OnnxBundleDir = _settings.Current.OnnxBundleDir;
            NfaBundleDir = _settings.Current.NfaBundleDir;
            RenderMarkdown = _settings.Current.RenderMarkdown;
        }
        finally { _loadingSettings = false; }
    }

    private void PersistSettings()
    {
        if (_loadingSettings) return;
        _settings.Current.VoicePath = VoicePath;
        _settings.Current.OnnxBundleDir = OnnxBundleDir;
        _settings.Current.NfaBundleDir = NfaBundleDir;
        _settings.Current.RenderMarkdown = RenderMarkdown;
        _settings.Save();
    }

    // Generated [ObservableProperty] partials let us hook each setter
    // for the autosave + (for bundle paths) cached-service invalidation.
    partial void OnVoicePathChanged(string value) => PersistSettings();
    partial void OnOnnxBundleDirChanged(string value)
    {
        InvalidateSynthService();
        PersistSettings();
    }
    partial void OnNfaBundleDirChanged(string value)
    {
        InvalidateSynthService();
        PersistSettings();
    }
    partial void OnRenderMarkdownChanged(bool value) => PersistSettings();

    private void InvalidateSynthService()
    {
        var stale = _synthService;
        _synthService = null;
        stale?.Dispose();
    }

    // ── Pickers ──────────────────────────────────────────────────────

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

    // ── Synthesis (streaming) ────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanSynthesize))]
    private async Task SynthesizeAsync()
    {
        IsBusy = true;
        StatusMessage = _synthService is null ? "Loading models (one-time)..." : "Synthesizing...";
        // Stop any in-progress playback from a prior run; clear the
        // highlight + word list so the new synthesis streams in fresh.
        _playback.Stop();
        Words.Clear();
        _currentWordIndex = -1;
        HasSynthesizedAudio = false;
        _lastAlignment = null;
        _lastAudioPath = null;
        ChunksDone = 0;
        TotalChunks = 0;
        // Accumulators for the partial-on-cancel path. The alignment
        // chain emits chunks in input order so a List append is correct;
        // we lock just for the memory barrier between the chain thread
        // and the catch handler.
        var receivedAudio = new List<float[]>();
        var receivedWords = new List<AlignedWord>();
        var receivedLock = new object();
        _synthCts?.Dispose();
        _synthCts = new CancellationTokenSource();
        var token = _synthCts.Token;

        try
        {
            _synthService ??= new SynthesisService(
                OnnxBundleDir,
                nfaBundleDir: string.IsNullOrWhiteSpace(NfaBundleDir) ? null : NfaBundleDir);

            string outWav = Path.Combine(Path.GetTempPath(),
                $"chatterbox_app_{DateTime.UtcNow:yyyyMMddHHmmss}.wav");
            bool streamingStarted = false;

            var result = await _synthService.SynthesizeStreamingAsync(
                VoicePath, Text, outWav,
                onChunkProduced: ev =>
                {
                    // Audio appending runs HERE — on the alignment-chain
                    // thread (background), NOT on the dispatcher. On the
                    // ffplay backend, AppendSamples blocks under pipe
                    // backpressure (ffplay only drains as fast as it
                    // plays). If we did this on the UI thread, the
                    // progress bar would freeze, queued Word adds would
                    // stack up, and the user would see chunks arrive in
                    // bursts instead of streaming live.
                    try
                    {
                        if (!streamingStarted)
                        {
                            _playback.StartStreaming(ChatterboxConstants.S3GenSr, channels: 1);
                            streamingStarted = true;
                        }
                        _playback.AppendSamples(ev.Audio24k);
                    }
                    catch (Exception ex)
                    {
                        Dispatcher.UIThread.Post(() =>
                            StatusMessage = $"Playback failed: {ex.Message}");
                    }

                    // Accumulate for the partial-WAV-on-cancel path.
                    lock (receivedLock)
                    {
                        receivedAudio.Add(ev.Audio24k);
                        if (ev.Words.Count > 0) receivedWords.AddRange(ev.Words);
                    }

                    // Word adds touch the ObservableCollection → must go
                    // through the dispatcher. Capture ev by local so the
                    // closure doesn't race a subsequent chunk's event.
                    var wordsForChunk = ev.Words;
                    int chunksDone = ev.ChunkIndex + 1;
                    int totalChunks = ev.TotalChunks;
                    Dispatcher.UIThread.Post(() =>
                    {
                        foreach (var w in wordsForChunk)
                            Words.Add(new WordItemViewModel(w, Words.Count, SeekToWord));
                        ChunksDone = chunksDone;
                        TotalChunks = totalChunks;
                    });
                },
                onProgress: p => Dispatcher.UIThread.Post(() =>
                {
                    StatusMessage = p.ChunkIndex is int idx && p.TotalChunks is int total
                        ? $"{p.Phase} ({idx}/{total})"
                        : p.Phase;
                    // Pick up the total at the "chunked into N paragraphs"
                    // phase so the progress bar can render proportionally
                    // before any chunk has landed.
                    if (p.TotalChunks is int t && t > TotalChunks)
                        TotalChunks = t;
                }),
                cancellationToken: token);

            _lastAlignment = result.Alignment;
            _lastAudioPath = result.AudioPath;
            HasSynthesizedAudio = true;
            // Tell the playback service no more samples are coming so it
            // can fire PlaybackStopped naturally when the buffer drains.
            if (streamingStarted) _playback.EndOfStream();
            StatusMessage = $"Done. {result.Alignment.AudioDurationSeconds:F2}s audio, "
                + $"{result.Alignment.Words.Count} words.";
        }
        catch (OperationCanceledException)
        {
            // Persist whatever chunks landed so the user can still play
            // back what was synthesized before they hit Cancel. Without
            // this, Cancel would discard a long-running partial result.
            List<float[]> audioSnapshot;
            List<AlignedWord> wordsSnapshot;
            lock (receivedLock)
            {
                audioSnapshot = new List<float[]>(receivedAudio);
                wordsSnapshot = new List<AlignedWord>(receivedWords);
            }
            string outWavCancelled = Path.Combine(Path.GetTempPath(),
                $"chatterbox_app_{DateTime.UtcNow:yyyyMMddHHmmss}_partial.wav");
            if (audioSnapshot.Count > 0)
            {
                try
                {
                    SynthesisService.WriteWavFromChunks(outWavCancelled, audioSnapshot,
                        ChatterboxConstants.S3GenSr);
                    int totalSamples = 0;
                    foreach (var c in audioSnapshot) totalSamples += c.Length;
                    double duration = totalSamples / (double)ChatterboxConstants.S3GenSr;
                    _lastAudioPath = outWavCancelled;
                    _lastAlignment = new AlignmentSidecar
                    {
                        AudioPath = outWavCancelled,
                        SampleRate = ChatterboxConstants.S3GenSr,
                        AudioDurationSeconds = duration,
                        Aligner = string.IsNullOrWhiteSpace(NfaBundleDir) ? "none (partial)" : "nemo_nfa (partial)",
                        Words = wordsSnapshot,
                    };
                    HasSynthesizedAudio = true;
                    StatusMessage = $"Cancelled — saved {audioSnapshot.Count} chunk(s), {duration:F2}s playable.";
                }
                catch (Exception saveEx)
                {
                    StatusMessage = $"Cancelled; partial save failed: {saveEx.Message}";
                }
            }
            else
            {
                StatusMessage = "Synthesis cancelled.";
            }
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

    /// <summary>Cancels in-flight synthesis. Cancellation is checked
    /// at chunk boundaries — the LM rollout for the chunk that's
    /// currently being generated can't be interrupted, so a click here
    /// takes effect after the current chunk finishes (a few seconds at
    /// worst). Any chunks that completed before cancel are written to a
    /// partial WAV so Play still works.</summary>
    [RelayCommand(CanExecute = nameof(CanCancelSynthesis))]
    private void CancelSynthesis()
    {
        _synthCts?.Cancel();
        StatusMessage = "Cancelling — will stop after the current chunk.";
    }

    private bool CanCancelSynthesis() => IsBusy;

    // ── Playback ─────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanPlay))]
    private void Play()
    {
        // Replay path: synthesis is done, audio is on disk. Use
        // SeekIntoFile(0) to play the saved WAV from the beginning.
        if (_lastAudioPath is null || _lastAlignment is null) return;
        try
        {
            _playback.SeekIntoFile(_lastAudioPath, _lastAlignment.AudioDurationSeconds, 0);
            StatusMessage = "Playing.";
        }
        catch (Exception ex) { StatusMessage = $"Play failed: {ex.Message}"; }
    }

    // Play is enabled when we have audio AND nothing is currently
    // playing/synthesizing — i.e. not while a previous Synthesize is
    // still streaming and not while the existing playback is active.
    private bool CanPlay() => HasSynthesizedAudio && !IsBusy && !IsPlayingBack
        && _playback.CanPlayOnThisPlatform;

    /// <summary>Stops audio playback. Immediate — does NOT cancel any
    /// in-flight synthesis (use the Cancel button for that). The two
    /// were combined in an earlier iteration; the user fed back that
    /// the controls felt muddled, so they're separate now.</summary>
    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop() => _playback.Stop();

    // Stop is enabled iff something is actually playing.
    private bool CanStop() => IsPlayingBack;

    // Click-to-seek wiring. Each WordItemViewModel calls this back via a
    // delegate set at construction (avoids per-word command boilerplate +
    // keeps the VM list as a pure data collection).
    private void SeekToWord(WordItemViewModel word)
    {
        // After synthesis completes we have an on-disk WAV and can do a
        // real audio seek. Mid-synthesis we can only re-anchor the
        // highlight clock (audio is being buffered as we go and seeking
        // into a streaming buffer would be a much bigger surgery).
        if (HasSynthesizedAudio && _lastAudioPath is not null && _lastAlignment is not null)
        {
            try
            {
                _playback.SeekIntoFile(_lastAudioPath, _lastAlignment.AudioDurationSeconds,
                    word.StartSeconds);
                StatusMessage = $"Seek → {word.StartSeconds:F2}s.";
            }
            catch (Exception ex) { StatusMessage = $"Seek failed: {ex.Message}"; }
        }
        else
        {
            _playback.SeekTo(word.StartSeconds);
        }
    }

    private void OnPlaybackPositionChanged(double posSec)
    {
        int idx = FindWordAt(posSec);
        if (idx != _currentWordIndex)
        {
            if (_currentWordIndex >= 0 && _currentWordIndex < Words.Count)
                Words[_currentWordIndex].IsCurrent = false;
            _currentWordIndex = idx;
            if (idx >= 0 && idx < Words.Count) Words[idx].IsCurrent = true;
        }
        // Use the live playback total — it grows as chunks append during
        // streaming, and matches _lastAlignment.AudioDurationSeconds once
        // synthesis finishes. (Reading _lastAlignment alone would show
        // "/ 0.00 s" for the entire stream.)
        PositionLabel = $"{posSec:F2} / {_playback.TotalEstimatedSeconds:F2} s";
    }

    private int FindWordAt(double posSec)
    {
        if (Words.Count == 0) return -1;
        int lo = 0, hi = Words.Count - 1, best = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (Words[mid].StartSeconds <= posSec) { best = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return best;
    }

    // ── Picker plumbing ──────────────────────────────────────────────

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
