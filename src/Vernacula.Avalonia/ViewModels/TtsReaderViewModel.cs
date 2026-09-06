using System.Collections.ObjectModel;
using System.Text.Json;
using Avalonia.Media;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.ViewModels;

/// <summary>
/// The reader panel for one text-to-speech job: the document as a structured word-by-word
/// (karaoke) view or rendered markdown, with playback that highlights the spoken word and
/// click-to-seek. Opens on a finished job (audio + alignment from the sidecar) or attaches to
/// a job still rendering (chunks stream in from the queue's <see cref="TtsJobUiState"/>, and
/// Play streams them as they arrive). No engine settings live here — those were fixed when the
/// job was created.
/// <para>
/// The display/annotation/highlight machinery is the standalone reader's
/// (Vernacula.Tts.Avalonia's MainViewModel), minus its pickers and synthesis, which the job
/// queue now owns.
/// </para>
/// </summary>
internal sealed partial class TtsReaderViewModel : ObservableObject, IDisposable
{
    private readonly JobQueueService _queue;
    private readonly SettingsService _settings;
    private readonly PlaybackService _playback = new();

    // ── Header / status ──────────────────────────────────────────────────────

    [ObservableProperty] private string _jobTitle      = "";
    /// <summary>"Kokoro · af_heart · 1.00×" — the choices the job was rendered with.</summary>
    [ObservableProperty] private string _jobInfo       = "";
    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private string _progressText  = "";

    /// <summary>True while the watched job is queued or rendering.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(CancelJobNowCommand), nameof(ExportCommand))]
    private bool _isRunning;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    private string? _errorMessage;
    public bool HasError => ErrorMessage != null;

    // ── View preferences (persisted, not per job) ────────────────────────────

    /// <summary>Show the document's source text verbatim instead of the word-by-word view.</summary>
    [ObservableProperty] private bool _showRawMarkdown;
    [ObservableProperty] private bool _showIpaAnnotation;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasIpaAnnotationNotice))]
    private string _ipaAnnotationNotice = "";
    public bool HasIpaAnnotationNotice => !string.IsNullOrEmpty(IpaAnnotationNotice);

    // The font and direction the raw view is set in — the UI font unless the document is in a
    // script that needs a specific face; right-to-left when the text or language says so.
    [ObservableProperty] private FontFamily    _textFontFamily    = ScriptFonts.Default;
    [ObservableProperty] private FlowDirection _textFlowDirection = FlowDirection.LeftToRight;

    /// <summary>The document source, verbatim, for the raw view.</summary>
    [ObservableProperty] private string _sourceText = "";

    // ── Playback ─────────────────────────────────────────────────────────────

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayPauseCommand), nameof(ExportCommand))]
    private bool _hasAudio;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayPauseCommand), nameof(StopCommand))]
    [NotifyPropertyChangedFor(nameof(PlayPauseLabel))]
    private bool _isPlayingBack;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayPauseCommand), nameof(StopCommand))]
    [NotifyPropertyChangedFor(nameof(PlayPauseLabel))]
    private bool _isPausedBack;

    public string PlayPauseLabel => IsPausedBack ? "▶ Resume"
        : IsPlayingBack ? "⏸ Pause"
        : "▶ Play";

    [ObservableProperty] private string _positionLabel = "0.00 / 0.00 s";

    // ── Words ────────────────────────────────────────────────────────────────

    /// <summary>Flat, in-order word list — the same objects as in DisplayBlocks. FindWordAt indexes it by StartSeconds.</summary>
    public ObservableCollection<WordItemViewModel> Words { get; } = new();
    /// <summary>The structured (markdown-styled) karaoke view: blocks of words.</summary>
    public ObservableCollection<BlockItemViewModel> DisplayBlocks { get; } = new();

    // ── Hooks (wired by MainViewModel) ───────────────────────────────────────

    public Action?      NavigateBack { get; set; }
    public Action<int>? CancelJob    { get; set; }

    // ── Private state ────────────────────────────────────────────────────────

    private JobRecord?      _job;
    private int?            _watchedJobId;
    private TtsJobUiState?  _state;
    private string          _text = "";
    private string          _lang = "en";
    private string?         _audioPath;
    private double          _audioDuration;
    private AlignmentSidecar? _sidecar;
    private int             _sampleRate = ChatterboxConstants.S3GenSr;

    // Chunks received while the job renders: audio for streaming/replay, words for timing.
    private readonly List<float[]>     _receivedAudio = new();
    private readonly List<AlignedWord> _receivedWords = new();
    private readonly object            _receivedLock  = new();
    private bool                       _streamingPlayback;
    private int                        _streamWordCursor;
    private int                        _currentWordIndex = -1;
    private readonly List<BlockItemViewModel> _wordBlock = new();   // word index → its block
    private BlockItemViewModel?        _currentBlock;
    private CancellationTokenSource?   _annotationCts;

    public TtsReaderViewModel(JobQueueService queue, SettingsService settings)
    {
        _queue    = queue;
        _settings = settings;

        _playback.PositionChanged += OnPlaybackPositionChanged;
        _playback.PlaybackStopped += _ =>
        {
            ClearCurrentHighlight();
            _streamingPlayback = false;
        };
        _playback.IsPlayingChanged += playing => Dispatcher.UIThread.Post(() => IsPlayingBack = playing);
        _playback.IsPausedChanged  += paused  => Dispatcher.UIThread.Post(() => IsPausedBack = paused);
        _playback.TotalChanged     += total   => Dispatcher.UIThread.Post(() =>
            PositionLabel = $"{_playback.PositionSeconds:F2} / {total:F2} s");

        _showRawMarkdown   = settings.Current.TtsShowRawMarkdown;
        _showIpaAnnotation = settings.Current.TtsShowIpaAnnotation;
    }

    // ── Opening a job ────────────────────────────────────────────────────────

    /// <summary>Shows <paramref name="job"/>: a finished one from its sidecar, a queued/running one live.</summary>
    public void Open(JobRecord job)
    {
        Detach();
        _job          = job;
        JobTitle      = job.JobTitle;
        JobInfo       = DescribeJob(job);
        ErrorMessage  = null;
        ProgressText  = "";
        StatusMessage = "";
        HasAudio      = false;
        _audioPath    = null;
        _audioDuration = 0;
        _sidecar      = null;
        _lang         = AnnotationLangFor(job);
        _sampleRate   = TtsJobRunner.SampleRateFor(new TtsJobSettings(job.TtsBackend, job.TtsLanguage, job.TtsVoice));
        lock (_receivedLock) { _receivedAudio.Clear(); _receivedWords.Clear(); }
        _streamWordCursor = 0;

        if (!_playback.CanPlayOnThisPlatform)
            StatusMessage = _playback.UnavailableReason!;

        switch (job.Status)
        {
            case JobStatus.Complete:
                LoadCompleted(job);
                break;
            case JobStatus.Running:
            case JobStatus.Queued:
                Attach(job);
                break;
            default:
                SetText(ReadDocument(job));
                if (job.Status == JobStatus.Failed) { ErrorMessage = job.ErrorMessage; StatusMessage = Loc.Instance["tts_status_failed"]; }
                else StatusMessage = Loc.Instance["status_cancelled"];
                break;
        }
    }

    private static string DescribeJob(JobRecord job)
    {
        var parts = new List<string> { job.TtsBackend };
        if (!string.IsNullOrWhiteSpace(job.TtsLanguage))
            parts.Add(LanguageCatalog.ByCode(job.TtsLanguage)?.Name ?? job.TtsLanguage);
        if (!string.IsNullOrWhiteSpace(job.TtsVoice))
            parts.Add(TtsJobRunner.ParseBackend(job.TtsBackend) == TtsBackendKind.Chatterbox
                ? Path.GetFileName(job.TtsVoice) : job.TtsVoice);
        if (TtsJobRunner.ParseBackend(job.TtsBackend) == TtsBackendKind.Kokoro)
            parts.Add($"{job.TtsSpeed:F2}×");
        if (TtsJobRunner.ParseBackend(job.TtsBackend) == TtsBackendKind.OmniVoice)
            parts.Add($"{job.TtsNumStep} steps");
        return string.Join("  ·  ", parts);
    }

    /// <summary>The language the IPA annotation is read in: OmniVoice's picked language, Kokoro's
    /// en/en-GB (its British voices are the bf_/bm_ ones), and en for Chatterbox.</summary>
    private static string AnnotationLangFor(JobRecord job) => TtsJobRunner.ParseBackend(job.TtsBackend) switch
    {
        TtsBackendKind.OmniVoice => string.IsNullOrWhiteSpace(job.TtsLanguage) ? "en" : job.TtsLanguage.Trim(),
        TtsBackendKind.Kokoro    => job.TtsVoice.StartsWith("bf_", StringComparison.Ordinal)
                                    || job.TtsVoice.StartsWith("bm_", StringComparison.Ordinal) ? "en-GB" : "en",
        _                        => "en",
    };

    private static string ReadDocument(JobRecord job)
    {
        try { return File.Exists(job.AudioFilePath) ? File.ReadAllText(job.AudioFilePath) : ""; }
        catch (Exception ex) { Console.Error.WriteLine($"[TtsReader] read failed: {ex.Message}"); return ""; }
    }

    private void LoadCompleted(JobRecord job)
    {
        AlignmentSidecar? sidecar = null;
        try
        {
            if (File.Exists(job.ResultsFile))
                sidecar = JsonSerializer.Deserialize<AlignmentSidecar>(File.ReadAllText(job.ResultsFile));
        }
        catch (Exception ex) { Console.Error.WriteLine($"[TtsReader] sidecar unreadable: {ex.Message}"); }

        SetText(sidecar?.SourceText ?? ReadDocument(job));

        if (sidecar is null)
        {
            ErrorMessage = $"Alignment sidecar not found or unreadable: {job.ResultsFile}";
            return;
        }
        AttachTimings(sidecar);
        _sidecar       = sidecar;
        _audioPath     = job.OutputAudioPath;
        _audioDuration = sidecar.AudioDurationSeconds;
        HasAudio       = File.Exists(_audioPath);
        PositionLabel  = $"0.00 / {_audioDuration:F2} s";
        StatusMessage  = Loc.Instance.T("tts_status_complete", new()
        {
            ["duration"] = $"{_audioDuration:F1}s",
            ["words"]    = sidecar.Words.Count.ToString(),
        });
        if (!HasAudio) ErrorMessage = $"Rendered audio not found: {_audioPath}";
    }

    // ── Watching a running job ───────────────────────────────────────────────

    private void Attach(JobRecord job)
    {
        SetText(ReadDocument(job));
        _watchedJobId = job.JobId;
        IsRunning     = true;
        StatusMessage = Loc.Instance["tts_status_running"];

        // Subscribe to lifecycle first so a transition cannot slip between snapshot and live.
        _queue.JobStatusChanged      += OnWatchedJobStatusChanged;
        _queue.JobTtsProgressUpdated += OnWatchedJobProgress;

        _state = _queue.GetTtsJobUiState(job.JobId);
        if (_state is null)
        {
            ProgressText = Loc.Instance["tts_progress_queued"];
            return;
        }

        var snapshot = _state.Subscribe(OnStateAction);
        _sampleRate = snapshot.SampleRate;
        foreach (var chunk in snapshot.Chunks)
            AcceptChunk(chunk);
        if (snapshot.LastProgress is { } p) ProgressText = FormatProgress(p);
    }

    /// <summary>
    /// Runs on the WORKER thread. Audio is appended to a streaming playback here, not on the
    /// dispatcher: on the ffplay backend AppendSamples blocks under pipe backpressure, and
    /// doing that on the UI thread would freeze the panel between chunks.
    /// </summary>
    private void OnStateAction(TtsJobUiAction action)
    {
        if (action is TtsChunkProducedAction { Chunk: var chunk })
        {
            bool streaming;
            lock (_receivedLock) streaming = _streamingPlayback;
            if (streaming)
            {
                try { _playback.AppendSamples(chunk.Audio24k); }
                catch (Exception ex) { Dispatcher.UIThread.Post(() => StatusMessage = $"Playback failed: {ex.Message}"); }
            }
            Dispatcher.UIThread.Post(() => AcceptChunk(chunk));
        }
        else if (action is TtsProgressAction { Progress: var p })
        {
            Dispatcher.UIThread.Post(() => ProgressText = FormatProgress(p));
        }
    }

    private static string FormatProgress(ProgressEvent p) =>
        p.ChunkIndex is int idx && p.TotalChunks is int total ? $"{p.Phase} ({idx}/{total})" : p.Phase;

    private void OnWatchedJobProgress(int jobId, ProgressEvent p)
    {
        // The UI-state subscription already carries progress for a running job; this event
        // is what a queued job (no state yet) reports through once it starts.
        if (jobId != _watchedJobId) return;
        if (_state is null)
            Dispatcher.UIThread.Post(() =>
            {
                ProgressText = FormatProgress(p);
                if (_state is null && _queue.GetTtsJobUiState(jobId) is { } state)
                {
                    _state = state;
                    var snapshot = state.Subscribe(OnStateAction);
                    _sampleRate = snapshot.SampleRate;
                    foreach (var chunk in snapshot.Chunks) AcceptChunk(chunk);
                }
            });
    }

    /// <summary>UI thread: accumulate a chunk (replay + timing) and light up its words.</summary>
    private void AcceptChunk(ChunkProducedEvent chunk)
    {
        bool known;
        lock (_receivedLock)
        {
            known = chunk.ChunkIndex < _receivedAudio.Count;
            if (!known)
            {
                _receivedAudio.Add(chunk.Audio24k);
                if (chunk.Words.Count > 0) _receivedWords.AddRange(chunk.Words);
            }
        }
        if (known) return;

        foreach (var w in chunk.Words)
        {
            if (_streamWordCursor < Words.Count)
            {
                var vm = Words[_streamWordCursor];
                vm.StartSeconds = w.StartSeconds;
                vm.EndSeconds   = w.EndSeconds;
            }
            _streamWordCursor++;
        }
        if (chunk.ChunkIndex < DisplayBlocks.Count)
            DisplayBlocks[chunk.ChunkIndex].SetTiming(chunk.AudioStartSeconds,
                chunk.AudioStartSeconds + chunk.Audio24k.Length / (double)_sampleRate);
        HasAudio = true;
    }

    private void OnWatchedJobStatusChanged(int jobId, JobStatus status, string? error, int? runTimeSeconds)
    {
        if (jobId != _watchedJobId) return;
        if (status is not (JobStatus.Complete or JobStatus.Failed or JobStatus.Cancelled)) return;

        Dispatcher.UIThread.Post(() =>
        {
            if (jobId != _watchedJobId || _job is null) return;
            _state?.Unsubscribe();
            _state = null;
            IsRunning    = false;
            ProgressText = "";
            _job.Status  = status;

            bool wasStreaming;
            lock (_receivedLock) wasStreaming = _streamingPlayback;
            if (wasStreaming) _playback.EndOfStream();

            switch (status)
            {
                case JobStatus.Complete:
                    // The sidecar has the final timings and the WAV is on disk: reload from it
                    // so Play/seek use the file (streaming replay is only for a job in flight).
                    LoadCompleted(_job);
                    break;
                case JobStatus.Failed:
                    ErrorMessage  = error;
                    StatusMessage = Loc.Instance["tts_status_failed"];
                    break;
                case JobStatus.Cancelled:
                    StatusMessage = Loc.Instance["tts_status_cancelled"];
                    break;
            }
        });
    }

    /// <summary>Leaves the watched job (if any) and stops playback. Safe to call repeatedly.</summary>
    public void Detach()
    {
        _queue.JobStatusChanged      -= OnWatchedJobStatusChanged;
        _queue.JobTtsProgressUpdated -= OnWatchedJobProgress;
        _state?.Unsubscribe();
        _state        = null;
        _watchedJobId = null;
        IsRunning     = false;
        _playback.Stop();
        lock (_receivedLock) _streamingPlayback = false;
        _annotationCts?.Cancel();
        _annotationCts?.Dispose();
        _annotationCts = null;
    }

    // ── Text / display ───────────────────────────────────────────────────────

    private void SetText(string text)
    {
        _text = text ?? "";
        TextFontFamily    = ScriptFonts.For(_text);
        TextFlowDirection = TextDirection.Resolve(_text, LanguageCatalog.IsRightToLeft(_lang))
            ? FlowDirection.RightToLeft : FlowDirection.LeftToRight;
        BuildDisplayStructure(_text);
        _streamWordCursor = 0;
        _currentWordIndex = -1;
        SourceText = _text;
    }

    partial void OnShowRawMarkdownChanged(bool value)
    {
        _settings.Current.TtsShowRawMarkdown = value;
        _settings.Save();
    }

    partial void OnShowIpaAnnotationChanged(bool value)
    {
        _settings.Current.TtsShowIpaAnnotation = value;
        _settings.Save();
        if (!value) IpaAnnotationNotice = "";
        RefreshIpaAnnotation();
    }

    /// <summary>
    /// Rebuild DisplayBlocks + the flat Words from the markdown so the whole document renders
    /// immediately: one block per <see cref="ParagraphSegmenter"/> segment — the same cut the
    /// engine renders and stores one file for — and its words in order. Words start un-timed
    /// (StartSeconds = +∞, never the highlight target); timing attaches by running index,
    /// which works because the segmenter guarantees the per-segment words are the whole
    /// extraction's words in order.
    /// </summary>
    private void BuildDisplayStructure(string text)
    {
        DisplayBlocks.Clear();
        Words.Clear();
        _wordBlock.Clear();
        var extract = MarkdownTextExtractor.Extract(text ?? "");
        var et = extract.Text;
        var ranges = extract.Ranges;

        foreach (var seg in ParagraphSegmenter.Segment(extract))
        {
            var block = new BlockItemViewModel(seg.Kind, seg.Level) { Index = seg.Index };
            DisplayBlocks.Add(block);

            int i = seg.OutputStart, segEnd = seg.OutputStart + seg.OutputLength;
            while (i < segEnd)
            {
                while (i < segEnd && char.IsWhiteSpace(et[i])) i++;
                if (i >= segEnd) break;
                int start = i;
                while (i < segEnd && !char.IsWhiteSpace(et[i])) i++;

                var style = ParagraphSegmenter.StyleAt(ranges, start);
                var w = new WordItemViewModel(et.Substring(start, i - start), Words.Count, seg.Kind, seg.Level, style, SeekToWord)
                {
                    StartSeconds = double.MaxValue,
                };
                block.Words.Add(w);
                Words.Add(w);
                _wordBlock.Add(block);
            }
        }

        var langRtl = LanguageCatalog.IsRightToLeft(_lang);
        foreach (var b in DisplayBlocks) b.UpdateFlowDirection(langRtl);

        RefreshIpaAnnotation();
    }

    private void AttachTimings(AlignmentSidecar sidecar)
    {
        var words = sidecar.Words;
        for (int k = 0; k < words.Count && k < Words.Count; k++)
        {
            Words[k].StartSeconds = words[k].StartSeconds;
            Words[k].EndSeconds   = words[k].EndSeconds;
        }
        _streamWordCursor = Math.Min(words.Count, Words.Count);
        foreach (var c in sidecar.Chunks)
            if (c.Index < DisplayBlocks.Count)
                DisplayBlocks[c.Index].SetTiming(c.AudioStartSeconds, c.AudioEndSeconds);
    }

    // ── IPA annotation (ruby text above each word) ───────────────────────────

    /// <summary>
    /// Recompute the ruby text over every word, or clear it when the option is off. Runs off
    /// the UI thread and per block; each call cancels the one before it, and a result that
    /// arrives after its text has changed is dropped rather than drawn over the new words.
    /// </summary>
    private void RefreshIpaAnnotation()
    {
        _annotationCts?.Cancel();
        _annotationCts?.Dispose();
        _annotationCts = null;
        if (!ShowIpaAnnotation)
        {
            foreach (var w in Words) w.SetRuby(null);
            ReapplyCurrentHighlight();
            return;
        }
        IpaAnnotationNotice = "";
        if (Words.Count == 0) return;

        var cts = new CancellationTokenSource();
        _annotationCts = cts;
        var token = cts.Token;
        var lang = _lang;
        var dataDir = _settings.GetPhonemizerDataDir();
        var blocks = DisplayBlocks.Select(b => b.Words.ToList()).ToList();

        _ = Task.Run(async () =>
        {
            try
            {
                await Task.Delay(100, token).ConfigureAwait(false);
                foreach (var block in blocks)
                {
                    token.ThrowIfCancellationRequested();
                    var words = block.Select(w => w.Text).ToList();
                    var ipa = IpaAnnotator.Annotate(words, lang, dataDir);
                    if (ipa is null)
                    {
                        await Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            if (token.IsCancellationRequested) return;
                            foreach (var w in Words) w.SetRuby(null);
                            ReapplyCurrentHighlight();
                            IpaAnnotationNotice = $"No IPA annotation for language \"{lang}\".";
                        });
                        return;
                    }
                    await Dispatcher.UIThread.InvokeAsync(() =>
                    {
                        if (token.IsCancellationRequested) return;
                        IpaAnnotationNotice = "";
                        for (var k = 0; k < block.Count; k++) block[k].SetRuby(ipa[k]);
                        ReapplyCurrentHighlight();
                    });
                }
            }
            catch (OperationCanceledException) { /* superseded */ }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[IPA] annotation failed: {ex.Message}");
            }
        }, token);
    }

    // ── Playback ─────────────────────────────────────────────────────────────

    /// <summary>Play / pause / resume. Idle: a finished job plays its WAV from the start; a job
    /// still rendering streams the chunks received so far and keeps going as more arrive.</summary>
    [RelayCommand(CanExecute = nameof(CanPlayPause))]
    private void PlayPause()
    {
        if (IsPausedBack)  { _playback.Resume(); return; }
        if (IsPlayingBack) { _playback.Pause();  return; }

        _playback.Stop();

        if (_audioPath is not null && File.Exists(_audioPath))
        {
            try { _playback.SeekIntoFile(_audioPath, _audioDuration, 0); }
            catch (Exception ex) { StatusMessage = $"Play failed: {ex.Message}"; }
            return;
        }

        List<float[]> snapshot;
        lock (_receivedLock)
        {
            if (_receivedAudio.Count == 0) return;
            snapshot = new List<float[]>(_receivedAudio);
        }

        if (IsRunning)
        {
            // Stream: what exists now, then every chunk as it lands (OnStateAction appends).
            try
            {
                _playback.StartStreaming(_sampleRate, channels: 1);
                lock (_receivedLock) _streamingPlayback = true;
                foreach (var c in snapshot) _playback.AppendSamples(c);
            }
            catch (Exception ex) { StatusMessage = $"Play failed: {ex.Message}"; }
            return;
        }

        // A cancelled or failed job with partial audio: play what was rendered from a temp WAV.
        try
        {
            var outWav = Path.Combine(Path.GetTempPath(), $"vernacula_tts_{DateTime.UtcNow:yyyyMMddHHmmss_fff}_play.wav");
            ChatterboxSynthesisService.WriteWavFromChunks(outWav, snapshot, _sampleRate);
            int totalSamples = 0;
            foreach (var c in snapshot) totalSamples += c.Length;
            _audioPath     = outWav;
            _audioDuration = totalSamples / (double)_sampleRate;
            _playback.SeekIntoFile(_audioPath, _audioDuration, 0);
        }
        catch (Exception ex) { StatusMessage = $"Play failed: {ex.Message}"; }
    }

    private bool CanPlayPause() => HasAudio && _playback.CanPlayOnThisPlatform;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop()
    {
        lock (_receivedLock) _streamingPlayback = false;
        _playback.Stop();
    }

    private bool CanStop() => IsPlayingBack || IsPausedBack;

    [RelayCommand(CanExecute = nameof(CanCancelJob))]
    private void CancelJobNow()
    {
        if (_job is not null) CancelJob?.Invoke(_job.JobId);
    }

    private bool CanCancelJob() => IsRunning;

    [RelayCommand]
    private void GoBack()
    {
        Detach();
        NavigateBack?.Invoke();
    }

    // ── Export ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Writes the rendered WAV and a sentence-by-sentence CSV (text, phonemes, timing) next to
    /// each other under one chosen name. Finished jobs only: the timing comes from the sidecar.
    /// </summary>
    [RelayCommand(CanExecute = nameof(CanExport))]
    private async Task Export()
    {
        if (_job is null || _sidecar is null || _audioPath is null) return;
        string stem = string.Concat(_job.JobTitle.Split(Path.GetInvalidFileNameChars())).Trim();
        if (stem.Length == 0) stem = "tts-export";
        var csvPath = await StoragePickers.SaveFileAsync(Loc.Instance["btn_export_tts"], stem + ".csv",
            StoragePickers.CsvFiles, StoragePickers.AllFiles);
        if (csvPath is null) return;
        string wavPath = Path.ChangeExtension(csvPath, ".wav");

        var job = _job; var sidecar = _sidecar; string audioPath = _audioPath;
        StatusMessage = Loc.Instance["tts_export_running"];
        try
        {
            await Task.Run(() =>
            {
                var kind = TtsJobRunner.ParseBackend(job.TtsBackend);
                var sentences = TtsExportService.SplitSentences(sidecar.SourceText ?? _text, sidecar.Words);
                var rows = TtsExportService.BuildRows(sentences, kind, _lang, job.TtsVoice, _settings.GetPhonemizerDataDir());
                TtsExportService.WriteCsv(csvPath, rows, TtsExportService.PhonemeScheme(kind));
                File.Copy(audioPath, wavPath, overwrite: true);
            });
            StatusMessage = Loc.Instance.T("tts_export_done", new()
            {
                ["csv"] = Path.GetFileName(csvPath),
                ["wav"] = Path.GetFileName(wavPath),
            });
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[TtsReader] export failed: {ex}");
            StatusMessage = $"Export failed: {ex.Message}";
        }
    }

    private bool CanExport() => HasAudio && !IsRunning && _sidecar is not null;

    private void SeekToWord(WordItemViewModel word)
    {
        if (word.StartSeconds is double.MaxValue or <= 0 && word.EndSeconds == 0) return;

        if (_audioPath is not null && File.Exists(_audioPath))
        {
            try { _playback.SeekIntoFile(_audioPath, _audioDuration, word.StartSeconds); }
            catch (Exception ex) { StatusMessage = $"Seek failed: {ex.Message}"; }
        }
        else
        {
            // Mid-render: only the highlight clock can be re-anchored within the streamed buffer.
            _playback.SeekTo(word.StartSeconds);
        }
    }

    private void OnPlaybackPositionChanged(double posSec)
    {
        int idx = FindWordAt(posSec);
        if (idx != _currentWordIndex)
        {
            ClearCurrentHighlight();
            _currentWordIndex = idx;
            // A split word (Japanese, Chinese) highlights piece by piece instead of all at once.
            if (idx >= 0 && idx < Words.Count) Words[idx].IsCurrent = !Words[idx].HasPieces;
            // And the box the word sits in, so the paragraph being read stands out at a glance.
            var block = idx >= 0 && idx < _wordBlock.Count ? _wordBlock[idx] : null;
            if (!ReferenceEquals(block, _currentBlock))
            {
                if (_currentBlock is not null) _currentBlock.IsCurrent = false;
                _currentBlock = block;
                if (_currentBlock is not null) _currentBlock.IsCurrent = true;
            }
        }
        if (idx >= 0 && idx < Words.Count) Words[idx].HighlightPieceAt(posSec);
        PositionLabel = $"{posSec:F2} / {_playback.TotalEstimatedSeconds:F2} s";
    }

    private void ClearCurrentHighlight()
    {
        if (_currentWordIndex >= 0 && _currentWordIndex < Words.Count)
        {
            Words[_currentWordIndex].IsCurrent = false;
            Words[_currentWordIndex].ClearPieceHighlight();
        }
        _currentWordIndex = -1;
        if (_currentBlock is not null) { _currentBlock.IsCurrent = false; _currentBlock = null; }
    }

    /// <summary>Put the highlight back on whichever half of the current word now draws it
    /// (gaining ruby pieces moves it from the word to a piece, losing them moves it back).</summary>
    private void ReapplyCurrentHighlight()
    {
        if (_currentWordIndex < 0 || _currentWordIndex >= Words.Count) return;
        var word = Words[_currentWordIndex];
        word.IsCurrent = !word.HasPieces;
        if (word.HasPieces) word.HighlightPieceAt(_playback.PositionSeconds);
        else word.ClearPieceHighlight();
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

    public void Dispose()
    {
        Detach();
        _playback.Dispose();
    }
}
