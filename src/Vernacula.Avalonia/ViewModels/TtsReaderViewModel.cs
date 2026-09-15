using Vernacula.Tts.Base.Alignment;
using System.Collections.ObjectModel;
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
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowRawEditor), nameof(ShowRawReader))]
    private bool _showRawMarkdown;
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
        _lang         = TtsEngines.For(job).AnnotationLanguage(job);
        _sampleRate   = TtsEngines.For(job).SampleRate;
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

    /// <summary>
    /// "Kokoro-82M · af_heart · 1.00×" — the engine, then whatever it says identifies the job.
    /// A job whose engine this build does not have is named by what it stored, not relabelled
    /// as the fallback engine, and only its voice is shown (no engine is there to interpret it).
    /// </summary>
    private static string DescribeJob(JobRecord job)
    {
        if (TtsEngines.TryFor(job.TtsBackend) is not { } engine)
            return string.Join("  ·  ", new[] { job.TtsBackend, job.TtsVoice }.Where(p => !string.IsNullOrWhiteSpace(p)));
        return string.Join("  ·  ", new[] { engine.DisplayName }.Concat(engine.DescribeJob(job)));
    }

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
                sidecar = AlignmentSidecar.Load(job.ResultsFile);
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
        // Finished jobs only: editing re-renders against a sidecar, and a running job has none yet.
        CanEdit = true;
        BeginEditing(job);
        WireBlockEditing();
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
        // ⚠ Whatever is waiting out the save clock is written NOW. Leaving the reader is exactly when
        // an edit would otherwise be lost, and it is the moment the user is least expecting that.
        _ = _saveDebounce?.FlushAsync();
        EndEditing();
        IsEditing = false;
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

        if (IsRunning)
        {
            // Stream: what exists now, then every chunk as it lands (OnStateAction appends).
            // ⚠ ONE LOCK HOLD for snapshot + flag + replay. AppendSamples only queues (the
            // writer task drains it), so holding the lock is cheap — and it is what keeps a
            // chunk landing right now from being neither in the replay nor appended live, or
            // appended ahead of the chunks before it.
            try
            {
                lock (_receivedLock)
                {
                    if (_receivedAudio.Count == 0) return;
                    _playback.StartStreaming(_sampleRate, channels: 1);
                    foreach (var c in _receivedAudio) _playback.AppendSamples(c);
                    _streamingPlayback = true;
                }
            }
            catch (Exception ex) { StatusMessage = $"Play failed: {ex.Message}"; }
            return;
        }

        List<float[]> snapshot;
        lock (_receivedLock)
        {
            if (_receivedAudio.Count == 0) return;
            snapshot = new List<float[]>(_receivedAudio);
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

    // ── Editing ──────────────────────────────────────────────────────────────

    /// <summary>Listening (the aligned karaoke view) vs Editing (the markdown, editable).</summary>
    /// <summary>Whether the mode toggle is offered at all — a finished job with a sidecar.</summary>
    [ObservableProperty] private bool _canEdit;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowRawEditor), nameof(ShowRawReader))]
    private bool _isEditing;

    /// <summary>Editing the whole document as markdown — the RESTRUCTURING view (move, split, merge,
    /// change a paragraph into a heading).</summary>
    public bool ShowRawEditor => IsEditing && ShowRawMarkdown;

    /// <summary>The source, read-only, as it has always been shown in listening.</summary>
    public bool ShowRawReader => !IsEditing && ShowRawMarkdown;
    [ObservableProperty] private string _editableText = "";
    /// <summary>What the corner indicator says; empty when there is nothing to say.</summary>
    [ObservableProperty] private string _editStatus = "";
    /// <summary>A re-render is running — the indicator spins.</summary>
    [ObservableProperty] private bool _isReRendering;
    /// <summary>Edits are waiting out the debounce — the indicator pulses.</summary>
    [ObservableProperty] private bool _isEditPending;

    /// <summary>Idle after the text stops changing before the changed paragraphs are re-rendered.</summary>
    private static readonly TimeSpan ReRenderIdle = TimeSpan.FromSeconds(10);
    /// <summary>…and the shorter one for writing the document back to disk. Saving is cheap and
    /// losing keystrokes is not, so it does not wait for the render clock.</summary>
    private static readonly TimeSpan SaveIdle = TimeSpan.FromSeconds(3);

    private TtsJobRunner?            _editRunner;
    private Debouncer?               _saveDebounce;
    private Debouncer?               _reRenderDebounce;
    private CancellationTokenSource? _reRenderCts;
    /// <summary>Set while the VM itself writes EditableText, so a reload does not look like a user edit.</summary>
    private bool _loadingEditableText;
    /// <summary>Whether <see cref="EditableText"/> has been filled from the document yet.</summary>
    private bool _editorSeeded;

    /// <summary>
    /// ⚠ CALLED ON EVERY LOAD, INCLUDING THE RELOAD AFTER A RE-RENDER, so it must be safe to enter
    /// while its own debounced work is in flight. Two things follow, and both were bugs first:
    /// the debouncers are created ONCE (tearing them down here would dispose the very debouncer
    /// whose action is running), and the editor is seeded ONCE — a reload must never overwrite what
    /// the user has typed since the render it is reloading began.
    /// </summary>
    private void BeginEditing(JobRecord job)
    {
        if (_saveDebounce is null)
        {
            _editRunner = new TtsJobRunner(_settings);
            _saveDebounce = new Debouncer(SaveIdle, _ => SaveSourceAsync(job), ReportEditError);
            _reRenderDebounce = new Debouncer(ReRenderIdle, ct => ReRenderAsync(job, ct), ReportEditError);
        }
        if (_editorSeeded) return;
        _loadingEditableText = true;
        EditableText = _text;
        _loadingEditableText = false;
        _editorSeeded = true;
    }

    private void EndEditing()
    {
        _reRenderCts?.Cancel();
        _saveDebounce?.Dispose();
        _reRenderDebounce?.Dispose();
        _saveDebounce = null;
        _reRenderDebounce = null;
        _editRunner = null;
        _editorSeeded = false;
    }

    partial void OnEditableTextChanged(string value)
    {
        if (_loadingEditableText) return;
        IsEditPending = true;
        EditStatus = Loc.Instance["tts_edit_pending"];
        _saveDebounce?.Bump();
        _reRenderDebounce?.Bump();
    }

    partial void OnIsEditingChanged(bool value)
    {
        // ⚠ CLICKING A WORD SEEKS, and that must keep being true in Listening. The cards carry this
        // flag so the editing overlay — which is what takes the click instead — exists only here.
        foreach (var b in DisplayBlocks) b.CardEditingEnabled = value;
        if (!value)
        {
            CommitOpenBlock();
            // Leaving the editor writes immediately rather than on the clock — the user has moved on.
            _ = _saveDebounce?.FlushAsync();
        }
    }

    /// <summary>Opens one card for editing, seeded with its own markdown.</summary>
    private void OnBlockEditRequested(BlockItemViewModel block)
    {
        if (!IsEditing) return;
        CommitOpenBlock();
        var span = MarkdownSegmentSpans.For(EditableText).FirstOrDefault(s => s.Index == block.Index);
        if (span is null)
        {
            // No mapped extent — leave the card read-only rather than guess at where it lives.
            EditStatus = Loc.Instance["tts_edit_card_unmapped"];
            return;
        }
        block.EditText = MarkdownSegmentSpans.TextOf(EditableText, span);
        block.IsEditingBlock = true;
    }

    /// <summary>
    /// Splices the card's text back into the document. Setting EditableText is what starts both
    /// clocks, so a card edit debounces exactly like a raw edit — one path, not two.
    /// </summary>
    private void OnBlockEditCommitted(BlockItemViewModel block)
    {
        if (!block.IsEditingBlock) return;
        block.IsEditingBlock = false;
        var span = MarkdownSegmentSpans.For(EditableText).FirstOrDefault(s => s.Index == block.Index);
        if (span is null) return;
        string current = MarkdownSegmentSpans.TextOf(EditableText, span);
        if (string.Equals(current, block.EditText, StringComparison.Ordinal)) return;
        EditableText = MarkdownSegmentSpans.Splice(EditableText, span, block.EditText);
    }

    private static void OnBlockEditCancelled(BlockItemViewModel block) => block.IsEditingBlock = false;

    /// <summary>
    /// Closes whatever card is open, keeping its text. The view calls this when a click lands
    /// anywhere outside the open editor.
    ///
    /// ⚠ LOSING FOCUS IS NOT THE SAME EVENT AS BEING CLICKED AWAY FROM, which is why the text box's
    /// own LostFocus is not enough. Clicking the card's caption, the gap between cards, the
    /// scroll area or any other inert surface moves focus NOWHERE — the box keeps it and stays open,
    /// so the card appears stuck in editing until something focusable is clicked. Reported as
    /// "clicking outside a currently active editing text box should put it back into the
    /// non-editing rendered state".
    /// </summary>
    public void CommitOpenCard() => CommitOpenBlock();

    private void CommitOpenBlock()
    {
        foreach (var b in DisplayBlocks)
            if (b.IsEditingBlock) { OnBlockEditCommitted(b); break; }
    }

    /// <summary>Gives every freshly built card its editing hooks and the current mode.</summary>
    private void WireBlockEditing()
    {
        foreach (var b in DisplayBlocks)
        {
            b.EditRequested = OnBlockEditRequested;
            b.EditCommitted = OnBlockEditCommitted;
            b.EditCancelled = OnBlockEditCancelled;
            b.CardEditingEnabled = IsEditing;
        }
    }

    /// <summary>Writes the edited markdown back over the job's own input document.</summary>
    private async Task SaveSourceAsync(JobRecord job)
    {
        string path = job.AudioFilePath;   // for a TTS job this is the input document
        if (string.IsNullOrWhiteSpace(path)) return;
        string text = EditableText;
        await File.WriteAllTextAsync(path, text).ConfigureAwait(false);
        _text = text;
        Dispatcher.UIThread.Post(() =>
        {
            SourceText = text;
            if (!IsReRendering) EditStatus = Loc.Instance["tts_edit_saved"];
        });
    }

    /// <summary>
    /// Re-renders only the paragraphs that changed and reloads the view from the new sidecar.
    /// ⚠ Playback stops first: the run REPLACES the rendered WAV underneath it.
    /// </summary>
    private async Task ReRenderAsync(JobRecord job, CancellationToken outer)
    {
        if (_sidecar is null || job.TtsSettings is null) return;

        // ⚠ AN OPEN CARD IS COMMITTED FIRST, AND THAT ORDER IS LOAD-BEARING. A successful re-render
        // ends in LoadCompleted, which REBUILDS DisplayBlocks — every card object is replaced. A card
        // left open across that rebuild loses whatever was being typed into it, silently, and the
        // window for it is ordinary: edit one paragraph, start on the next, and the first one's clock
        // runs out while you are still typing. Committing folds the open card into EditableText, so
        // nothing is lost and this render includes it.
        await Dispatcher.UIThread.InvokeAsync(CommitOpenBlock);

        string text = EditableText;
        var previous = _sidecar;

        var changed = TtsSegmentReuse.ChangedSegments(text, previous);
        if (changed.Count == 0)
        {
            Dispatcher.UIThread.Post(() => { IsEditPending = false; EditStatus = ""; });
            return;
        }

        _reRenderCts?.Cancel();
        _reRenderCts?.Dispose();
        _reRenderCts = CancellationTokenSource.CreateLinkedTokenSource(outer);
        var ct = _reRenderCts.Token;

        await Dispatcher.UIThread.InvokeAsync(() =>
        {
            _playback.Stop();
            IsEditPending = false;
            IsReRendering = true;
            EditStatus = Loc.Instance.T("tts_edit_rendering", new()
            {
                ["count"] = changed.Count.ToString(),
            });
        });

        try
        {
            var sidecar = await _editRunner!.ReRenderAsync(
                text, job.ResultsFile, job.TtsSettings, previous,
                _ => { }, ct).ConfigureAwait(false);

            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                _sidecar = sidecar;
                LoadCompleted(job);
                IsReRendering = false;
                EditStatus = Loc.Instance["tts_edit_rendered"];
            });
        }
        catch (OperationCanceledException)
        {
            Dispatcher.UIThread.Post(() => { IsReRendering = false; EditStatus = ""; });
        }
    }

    private void ReportEditError(Exception ex)
    {
        Console.Error.WriteLine($"[TtsReader] edit: {ex}");
        Dispatcher.UIThread.Post(() =>
        {
            IsReRendering = false;
            IsEditPending = false;
            EditStatus = Loc.Instance.T("tts_edit_failed", new() { ["error"] = ex.Message });
        });
    }

    // ── Export ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Writes ONE file: the rendered WAV, the sentence-by-sentence CSV (text, phonemes, timing), or
    /// the source markdown. Finished jobs only — the CSV's timing comes from the sidecar.
    ///
    /// ⚠ THE PICKER'S FILE TYPE IS THE CHOICE OF WHAT TO WRITE, and it is the only UI this needs.
    /// The dialog used to be picking a LOCATION AND STEM while both files were written regardless,
    /// which is not what a file-type dropdown means anywhere else.
    /// </summary>
    [RelayCommand(CanExecute = nameof(CanExport))]
    private async Task Export()
    {
        if (_job is null || _sidecar is null || _audioPath is null) return;
        string stem = string.Concat(_job.JobTitle.Split(Path.GetInvalidFileNameChars())).Trim();
        if (stem.Length == 0) stem = "tts-export";
        var chosen = await StoragePickers.SaveFileAsync(Loc.Instance["tts_export_title"], stem + ".wav",
            StoragePickers.WavFiles, StoragePickers.CsvFiles, StoragePickers.MarkdownFiles);
        if (chosen is null) return;

        if (TtsExportService.KindOf(chosen) is not { } kind)
        {
            // A name typed with no extension at all has none to quote back; name the file instead,
            // so the message reads the same either way.
            StatusMessage = Loc.Instance.T("tts_export_unknown_type", new()
            {
                ["ext"] = Path.GetExtension(chosen) is { Length: > 0 } ext ? ext : Path.GetFileName(chosen),
            });
            return;
        }

        var job = _job; var sidecar = _sidecar; string audioPath = _audioPath;
        // ⚠ Read on the UI thread: EditableText is what the user has typed, and the export of the
        // markdown is meant to be of the document as it stands, open card and all.
        CommitOpenBlock();
        string markdown = _editorSeeded ? EditableText : _text;
        string written = "";
        StatusMessage = Loc.Instance["tts_export_running"];
        try
        {
            await Task.Run(() =>
            {
                written = kind switch
                {
                    TtsExportService.ExportKind.Audio    => TtsExportService.WriteAudio(chosen, audioPath),
                    TtsExportService.ExportKind.Markdown => TtsExportService.WriteMarkdown(chosen, markdown),
                    _ => WriteTranscript(),
                };

                string WriteTranscript()
                {
                    var engine = TtsEngines.For(job);
                    var settings = new TtsJobSettings(job.TtsBackend, job.TtsLanguage, job.TtsVoice, job.TtsSpeed, job.TtsNumStep);
                    var sentences = TtsExportService.SplitSentences(sidecar.SourceText ?? _text, sidecar.Words);
                    var rows = TtsExportService.BuildRows(sentences, _settings, settings);
                    return TtsExportService.WriteTranscript(chosen, rows, engine.PhonemeScheme);
                }
            });
            StatusMessage = Loc.Instance.T("tts_export_done", new() { ["file"] = Path.GetFileName(written) });
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
