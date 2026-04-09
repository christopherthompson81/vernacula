using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Avalonia;
using Avalonia.Media;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using NAudio.Wave;
using SoundTouch;
using Vernacula.Base;
using Vernacula.App.Models;
using Vernacula.App.Services;
using ParakeetAsr = Vernacula.Base.Parakeet;

namespace Vernacula.App.ViewModels;

internal partial class TranscriptEditorViewModel : ObservableObject, IDisposable
{
    private static readonly TimeSpan InitialPlaybackWarmupLeadIn = TimeSpan.FromMilliseconds(400);
    internal sealed record LoadSnapshot(
        List<EditorSegment> Segments,
        List<(int SpeakerId, string Name)> Speakers,
        string? AudioPath);

    private static readonly string? FfplayPath = FindExecutable("ffplay");

    public static bool SupportsAudioPlayback => OperatingSystem.IsWindows() || FfplayPath is not null;
    public static string PlaybackUnavailableReason =>
        "Playback requires Windows audio output or an `ffplay` executable in PATH.";

    private string?          _dbPath;
    private string?          _audioPath;
    private float[]?         _fullAudio;
    private int              _audioSampleRate = Config.SampleRate;
    private int              _audioChannels   = 1;
    private WaveOutEvent?    _waveOut;
    private Process?         _playbackProcess;
    private DispatcherTimer? _playbackTimer;
    private bool             _preserveContinuousPlaybackPosition;
    private bool             _suppressContinuousPositionFocusSync;
    private bool             _needsInitialPlaybackWarmup = true;

    public ObservableCollection<EditorSegment> Segments    { get; } = [];
    public List<(int SpeakerId, string Name)>  AllSpeakers { get; private set; } = [];

    [ObservableProperty] private int           _focusedIndex     = 0;
    [ObservableProperty] private PlaybackMode  _playbackMode     = PlaybackMode.Continuous;
    [ObservableProperty] private bool          _isPlaying        = false;
    [ObservableProperty] private double        _playbackPosition = 0.0;  // 0..1
    [ObservableProperty] private double        _playbackSpeed    = 1.0;  // 0.2..2.0
    [ObservableProperty] private int           _highlightedToken = -1;
    [ObservableProperty] private string        _audioBaseName    = "";
    [ObservableProperty] private EditorSegment? _focusedSegment;

    // Raised so the window can animate scroll + rebuild ASR panel
    public event Action<int>? FocusedIndexChanging;

    // ── Loading ───────────────────────────────────────────────────────────────

    public static LoadSnapshot LoadSnapshotData(string dbPath)
    {
        var segments = new List<EditorSegment>();
        using var db = new TranscriptionDb(dbPath);
        foreach (var row in db.GetCardsForEditor())
        {
            segments.Add(new EditorSegment
            {
                CardId             = row.CardId,
                SpeakerId          = row.SpeakerId,
                SpeakerDisplayName = row.SpeakerName,
                PlayStart          = row.PlayStart,
                PlayEnd            = row.PlayEnd,
                AsrContent         = row.AsrContent,
                Content            = row.Content,
                Tokens             = row.Tokens,
                Timestamps         = row.Timestamps,
                Logprobs           = row.Logprobs,
                Sources            = row.Sources,
                Verified           = row.Verified,
                IsSuppressed       = row.IsSuppressed,
            });
        }

        string? audioPath;
        using (var db2 = new TranscriptionDb(dbPath))
            audioPath = db2.GetAudioFilePath();

        List<(int SpeakerId, string Name)> speakers;
        using (var db3 = new TranscriptionDb(dbPath))
            speakers = db3.GetSpeakers();

        return new LoadSnapshot(segments, speakers, audioPath);
    }

    public void ApplyLoadedData(string dbPath, string audioBaseName, LoadSnapshot snapshot)
    {
        _dbPath       = dbPath;
        AudioBaseName = audioBaseName;
        _audioPath    = snapshot.AudioPath;
        _fullAudio = null;
        _audioSampleRate = Config.SampleRate;
        _audioChannels = 1;
        _needsInitialPlaybackWarmup = true;
        Segments.Clear();
        foreach (var segment in snapshot.Segments)
            Segments.Add(segment);

        AllSpeakers = snapshot.Speakers;
        FocusedIndex = 0;
        FocusedSegment = Segments.Count > 0 ? Segments[0] : null;

        NextSegmentCommand.NotifyCanExecuteChanged();
        PrevSegmentCommand.NotifyCanExecuteChanged();
        PlayCommand.NotifyCanExecuteChanged();

        if (_audioPath != null && File.Exists(_audioPath))
        {
            Task.Run(() =>
            {
                try
                {
                    var (samples, rate, channels) = AudioUtils.ReadAudio(_audioPath);
                    _fullAudio       = samples;
                    _audioSampleRate = rate;
                    _audioChannels   = channels;
                    Dispatcher.UIThread.InvokeAsync(
                        () => PlayCommand.NotifyCanExecuteChanged());
                }
                catch { /* audio unavailable */ }
            });
        }
    }

    public void Load(string dbPath, string audioBaseName)
        => ApplyLoadedData(dbPath, audioBaseName, LoadSnapshotData(dbPath));

    // ── Navigation ────────────────────────────────────────────────────────────

    public void NavigateTo(int index)
    {
        if (index < 0 || index >= Segments.Count || index == FocusedIndex) return;
        if (IsPlaying && PlaybackMode == PlaybackMode.Continuous)
        {
            JumpToContinuousSegment(index);
            return;
        }

        StopPlayback();
        FocusedIndexChanging?.Invoke(index);
        FocusedIndex = index;
    }

    [RelayCommand(CanExecute = nameof(CanGoPrev))]
    private void PrevSegment()
    {
        int next = FocusedIndex - 1;
        if (IsPlaying && PlaybackMode == PlaybackMode.Continuous)
        {
            JumpToContinuousSegment(next);
            return;
        }

        StopPlayback();
        FocusedIndexChanging?.Invoke(next);
        FocusedIndex = next;
    }
    private bool CanGoPrev() => FocusedIndex > 0;

    [RelayCommand(CanExecute = nameof(CanGoNext))]
    private void NextSegment()
    {
        int next = FocusedIndex + 1;
        if (IsPlaying && PlaybackMode == PlaybackMode.Continuous)
        {
            JumpToContinuousSegment(next);
            return;
        }

        StopPlayback();
        FocusedIndexChanging?.Invoke(next);
        FocusedIndex = next;
    }
    private bool CanGoNext() => FocusedIndex < Segments.Count - 1;

    private void JumpToContinuousSegment(int index)
    {
        if (index < 0 || index >= Segments.Count)
        {
            return;
        }

        StopPlayback();
        FocusedIndexChanging?.Invoke(index);
        FocusedIndex = index;
        Play();
    }

    partial void OnPlaybackSpeedChanged(double value)
    {
        if (IsPlaying) Play();
    }

    partial void OnPlaybackModeChanged(PlaybackMode value)
    {
        StopPlayback();
        PlaybackPosition = 0;
    }

    partial void OnFocusedIndexChanged(int value)
    {
        if (PlaybackMode == PlaybackMode.Continuous)
        {
            // Keep the global playback cursor when focus is being driven from it, such as
            // during continuous-mode seeking.
            if (!IsPlaying
                && !_preserveContinuousPlaybackPosition
                && value >= 0
                && value < Segments.Count
                && _fullAudio != null)
            {
                double totalDuration = (double)_fullAudio.Length / (_audioSampleRate * _audioChannels);
                if (totalDuration > 0)
                    PlaybackPosition = Segments[value].PlayStart / totalDuration;
            }
        }
        else
        {
            PlaybackPosition = 0;
        }
        HighlightedToken = -1;
        FocusedSegment   = value >= 0 && value < Segments.Count ? Segments[value] : null;
        PrevSegmentCommand.NotifyCanExecuteChanged();
        NextSegmentCommand.NotifyCanExecuteChanged();
        PlayCommand.NotifyCanExecuteChanged();
    }

    partial void OnPlaybackPositionChanged(double value)
    {
        // When not playing, seek bar drag updates token highlight (and focus in continuous mode)
        if (IsPlaying) return;
        if (FocusedIndex < 0 || FocusedIndex >= Segments.Count) return;
        if (PlaybackMode == PlaybackMode.Continuous)
        {
            if (_fullAudio == null) return;
            double totalDuration = (double)_fullAudio.Length / (_audioSampleRate * _audioChannels);
            double curSec = value * totalDuration;
            if (!_suppressContinuousPositionFocusSync)
            {
                _preserveContinuousPlaybackPosition = true;
                try
                {
                    UpdateContinuousFocus(curSec);
                }
                finally
                {
                    _preserveContinuousPlaybackPosition = false;
                }
            }
            if (FocusedIndex >= 0 && FocusedIndex < Segments.Count)
            {
                var cseg = Segments[FocusedIndex];
                double intoSeg = curSec - cseg.PlayStart;
                if (intoSeg >= 0) UpdateHighlightedToken(cseg, intoSeg);
            }
        }
        else
        {
            var seg = Segments[FocusedIndex];
            double segDuration = seg.PlayEnd - seg.PlayStart;
            if (segDuration > 0)
                UpdateHighlightedToken(seg, value * segDuration);
        }
    }

    // ── Playback ──────────────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanPlay))]
    private void Play()
    {
        if (PlaybackMode == PlaybackMode.Continuous) { PlayContinuous(); return; }

        if (_fullAudio is null) return;
        if (FocusedIndex < 0 || FocusedIndex >= Segments.Count) return;
        if (PlaybackPosition >= 1.0) PlaybackPosition = 0;
        var seg = Segments[FocusedIndex];

        double segDuration = seg.PlayEnd - seg.PlayStart;
        if (segDuration <= 0) return;

        double segOffsetSec = PlaybackPosition * segDuration;
        int startSample = (int)((seg.PlayStart + segOffsetSec) * _audioSampleRate) * _audioChannels;
        int endSample   = (int)(seg.PlayEnd * _audioSampleRate) * _audioChannels;
        startSample = Math.Clamp(startSample, 0, _fullAudio.Length);
        endSample   = Math.Clamp(endSample,   0, _fullAudio.Length);
        int count = endSample - startSample;
        if (count <= 0) return;

        StopPlayback();
        TimeSpan leadIn = ConsumeInitialPlaybackLeadIn();
        if (OperatingSystem.IsWindows())
        {
            var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(_audioSampleRate, _audioChannels);
            var provider   = new SoundTouchWaveProvider(_fullAudio, startSample, count,
                                                        waveFormat, _audioChannels, PlaybackSpeed,
                                                        leadIn);

            _waveOut = new WaveOutEvent();
            _waveOut.Init(provider);
            _waveOut.PlaybackStopped += OnWaveOutStopped;
            _waveOut.Play();
        }
        else if (!StartFfplayPlayback(seg.PlayStart + segOffsetSec, segDuration - segOffsetSec, leadIn))
        {
            return;
        }

        IsPlaying = true;
        PlayCommand.NotifyCanExecuteChanged();
        PauseCommand.NotifyCanExecuteChanged();

        double posAtStart = PlaybackPosition;
        var    startedAt  = DateTime.UtcNow + leadIn;
        double speed      = PlaybackSpeed;

        UpdateHighlightedToken(seg, segOffsetSec);

        _playbackTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
        _playbackTimer.Tick += (_, _) =>
        {
            double elapsed = Math.Max(0.0, (DateTime.UtcNow - startedAt).TotalSeconds);
            double newPos  = posAtStart + elapsed * speed / segDuration;
            PlaybackPosition = Math.Min(1.0, newPos);

            double intoSeg = segOffsetSec + elapsed * speed;
            UpdateHighlightedToken(seg, intoSeg);

            if (PlaybackPosition >= 1.0)
                OnSegmentPlaybackComplete();
        };
        _playbackTimer.Start();
    }
    private bool CanPlay() => SupportsAudioPlayback
                           && _fullAudio != null && !IsPlaying
                           && (PlaybackMode == PlaybackMode.Continuous
                               || (FocusedIndex >= 0 && FocusedIndex < Segments.Count));

    private void PlayContinuous()
    {
        if (_fullAudio is null) return;

        double totalDuration = (double)_fullAudio.Length / (_audioSampleRate * _audioChannels);
        if (totalDuration <= 0) return;

        // Resolve start position in the file
        double startSec;
        if (PlaybackPosition <= 0 || PlaybackPosition >= 1.0)
        {
            startSec = FocusedIndex >= 0 && FocusedIndex < Segments.Count
                ? Segments[FocusedIndex].PlayStart
                : 0.0;
            PlaybackPosition = startSec / totalDuration;
        }
        else
        {
            startSec = PlaybackPosition * totalDuration;
        }

        int startSample = Math.Clamp((int)(startSec * _audioSampleRate) * _audioChannels,
                                     0, _fullAudio.Length);
        int count = _fullAudio.Length - startSample;
        if (count <= 0) return;

        StopPlayback();
        TimeSpan leadIn = ConsumeInitialPlaybackLeadIn();
        double speed = PlaybackSpeed;
        if (OperatingSystem.IsWindows())
        {
            var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(_audioSampleRate, _audioChannels);
            var provider   = new SoundTouchWaveProvider(_fullAudio, startSample, count,
                                                        waveFormat, _audioChannels, speed,
                                                        leadIn);

            _waveOut = new WaveOutEvent();
            _waveOut.Init(provider);
            _waveOut.PlaybackStopped += OnWaveOutStopped;
            _waveOut.Play();
        }
        else if (!StartFfplayPlayback(startSec, null, leadIn))
        {
            return;
        }

        IsPlaying = true;
        PlayCommand.NotifyCanExecuteChanged();
        PauseCommand.NotifyCanExecuteChanged();

        var startedAt = DateTime.UtcNow + leadIn;

        UpdateContinuousFocus(startSec);

        if (FocusedIndex >= 0 && FocusedIndex < Segments.Count)
        {
            var currentSegment = Segments[FocusedIndex];
            UpdateHighlightedToken(currentSegment, Math.Max(0.0, startSec - currentSegment.PlayStart));
        }

        _playbackTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
        _playbackTimer.Tick += (_, _) =>
        {
            double elapsed = Math.Max(0.0, (DateTime.UtcNow - startedAt).TotalSeconds);
            double curSec = startSec + elapsed * speed;
            PlaybackPosition = Math.Min(1.0, curSec / totalDuration);

            UpdateContinuousFocus(curSec);

            if (FocusedIndex >= 0 && FocusedIndex < Segments.Count)
            {
                var s = Segments[FocusedIndex];
                double intoSeg = curSec - s.PlayStart;
                if (intoSeg >= 0) UpdateHighlightedToken(s, intoSeg);
            }

            if (PlaybackPosition >= 1.0)
                OnSegmentPlaybackComplete();
        };
        _playbackTimer.Start();
    }

    private TimeSpan ConsumeInitialPlaybackLeadIn()
    {
        if (!_needsInitialPlaybackWarmup)
        {
            return TimeSpan.Zero;
        }

        _needsInitialPlaybackWarmup = false;
        return InitialPlaybackWarmupLeadIn;
    }

    /// <summary>
    /// Updates FocusedIndex to the segment that contains <paramref name="timeSec"/>,
    /// or to the next upcoming segment when in a gap.
    /// </summary>
    private void UpdateContinuousFocus(double timeSec)
    {
        // Inside a segment?
        for (int i = 0; i < Segments.Count; i++)
        {
            var s = Segments[i];
            if (timeSec >= s.PlayStart && timeSec < s.PlayEnd)
            {
                if (FocusedIndex != i) { FocusedIndexChanging?.Invoke(i); FocusedIndex = i; }
                return;
            }
        }
        // In a gap — focus the next upcoming segment
        for (int i = 0; i < Segments.Count; i++)
        {
            var s = Segments[i];
            if (s.PlayStart > timeSec)
            {
                if (FocusedIndex != i) { FocusedIndexChanging?.Invoke(i); FocusedIndex = i; }
                return;
            }
        }
        // Past all segments — keep current focus
    }

    [RelayCommand(CanExecute = nameof(CanPause))]
    private void Pause()
    {
        if (OperatingSystem.IsWindows())
        {
            _waveOut?.Pause();
        }
        else
        {
            StopFfplayPlayback();
        }
        _playbackTimer?.Stop();
        IsPlaying = false;
        PlayCommand.NotifyCanExecuteChanged();
        PauseCommand.NotifyCanExecuteChanged();
    }
    private bool CanPause() => IsPlaying;

    public void Seek(double position)
    {
        if (FocusedIndex < 0 || FocusedIndex >= Segments.Count) return;
        bool wasPlaying = IsPlaying;
        StopPlayback();
        PlaybackPosition = Math.Clamp(position, 0.0, 1.0);
        if (wasPlaying) Play();
    }

    public bool SeekContinuous(double position, bool resumePlayback = true)
    {
        if (_fullAudio == null || Segments.Count == 0)
            return false;

        bool wasPlaying = IsPlaying;
        double clampedPosition = Math.Clamp(position, 0.0, 1.0);
        int focusIndex = GetContinuousFocusIndex(clampedPosition);

        StopPlayback();

        _suppressContinuousPositionFocusSync = true;
        _preserveContinuousPlaybackPosition = true;
        try
        {
            PlaybackPosition = clampedPosition;
            if (focusIndex >= 0 && focusIndex < Segments.Count && focusIndex != FocusedIndex)
            {
                FocusedIndexChanging?.Invoke(focusIndex);
                FocusedIndex = focusIndex;
            }
        }
        finally
        {
            _preserveContinuousPlaybackPosition = false;
            _suppressContinuousPositionFocusSync = false;
        }

        if (wasPlaying && resumePlayback)
            Play();

        return wasPlaying;
    }

    public int GetContinuousFocusIndex(double position)
    {
        if (_fullAudio == null || Segments.Count == 0)
            return FocusedIndex;

        double totalDuration = (double)_fullAudio.Length / (_audioSampleRate * _audioChannels);
        if (totalDuration <= 0)
            return FocusedIndex;

        double timeSec = Math.Clamp(position, 0.0, 1.0) * totalDuration;

        for (int i = 0; i < Segments.Count; i++)
        {
            var segment = Segments[i];
            if (timeSec >= segment.PlayStart && timeSec < segment.PlayEnd)
                return i;
        }

        for (int i = 0; i < Segments.Count; i++)
        {
            if (Segments[i].PlayStart > timeSec)
                return i;
        }

        return Segments.Count - 1;
    }

    private void UpdateHighlightedToken(EditorSegment seg, double secondsIntoSegment)
    {
        if (seg.Timestamps.Count == 0) return;
        const double frameSeconds = Config.HopLength * 8.0 / Config.SampleRate;
        int best = -1;
        for (int i = 0; i < seg.Timestamps.Count; i++)
        {
            if (seg.Timestamps[i] * frameSeconds <= secondsIntoSegment)
                best = i;
            else
                break;
        }
        HighlightedToken = best;
    }

    private void OnWaveOutStopped(object? sender, StoppedEventArgs e)
    {
        Dispatcher.UIThread.InvokeAsync(() =>
        {
            PlaybackPosition = 1.0;
            OnSegmentPlaybackComplete();
        });
    }

    private void OnSegmentPlaybackComplete()
    {
        if (!IsPlaying) return;
        StopPlayback();
        if (PlaybackMode == PlaybackMode.Single) return;
        if (PlaybackMode == PlaybackMode.Continuous) return; // end of file reached

        if (PlaybackMode == PlaybackMode.AutoAdvance)
            MarkVerified(FocusedIndex, true);

        int next = FocusedIndex + 1;

        if (next >= Segments.Count) return;

        FocusedIndexChanging?.Invoke(next);
        FocusedIndex = next;

        int delayMs = PlaybackMode == PlaybackMode.AutoAdvance ? 400 : 0;
        var delay = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(delayMs) };
        delay.Tick += (_, _) => { delay.Stop(); Play(); };
        delay.Start();
    }

    private void StopPlayback()
    {
        _playbackTimer?.Stop();
        _playbackTimer = null;
        if (OperatingSystem.IsWindows())
        {
            if (_waveOut != null)
            {
                _waveOut.PlaybackStopped -= OnWaveOutStopped;
                _waveOut.Stop();
                _waveOut.Dispose();
                _waveOut = null;
            }
        }
        else
        {
            StopFfplayPlayback();
        }
        IsPlaying = false;
        PlayCommand.NotifyCanExecuteChanged();
        PauseCommand.NotifyCanExecuteChanged();
    }

    // ── Verified ──────────────────────────────────────────────────────────────

    [RelayCommand]
    private void ToggleVerified()
    {
        if (FocusedIndex < 0 || FocusedIndex >= Segments.Count) return;
        MarkVerified(FocusedIndex, !Segments[FocusedIndex].Verified);
    }

    private void MarkVerified(int index, bool verified)
    {
        if (index < 0 || index >= Segments.Count || _dbPath is null) return;
        var seg = Segments[index];
        seg.Verified = verified;
        using var db = new TranscriptionDb(_dbPath);
        db.UpdateCardVerified(seg.CardId, verified);
    }

    // ── Save edits ────────────────────────────────────────────────────────────

    public bool SaveContent(int index, string text)
    {
        if (index < 0 || index >= Segments.Count || _dbPath is null) return false;
        var seg = Segments[index];
        if (seg.Content == text) return false;
        seg.Content = text;
        string? dbValue = text == seg.AsrContent ? null : text;
        using var db = new TranscriptionDb(_dbPath);
        db.UpdateCardContent(seg.CardId, dbValue);
        return true;
    }

    public bool SaveDraft(TranscriptEditorCardState card)
    {
        bool changed = false;
        string draftContent = card.DraftContent ?? string.Empty;
        if (SaveContent(card.Index, draftContent))
            changed = true;

        string speakerName = (card.DraftSpeakerName ?? string.Empty).Trim();
        if (!string.IsNullOrEmpty(speakerName) && SaveSpeakerName(card.Index, speakerName))
            changed = true;

        card.SyncDraftsFromSegment();
        return changed;
    }

    // ── Speaker management ────────────────────────────────────────────────────

    public int AddSpeaker(string name)
    {
        if (_dbPath is null) return -1;
        using var db = new TranscriptionDb(_dbPath);
        int id = db.AddSpeaker(name);
        AllSpeakers = db.GetSpeakers();
        return id;
    }

    public bool ReassignSegment(int segmentIndex, int newSpeakerId)
    {
        if (segmentIndex < 0 || segmentIndex >= Segments.Count || _dbPath is null) return false;
        var seg = Segments[segmentIndex];
        if (seg.SpeakerId == newSpeakerId) return false;

        string newName = AllSpeakers.FirstOrDefault(s => s.SpeakerId == newSpeakerId).Name ?? "";

        using var db = new TranscriptionDb(_dbPath);
        db.UpdateCardSpeaker(seg.CardId, newSpeakerId);

        seg.SpeakerId          = newSpeakerId;
        seg.SpeakerDisplayName = newName;
        return true;
    }

    public bool SaveSpeakerName(int index, string name)
    {
        if (index < 0 || index >= Segments.Count || _dbPath is null) return false;
        var seg = Segments[index];
        if (seg.SpeakerDisplayName == name) return false;

        int speakerId = seg.SpeakerId;

        foreach (var s in Segments)
            if (s.SpeakerId == speakerId)
                s.SpeakerDisplayName = name;

        for (int i = 0; i < AllSpeakers.Count; i++)
            if (AllSpeakers[i].SpeakerId == speakerId)
                AllSpeakers[i] = (speakerId, name);

        using var db = new TranscriptionDb(_dbPath);
        db.UpdateSpeaker(speakerId, name);
        return true;
    }

    public bool RenameSpeaker(TranscriptEditorCardState card)
    {
        string name = (card.DraftSpeakerName ?? string.Empty).Trim();
        if (string.IsNullOrEmpty(name))
            return false;

        bool changed = SaveSpeakerName(card.Index, name);
        card.SyncDraftsFromSegment();
        return changed;
    }

    public bool ReassignSpeaker(TranscriptEditorCardState card)
    {
        if (card.SelectedSpeaker is null)
            return false;

        bool changed = ReassignSegment(card.Index, card.SelectedSpeaker.SpeakerId);
        card.SyncDraftsFromSegment();
        return changed;
    }

    // ── Segment operations ────────────────────────────────────────────────────

    public bool ToggleSuppressed(int index)
    {
        if (index < 0 || index >= Segments.Count || _dbPath is null) return false;
        var seg = Segments[index];
        seg.IsSuppressed = !seg.IsSuppressed;
        using var db = new TranscriptionDb(_dbPath);
        db.UpdateCardSuppressed(seg.CardId, seg.IsSuppressed);
        return true;
    }

    public bool ToggleSuppressed(TranscriptEditorCardState card)
        => ToggleSuppressed(card.Index);

    public bool AdjustSegmentTimes(int index, double newStart, double newEnd)
    {
        if (index < 0 || index >= Segments.Count || _dbPath is null) return false;
        if (newStart < 0 || newEnd <= newStart) return false;
        var seg = Segments[index];
        seg.PlayStart = newStart;
        seg.PlayEnd   = newEnd;
        using var db = new TranscriptionDb(_dbPath);
        db.UpdateCardTimes(seg.CardId, newStart, newEnd);
        return true;
    }

    public bool AdjustSegmentTimes(TranscriptEditorCardState card, double newStart, double newEnd)
        => AdjustSegmentTimes(card.Index, newStart, newEnd);

    /// <summary>
    /// Merges Segments[index] with Segments[index+1] into a new overlay card.
    /// The results table is not modified.
    /// </summary>
    public bool MergeWithNext(int index)
    {
        if (index < 0 || index >= Segments.Count - 1 || _dbPath is null) return false;
        StopPlayback();
        var a = Segments[index];
        var b = Segments[index + 1];

        const double frameSeconds = Config.HopLength * 8.0 / Config.SampleRate;
        // Offset to convert B's in-memory timestamps (relative to B.PlayStart)
        // to be relative to A.PlayStart (the merged card's PlayStart).
        int bTsOffset = (int)Math.Round((b.PlayStart - a.PlayStart) / frameSeconds);

        // Merged in-memory token arrays
        var mergedTokens     = a.Tokens.Concat(b.Tokens).ToList();
        var mergedTimestamps = a.Timestamps
            .Concat(b.Timestamps.Select(t => t + bTsOffset))
            .ToList();
        var mergedLogprobs = a.Logprobs.Concat(b.Logprobs).ToList();

        // Merged content — always stored explicitly for multi-source cards because
        // the export query only joins source_order=0 and can't reconstruct the full text.
        string mergedAsr = (a.AsrContent.TrimEnd() + " " + b.AsrContent.TrimStart()).Trim();
        string mergedCon = (a.Content.TrimEnd() + " " + b.Content.TrimStart()).Trim();
        string? dbContent = mergedCon;

        // Build new source list: A's sources unchanged, B's sources with added bTsOffset.
        // If A and B share a ResultId (e.g. re-merging two halves of a split), consolidate
        // into one source spanning the full token range to avoid a PK conflict in card_sources.
        var newSources = new List<CardSource>(a.Sources);
        var aResultIds = new HashSet<int>(a.Sources.Select(s => s.ResultId));
        int nextOrder  = a.Sources.Count;
        foreach (var src in b.Sources)
        {
            int adjustedOffset = src.TsFrameOffset + bTsOffset;
            if (aResultIds.Contains(src.ResultId))
            {
                // Collapse: extend the existing A source's token range to cover B's tokens too.
                int existingIdx = newSources.FindIndex(s => s.ResultId == src.ResultId);
                var existing    = newSources[existingIdx];
                int? mergedEnd  = existing.TokenEnd is null || src.TokenEnd is null
                    ? null
                    : Math.Max(existing.TokenEnd.Value, src.TokenEnd.Value);
                newSources[existingIdx] = new CardSource(existing.ResultId, existing.TokenStart,
                                                         mergedEnd, existing.TsFrameOffset,
                                                         existing.SourceOrder);
            }
            else
            {
                aResultIds.Add(src.ResultId);
                newSources.Add(new CardSource(src.ResultId, src.TokenStart, src.TokenEnd,
                                              adjustedOffset, nextOrder++));
            }
        }

        int newCardId;
        using (var db = new TranscriptionDb(_dbPath))
        {
            newCardId = db.CreateCard(a.SpeakerId, a.PlayStart, b.PlayEnd);
            foreach (var src in newSources)
                db.AddCardSource(newCardId, src.ResultId, src.TokenStart, src.TokenEnd,
                                 src.TsFrameOffset, src.SourceOrder);
            if (dbContent != null)
                db.UpdateCardContent(newCardId, dbContent);
            db.DeleteCard(a.CardId);
            db.DeleteCard(b.CardId);
        }

        // Update kept segment in-memory
        a.CardId      = newCardId;
        a.PlayEnd     = b.PlayEnd;
        a.AsrContent  = mergedAsr;
        a.Content     = mergedCon;
        a.Tokens      = mergedTokens;
        a.Timestamps  = mergedTimestamps;
        a.Logprobs    = mergedLogprobs;
        a.Sources     = newSources;

        Segments.RemoveAt(index + 1);

        // Renumber sort orders to match current in-memory order
        if (_dbPath != null)
            using (var db = new TranscriptionDb(_dbPath))
                db.RenumberSortOrders(Segments.Select(s => s.CardId));

        return true;
    }

    /// <summary>Merges Segments[index-1] with Segments[index].</summary>
    public bool MergeWithPrev(int index)
    {
        if (index <= 0 || index >= Segments.Count || _dbPath is null) return false;
        return MergeWithNext(index - 1);
    }

    /// <summary>True when the full audio waveform is loaded and can be used for re-ASR.</summary>
    public bool HasAudio => _fullAudio != null;

    /// <summary>Total audio length in seconds. Null until audio is loaded.</summary>
    public double? TotalAudioSeconds =>
        _fullAudio != null ? (double)_fullAudio.Length / (_audioSampleRate * _audioChannels) : null;

    public List<TranscriptEditorCardState.SpeakerChoice> CreateSpeakerChoices(int selectedSpeakerId)
    {
        return AllSpeakers
            .Select(s => new TranscriptEditorCardState.SpeakerChoice(
                s.SpeakerId,
                string.IsNullOrWhiteSpace(s.Name) ? $"speaker_{s.SpeakerId - 1}" : s.Name))
            .OrderByDescending(s => s.SpeakerId == selectedSpeakerId)
            .ThenBy(s => s.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public void RefreshCardState(TranscriptEditorCardState card, bool preserveDrafts,
        int totalCardCount, bool redoAsrRunning, bool asrModelsAvailable, bool hasVocab)
    {
        var seg = card.Segment;
        card.RefreshFromSegment(
            preserveDrafts,
            CreateSpeakerChoices(seg.SpeakerId),
            Loc.Instance["editor_suppress"],
            Loc.Instance["editor_unsuppress"],
            canMergePrev: !redoAsrRunning && card.Index > 0,
            canMergeNext: !redoAsrRunning && card.Index < totalCardCount - 1,
            canSplit: !redoAsrRunning && hasVocab && seg.Tokens.Count > 1,
            canRedoAsr: !redoAsrRunning && asrModelsAvailable && HasAudio,
            canAdjustTimes: !redoAsrRunning);
    }

    public void RefreshAdjacentCardHighlighting(TranscriptEditorCardState card, VocabService? vocab,
        Color confidenceLowColor, Color greenColor, Color redColor, IBrush subtextBrush, IBrush textBrush)
    {
        var seg = card.Segment;
        if (seg.IsSuppressed)
        {
            card.SetAdjacentPlainText(
                seg.Content,
                new SolidColorBrush(Color.FromArgb(30, redColor.R, redColor.G, redColor.B)),
                subtextBrush,
                TextDecorations.Strikethrough);
            return;
        }

        if (seg.Verified)
        {
            card.SetAdjacentPlainText(
                seg.Content,
                new SolidColorBrush(Color.FromArgb(40, greenColor.R, greenColor.G, greenColor.B)),
                textBrush);
            return;
        }

        if (seg.Content != seg.AsrContent)
        {
            card.SetAdjacentPlainText(seg.Content, Brushes.Transparent, textBrush);
            return;
        }

        if (vocab == null || seg.Tokens.Count == 0)
        {
            card.SetAdjacentPlainText(seg.Content, Brushes.Transparent, subtextBrush);
            return;
        }

        var runs = vocab.GetTokenRuns(seg.Tokens, seg.Logprobs)
            .Select(r => (
                r.text,
                VocabService.GetConfidenceHighlight(r.logprob, confidenceLowColor)))
            .ToList();

        card.AdjacentBackground = Brushes.Transparent;
        card.RebuildAdjacentRuns(runs, subtextBrush);
    }

    public void RefreshFocusedCardAppearance(TranscriptEditorCardState card,
        IBrush surfaceBrush, IBrush accentBrush, IBrush greenBrush, IBrush redBrush,
        Color greenColor, Color redColor, IBrush textBrush)
    {
        Color surfaceColor = surfaceBrush is ISolidColorBrush sb ? sb.Color : Colors.Transparent;
        var seg = card.Segment;
        if (seg.IsSuppressed)
        {
            card.SetFocusedAppearance(
                new SolidColorBrush(Blend(surfaceColor, redColor, 0.18)),
                redBrush,
                redBrush);
            return;
        }

        if (seg.Verified)
        {
            card.SetFocusedAppearance(
                new SolidColorBrush(Blend(surfaceColor, greenColor, 0.18)),
                greenBrush,
                textBrush);
            return;
        }

        card.SetFocusedAppearance(surfaceBrush, accentBrush, textBrush);
    }

    private static Color Blend(Color baseColor, Color tintColor, double tintAmount)
    {
        byte BlendChannel(byte a, byte b) => (byte)Math.Clamp(
            (int)Math.Round(a * (1.0 - tintAmount) + b * tintAmount), 0, 255);

        return Color.FromArgb(
            255,
            BlendChannel(baseColor.R, tintColor.R),
            BlendChannel(baseColor.G, tintColor.G),
            BlendChannel(baseColor.B, tintColor.B));
    }

    public void RefreshFocusedCardAsrHighlighting(TranscriptEditorCardState card, VocabService? vocab,
        Color confidenceLowColor, Color accentColor, IBrush textBrush, int highlightedToken)
    {
        if (vocab == null || card.Segment.Tokens.Count == 0)
        {
            card.RebuildAsrRuns([(card.Segment.AsrContent, Colors.Transparent)], textBrush);
            return;
        }

        var runs = vocab.GetTokenRuns(card.Segment.Tokens, card.Segment.Logprobs)
            .Select(r => (
                r.text,
                VocabService.GetConfidenceHighlight(r.logprob, confidenceLowColor)))
            .ToList();

        card.RebuildAsrRuns(runs, textBrush);
        card.ApplyHighlightedAsrToken(highlightedToken, accentColor);
    }

    public void UpdateFocusedCardTokenHighlight(TranscriptEditorCardState card, Color accentColor, int highlightedToken)
        => card.ApplyHighlightedAsrToken(highlightedToken, accentColor);

    /// <summary>
    /// Re-runs ASR on the segment's audio and replaces all its sources with a single fresh result.
    /// <para>MUST be called from a background thread — loads and runs the ONNX model.</para>
    /// Returns (newResultId, asrContent, tokens, timestamps, logprobs) on success, or null.
    /// </summary>
    public (int newResultId, string asrContent, List<int> tokens, List<int> timestamps, List<float> logprobs)?
        PerformRedoAsr(
            int index,
            string asrModel,
            string parakeetModelsDir,
            string encoderFile,
            string decoderJointFile,
            string? cohereModelsDir = null,
            string? cohereLanguage = null)
    {
        if (index < 0 || index >= Segments.Count || _dbPath is null || _fullAudio is null)
            return null;

        var seg = Segments[index];

        // Extract the segment's audio slice at the original rate, then convert to 16 kHz mono
        int startSample = (int)(seg.PlayStart * _audioSampleRate) * _audioChannels;
        int endSample   = (int)(seg.PlayEnd   * _audioSampleRate) * _audioChannels;
        startSample = Math.Clamp(startSample, 0, _fullAudio.Length);
        endSample   = Math.Clamp(endSample,   0, _fullAudio.Length);
        if (endSample <= startSample) return null;

        var slice = new float[endSample - startSample];
        Array.Copy(_fullAudio, startSample, slice, 0, slice.Length);
        float[] mono16k = AudioUtils.AudioTo16000Mono(slice, _audioSampleRate, _audioChannels);

        var asrSeg = new List<(double start, double end, string spk)>
        {
            (0.0, seg.PlayEnd - seg.PlayStart, $"speaker_{seg.SpeakerId - 1}")
        };

        string text = "";
        List<int> tokens = [];
        List<int> timestamps = [];
        List<float> logprobs = [];
        string? language = null;
        string? emotion = null;
        string? asrMeta = null;

        if (string.Equals(asrModel, "CohereLabs/cohere-transcribe-03-2026", StringComparison.Ordinal))
        {
            if (string.IsNullOrWhiteSpace(cohereModelsDir))
                return null;

            using var cohere = new CohereTranscribe(cohereModelsDir);
            foreach (var result in cohere.RecognizeDetailed(asrSeg, mono16k, forceLanguage: cohereLanguage))
            {
                text = result.Text;
                tokens = result.TextTokens.ToList();
                timestamps = BuildSyntheticTokenTimestamps(seg.PlayEnd - seg.PlayStart, result.TextTokens.Count);
                logprobs = result.TextLogprobs.ToList();
                language = result.Meta.Language;
                emotion = result.Meta.Emotion;
                asrMeta = result.Meta.ToJson(
                    rawDecoderTokens: result.RawTokens,
                    storedTextTokens: result.TextTokens,
                    syntheticTimestamps: true,
                    timestampMode: CohereSyntheticTimestampMode);
            }
        }
        else
        {
            // Run ASR — the slice starts at t=0, so pass a 0-based time range
            using var parakeet = new ParakeetAsr(parakeetModelsDir, encoderFile, decoderJointFile);
            foreach (var (_, t, tk, ts, lp) in parakeet.Recognize(asrSeg, mono16k))
            {
                text = t; tokens = tk; timestamps = ts; logprobs = lp;
            }
        }

        // Persist: insert new result, swap card sources, clean up orphaned old results
        var oldResultIds = seg.Sources.Select(s => s.ResultId).Distinct().ToList();
        int newResultId;
        using (var db = new TranscriptionDb(_dbPath))
        {
            newResultId = db.InsertResultReturningId(
                seg.SpeakerId, seg.PlayStart, seg.PlayEnd,
                asrContent: text,
                tokens:     JsonSerializer.Serialize(tokens),
                timestamps: JsonSerializer.Serialize(timestamps),
                logprobs:   JsonSerializer.Serialize(logprobs),
                language:   language,
                emotion:    emotion,
                asrMeta:    asrMeta);

            db.DeleteCardSources(seg.CardId);
            db.AddCardSource(seg.CardId, newResultId, 0, null, 0, 0);
            db.UpdateCardContent(seg.CardId, null); // clear any user override so new ASR text shows
            db.DeleteOrphanedResults(oldResultIds);
        }

        return (newResultId, text, tokens, timestamps, logprobs);
    }

    private const string CohereSyntheticTimestampMode = "uniform_segment_frames_v1";

    private static List<int> BuildSyntheticTokenTimestamps(double segmentDurationSeconds, int tokenCount)
    {
        if (tokenCount <= 0)
            return [];

        const double frameSeconds = Config.HopLength * 8.0 / Config.SampleRate;
        int maxFrame = Math.Max((int)Math.Round(segmentDurationSeconds / frameSeconds), 0);
        if (tokenCount == 1)
            return [0];

        var timestamps = new List<int>(tokenCount);
        for (int i = 0; i < tokenCount; i++)
        {
            double fraction = i / (double)(tokenCount - 1);
            int frame = (int)Math.Round(fraction * maxFrame);
            if (timestamps.Count > 0 && frame < timestamps[^1])
                frame = timestamps[^1];
            timestamps.Add(frame);
        }

        return timestamps;
    }

    /// <summary>
    /// Applies redo-ASR results to the in-memory segment. Call on the UI thread after
    /// <see cref="PerformRedoAsr"/> completes.
    /// </summary>
    public void ApplyRedoAsr(int index, int newResultId, string asrContent,
        List<int> tokens, List<int> timestamps, List<float> logprobs)
    {
        if (index < 0 || index >= Segments.Count) return;
        var seg = Segments[index];
        seg.AsrContent = asrContent;
        seg.Content    = asrContent;
        seg.Tokens     = tokens;
        seg.Timestamps = timestamps;
        seg.Logprobs   = logprobs;
        seg.Sources    = new List<CardSource>
        {
            new CardSource(newResultId, 0, null, 0, 0)
        };
    }

    /// <summary>
    /// Splits Segments[index] before token splitIndex.
    /// Creates two new overlay cards; the results table is not modified.
    /// Only supported for single-source cards.
    /// </summary>
    public bool SplitSegment(int index, int splitIndex, Services.VocabService? vocab)
    {
        if (index < 0 || index >= Segments.Count || _dbPath is null) return false;
        var seg = Segments[index];
        if (splitIndex <= 0 || splitIndex >= seg.Tokens.Count) return false;
        if (seg.Sources.Count != 1) return false; // multi-source splits not yet supported
        StopPlayback();

        var src = seg.Sources[0];
        const double frameSeconds = Config.HopLength * 8.0 / Config.SampleRate;

        // Split time is where the second half starts
        double splitOffset = seg.Timestamps.Count > splitIndex
            ? seg.Timestamps[splitIndex] * frameSeconds
            : (seg.PlayEnd - seg.PlayStart) / 2.0;
        double splitTime   = seg.PlayStart + splitOffset;

        // In-memory token arrays for each half
        var firstTokens  = seg.Tokens.Take(splitIndex).ToList();
        var secondTokens = seg.Tokens.Skip(splitIndex).ToList();
        var firstTs      = seg.Timestamps.Take(splitIndex).ToList();
        // Second card's timestamps are made relative to the split point
        int tAtSplit     = seg.Timestamps.Count > splitIndex ? seg.Timestamps[splitIndex] : 0;
        var secondTs     = seg.Timestamps.Skip(splitIndex).Select(t => t - tAtSplit).ToList();
        var firstLp      = seg.Logprobs.Take(splitIndex).ToList();
        var secondLp     = seg.Logprobs.Skip(splitIndex).ToList();

        // Decoded content for each half
        string firstAsr, secondAsr, firstCon, secondCon;
        if (vocab != null && firstTokens.Count > 0 && secondTokens.Count > 0)
        {
            firstAsr  = vocab.DecodeTokens(firstTokens);
            secondAsr = vocab.DecodeTokens(secondTokens);
            firstCon  = firstAsr;
            secondCon = secondAsr;
        }
        else
        {
            int charSplit = Math.Clamp(
                (int)((double)splitIndex / seg.Tokens.Count * seg.Content.Length),
                0, seg.Content.Length);
            while (charSplit > 0 && charSplit < seg.Content.Length && seg.Content[charSplit - 1] != ' ')
                charSplit--;
            firstAsr  = seg.AsrContent[..Math.Min(charSplit, seg.AsrContent.Length)].TrimEnd();
            secondAsr = charSplit < seg.AsrContent.Length ? seg.AsrContent[charSplit..].TrimStart() : "";
            firstCon  = seg.Content[..charSplit].TrimEnd();
            secondCon = charSplit < seg.Content.Length ? seg.Content[charSplit..].TrimStart() : "";
        }

        // Absolute token indices within the source result
        int absFirstTEnd    = src.TokenStart + splitIndex;
        int absSecondTStart = absFirstTEnd;

        // ts_frame_offset for second card: makes second timestamps relative to splitTime
        // seg.Timestamps[splitIndex] is already relative to seg.PlayStart, which equals
        // src.TsFrameOffset + srcTimestamps[splitIndex]. We need the second card's offset
        // such that srcTimestamps[absSecondTStart] + secondTsOffset = 0 (relative to splitTime).
        int secondTsOffset = src.TsFrameOffset - tAtSplit;

        double origPlayEnd = seg.PlayEnd;

        int newFirstCardId, newSecondCardId;
        using (var db = new TranscriptionDb(_dbPath))
        {
            newFirstCardId = db.CreateCard(seg.SpeakerId, seg.PlayStart, splitTime);
            db.AddCardSource(newFirstCardId, src.ResultId, src.TokenStart, absFirstTEnd,
                             src.TsFrameOffset, 0);
            db.UpdateCardContent(newFirstCardId, firstCon);

            newSecondCardId = db.CreateCard(seg.SpeakerId, splitTime, origPlayEnd);
            db.AddCardSource(newSecondCardId, src.ResultId, absSecondTStart, src.TokenEnd,
                             secondTsOffset, 0);
            db.UpdateCardContent(newSecondCardId, secondCon);

            db.DeleteCard(seg.CardId);
        }

        // Update first half in-memory
        seg.CardId     = newFirstCardId;
        seg.PlayEnd    = splitTime;
        seg.AsrContent = firstAsr;
        seg.Content    = firstCon;
        seg.Tokens     = firstTokens;
        seg.Timestamps = firstTs;
        seg.Logprobs   = firstLp;
        seg.Sources    = new List<CardSource>
        {
            new CardSource(src.ResultId, src.TokenStart, absFirstTEnd, src.TsFrameOffset, 0)
        };

        // Insert second half
        Segments.Insert(index + 1, new EditorSegment
        {
            CardId             = newSecondCardId,
            SpeakerId          = seg.SpeakerId,
            SpeakerDisplayName = seg.SpeakerDisplayName,
            PlayStart          = splitTime,
            PlayEnd            = origPlayEnd,
            AsrContent         = secondAsr,
            Content            = secondCon,
            Tokens             = secondTokens,
            Timestamps         = secondTs,
            Logprobs           = secondLp,
            Sources            = new List<CardSource>
            {
                new CardSource(src.ResultId, absSecondTStart, src.TokenEnd, secondTsOffset, 0)
            },
        });

        // Renumber sort orders
        if (_dbPath != null)
            using (var db = new TranscriptionDb(_dbPath))
                db.RenumberSortOrders(Segments.Select(s => s.CardId));

        return true;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private bool StartFfplayPlayback(double startSec, double? durationSec = null, TimeSpan? leadingSilence = null)
    {
        if (FfplayPath is null || string.IsNullOrWhiteSpace(_audioPath) || !File.Exists(_audioPath))
            return false;

        var psi = new ProcessStartInfo(FfplayPath)
        {
            UseShellExecute        = false,
            RedirectStandardError  = true,
            RedirectStandardOutput = true,
            CreateNoWindow         = true,
        };

        psi.ArgumentList.Add("-nodisp");
        psi.ArgumentList.Add("-autoexit");
        psi.ArgumentList.Add("-loglevel");
        psi.ArgumentList.Add("error");
        psi.ArgumentList.Add("-ss");
        psi.ArgumentList.Add(startSec.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture));

        if (durationSec is > 0)
        {
            psi.ArgumentList.Add("-t");
            psi.ArgumentList.Add(durationSec.Value.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture));
        }

        string? audioFilter = BuildPlaybackAudioFilter(PlaybackSpeed, leadingSilence ?? TimeSpan.Zero);
        if (!string.IsNullOrEmpty(audioFilter))
        {
            psi.ArgumentList.Add("-af");
            psi.ArgumentList.Add(audioFilter);
        }

        psi.ArgumentList.Add(_audioPath);

        var process = new Process
        {
            StartInfo = psi,
            EnableRaisingEvents = true,
        };

        process.Exited += OnFfplayExited;
        if (!process.Start())
        {
            process.Exited -= OnFfplayExited;
            process.Dispose();
            return false;
        }

        _ = process.StandardOutput.ReadToEndAsync();
        _ = process.StandardError.ReadToEndAsync();
        _playbackProcess = process;
        return true;
    }

    private void StopFfplayPlayback()
    {
        if (_playbackProcess is null)
            return;

        var process = _playbackProcess;
        _playbackProcess = null;
        process.Exited -= OnFfplayExited;
        try
        {
            if (!process.HasExited)
                process.Kill(entireProcessTree: true);
        }
        catch
        {
            // Ignore shutdown races during pause/stop/dispose.
        }
        finally
        {
            process.Dispose();
        }
    }

    private void OnFfplayExited(object? sender, EventArgs e)
    {
        if (sender is not Process process)
            return;

        process.Exited -= OnFfplayExited;
        process.Dispose();

        if (!ReferenceEquals(_playbackProcess, process))
            return;

        _playbackProcess = null;
        Dispatcher.UIThread.InvokeAsync(() =>
        {
            PlaybackPosition = 1.0;
            OnSegmentPlaybackComplete();
        });
    }

    private static string BuildAtempoFilter(double speed)
    {
        var factors = new List<double>();
        double remaining = speed;

        while (remaining < 0.5)
        {
            factors.Add(0.5);
            remaining /= 0.5;
        }

        while (remaining > 2.0)
        {
            factors.Add(2.0);
            remaining /= 2.0;
        }

        factors.Add(remaining);

        return string.Join(",",
            factors.Select(f => $"atempo={f.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture)}"));
    }

    private static string? BuildPlaybackAudioFilter(double speed, TimeSpan leadingSilence)
    {
        var filters = new List<string>();

        if (Math.Abs(speed - 1.0) >= 0.005)
        {
            filters.Add(BuildAtempoFilter(speed));
        }

        if (leadingSilence > TimeSpan.Zero)
        {
            int delayMs = (int)Math.Ceiling(leadingSilence.TotalMilliseconds);
            filters.Add($"adelay={delayMs}:all=1");
        }

        return filters.Count == 0 ? null : string.Join(",", filters);
    }

    private static string? FindExecutable(string fileName)
    {
        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrWhiteSpace(pathEnv))
            return null;

        foreach (string dir in pathEnv.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            string candidate = Path.Combine(dir, fileName);
            if (File.Exists(candidate))
                return candidate;

            if (OperatingSystem.IsWindows())
            {
                string candidateExe = candidate + ".exe";
                if (File.Exists(candidateExe))
                    return candidateExe;
            }
        }

        return null;
    }

    /// <summary>
    /// Streams audio through SoundTouch on-the-fly, so tempo-stretching never
    /// blocks the UI thread. The source array is read directly (no copy).
    /// </summary>
    private sealed class SoundTouchWaveProvider : IWaveProvider
    {
        private readonly float[]              _src;
        private readonly int                  _srcEnd;
        private readonly int                  _channels;
        private readonly SoundTouchProcessor? _st;
        private readonly int                  _leadingSilenceFloats;
        private          int                  _srcPos;
        private          bool                 _flushed;
        private          int                  _leadingSilenceFloatsRemaining;
        private readonly Queue<float>         _outQ = new();

        public WaveFormat WaveFormat { get; }

        public SoundTouchWaveProvider(float[] src, int offset, int length,
                                      WaveFormat waveFormat, int channels, double speed,
                                      TimeSpan leadingSilence)
        {
            _src      = src;
            _srcPos   = offset;
            _srcEnd   = offset + length;
            _channels = channels;
            WaveFormat = waveFormat;
            _leadingSilenceFloats = (int)Math.Ceiling(
                leadingSilence.TotalSeconds * waveFormat.SampleRate * channels);
            _leadingSilenceFloatsRemaining = _leadingSilenceFloats;

            if (Math.Abs(speed - 1.0) >= 0.005)
            {
                _st            = new SoundTouchProcessor();
                _st.SampleRate = waveFormat.SampleRate;
                _st.Channels   = channels;
                _st.Tempo      = speed;
            }
        }

        public int Read(byte[] buffer, int offset, int count)
        {
            if (_leadingSilenceFloatsRemaining > 0)
            {
                int silenceFloats = Math.Min(count / 4, _leadingSilenceFloatsRemaining);
                buffer.AsSpan(offset, silenceFloats * 4).Clear();
                _leadingSilenceFloatsRemaining -= silenceFloats;
                return silenceFloats * 4;
            }

            if (_st is null)
            {
                // 1.0× — copy raw bytes directly, no processing needed
                int avail = (_srcEnd - _srcPos) * 4;
                int copy  = Math.Min(count, avail);
                if (copy <= 0) return 0;
                Buffer.BlockCopy(_src, _srcPos * 4, buffer, offset, copy);
                _srcPos += copy / 4;
                return copy;
            }

            int floatsNeeded = count / 4;
            const int FeedFrames = 4096;

            while (_outQ.Count < floatsNeeded)
            {
                if (_srcPos < _srcEnd)
                {
                    int frames = Math.Min(FeedFrames, (_srcEnd - _srcPos) / _channels);
                    if (frames <= 0) break;
                    var chunk = new float[frames * _channels];
                    Array.Copy(_src, _srcPos, chunk, 0, chunk.Length);
                    _st.PutSamples(chunk, frames);
                    _srcPos += chunk.Length;
                    Drain();
                }
                else
                {
                    if (!_flushed) { _st.Flush(); _flushed = true; Drain(); }
                    break;
                }
            }

            int toWrite = Math.Min(floatsNeeded, _outQ.Count);
            var outSpan = MemoryMarshal.Cast<byte, float>(buffer.AsSpan(offset, toWrite * 4));
            for (int i = 0; i < toWrite; i++)
                outSpan[i] = _outQ.Dequeue();
            return toWrite * 4;
        }

        private void Drain()
        {
            var recv = new float[4096 * _channels];
            int n;
            while ((n = _st!.ReceiveSamples(recv, recv.Length / _channels)) > 0)
                for (int i = 0; i < n * _channels; i++)
                    _outQ.Enqueue(recv[i]);
        }
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose()
    {
        StopPlayback();
        _fullAudio = null;
    }
}
