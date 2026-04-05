using Vernacula.App.Models;

namespace Vernacula.App.Services;

// ── Actions ───────────────────────────────────────────────────────────────────

internal abstract record JobUiAction;

internal record SegmentAddedAction(
    int    SegmentId,
    string SpeakerTag,
    string SpeakerDisplayName,
    double StartTime,
    double EndTime) : JobUiAction;

internal record SegmentTextUpdatedAction(
    int    SegmentId,
    string Text) : JobUiAction;

internal record ProgressUpdatedAction(
    TranscriptionProgress Progress) : JobUiAction;

// ── State ─────────────────────────────────────────────────────────────────────

/// <summary>
/// Reducer-style state for one active transcription job.
/// <para>
/// All mutations arrive via <see cref="Dispatch"/> on the worker thread.
/// The single subscriber (the progress ViewModel) receives each action after
/// it has been applied, and must marshal to the UI thread itself.
/// </para>
/// <para>
/// Segment data is stored as plain value records — no WPF observable objects —
/// so mutations are safe to make from the worker thread without triggering
/// PropertyChanged notifications on the wrong thread.
/// </para>
/// </summary>
internal sealed class JobUiState
{
    private record SegmentData(
        int    SegmentId,
        string SpeakerTag,
        string SpeakerDisplayName,
        double StartTime,
        double EndTime,
        string Text);

    private readonly object           _lock     = new();
    private readonly List<SegmentData> _segments = new();
    private double                    _percent;
    private TranscriptionProgress?    _lastProgress;
    private Action<JobUiAction>?      _subscriber;

    // ── Read-only accessors (safe on any thread) ──────────────────────────────

    public double                 Percent      { get { lock (_lock) return _percent; } }
    public TranscriptionProgress? LastProgress { get { lock (_lock) return _lastProgress; } }

    // ── Dispatch (called from worker thread) ──────────────────────────────────

    public void Dispatch(JobUiAction action)
    {
        Action<JobUiAction>? sub;
        lock (_lock)
        {
            Apply(action);
            sub = _subscriber;
        }
        sub?.Invoke(action);   // outside lock — subscriber does its own thread marshalling
    }

    // ── Subscribe / unsubscribe (called from UI thread) ───────────────────────

    /// <summary>
    /// Atomically snapshots the current accumulated state and registers
    /// <paramref name="subscriber"/> for all future actions.  No action can
    /// be lost in the gap between reading the snapshot and going live.
    /// </summary>
    public JobUiSnapshot Subscribe(Action<JobUiAction> subscriber)
    {
        lock (_lock)
        {
            _subscriber = subscriber;
            return new JobUiSnapshot(
                _segments.Select(s => new SegmentRow
                {
                    SegmentId          = s.SegmentId,
                    SpeakerTag         = s.SpeakerTag,
                    SpeakerDisplayName = s.SpeakerDisplayName,
                    StartTime          = s.StartTime,
                    EndTime            = s.EndTime,
                    Text               = s.Text,
                }).ToList(),
                _percent,
                _lastProgress);
        }
    }

    public void Unsubscribe()
    {
        lock (_lock) _subscriber = null;
    }

    // ── Private reducer ───────────────────────────────────────────────────────

    private void Apply(JobUiAction action)
    {
        switch (action)
        {
            case SegmentAddedAction a:
                // Guard against duplicate dispatches if the pipeline replays
                // pre-existing segments on resume.
                if (a.SegmentId >= _segments.Count)
                    _segments.Add(new SegmentData(
                        a.SegmentId, a.SpeakerTag, a.SpeakerDisplayName,
                        a.StartTime, a.EndTime, ""));
                break;

            case SegmentTextUpdatedAction a:
                if (a.SegmentId >= 0 && a.SegmentId < _segments.Count)
                {
                    var s = _segments[a.SegmentId];
                    _segments[a.SegmentId] = s with { Text = a.Text };
                }
                break;

            case ProgressUpdatedAction a:
                _percent      = a.Progress.Percent;
                _lastProgress = a.Progress;
                break;
        }
    }
}

/// <summary>Immutable snapshot returned by <see cref="JobUiState.Subscribe"/>.</summary>
internal record JobUiSnapshot(
    IReadOnlyList<SegmentRow>  Segments,
    double                     Percent,
    TranscriptionProgress?     LastProgress);
