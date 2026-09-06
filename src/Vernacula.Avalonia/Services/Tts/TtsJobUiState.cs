using Vernacula.App.Models;

namespace Vernacula.App.Services.Tts;

// ── Actions ───────────────────────────────────────────────────────────────────

internal abstract record TtsJobUiAction;

internal record TtsChunkProducedAction(ChunkProducedEvent Chunk) : TtsJobUiAction;

internal record TtsProgressAction(ProgressEvent Progress) : TtsJobUiAction;

// ── State ─────────────────────────────────────────────────────────────────────

/// <summary>
/// Reducer-style state for one active text-to-speech job — the TTS counterpart of
/// <see cref="JobUiState"/>. Accumulates every chunk the backend has produced (audio + word
/// timings) so the reader panel can attach to a job mid-render, play what exists, and keep
/// receiving chunks live. All mutations arrive via <see cref="Dispatch"/> on the worker
/// thread; the single subscriber marshals to the UI thread itself.
/// <para>
/// The audio is held in memory only while the job runs (24 kHz mono float32 ≈ 5.8 MB per
/// minute); once the job completes the WAV on disk is the source and this state is dropped.
/// </para>
/// </summary>
internal sealed class TtsJobUiState
{
    private readonly object                    _lock     = new();
    private readonly List<ChunkProducedEvent>  _chunks   = new();
    private ProgressEvent?                     _lastProgress;
    private Action<TtsJobUiAction>?            _subscriber;

    public int SampleRate { get; }

    public TtsJobUiState(int sampleRate) => SampleRate = sampleRate;

    // ── Read-only accessors (safe on any thread) ──────────────────────────────

    public ProgressEvent? LastProgress { get { lock (_lock) return _lastProgress; } }

    /// <summary>Chunks done / total, as a percentage; 0 until the total is known.</summary>
    public double Percent
    {
        get
        {
            lock (_lock)
            {
                int total = _lastProgress?.TotalChunks ?? 0;
                return total > 0 ? Math.Min(100.0, 100.0 * _chunks.Count / total) : 0;
            }
        }
    }

    // ── Dispatch (called from worker thread) ──────────────────────────────────

    public void Dispatch(TtsJobUiAction action)
    {
        Action<TtsJobUiAction>? sub;
        lock (_lock)
        {
            Apply(action);
            sub = _subscriber;
        }
        sub?.Invoke(action);   // outside lock — subscriber does its own thread marshalling
    }

    // ── Subscribe / unsubscribe (called from UI thread) ───────────────────────

    /// <summary>
    /// Atomically snapshots the chunks received so far and registers
    /// <paramref name="subscriber"/> for all future actions, so no chunk can be lost in the
    /// gap between reading the snapshot and going live.
    /// </summary>
    public TtsJobUiSnapshot Subscribe(Action<TtsJobUiAction> subscriber)
    {
        lock (_lock)
        {
            _subscriber = subscriber;
            return new TtsJobUiSnapshot(_chunks.ToList(), _lastProgress, SampleRate);
        }
    }

    public void Unsubscribe()
    {
        lock (_lock) _subscriber = null;
    }

    // ── Private reducer ───────────────────────────────────────────────────────

    private void Apply(TtsJobUiAction action)
    {
        switch (action)
        {
            case TtsChunkProducedAction a:
                // Chunks arrive in input order; a replay of an index already held is dropped.
                if (a.Chunk.ChunkIndex >= _chunks.Count)
                    _chunks.Add(a.Chunk);
                break;

            case TtsProgressAction a:
                _lastProgress = a.Progress;
                break;
        }
    }
}

/// <summary>Immutable snapshot returned by <see cref="TtsJobUiState.Subscribe"/>.</summary>
internal record TtsJobUiSnapshot(
    IReadOnlyList<ChunkProducedEvent> Chunks,
    ProgressEvent?                    LastProgress,
    int                               SampleRate);
