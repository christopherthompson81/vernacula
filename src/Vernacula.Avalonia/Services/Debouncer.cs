namespace Vernacula.App.Services;

/// <summary>
/// "Do this once the user has stopped for N seconds." Each <see cref="Bump"/> restarts the clock;
/// the action runs on a background thread when the clock finally runs out.
///
/// <para>⚠ ONE RUN AT A TIME, AND A BUMP DURING A RUN RE-ARMS. Typing while a re-render is under way
/// is the normal case, not the edge case, so the action is never entered twice concurrently and a
/// change that lands mid-run is not lost — it simply starts the clock again once the run finishes.</para>
/// </summary>
internal sealed class Debouncer : IDisposable
{
    private readonly TimeSpan _delay;
    private readonly Func<CancellationToken, Task> _action;
    private readonly Action<Exception>? _onError;
    private readonly object _gate = new();

    private CancellationTokenSource? _pending;
    private bool _running;
    private bool _bumpedDuringRun;
    private bool _disposed;

    public Debouncer(TimeSpan delay, Func<CancellationToken, Task> action, Action<Exception>? onError = null)
    {
        _delay = delay;
        _action = action;
        _onError = onError;
    }

    /// <summary>True while the clock is running — what an "unsaved / pending" indicator binds to.</summary>
    public bool IsPending { get { lock (_gate) return _pending is not null; } }

    /// <summary>Restart the clock.</summary>
    public void Bump()
    {
        lock (_gate)
        {
            if (_disposed) return;
            if (_running) { _bumpedDuringRun = true; return; }
            _pending?.Cancel();
            _pending?.Dispose();
            _pending = new CancellationTokenSource();
            _ = WaitThenRun(_pending.Token);
        }
    }

    /// <summary>Drop a pending run without performing it.</summary>
    public void Cancel()
    {
        lock (_gate)
        {
            _pending?.Cancel();
            _pending?.Dispose();
            _pending = null;
        }
    }

    /// <summary>Run now if anything is pending, and wait for it. For "save before closing".</summary>
    public async Task FlushAsync()
    {
        bool due;
        lock (_gate) { due = _pending is not null; Cancel(); }
        if (due) await RunOnce(CancellationToken.None).ConfigureAwait(false);
    }

    private async Task WaitThenRun(CancellationToken ct)
    {
        try { await Task.Delay(_delay, ct).ConfigureAwait(false); }
        catch (OperationCanceledException) { return; }

        lock (_gate)
        {
            if (_disposed || ct.IsCancellationRequested) return;
            _pending?.Dispose();
            _pending = null;
            _running = true;
            _bumpedDuringRun = false;
        }

        await RunOnce(ct).ConfigureAwait(false);

        bool again;
        lock (_gate) { _running = false; again = _bumpedDuringRun && !_disposed; }
        if (again) Bump();
    }

    private async Task RunOnce(CancellationToken ct)
    {
        try { await _action(ct).ConfigureAwait(false); }
        catch (OperationCanceledException) { }
        catch (Exception ex) { _onError?.Invoke(ex); }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
            _pending?.Cancel();
            _pending?.Dispose();
            _pending = null;
        }
    }
}
