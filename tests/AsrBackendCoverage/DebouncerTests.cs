using Vernacula.App.Services;
using Xunit;

namespace AsrBackendCoverage;

public class DebouncerTests
{
    [Fact]
    public async Task ItRunsOnceAfterTheBumpsStop()
    {
        int runs = 0;
        using var d = new Debouncer(TimeSpan.FromMilliseconds(60), _ => { runs++; return Task.CompletedTask; });
        for (int i = 0; i < 5; i++) { d.Bump(); await Task.Delay(10); }
        await Task.Delay(300);
        Assert.Equal(1, runs);
    }

    /// <summary>⚠ Typing DURING a run is the normal case. The change must not be lost, and the action
    /// must not be entered twice at once.</summary>
    [Fact]
    public async Task ABumpDuringARunReArmsInsteadOfBeingDropped()
    {
        int runs = 0, concurrent = 0, maxConcurrent = 0;
        var started = new TaskCompletionSource();
        using var d = new Debouncer(TimeSpan.FromMilliseconds(40), async _ =>
        {
            maxConcurrent = Math.Max(maxConcurrent, Interlocked.Increment(ref concurrent));
            if (Interlocked.Increment(ref runs) == 1) started.TrySetResult();
            await Task.Delay(120);
            Interlocked.Decrement(ref concurrent);
        });

        d.Bump();
        await started.Task;     // first run is in flight
        d.Bump();               // …and the user types again
        await Task.Delay(500);

        Assert.Equal(2, runs);
        Assert.Equal(1, maxConcurrent);
    }

    [Fact]
    public async Task CancelDropsThePendingRun()
    {
        int runs = 0;
        using var d = new Debouncer(TimeSpan.FromMilliseconds(60), _ => { runs++; return Task.CompletedTask; });
        d.Bump();
        d.Cancel();
        await Task.Delay(200);
        Assert.Equal(0, runs);
    }

    /// <summary>"Save before closing": whatever is pending happens now, and the caller can await it.</summary>
    [Fact]
    public async Task FlushRunsThePendingWorkImmediately()
    {
        int runs = 0;
        using var d = new Debouncer(TimeSpan.FromSeconds(30), _ => { runs++; return Task.CompletedTask; });
        d.Bump();
        await d.FlushAsync();
        Assert.Equal(1, runs);
    }

    [Fact]
    public async Task FlushWithNothingPendingDoesNothing()
    {
        int runs = 0;
        using var d = new Debouncer(TimeSpan.FromSeconds(30), _ => { runs++; return Task.CompletedTask; });
        await d.FlushAsync();
        Assert.Equal(0, runs);
    }

    /// <summary>An action that throws must not take the app down, and must not wedge the debouncer.</summary>
    [Fact]
    public async Task AThrowingActionIsReportedAndTheDebouncerKeepsWorking()
    {
        Exception? seen = null;
        int runs = 0;
        using var d = new Debouncer(TimeSpan.FromMilliseconds(40),
            _ => { runs++; throw new InvalidOperationException("boom"); },
            ex => seen = ex);
        d.Bump();
        await Task.Delay(250);
        Assert.IsType<InvalidOperationException>(seen);
        d.Bump();
        await Task.Delay(250);
        Assert.Equal(2, runs);
    }
}
