using System;
using System.Linq;
using Avalonia;

namespace Vernacula.Tts.App;

/// <summary>
/// Avalonia entry point. The actual UI lives in <see cref="App"/>;
/// this is just the static AppBuilder boilerplate, plus the guard that keeps a
/// shutdown-time D-Bus race from turning a normal window close into a crash.
/// </summary>
internal static class Program
{
    private static volatile bool s_shuttingDown;
    private static volatile int s_exitCode;

    /// <summary>
    /// Called when the desktop lifetime is actually exiting -- not merely when a shutdown is
    /// requested, since such a request can still be cancelled and the app go on running. After
    /// this point the dispatcher stops accepting work, so anything still trying to marshal to
    /// the UI thread will fail.
    /// </summary>
    internal static void BeginShutdown(int exitCode)
    {
        s_exitCode = exitCode;
        s_shuttingDown = true;
    }

    [System.STAThread]
    public static int Main(string[] args)
    {
        // Avalonia's FreeDesktop backend keeps a D-Bus connection whose signal observers were
        // subscribed on the UI thread. When that connection drops during exit it reports the
        // disconnect to each observer through the captured (Avalonia) synchronization context;
        // with the dispatcher already shutting down, the Send is cancelled and the resulting
        // TaskCanceledException is rethrown on a thread pool thread, where nothing can catch it.
        // The app is done at that point, so treat it as a clean exit -- but only for exceptions
        // that actually came through the D-Bus stack, and only once shutdown has begun, so real
        // faults are still reported.
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            if (!s_shuttingDown || e.ExceptionObject is not Exception ex || !IsDBusTeardown(ex)) return;
            Console.Error.WriteLine($"[shutdown] ignoring D-Bus teardown error: {ex.GetType().Name}: {ex.Message}");
            // Exit with the code the run had already settled on: a failing run must not be
            // reported as a success just because teardown raised on the way out.
            Environment.Exit(s_exitCode);
        };

        var exitCode = BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        s_exitCode = exitCode;
        // The UI is gone and the exit code is settled; leave before any background teardown
        // (D-Bus readers, ONNX session finalizers) can raise on a thread we do not own.
        Console.Out.Flush();
        Environment.Exit(exitCode);
        return exitCode;
    }

    /// <summary>True when <paramref name="ex"/>, or anything it wraps, was raised inside the
    /// D-Bus stack. Aggregates are flattened: one disconnect can fault several observers at once,
    /// and the D-Bus one need not be first.</summary>
    private static bool IsDBusTeardown(Exception ex)
    {
        if (ex.StackTrace?.Contains("Tmds.DBus", StringComparison.Ordinal) == true) return true;
        if (ex is AggregateException agg)
            return agg.Flatten().InnerExceptions.Any(IsDBusTeardown);
        return ex.InnerException is { } inner && IsDBusTeardown(inner);
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
