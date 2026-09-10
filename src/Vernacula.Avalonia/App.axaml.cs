using System.Globalization;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Vernacula.App.ViewModels;
using Vernacula.App.Views;
using Vernacula.Base;

namespace Vernacula.App;

public partial class App : Application
{
    private bool _servicesDisposed;

    public new static App Current => (App)Application.Current!;

    internal SettingsService      Settings      { get; } = new();
    internal ControlDb            ControlDb     { get; private set; } = null!;
    internal ModelManagerService  ModelManager  { get; private set; } = null!;
    internal LangIdService        LangId        { get; private set; } = null!;
    internal TranscriptionService Transcription { get; private set; } = null!;
    internal TtsJobRunner         TtsRunner     { get; private set; } = null!;
    internal JobQueueService      JobQueue      { get; private set; } = null!;
    internal ExportService        Export        { get; } = new();

    private static volatile bool s_shuttingDown;
    private static volatile int s_exitCode;

    /// <summary>
    /// Called when the desktop lifetime is actually exiting -- not merely when a shutdown is
    /// requested, since such a request can still be cancelled and the app go on running. After
    /// this point the dispatcher stops accepting work, so anything still trying to marshal to
    /// the UI thread will fail.
    /// </summary>
    private static void BeginShutdown(int exitCode)
    {
        s_exitCode = exitCode;
        s_shuttingDown = true;
    }

    public static int Main(string[] args)
    {
        SetupGlobalExceptionHandlers();
        var exitCode = BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        s_exitCode = exitCode;
        Console.WriteLine("[App] StartWithClassicDesktopLifetime returned");
        // The UI is gone and the exit code is settled; leave before any background teardown
        // (D-Bus readers, ONNX session finalizers) can raise on a thread we do not own.
        Console.Out.Flush();
        Environment.Exit(exitCode);
        return exitCode;
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();

    private static void SetupGlobalExceptionHandlers()
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
            if (s_shuttingDown && e.ExceptionObject is Exception ex && IsDBusTeardown(ex))
            {
                Console.Error.WriteLine($"[shutdown] ignoring D-Bus teardown error: {ex.GetType().Name}: {ex.Message}");
                // Exit with the code the run had already settled on: a failing run must not be
                // reported as a success just because teardown raised on the way out.
                Environment.Exit(s_exitCode);
            }
            Console.WriteLine($"[UNHANDLED] AppDomain exception: {e.ExceptionObject}");
        };

        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            if (IsIgnorableLinuxDesktopIntegrationException(e.Exception))
            {
                e.SetObserved();
                Console.WriteLine($"[INFO] Ignored desktop integration exception: {e.Exception.GetBaseException().Message}");
                return;
            }

            Console.WriteLine($"[UNHANDLED] Unobserved task exception: {e.Exception}");
        };
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

    private static bool IsIgnorableLinuxDesktopIntegrationException(Exception ex)
    {
        string text = ex.ToString();
        return text.Contains("com.canonical.AppMenu.Registrar", StringComparison.Ordinal)
               && text.Contains("org.freedesktop.DBus.Error.ServiceUnknown", StringComparison.Ordinal);
    }

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        base.OnFrameworkInitializationCompleted();

        try
        {
            Settings.Load();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Failed to load settings: {ex}");
        }

        // Initialize localization BEFORE creating any windows
        var lang = Settings.Current.Language;
        if (string.IsNullOrEmpty(lang))
        {
            var systemCode = CultureInfo.CurrentUICulture.TwoLetterISOLanguageName;
            lang = Loc.Languages.Any(l => l.Code == systemCode) ? systemCode : "en";
            Settings.Current.Language = lang;
            Settings.Save();
        }
        
        try
        {
            Loc.Instance.SetLanguage(lang);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Failed to set language {lang}: {ex}");
            Loc.Instance.SetLanguage("en");
        }

        ThemeManager.Apply(Settings.Current.Theme);

        FFmpegDecoder.Initialize(AppContext.BaseDirectory);

        ModelManagerService.AddCudaToSearchPath();

        ControlDb     = new ControlDb(Settings.GetControlDbPath());
        ModelManager  = new ModelManagerService(Settings);
        LangId        = new LangIdService(Settings);
        Transcription = new TranscriptionService(Settings, LangId);
        TtsRunner     = new TtsJobRunner(Settings);
        JobQueue      = new JobQueueService(Transcription, TtsRunner, ControlDb, Settings);

        // Reclaim what a previous version left behind — retired model files and, far more
        // importantly, their compiled CoreML bundles at ~4.4 GB each. Nothing else reclaims
        // those once a model stops being used, and the affected users already have every
        // model on disk, so they would never hit the download path that also does this.
        _ = ModelManager.RemoveRetiredAssetsAsync();

        // Warm up Sortformer model on a background thread so the first
        // transcription starts without the usual ONNX Runtime initialisation
        // delay (graph optimisation, CUDA/DML provider setup, memory alloc).
        Task.Run(() =>
        {
            try
            {
                var sortformerDir = Settings.GetSortformerModelsDir();
                if (Directory.Exists(sortformerDir))
                {
                    // Same provider the real transcription will use, or the warm-up
                    // compiles a graph for a provider nothing goes on to use.
                    using var streamer = new SortformerStreamer(
                        sortformerDir, Settings.Current.ResolvedExecutionProvider);
                    streamer.Warmup();
                    Console.WriteLine("[App] Sortformer model warmup complete.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[App] Sortformer warmup failed (non-fatal): {ex.Message}");
            }
        });

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.Exit += (_, e) =>
            {
                Console.WriteLine($"[App] Desktop Exit event! ExitCode={e.ApplicationExitCode}");
                // Only Exit arms the guard: a shutdown *request* can be cancelled, and the guard
                // must not stay armed while the app carries on running.
                BeginShutdown(e.ApplicationExitCode);
                DisposeServices();
            };
            var mainVm = new MainViewModel(Settings, ControlDb, ModelManager, Transcription, JobQueue, Export, TtsRunner);
            desktop.MainWindow = new MainWindow { DataContext = mainVm };
            Console.WriteLine("[App] MainWindow set");

            // Wire the LID mismatch popup. TranscriptionService runs its
            // pipeline on a worker thread; the callback marshals to the UI
            // thread, shows a modal dialog owned by the main window, and
            // awaits the user's choice before the worker continues.
            Transcription.OnAsrLanguageMismatch = async (lidResult, currentBackend, suggestedBackend) =>
            {
                Views.Dialogs.AsrMismatchResult? result = null;
                await Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(async () =>
                {
                    var dialog = new Views.Dialogs.AsrMismatchDialog();
                    dialog.Configure(
                        detectedIso:        lidResult.Iso,
                        detectedName:       lidResult.Top.Name,
                        detectedProbability: lidResult.TopProbability,
                        currentBackend:     currentBackend,
                        suggestedBackend:   suggestedBackend);
                    result = await dialog.ShowDialog<Views.Dialogs.AsrMismatchResult?>(desktop.MainWindow!);
                });
                return result;
            };
        }
        else if (ApplicationLifetime is ISingleViewApplicationLifetime singleView)
        {
            // For mobile/SWA platforms
        }
    }

    private void DisposeServices()
    {
        if (_servicesDisposed)
        {
            return;
        }

        _servicesDisposed = true;
        TtsRunner?.Dispose();
        ControlDb?.Dispose();
    }
}
