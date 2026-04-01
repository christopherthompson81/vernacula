using System.Globalization;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using ParakeetCSharp.Services;
using ParakeetCSharp.ViewModels;
using ParakeetCSharp.Views;

namespace ParakeetCSharp;

public partial class App : Application
{
    public new static App Current => (App)Application.Current!;

    internal SettingsService      Settings      { get; } = new();
    internal ControlDb            ControlDb     { get; private set; } = null!;
    internal ModelManagerService  ModelManager  { get; private set; } = null!;
    internal TranscriptionService Transcription { get; private set; } = null!;
    internal JobQueueService      JobQueue      { get; private set; } = null!;
    internal ExportService        Export        { get; } = new();

    public static void Main(string[] args)
    {
        SetupGlobalExceptionHandlers();
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        Console.WriteLine("[App] StartWithClassicDesktopLifetime returned");
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();

    private static void SetupGlobalExceptionHandlers()
    {
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
            Console.WriteLine($"[UNHANDLED] AppDomain exception: {e.ExceptionObject}");

        TaskScheduler.UnobservedTaskException += (_, e) =>
            Console.WriteLine($"[UNHANDLED] Unobserved task exception: {e.Exception}");
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
        Transcription = new TranscriptionService(Settings);
        JobQueue      = new JobQueueService(Transcription, ControlDb, Settings);

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.Exit += (_, e) => Console.WriteLine($"[App] Desktop Exit event! ExitCode={e.ApplicationExitCode}");
            var mainVm = new MainViewModel(Settings, ControlDb, ModelManager, Transcription, JobQueue, Export);
            desktop.MainWindow = new MainWindow { DataContext = mainVm };
            Console.WriteLine("[App] MainWindow set");
        }
        else if (ApplicationLifetime is ISingleViewApplicationLifetime singleView)
        {
            // For mobile/SWA platforms
        }
    }

    public void OnExit()
    {
        ControlDb?.Dispose();
    }
}
