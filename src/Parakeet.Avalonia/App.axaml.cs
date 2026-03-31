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

    public static void Main(string[] args) => BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
        // Theme resources and styles are loaded via App.axaml
    }

    public override void OnFrameworkInitializationCompleted()
    {
        base.OnFrameworkInitializationCompleted();

        Settings.Load();
        ThemeManager.Apply(Settings.Current.Theme);

        // If no language preference has been saved yet, try to match the system
        // UI language; fall back to English if no supported locale matches.
        // Persist the resolved value so subsequent launches skip detection.
        var lang = Settings.Current.Language;
        if (string.IsNullOrEmpty(lang))
        {
            var systemCode = CultureInfo.CurrentUICulture.TwoLetterISOLanguageName;
            lang = Loc.Languages.Any(l => l.Code == systemCode) ? systemCode : "en";
            Settings.Current.Language = lang;
            Settings.Save();
        }
        Loc.Instance.SetLanguage(lang);

        FFmpegDecoder.Initialize(AppContext.BaseDirectory);

        ModelManagerService.AddCudaToSearchPath();

        ControlDb     = new ControlDb(Settings.GetControlDbPath());
        ModelManager  = new ModelManagerService(Settings);
        Transcription = new TranscriptionService(Settings);
        JobQueue      = new JobQueueService(Transcription, ControlDb, Settings);

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var mainVm = new MainViewModel(Settings, ControlDb, ModelManager, Transcription, JobQueue, Export);
            desktop.MainWindow = new MainWindow { DataContext = mainVm };
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
