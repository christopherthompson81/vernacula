using Avalonia;

namespace Vernacula.Tts.App;

/// <summary>
/// Avalonia entry point. The actual UI lives in <see cref="App"/>;
/// this is just the static AppBuilder boilerplate.
/// </summary>
internal static class Program
{
    [System.STAThread]
    public static int Main(string[] args)
        => BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
