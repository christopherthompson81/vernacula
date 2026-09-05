using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Vernacula.Tts.App.ViewModels;
using Vernacula.Tts.App.Views;

namespace Vernacula.Tts.App;

public sealed class App : Application
{
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.MainWindow = new MainWindow
            {
                DataContext = new MainViewModel(),
            };
            // Let Program know the dispatcher is on its way out, so a D-Bus disconnect racing
            // the teardown is treated as a clean exit rather than an unhandled crash.
            desktop.ShutdownRequested += (_, _) => Program.BeginShutdown();
            desktop.Exit += (_, _) => Program.BeginShutdown();
        }
        base.OnFrameworkInitializationCompleted();
    }
}
