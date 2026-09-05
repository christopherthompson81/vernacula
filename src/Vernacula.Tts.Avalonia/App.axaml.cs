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
            // the teardown is treated as a clean exit rather than an unhandled crash. Only Exit
            // counts: a shutdown *request* can be cancelled, and the guard must not stay armed
            // while the app carries on running.
            desktop.Exit += (_, e) => Program.BeginShutdown(e.ApplicationExitCode);
        }
        base.OnFrameworkInitializationCompleted();
    }
}
