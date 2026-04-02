using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using ParakeetCSharp.Models;

namespace ParakeetCSharp;

public static class ThemeManager
{
    private const string MochaUri = "avares://Parakeet.Avalonia/Themes/MochaTheme.axaml";
    private const string LatteUri = "avares://Parakeet.Avalonia/Themes/LatteTheme.axaml";

    public static event Action<AppTheme>? ThemeChanged;

    public static void Apply(AppTheme theme)
    {
        string uri = theme == AppTheme.Dark ? MochaUri : LatteUri;
        var app = Application.Current as App;
        if (app == null) return;

        var dicts = app.Resources.MergedDictionaries;
        // Replace the theme dictionary (index 0)
        if (dicts.Count > 0)
        {
            var resourceDict = (ResourceDictionary)AvaloniaXamlLoader.Load(new Uri(uri));
            dicts[0] = resourceDict;
        }

        bool dark = theme == AppTheme.Dark;

        // Update all open windows
        foreach (var window in FindAllWindows())
        {
            WindowHelper.SetDarkMode(window, dark);
        }

        ThemeChanged?.Invoke(theme);
    }

    private static IEnumerable<Window> FindAllWindows()
    {
        var app = Application.Current as App;
        if (app?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            foreach (var window in desktop.Windows.OfType<Window>().Distinct())
            {
                yield return window;
            }
        }
    }
}
