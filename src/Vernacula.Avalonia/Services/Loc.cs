using System.ComponentModel;
using System.Reflection;
using System.Text.Json;

namespace Vernacula.Avalonia;

/// <summary>
/// Singleton localization service. Loads JSON locale files embedded in the assembly
/// and notifies WPF bindings when the language changes via INotifyPropertyChanged.
/// </summary>
public sealed class Loc : INotifyPropertyChanged
{
    public static readonly Loc Instance = new();
    private Loc() 
    {
        // Initialize with English as default
        SetLanguage("en");
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private Dictionary<string, string> _current  = new();
    private Dictionary<string, string> _fallback = new();

    public string CurrentLanguage { get; private set; } = "en";

    /// <summary>Indexer used by {loc:Loc key} bindings in XAML.</summary>
    public string this[string key]
    {
        get
        {
            if (_current.TryGetValue(key, out var val))  return val;
            if (_fallback.TryGetValue(key, out val))     return val;
            return key; // Raw key as last-resort fallback
        }
    }

    /// <summary>
    /// Returns a translated string with {placeholder} values substituted.
    /// </summary>
    public string T(string key, Dictionary<string, string> ps)
    {
        var str = this[key];
        foreach (var (k, v) in ps)
            str = str.Replace($"{{{k}}}", v);
        return str;
    }

    public void SetLanguage(string lang)
    {
        CurrentLanguage = lang;
        _current  = LoadLocale(lang);
        _fallback = lang == "en" ? new() : LoadLocale("en");

        // Notify all indexer bindings (WPF re-evaluates all [key] bindings)
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Item[]"));
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(CurrentLanguage)));
    }

    private static Dictionary<string, string> LoadLocale(string lang)
    {
        // Assembly name = project file name ("parakeet_csharp"), not the CLR namespace.
        var assembly = Assembly.GetExecutingAssembly();
        var suffix   = $".Locales.{lang}.json";
        var name     = assembly.GetManifestResourceNames()
                               .FirstOrDefault(n => n.EndsWith(suffix, StringComparison.Ordinal));
        if (name is null) return new();
        try
        {
            using var stream = assembly.GetManifestResourceStream(name)!;
            using var reader = new StreamReader(stream, Encoding.UTF8);
            return JsonSerializer.Deserialize<Dictionary<string, string>>(reader.ReadToEnd())
                   ?? new();
        }
        catch
        {
            return new();
        }
    }

    // ── Static language list (code + native display name) ────────────────────

    public record LanguageInfo(string Code, string DisplayName);

    public static readonly IReadOnlyList<LanguageInfo> Languages = new[]
    {
        new LanguageInfo("bg", "Български"),
        new LanguageInfo("cs", "Čeština"),
        new LanguageInfo("da", "Dansk"),
        new LanguageInfo("de", "Deutsch"),
        new LanguageInfo("en", "English"),
        new LanguageInfo("el", "Ελληνικά"),
        new LanguageInfo("es", "Español"),
        new LanguageInfo("et", "Eesti keel"),
        new LanguageInfo("fr", "Français"),
        new LanguageInfo("hr", "Hrvatski"),
        new LanguageInfo("it", "Italiano"),
        new LanguageInfo("lv", "Latviešu"),
        new LanguageInfo("lt", "Lietuvių"),
        new LanguageInfo("mt", "Malti"),
        new LanguageInfo("hu", "Magyar"),
        new LanguageInfo("nl", "Nederlands"),
        new LanguageInfo("pl", "Polski"),
        new LanguageInfo("pt", "Português"),
        new LanguageInfo("ro", "Română"),
        new LanguageInfo("ru", "Русский"),
        new LanguageInfo("sk", "Slovenčina"),
        new LanguageInfo("sl", "Slovenščina"),
        new LanguageInfo("fi", "Suomi"),
        new LanguageInfo("sv", "Svenska"),
        new LanguageInfo("uk", "Українська"),
    };
}
