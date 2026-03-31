using Avalonia.Markup.Xaml;

namespace ParakeetCSharp;

/// <summary>
/// XAML markup extension for localized strings.
/// Usage:  Text="{local:Loc some_key}"
/// Returns the localized string directly.
/// </summary>
public class LocExtension : MarkupExtension
{
    public string Key { get; set; }

    public LocExtension(string key) { Key = key; }

    public override object ProvideValue(IServiceProvider serviceProvider)
    {
        // Return the localized string directly
        return Loc.Instance[Key];
    }
}
