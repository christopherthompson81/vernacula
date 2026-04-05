using Avalonia.Markup.Xaml;
using Avalonia.Data;

namespace Vernacula.Avalonia;

/// <summary>
/// XAML markup extension for localized strings.
/// Usage:  Text="{local:Loc some_key}"
/// Returns a live binding so open views update immediately when the language changes.
/// </summary>
public class LocExtension : MarkupExtension
{
    public string Key { get; set; }

    public LocExtension(string key) { Key = key; }

    public override object ProvideValue(IServiceProvider serviceProvider)
    {
        return new Binding
        {
            Source = Loc.Instance,
            Path = $"[{Key}]",
            Mode = BindingMode.OneWay,
        };
    }
}
