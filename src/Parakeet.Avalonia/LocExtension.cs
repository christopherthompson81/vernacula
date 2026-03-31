using Avalonia.Data;
using Avalonia.Markup;
using Avalonia.Markup.Xaml;

namespace ParakeetCSharp;

/// <summary>
/// XAML markup extension for localized strings.
/// Usage:  Text="{local:Loc some_key}"
/// Produces a OneWay Binding to Loc.Instance["some_key"].
/// All bindings refresh automatically when Loc.Instance.SetLanguage() is called.
/// </summary>
public class LocExtension : MarkupExtension
{
    public string Key { get; set; }

    public LocExtension(string key) { Key = key; }

    public override object ProvideValue(IServiceProvider serviceProvider)
    {
        var binding = new Binding($"[{Key}]")
        {
            Source = Loc.Instance,
            Mode   = BindingMode.OneWay,
        };
        return binding;
    }
}
