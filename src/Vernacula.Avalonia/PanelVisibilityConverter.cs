using Avalonia.Data;
using Avalonia.Data.Converters;
using Vernacula.App.ViewModels;
using System.Globalization;

namespace Vernacula.App;

public class PanelVisibilityConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not AppPanel panel)
            return (bool?)false;

        var param = parameter?.ToString() ?? "";
        return (bool?)(param switch
        {
            "Home"     => panel == AppPanel.Home,
            "Progress" => panel == AppPanel.Progress,
            "Results"  => panel == AppPanel.Results,
            _          => false
        });
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return BindingOperations.DoNothing;
    }
}
