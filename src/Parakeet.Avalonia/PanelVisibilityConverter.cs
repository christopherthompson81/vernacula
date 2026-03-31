using Avalonia.Data;
using Avalonia.Data.Converters;
using ParakeetCSharp.ViewModels;
using System.Globalization;

namespace ParakeetCSharp;

public class PanelVisibilityConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not AppPanel panel)
            return false;

        var param = parameter?.ToString() ?? "";
        return param switch
        {
            "Home" => panel == AppPanel.Home,
            "Progress" => panel == AppPanel.Progress,
            "Results" => panel == AppPanel.Results,
            _ => false
        };
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return null;
    }
}
