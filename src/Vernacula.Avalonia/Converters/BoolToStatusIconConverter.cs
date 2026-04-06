using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace Vernacula.App.Converters;

/// <summary>
/// Converts a boolean value to a status icon character.
/// Returns "✓" for true, "✗" for false.
/// </summary>
public class BoolToStatusIconConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool v = value is bool b && b;
        return v ? "✓" : "✗";
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

/// <summary>
/// Converts a boolean value to a status color brush.
/// Returns GreenBrush for true, and a configurable brush for false (default: YellowBrush).
/// </summary>
public class BoolToStatusBrushConverter : IValueConverter
{
    public string FalseBrushKey { get; set; } = "YellowBrush";

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not bool b) return Brushes.Transparent;

        string key = b ? "GreenBrush" : FalseBrushKey;
        var app = Avalonia.Application.Current;
        if (app?.Resources.TryGetResource(key, null, out var res) == true)
        {
            if (res is IBrush brush) return brush;
            if (res is Color color)  return new SolidColorBrush(color);
        }
        return Brushes.Transparent;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
