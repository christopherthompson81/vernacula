using System.Globalization;
using Avalonia.Data.Converters;

namespace ParakeetCSharp.Converters;

public class BoolToVisibilityConverter : IValueConverter
{
    public bool Invert { get; set; }

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool v = value is bool b && b;
        if (Invert) v = !v;
        // Return string values that XAML can convert to Visibility
        return v ? "Visible" : "Collapsed";
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
