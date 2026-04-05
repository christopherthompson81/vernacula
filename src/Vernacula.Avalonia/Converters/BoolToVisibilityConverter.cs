using System.Globalization;
using Avalonia.Data.Converters;

namespace Vernacula.App.Converters;

public class BoolToVisibilityConverter : IValueConverter
{
    public bool Invert { get; set; }

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool v = value switch
        {
            bool b  => b,
            int  i  => i != 0,
            _       => false,
        };
        if (Invert) v = !v;
        return v;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
