using System.Globalization;
using Avalonia.Data.Converters;

namespace Vernacula.App.Converters;

public class BoolToOpacityConverter : IValueConverter
{
    public bool Invert { get; set; }
    public double TrueOpacity { get; set; } = 1.0;
    public double FalseOpacity { get; set; } = 0.0;

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool flag = value is true;
        if (Invert)
            flag = !flag;

        return flag ? TrueOpacity : FalseOpacity;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
