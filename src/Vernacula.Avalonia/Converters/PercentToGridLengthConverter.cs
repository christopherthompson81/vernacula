using System.Globalization;
using Avalonia.Controls;
using Avalonia.Data.Converters;

namespace Vernacula.App.Converters;

public class PercentToGridLengthConverter : IValueConverter
{
    public bool Invert { get; set; }

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        double percent = value switch
        {
            double d => d,
            float f => f,
            int i => i,
            _ => 0,
        };

        percent = Math.Clamp(percent, 0, 100);
        double stars = Invert ? 100 - percent : percent;
        return new GridLength(stars, GridUnitType.Star);
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
