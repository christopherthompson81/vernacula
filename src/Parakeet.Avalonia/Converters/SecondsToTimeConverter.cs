using System.Globalization;
using Avalonia.Data.Converters;

namespace ParakeetCSharp.Converters;

public class SecondsToTimeConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not double seconds) return "";
        int h = (int)(seconds / 3600);
        int m = (int)(seconds % 3600 / 60);
        double s = seconds % 60;
        return h > 0 ? $"{h}:{m:D2}:{s:00.0}" : $"{m:D2}:{s:00.0}";
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
