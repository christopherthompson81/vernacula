using System.Globalization;
using System.Collections.Generic;
using Avalonia.Data.Converters;

namespace Vernacula.Avalonia.Converters;

/// <summary>
/// IMultiValueConverter that returns true when all bound values are equal strings.
/// Used by the Language menu to set IsChecked on the current language item.
/// </summary>
public class StringEqualityConverter : IMultiValueConverter
{
    public object? Convert(IList<object?> values, Type targetType, object? parameter, CultureInfo culture)
        => values.Count >= 2 && values[0]?.ToString() == values[1]?.ToString();

    public object?[] ConvertBack(object? value, Type[] targetTypes, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
