using System.Globalization;
using Avalonia;
using Avalonia.Data.Converters;
using Avalonia.Media;
using Vernacula.App.Models;

namespace Vernacula.App.Converters;

/// <summary>Converts JobStatus to a SolidColorBrush from the active theme.</summary>
public class StatusToBrushConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not JobStatus status) return Brushes.Transparent;

        string key = status switch
        {
            JobStatus.Complete  => "GreenBrush",
            JobStatus.Failed    => "RedBrush",
            JobStatus.Cancelled => "YellowBrush",
            JobStatus.Running   => "AccentBrush",
            JobStatus.Queued    => "AccentBrush",
            _                   => "SubtextBrush",
        };

        var app = Application.Current;
        if (app?.Resources.TryGetResource(key, null, out var resourceValue) == true)
        {
            if (resourceValue is IBrush brush)
            {
                return brush;
            }

            if (resourceValue is Color color)
            {
                return new SolidColorBrush(color);
            }
        }

        return Brushes.Transparent;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
