using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Converters;

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

        var app = Avalonia.Application.Current;
        return app?.Resources[key] as IBrush ?? Brushes.Transparent;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
