using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Platform;

namespace ParakeetCSharp;

internal static class WindowHelper
{
    /// <summary>
    /// Placeholder for dark mode support. 
    /// In Avalonia 11.2.1, dark mode title bars are handled differently.
    /// This will be implemented when needed.
    /// </summary>
    public static void SetDarkMode(Window window, bool dark)
    {
        // TODO: Implement Avalonia 11.2.1 dark mode support
        // For now, the theme is applied via resource dictionaries
    }

    public static PixelRect GetVirtualScreenBounds(Window window)
    {
        var screens = window.Screens;
        if (screens is null || screens.ScreenCount == 0)
        {
            return new PixelRect(0, 0, 1920, 1080);
        }

        var areas = screens.All.Select(s => s.WorkingArea).ToList();
        int left = areas.Min(a => a.X);
        int top = areas.Min(a => a.Y);
        int right = areas.Max(a => a.Right);
        int bottom = areas.Max(a => a.Bottom);
        return new PixelRect(left, top, right - left, bottom - top);
    }

    public static PixelPoint ClampToVisibleArea(
        Window window,
        PixelPoint desiredPosition,
        double width,
        double height,
        int minVisiblePixels = 100)
    {
        var bounds = GetVirtualScreenBounds(window);
        int windowWidth = Math.Max(minVisiblePixels, (int)Math.Ceiling(width));
        int windowHeight = Math.Max(minVisiblePixels, (int)Math.Ceiling(height));

        int minX = bounds.X - windowWidth + minVisiblePixels;
        int maxX = bounds.Right - minVisiblePixels;
        int minY = bounds.Y;
        int maxY = bounds.Bottom - minVisiblePixels;

        return new PixelPoint(
            Math.Clamp(desiredPosition.X, minX, maxX),
            Math.Clamp(desiredPosition.Y, minY, maxY));
    }
}
