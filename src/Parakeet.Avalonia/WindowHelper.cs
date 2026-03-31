using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

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

    /// <summary>
    /// Gets the virtual screen bounds for multi-monitor support.
    /// </summary>
    public static Rect GetVirtualScreenBounds()
    {
        // Return a default screen size for now
        // TODO: Implement proper multi-monitor support
        return new Rect(0, 0, 1920, 1080);
    }
}
