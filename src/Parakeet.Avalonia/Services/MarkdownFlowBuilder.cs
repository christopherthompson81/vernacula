using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace ParakeetCSharp.Services;

/// <summary>
/// ⚠️ AVALONIA MIGRATION REQUIRED ⚠️
///
/// This class builds a FlowDocument from markdown, but FlowDocument doesn't exist in Avalonia.
/// Options for Avalonia:
/// 1. Use a third-party markdown renderer like Avalonia.Markdown
/// 2. Build a custom control using TextBlock.Inlines for simple formatting
/// 3. Use WebView control to render HTML
///
/// For now, this returns a placeholder panel with a message.
/// </summary>
public static class MarkdownFlowBuilder
{
    public static Panel Build(string markdown, Action<string>? onLinkNavigate = null)
    {
        // TODO: Implement proper markdown rendering for Avalonia
        // Options:
        // 1. NuGet package: Avalonia.Markdown (if available)
        // 2. Custom implementation using TextBlock.Inlines
        // 3. WebView with HTML rendering

        var panel = new StackPanel();
        panel.Children.Add(new TextBlock
        {
            Text = "Markdown rendering not yet implemented for Avalonia.\n\n" +
                   "See MarkdownFlowBuilder class documentation for migration options.",
            TextWrapping = TextWrapping.Wrap
        });

        return panel;
    }
}
