using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Markdig;
using Markdig.Extensions.Tables;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;

namespace ParakeetCSharp.Services;

/// <summary>
/// Converts a Markdown string to an Avalonia UI using Markdig for parsing.
/// Colors are read from <see cref="Application.Current.Resources"/> at call time
/// so that re-calling after a theme swap produces correctly-themed output.
/// 
/// ⚠️ AVALONIA MIGRATION REQUIRED ⚠️
/// 
/// WPF FlowDocument does not exist in Avalonia. The following alternatives should be considered:
/// 
/// 1. Use a Markdown rendering library like:
///    - AvaloniaEdit with Markdown support
///    - Material.Icons for Avalonia
///    - Community Avalonia.Markdown library
/// 
/// 2. Render markdown to HTML and use a WebView control
/// 
/// 3. Build custom Avalonia Controls to represent markdown elements:
///    - Paragraph → TextBlock
///    - Heading → Styled TextBlock
///    - List → ItemsControl with custom template
///    - Table → DataGrid or custom grid
///    - Code blocks → TextBlock with monospace font
///    - Inline elements → Span-like constructs in TextBlock.Inlines
/// 
/// 4. Use a third-party Markdown renderer:
///    - Markdig can render to HTML which could be displayed in a WebView
///    - Consider using Avalonia.Markdown or similar NuGet packages
/// 
/// Current implementation uses WPF types that need to be replaced:
/// - FlowDocument → N/A (use alternative approach)
/// - Paragraph → TextBlock or custom control
/// - Section → Border with content
/// - Run → TextBlock/TextSpan
/// - Span → TextBlock with inlines
/// - Hyperlink → Hyperlink control (if available) or Button
/// - LineBreak → LineBreak or newline
/// - List → ItemsControl
/// - ListItem → Custom item template
/// - Table → DataGrid or custom grid layout
/// - TextDecorations → TextDecoration
/// - Thickness → Thickness (Avalonia has this)
/// - FontFamily, FontSize, FontWeight → Same in Avalonia.Media
/// - SolidColorBrush, Color → IBrush, Color in Avalonia.Media
/// - GridLength, GridUnitType → Avalonia.Layout
/// 
/// The Build method signature should change from returning FlowDocument to returning
/// an Avalonia Control (e.g., Panel, StackPanel, or a custom MarkdownPanel).
/// </summary>
internal static class MarkdownFlowBuilder
{
    private static readonly MarkdownPipeline Pipeline =
        new MarkdownPipelineBuilder().UseAdvancedExtensions().Build();

    private static readonly FontFamily MonoFamily =
        new FontFamily("Cascadia Mono,Consolas,Courier New");

    // ── Entry point ──────────────────────────────────────────────────────────

    /// <summary>
    /// Builds a control from <paramref name="markdown"/>.
    /// ⚠️ TODO: Implement Avalonia-specific markdown rendering.
    /// Currently returns null as a placeholder.
    /// </summary>
    /// <param name="markdown">Plain Markdown text (frontmatter already stripped).</param>
    /// <param name="navigateTo">
    ///   Callback invoked when a link is clicked. Receives the raw URL string;
    ///   relative .md paths and absolute http URLs are both passed through.
    /// </param>
    internal static Panel Build(string markdown, Action<string>? navigateTo = null)
    {
        // TODO: Implement Avalonia markdown rendering
        // Options:
        // 1. Use a WebView control with HTML output from Markdig
        // 2. Build a custom Panel with TextBlocks and other controls
        // 3. Use a third-party Avalonia Markdown library
        
        var panel = new StackPanel();
        panel.Children.Add(new TextBlock
        {
            Text = "Markdown rendering not yet implemented for Avalonia.\n\n" +
                   "See MarkdownFlowBuilder class documentation for migration options.",
            WrapText = true
        });
        
        return panel;
    }
}
