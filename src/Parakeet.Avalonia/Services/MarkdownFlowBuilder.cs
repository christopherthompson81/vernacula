using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Markdig;
using Markdig.Syntax;

namespace ParakeetCSharp.Services;

public static class MarkdownFlowBuilder
{
    private static IBrush? _defaultTextBrush;

    public static Panel Build(string markdown, Action<string>? onLinkNavigate = null)
    {
        _defaultTextBrush = GetTextBrush();

        var panel = new StackPanel 
        { 
            Spacing = 8, 
            Width = 600,
            Margin = new Thickness(16),
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Left
        };

        if (string.IsNullOrWhiteSpace(markdown))
            return panel;

        var pipeline = new MarkdownPipelineBuilder()
            .UseAdvancedExtensions()
            .Build();

        var document = Markdig.Markdown.Parse(markdown, pipeline);

        foreach (var block in document)
        {
            var control = CreateBlockControl(block, onLinkNavigate);
            if (control != null)
                panel.Children.Add(control);
        }

        return panel;
    }

    private static IBrush GetTextBrush()
    {
        try
        {
            var app = Application.Current;
            if (app != null)
            {
                if (app.Resources["TextBrush"] is IBrush brush)
                    return brush;
                
                foreach (var dict in app.Resources.MergedDictionaries)
                {
                    if (dict is Avalonia.Controls.ResourceDictionary rd && rd["TextBrush"] is IBrush b)
                        return b;
                }
            }
        }
        catch { }
        
        return Brushes.White;
    }

    private static Control? CreateBlockControl(Block block, Action<string>? onLinkNavigate) => block switch
    {
        HeadingBlock heading => CreateHeading(heading),
        ParagraphBlock paragraph => CreateParagraph(paragraph, onLinkNavigate),
        ListBlock listBlock => listBlock.IsOrdered 
            ? CreateOrderedList(listBlock, onLinkNavigate) 
            : CreateBulletList(listBlock, onLinkNavigate),
        CodeBlock codeBlock => CreateCodeBlock(codeBlock),
        ThematicBreakBlock => CreateHorizontalRule(),
        QuoteBlock quoteBlock => CreateQuote(quoteBlock, onLinkNavigate),
        _ => null
    };

    private static Control CreateHeading(HeadingBlock heading)
    {
        var fontSize = heading.Level switch
        {
            1 => 28,
            2 => 24,
            3 => 20,
            4 => 16,
            _ => 14
        };

        var fontWeight = heading.Level switch
        {
            1 or 2 => FontWeight.Bold,
            _ => FontWeight.SemiBold
        };

        return new TextBlock
        {
            FontSize = fontSize,
            FontWeight = fontWeight,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 8, 0, 4),
            Text = GetLeafBlockText(heading),
            Foreground = _defaultTextBrush
        };
    }

    private static Control CreateParagraph(ParagraphBlock paragraph, Action<string>? onLinkNavigate)
    {
        var text = GetLeafBlockText(paragraph);
        var links = ExtractLinks(text);

        if (links.Count > 0 && onLinkNavigate != null)
            return CreateParagraphWithLinks(text, links, onLinkNavigate);

        return new TextBlock
        {
            Text = text,
            TextWrapping = TextWrapping.Wrap,
            LineHeight = 24,
            Foreground = _defaultTextBrush
        };
    }

    private static string GetLeafBlockText(LeafBlock block)
    {
        var inline = block.Inline;
        if (inline == null)
            return string.Empty;

        var result = new System.Text.StringBuilder();
        ExtractInlineText(inline, result);
        return result.ToString();
    }

    private static string GetContainerBlockText(ContainerBlock block)
    {
        var result = new System.Text.StringBuilder();
        foreach (var child in block)
        {
            if (child is LeafBlock leaf)
                result.Append(GetLeafBlockText(leaf));
            else if (child is ContainerBlock container)
                result.Append(GetContainerBlockText(container));
        }
        return result.ToString();
    }

    private static void ExtractInlineText(object inline, System.Text.StringBuilder result)
    {
        var type = inline.GetType();
        var name = type.Name;

        if (name == "TextInline")
        {
            var prop = type.GetProperty("Content");
            if (prop != null)
                result.Append(prop.GetValue(inline)?.ToString() ?? "");
        }
        else if (name == "CodeInline")
        {
            var prop = type.GetProperty("Content");
            if (prop != null)
                result.Append(prop.GetValue(inline)?.ToString() ?? "");
        }
        else if (name == "LineBreakInline")
        {
            result.AppendLine();
        }
        else
        {
            var childrenProp = type.GetProperty("Children");
            if (childrenProp != null)
            {
                var children = childrenProp.GetValue(inline) as System.Collections.IEnumerable;
                if (children != null)
                {
                    foreach (var child in children)
                        ExtractInlineText(child, result);
                }
            }
        }
    }

    private static List<(int Start, int End, string Url)> ExtractLinks(string text)
    {
        var links = new List<(int Start, int End, string Url)>();
        int pos = 0;

        while (pos < text.Length)
        {
            int linkStart = text.IndexOf("](", pos);
            if (linkStart < 0) break;

            int textStart = text.LastIndexOf('[', linkStart);
            if (textStart < 0) break;

            int urlStart = linkStart + 2;
            int urlEnd = text.IndexOf(')', urlStart);
            if (urlEnd < 0) break;

            var url = text.Substring(urlStart, urlEnd - urlStart);
            var linkText = text.Substring(textStart + 1, linkStart - textStart - 1);

            if (!string.IsNullOrEmpty(url))
                links.Add((textStart, urlEnd + 1, url));

            pos = urlEnd + 1;
        }

        return links;
    }

    private static Control CreateParagraphWithLinks(string text, List<(int Start, int End, string Url)> links, Action<string> onLinkNavigate)
    {
        var sp = new StackPanel { Spacing = 0 };
        int lastEnd = 0;

        foreach (var (start, end, url) in links)
        {
            if (start > lastEnd)
                sp.Children.Add(new TextBlock { Text = text.Substring(lastEnd, start - lastEnd), TextWrapping = TextWrapping.Wrap, Foreground = _defaultTextBrush });

            int textStart = text.LastIndexOf('[', start);
            int linkEnd = text.IndexOf(']', textStart);
            string linkText = text.Substring(textStart + 1, linkEnd - textStart - 1);

            var button = new Button
            {
                Content = linkText,
                Background = Brushes.Transparent,
                BorderThickness = new Thickness(0),
                Padding = new Thickness(0),
                Tag = url
            };
            button.Click += (_, _) => onLinkNavigate(url);
            sp.Children.Add(button);

            lastEnd = end;
        }

        if (lastEnd < text.Length)
            sp.Children.Add(new TextBlock { Text = text.Substring(lastEnd), TextWrapping = TextWrapping.Wrap, Foreground = _defaultTextBrush });

        return sp;
    }

    private static string GetListItemText(Block item)
    {
        if (item is ContainerBlock container)
            return GetContainerBlockText(container);
        if (item is LeafBlock leaf)
            return GetLeafBlockText(leaf);
        return item.ToString() ?? string.Empty;
    }

    private static Control CreateBulletList(ListBlock list, Action<string>? onLinkNavigate)
    {
        var stackPanel = new StackPanel { Spacing = 4 };

        foreach (var item in list)
        {
            var row = new StackPanel { Orientation = Avalonia.Layout.Orientation.Horizontal, Spacing = 8 };
            row.Children.Add(new TextBlock { Text = "\u2022", FontSize = 14 });

            var text = GetListItemText(item).TrimStart('*', '-', ' ');
            if (text.Length > 0 && text[0] == ' ')
                text = text.Substring(1);

            var links = ExtractLinks(text);
            if (links.Count > 0 && onLinkNavigate != null)
                row.Children.Add(CreateParagraphWithLinks(text, links, onLinkNavigate!));
            else
                row.Children.Add(new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap, Foreground = _defaultTextBrush });

            stackPanel.Children.Add(row);
        }

        return stackPanel;
    }

    private static Control CreateOrderedList(ListBlock list, Action<string>? onLinkNavigate)
    {
        var stackPanel = new StackPanel { Spacing = 4 };
        int index = 1;

        foreach (var item in list)
        {
            var row = new StackPanel { Orientation = Avalonia.Layout.Orientation.Horizontal, Spacing = 8 };
            row.Children.Add(new TextBlock { Text = $"{index}.", FontSize = 14, Width = 24 });

            var text = GetListItemText(item).TrimStart('1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', ' ');
            if (text.Length > 0 && text[0] == ' ')
                text = text.Substring(1);

            var links = ExtractLinks(text);
            if (links.Count > 0 && onLinkNavigate != null)
                row.Children.Add(CreateParagraphWithLinks(text, links, onLinkNavigate!));
            else
                row.Children.Add(new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap, Foreground = _defaultTextBrush });

            stackPanel.Children.Add(row);
            index++;
        }

        return stackPanel;
    }

    private static Control CreateCodeBlock(CodeBlock codeBlock)
    {
        return new TextBlock
        {
            Text = codeBlock.Lines.ToString(),
            FontFamily = new FontFamily("Consolas, Courier New, monospace"),
            FontSize = 13,
            TextWrapping = TextWrapping.Wrap,
            Padding = new Thickness(8),
            Background = new SolidColorBrush(Color.Parse("#F5F5F5")),
            Margin = new Thickness(0, 4, 0, 4),
            Foreground = _defaultTextBrush
        };
    }

    private static Control CreateHorizontalRule()
    {
        return new Border
        {
            Height = 1,
            Background = new SolidColorBrush(Color.Parse("#E0E0E0")),
            Margin = new Thickness(0, 8, 0, 8)
        };
    }

    private static Control CreateQuote(QuoteBlock quote, Action<string>? onLinkNavigate)
    {
        var border = new Border
        {
            BorderBrush = new SolidColorBrush(Color.Parse("#E0E0E0")),
            BorderThickness = new Thickness(4, 0, 0, 0),
            Padding = new Thickness(12, 4, 4, 4),
            Margin = new Thickness(0, 4, 0, 4)
        };

        var stackPanel = new StackPanel { Spacing = 4 };

        foreach (var block in quote)
        {
            var control = CreateBlockControl(block, onLinkNavigate);
            if (control != null)
                stackPanel.Children.Add(control);
        }

        border.Child = stackPanel;
        return border;
    }
}
