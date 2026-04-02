using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Layout;
using Avalonia.Media;
using Markdig;
using Markdig.Extensions.Tables;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;

namespace ParakeetCSharp.Services;

internal static class MarkdownFlowBuilder
{
    private static readonly MarkdownPipeline Pipeline =
        new MarkdownPipelineBuilder().UseAdvancedExtensions().Build();

    private static readonly FontFamily MonoFamily =
        new("Cascadia Mono,Consolas,Courier New");

    internal static Panel Build(string markdown, Action<string>? navigateTo = null)
    {
        var root = new Grid
        {
            Margin = new Thickness(0, 0, 20, 56),
            MaxWidth = 860,
            HorizontalAlignment = HorizontalAlignment.Stretch,
            VerticalAlignment = VerticalAlignment.Top,
        };

        if (string.IsNullOrWhiteSpace(markdown))
        {
            return root;
        }

        var ast = Markdig.Markdown.Parse(markdown, Pipeline);
        int rowIndex = 0;
        foreach (var block in ast)
        {
            var control = ConvertBlock(block, navigateTo);
            if (control is not null)
            {
                root.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
                Grid.SetRow(control, rowIndex++);
                root.Children.Add(control);
            }
        }

        return root;
    }

    private static Control? ConvertBlock(MarkdownObject block, Action<string>? navigateTo) => block switch
    {
        HeadingBlock heading => ConvertHeading(heading, navigateTo),
        ParagraphBlock paragraph => ConvertParagraph(paragraph, navigateTo),
        ListBlock list => ConvertList(list, navigateTo),
        FencedCodeBlock fenced => ConvertCodeBlock(fenced.Lines.ToString()),
        CodeBlock code => ConvertCodeBlock(code.Lines.ToString()),
        QuoteBlock quote => ConvertQuote(quote, navigateTo),
        ThematicBreakBlock => new Border
        {
            Height = 1,
            Background = Brush("OverlayBrush"),
            Margin = new Thickness(0, 8, 0, 8),
        },
        Table table => ConvertTable(table, navigateTo),
        _ => null,
    };

    private static Control ConvertHeading(HeadingBlock heading, Action<string>? navigateTo)
    {
        var textBlock = new TextBlock
        {
            Text = GetInlinePlainText(heading.Inline),
            TextWrapping = TextWrapping.Wrap,
            Foreground = Brush("TextBrush"),
        };
        textBlock.Margin = heading.Level switch
        {
            1 => new Thickness(0, 16, 0, 6),
            2 => new Thickness(0, 14, 0, 4),
            _ => new Thickness(0, 10, 0, 2),
        };
        textBlock.FontSize = heading.Level switch
        {
            1 => 22,
            2 => 17,
            _ => 14,
        };
        textBlock.FontWeight = heading.Level == 1
            ? FontWeight.Bold
            : FontWeight.SemiBold;

        if (heading.Level == 1)
        {
            return new Border
            {
                BorderBrush = Brush("OverlayBrush"),
                BorderThickness = new Thickness(0, 0, 0, 1),
                Padding = new Thickness(0, 0, 0, 6),
                Margin = new Thickness(0, 0, 0, 8),
                Child = textBlock,
            };
        }

        return textBlock;
    }

    private static Control ConvertParagraph(ParagraphBlock paragraph, Action<string>? navigateTo)
    {
        return CreateTextWithLinks(paragraph.Inline, new Thickness(0, 0, 0, 8), navigateTo);
    }

    private static Control ConvertList(ListBlock list, Action<string>? navigateTo)
    {
        var panel = new StackPanel
        {
            Spacing = 4,
            Margin = new Thickness(0, 0, 0, 8),
        };

        int orderedIndex = 1;
        foreach (var item in list.OfType<ListItemBlock>())
        {
            var row = new Grid
            {
                ColumnDefinitions =
                {
                    new ColumnDefinition(GridLength.Auto),
                    new ColumnDefinition(new GridLength(1, GridUnitType.Star)),
                },
                Margin = new Thickness(0, 0, 0, 2),
            };

            var bullet = new TextBlock
            {
                Text = list.IsOrdered ? $"{orderedIndex}." : "\u2022",
                Margin = new Thickness(0, 0, 8, 0),
                VerticalAlignment = VerticalAlignment.Top,
                Foreground = Brush("TextBrush"),
            };
            Grid.SetColumn(bullet, 0);
            row.Children.Add(bullet);

            var itemHost = new StackPanel { Spacing = 2 };
            foreach (var child in item)
            {
                var control = ConvertBlock(child, navigateTo);
                if (control is not null)
                {
                    TightenBlockMargins(control);
                    itemHost.Children.Add(control);
                }
            }

            Grid.SetColumn(itemHost, 1);
            row.Children.Add(itemHost);
            panel.Children.Add(row);
            orderedIndex++;
        }

        return panel;
    }

    private static Control ConvertCodeBlock(string text)
    {
        return new Border
        {
            Background = Brush("SurfaceBrush"),
            Margin = new Thickness(0, 0, 0, 8),
            Padding = new Thickness(12, 8, 12, 8),
            Child = new SelectableTextBlock
            {
                Text = text.TrimEnd(),
                FontFamily = MonoFamily,
                FontSize = 12,
                Foreground = Brush("TextBrush"),
                TextWrapping = TextWrapping.Wrap,
            },
        };
    }

    private static Control ConvertQuote(QuoteBlock quote, Action<string>? navigateTo)
    {
        var host = new StackPanel { Spacing = 4 };
        foreach (var child in quote)
        {
            var control = ConvertBlock(child, navigateTo);
            if (control is not null)
            {
                TightenBlockMargins(control);
                host.Children.Add(control);
            }
        }

        return new Border
        {
            Background = Brush("SurfaceBrush"),
            BorderBrush = Brush("AccentBrush"),
            BorderThickness = new Thickness(3, 0, 0, 0),
            Margin = new Thickness(0, 4, 0, 8),
            Padding = new Thickness(14, 8, 12, 8),
            Child = host,
        };
    }

    private static Control ConvertTable(Table table, Action<string>? navigateTo)
    {
        var rows = table.OfType<TableRow>().ToList();
        int columnCount = Math.Max(1, rows.Count == 0 ? 0 : rows.Max(r => r.OfType<TableCell>().Count()));
        var grid = new Grid
        {
            ColumnSpacing = 0,
            RowSpacing = 0,
        };

        for (int i = 0; i < columnCount; i++)
        {
            grid.ColumnDefinitions.Add(new ColumnDefinition(new GridLength(1, GridUnitType.Star)));
        }

        int rowIndex = 0;
        foreach (var mdRow in rows)
        {
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));

            int columnIndex = 0;
            foreach (var mdCell in mdRow.OfType<TableCell>())
            {
                var content = new StackPanel { Spacing = 2 };
                foreach (var child in mdCell)
                {
                    var control = ConvertBlock(child, navigateTo);
                    if (control is not null)
                    {
                        TightenBlockMargins(control);
                        if (rowIndex == 0)
                        {
                            ApplyFontWeight(control, FontWeight.SemiBold);
                        }

                        content.Children.Add(control);
                    }
                }

                var cell = new Border
                {
                    Background = rowIndex == 0 ? Brush("SurfaceBrush") : Brush("BgBrush"),
                    BorderBrush = Brush("OverlayBrush"),
                    BorderThickness = new Thickness(0, 0, 1, 1),
                    Padding = new Thickness(8, 5, 8, 5),
                    Child = content,
                };

                Grid.SetRow(cell, rowIndex);
                Grid.SetColumn(cell, columnIndex);
                grid.Children.Add(cell);
                columnIndex++;
            }

            rowIndex++;
        }

        return new Border
        {
            BorderBrush = Brush("OverlayBrush"),
            BorderThickness = new Thickness(1),
            Margin = new Thickness(0, 0, 0, 12),
            Child = grid,
        };
    }

    private static Control CreateTextWithLinks(ContainerInline? inline, Thickness margin, Action<string>? navigateTo)
    {
        var segments = new List<(string Text, string? Url)>();
        CollectSegments(inline, segments);
        bool hasLinks = segments.Any(s => !string.IsNullOrWhiteSpace(s.Url));

        if (!hasLinks)
        {
            return new TextBlock
            {
                Text = string.Concat(segments.Select(s => s.Text)),
                TextWrapping = TextWrapping.Wrap,
                Foreground = Brush("TextBrush"),
                Margin = margin,
            };
        }

        var textBlock = new TextBlock
        {
            Foreground = Brush("TextBrush"),
            TextWrapping = TextWrapping.Wrap,
            Margin = margin,
        };

        foreach (var (text, url) in segments)
        {
            if (string.IsNullOrEmpty(text))
            {
                continue;
            }

            if (string.IsNullOrWhiteSpace(url))
            {
                textBlock.Inlines!.Add(new Run(text));
            }
            else
            {
                var linkText = new TextBlock
                {
                    Text = text,
                    Foreground = Brush("LinkBrush"),
                    TextWrapping = TextWrapping.NoWrap,
                    TextDecorations = TextDecorations.Underline,
                    FontWeight = FontWeight.SemiBold,
                    Cursor = new Avalonia.Input.Cursor(Avalonia.Input.StandardCursorType.Hand),
                    FontSize = 13,
                    FontFamily = new FontFamily("Segoe UI"),
                };
                linkText.PointerPressed += (_, _) => navigateTo?.Invoke(url);
                textBlock.Inlines!.Add(new InlineUIContainer
                {
                    Child = linkText,
                    BaselineAlignment = BaselineAlignment.TextBottom,
                });
            }
        }

        return textBlock;
    }

    private static void CollectSegments(ContainerInline? container, List<(string Text, string? Url)> segments)
    {
        if (container is null)
        {
            return;
        }

        foreach (var child in container)
        {
            CollectSegments(child, segments);
        }
    }

    private static void CollectSegments(Markdig.Syntax.Inlines.Inline inline, List<(string Text, string? Url)> segments)
    {
        switch (inline)
        {
            case LiteralInline literal:
                AddSegment(segments, literal.ToString(), null);
                break;
            case CodeInline code:
                AddSegment(segments, code.Content, null);
                break;
            case LineBreakInline lineBreak:
                AddSegment(segments, lineBreak.IsHard ? "\n" : " ", null);
                break;
            case HtmlInline html:
                AddSegment(segments, ConvertHtmlInline(html), null);
                break;
            case LinkInline link:
                AddSegment(segments, GetInlinePlainText(link), link.Url);
                break;
            case ContainerInline container:
                foreach (var child in container)
                {
                    CollectSegments(child, segments);
                }
                break;
            default:
                AddSegment(segments, inline.ToString() ?? string.Empty, null);
                break;
        }
    }

    private static void AddSegment(List<(string Text, string? Url)> segments, string text, string? url)
    {
        if (string.IsNullOrEmpty(text))
        {
            return;
        }

        if (segments.Count > 0 && segments[^1].Url == url)
        {
            segments[^1] = (segments[^1].Text + text, url);
            return;
        }

        segments.Add((text, url));
    }

    private static string ConvertHtmlInline(HtmlInline html)
    {
        var tag = html.Tag;
        const string marker = "ch=\"";
        int index = tag.IndexOf(marker, StringComparison.Ordinal);
        if (tag.StartsWith("<mdl2 ", StringComparison.Ordinal) && index >= 0)
        {
            int start = index + marker.Length;
            int end = tag.IndexOf('"', start);
            if (end > start && ushort.TryParse(tag[start..end], NumberStyles.HexNumber, null, out var codePoint))
            {
                return ((char)codePoint).ToString();
            }
        }

        return string.Empty;
    }

    private static string GetInlinePlainText(ContainerInline? container)
    {
        var result = new System.Text.StringBuilder();
        if (container is null)
        {
            return string.Empty;
        }

        foreach (var child in container)
        {
            AppendInlinePlainText(result, child);
        }

        return result.ToString();
    }

    private static void AppendInlinePlainText(System.Text.StringBuilder result, Markdig.Syntax.Inlines.Inline inline)
    {
        switch (inline)
        {
            case LiteralInline literal:
                result.Append(literal.ToString());
                break;

            case CodeInline code:
                result.Append(code.Content);
                break;

            case LineBreakInline lineBreak:
                result.Append(lineBreak.IsHard ? '\n' : ' ');
                break;

            case HtmlInline:
                result.Append(ConvertHtmlInline((HtmlInline)inline));
                break;

            case ContainerInline container:
                foreach (var child in container)
                {
                    AppendInlinePlainText(result, child);
                }

                break;

            default:
                result.Append(inline.ToString());
                break;
        }
    }

    private static void TightenBlockMargins(Control control)
    {
        switch (control)
        {
            case TextBlock textBlock:
                textBlock.Margin = new Thickness(0);
                break;
            case Border border:
                border.Margin = new Thickness(0);
                break;
            case Grid grid:
                grid.Margin = new Thickness(0);
                break;
        }
    }

    private static void ApplyFontWeight(Control control, FontWeight weight)
    {
        switch (control)
        {
            case TextBlock textBlock:
                textBlock.FontWeight = weight;
                break;
            case Border border when border.Child is Control child:
                ApplyFontWeight(child, weight);
                break;
            case Panel panel:
                foreach (var child in panel.Children.OfType<Control>())
                {
                    ApplyFontWeight(child, weight);
                }

                break;
        }
    }

    private static IBrush Brush(string brushKey)
    {
        if (TryFindResource(brushKey) is IBrush brush)
        {
            return brush;
        }

        string colorKey = brushKey.Replace("Brush", "Color", StringComparison.Ordinal);
        if (TryFindResource(colorKey) is Color color)
        {
            return new SolidColorBrush(color);
        }

        return Brushes.Gray;
    }

    private static object? TryFindResource(string key)
    {
        var app = Application.Current;
        if (app is null)
        {
            return null;
        }

        if (app.Resources.TryGetResource(key, null, out var value))
        {
            return value;
        }

        return null;
    }
}
