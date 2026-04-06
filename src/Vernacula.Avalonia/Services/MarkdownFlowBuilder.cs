using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Markdig;
using Markdig.Extensions.Tables;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;
using Vernacula.App.Models;

namespace Vernacula.App.Services;

internal static class MarkdownFlowBuilder
{
    private static readonly MarkdownPipeline Pipeline =
        new MarkdownPipelineBuilder().UseAdvancedExtensions().Build();

    private static readonly IReadOnlyDictionary<ushort, string> Mdl2Fallbacks = new Dictionary<ushort, string>
    {
        [0xE77B] = "\u25CC",
        [0xE916] = "\u23F2",
        [0xEA39] = "\u2297",
        [0xE72B] = "\u2190",
        [0xE72A] = "\u2192",
        [0xE8C6] = "\u2702",
        [0xE72C] = "\u21BB",
    };
    private static readonly Dictionary<string, Bitmap> ImageCache = new(StringComparer.Ordinal);

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
        var segments = new List<InlineSegment>();
        CollectSegments(inline, segments);
        bool hasLinks = segments.Any(s => !string.IsNullOrWhiteSpace(s.Url));
        bool hasIcons = segments.Any(s => s.IconUri is not null || s.IconGlyph is not null);

        if (!hasLinks && !hasIcons)
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

        foreach (var segment in segments)
        {
            if (segment.IconGlyph is not null)
            {
                textBlock.Inlines!.Add(new InlineUIContainer
                {
                    Child = new TextBlock
                    {
                        Text = segment.IconGlyph,
                        FontFamily = new FontFamily("Segoe MDL2 Assets"),
                        FontSize = 16,
                        Foreground = Brush("TextBrush"),
                        VerticalAlignment = VerticalAlignment.Center,
                    },
                    BaselineAlignment = BaselineAlignment.Center,
                });
                continue;
            }

            if (segment.IconUri is not null)
            {
                textBlock.Inlines!.Add(new InlineUIContainer
                {
                    Child = new Border
                    {
                        Background = new SolidColorBrush(Color.Parse("#3A404A")),
                        CornerRadius = new CornerRadius(3),
                        Padding = new Thickness(3),
                        VerticalAlignment = VerticalAlignment.Center,
                        Child = new Image
                        {
                            Source = LoadBitmap(segment.IconUri),
                            Width = 16,
                            Height = 16,
                            Stretch = Stretch.Uniform,
                        },
                    },
                    BaselineAlignment = BaselineAlignment.Center,
                });
                continue;
            }

            if (string.IsNullOrEmpty(segment.Text))
            {
                continue;
            }

            if (string.IsNullOrWhiteSpace(segment.Url))
            {
                textBlock.Inlines!.Add(new Run(segment.Text));
            }
            else
            {
                var linkText = new TextBlock
                {
                    Text = segment.Text,
                    Foreground = Brush("LinkBrush"),
                    TextWrapping = TextWrapping.NoWrap,
                    TextDecorations = TextDecorations.Underline,
                    FontWeight = FontWeight.SemiBold,
                    Cursor = new Avalonia.Input.Cursor(Avalonia.Input.StandardCursorType.Hand),
                    FontSize = 13,
                    FontFamily = new FontFamily("Segoe UI"),
                };
                linkText.PointerPressed += (_, _) => navigateTo?.Invoke(segment.Url);
                textBlock.Inlines!.Add(new InlineUIContainer
                {
                    Child = linkText,
                    BaselineAlignment = BaselineAlignment.TextBottom,
                });
            }
        }

        return textBlock;
    }

    private static void CollectSegments(ContainerInline? container, List<InlineSegment> segments)
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

    private static void CollectSegments(Markdig.Syntax.Inlines.Inline inline, List<InlineSegment> segments)
    {
        switch (inline)
        {
            case LiteralInline literal:
                AddSegment(segments, literal.ToString(), null, null, null);
                break;
            case CodeInline code:
                AddSegment(segments, code.Content, null, null, null);
                break;
            case LineBreakInline lineBreak:
                AddSegment(segments, lineBreak.IsHard ? "\n" : " ", null, null, null);
                break;
            case HtmlInline html:
                var icon = GetMdl2Icon(html);
                AddSegment(segments, icon.Text, null, icon.IconUri, icon.IconGlyph);
                break;
            case LinkInline link:
                AddSegment(segments, GetInlinePlainText(link), link.Url, null, null);
                break;
            case ContainerInline container:
                foreach (var child in container)
                {
                    CollectSegments(child, segments);
                }
                break;
            default:
                AddSegment(segments, inline.ToString() ?? string.Empty, null, null, null);
                break;
        }
    }

    private static void AddSegment(List<InlineSegment> segments, string? text, string? url, string? iconUri, string? iconGlyph)
    {
        if (iconUri is not null || iconGlyph is not null)
        {
            segments.Add(new InlineSegment(null, null, iconUri, iconGlyph));
            return;
        }

        if (string.IsNullOrEmpty(text))
        {
            return;
        }

        if (segments.Count > 0 && segments[^1].IconUri is null && segments[^1].IconGlyph is null && segments[^1].Url == url)
        {
            segments[^1] = segments[^1] with { Text = segments[^1].Text + text };
            return;
        }

        segments.Add(new InlineSegment(text, url, null, null));
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
                return Mdl2Fallbacks.TryGetValue(codePoint, out var fallback)
                    ? fallback
                    : string.Empty;
            }
        }

        return string.Empty;
    }

    private static (string? Text, string? IconUri, string? IconGlyph) GetMdl2Icon(HtmlInline html)
    {
        var tag = html.Tag;
        const string marker = "ch=\"";
        int index = tag.IndexOf(marker, StringComparison.Ordinal);
        if (!tag.StartsWith("<mdl2 ", StringComparison.Ordinal) || index < 0)
            return (ConvertHtmlInline(html), null, null);

        int start = index + marker.Length;
        int end = tag.IndexOf('"', start);
        if (end <= start || !ushort.TryParse(tag[start..end], NumberStyles.HexNumber, null, out var codePoint))
            return (ConvertHtmlInline(html), null, null);

        return codePoint switch
        {
            0xE77B => (null, "avares://Vernacula.Avalonia/Assets/toolbar_icons/add_speaker.png", null),
            0xE916 => (null, "avares://Vernacula.Avalonia/Assets/toolbar_icons/adjust_times.png", null),
            0xEA39 => (null, "avares://Vernacula.Avalonia/Assets/toolbar_icons/suppress.png", null),
            0xE72B => (null, "avares://Vernacula.Avalonia/Assets/toolbar_icons/merge_prev.png", null),
            0xE72A => (null, "avares://Vernacula.Avalonia/Assets/toolbar_icons/merge_next.png", null),
            0xE8C6 => (null, "avares://Vernacula.Avalonia/Assets/toolbar_icons/split.png", null),
            0xE72C => (null, "avares://Vernacula.Avalonia/Assets/toolbar_icons/redo.png", null),
            _ => (ConvertHtmlInline(html), null, null),
        };
    }

    private static Bitmap LoadBitmap(string uri)
    {
        if (ImageCache.TryGetValue(uri, out var bitmap))
            return bitmap;

        bitmap = new Bitmap(AssetLoader.Open(new Uri(uri)));
        ImageCache[uri] = bitmap;
        return bitmap;
    }

    private sealed record InlineSegment(string? Text, string? Url, string? IconUri, string? IconGlyph);

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
