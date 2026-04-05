using System.ComponentModel;
using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Threading;
using Vernacula.App;
using Vernacula.App.Models;
using Vernacula.App.Services;

namespace Vernacula.App.Views;

public partial class HelpWindow : Window
{
    private string? _currentTopicId;
    public string? CurrentTopicId => _currentTopicId;
    private readonly Dictionary<string, (Grid Root, Border Background, TextBlock Label)> _sidebarButtons = new();

    public HelpWindow()
    {
        InitializeComponent();
        Closing += Window_Closing;
    }

    public HelpWindow(string? topicId) : this()
    {
        _currentTopicId = topicId;
    }

    private void Window_Loaded(object? sender, RoutedEventArgs e)
    {
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
        Title = Loc.Instance["help_window_title"];
        BuildSidebar();
        DisplayTopic(_currentTopicId ?? HelpService.IndexTopic.TopicId);
        ThemeManager.ThemeChanged    += OnThemeChanged;
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
    }

    private void Window_Closing(object? sender, WindowClosingEventArgs e)
    {
        ThemeManager.ThemeChanged    -= OnThemeChanged;
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
    }

    // ── Theme change ─────────────────────────────────────────────────────────

    private void OnThemeChanged(AppTheme _)
    {
        // ThemeManager.Apply already called WindowHelper.SetDarkMode on all open windows.
        // We only need to re-render the FlowDocument with the new palette.
        Dispatcher.UIThread.InvokeAsync(() =>
        {
            if (_currentTopicId is not null)
                DisplayTopic(_currentTopicId);
        });
    }

    // ── Language change ──────────────────────────────────────────────────────

    private void OnLocalePropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage)) return;

        // Rebuild sidebar labels and reload the current page in the new language.
        // HelpService.LoadMarkdown falls back to English automatically if no localised
        // resource exists for the new language.
        Dispatcher.UIThread.InvokeAsync(() =>
        {
            Title = Loc.Instance["help_window_title"];
            BuildSidebar();
            if (_currentTopicId is not null)
            {
                DisplayTopic(_currentTopicId);
            }
        });
    }

    // ── Sidebar ──────────────────────────────────────────────────────────────

    private void BuildSidebar()
    {
        SidebarPanel.Children.Clear();
        _sidebarButtons.Clear();

        // "Help" home/index button at the top
        SidebarPanel.Children.Add(MakeSidebarButton(HelpService.IndexTopic));

        foreach (var group in HelpService.Groups)
        {
            SidebarPanel.Children.Add(new TextBlock
            {
                Text = Loc.Instance[group.GroupTitle],
                Foreground = GetBrush("SubtextBrush"),
                FontWeight = FontWeight.SemiBold,
                FontSize = 11,
                Margin = new Thickness(6, 10, 6, 2),
                Opacity = 1,
            });

            foreach (var topic in group.Topics)
                SidebarPanel.Children.Add(MakeSidebarButton(topic));
        }

        Console.WriteLine($"[HelpNav] sidebar children={SidebarPanel.Children.Count} buttonCount={_sidebarButtons.Count}");
    }

    private Grid MakeSidebarButton(HelpTopic topic)
    {
        string labelText = GetSidebarLabel(topic);
        Console.WriteLine($"[HelpNav] make topic={topic.TopicId} label='{labelText}'");

        var label = new TextBlock
        {
            Text = labelText,
            TextWrapping = TextWrapping.Wrap,
            Foreground = GetBrush("TextBrush"),
            FontSize = 13,
            Opacity = 1,
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Left,
            Margin = new Thickness(10, 5, 8, 5),
        };

        var background = new Border
        {
            CornerRadius = new CornerRadius(4),
            MinHeight = 28,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Stretch,
            Background = Brushes.Transparent,
        };

        var row = new Grid
        {
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Stretch,
            Margin = new Thickness(0, 1, 0, 1),
            MinHeight = 28,
            Cursor = new Avalonia.Input.Cursor(Avalonia.Input.StandardCursorType.Hand),
        };
        row.Children.Add(background);
        row.Children.Add(label);
        row.PointerPressed += (_, _) => DisplayTopic(topic.TopicId);

        _sidebarButtons[topic.TopicId] = (row, background, label);
        return row;
    }

    private static string GetSidebarLabel(HelpTopic topic)
        => Loc.Instance["help_topic_" + topic.TopicId] is { } text && !string.IsNullOrWhiteSpace(text) && text != "help_topic_" + topic.TopicId
            ? text
            : topic.Title;

    // ── Topic display ─────────────────────────────────────────────────────────

    public void DisplayTopic(string topicId)
    {
        var topic = HelpService.FindById(topicId) ?? HelpService.IndexTopic;
        _currentTopicId = topic.TopicId;
        try
        {
            string markdown = HelpService.LoadMarkdown(topic);
            var content = MarkdownFlowBuilder.Build(markdown, OnLinkNavigate);
            ContentViewer.Content = content;
            ContentScrollViewer.Offset = new Vector(0, 0);
        }
        catch (Exception ex)
        {
            ContentViewer.Content = new TextBlock
            {
                Text = $"Error loading help topic: {ex.Message}",
                Foreground = GetBrush("TextBrush"),
                TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(16),
            };
        }

        UpdateSidebarSelection(topic.TopicId);
    }

    private void OnLinkNavigate(string url)
    {
        if (url.StartsWith("http", StringComparison.OrdinalIgnoreCase))
        {
            Process.Start(new ProcessStartInfo { FileName = url, UseShellExecute = true });
            return;
        }

        var current = HelpService.FindById(_currentTopicId!) ?? HelpService.IndexTopic;
        var target  = HelpService.ResolveRelativeLink(current, url);
        if (target is not null)
            DisplayTopic(target.TopicId);
    }

    private void UpdateSidebarSelection(string topicId)
    {
        var overlayBrush = GetBrush("OverlayBrush");
        var accentBrush  = GetBrush("AccentBrush");
        var textBrush    = GetBrush("TextBrush");

        foreach (var (id, entry) in _sidebarButtons)
        {
            bool selected  = id == topicId;
            entry.Background.Background = selected ? overlayBrush : Brushes.Transparent;
            entry.Label.Foreground = selected ? accentBrush : textBrush;
        }

        if (_sidebarButtons.TryGetValue(topicId, out var current))
        {
            Console.WriteLine($"[HelpNav] selected={topicId} text='{current.Label.Text}' fg='{current.Label.Foreground}'");
        }
    }

    private static IBrush GetBrush(string key)
    {
        var app = Application.Current;
        if (app?.Resources.TryGetResource(key, null, out var value) == true && value is IBrush brush)
        {
            return brush;
        }

        return Brushes.Magenta;
    }

}
