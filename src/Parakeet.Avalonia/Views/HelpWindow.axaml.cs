using System.ComponentModel;
using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Threading;
using ParakeetCSharp;
using ParakeetCSharp.Models;
using ParakeetCSharp.Services;

namespace ParakeetCSharp.Views;

/// <summary>
/// ⚠️ AVALONIA MIGRATION REQUIRED ⚠️
/// 
/// This window uses FlowDocument which doesn't exist in Avalonia.
/// The following WPF types need to be replaced:
/// 
/// - Window → Avalonia.Controls.Window (already correct)
/// - TextBlock → Avalonia.Controls.TextBlock (already correct)
/// - Button → Avalonia.Controls.Button (already correct)
/// - ScrollViewer → Avalonia.Controls.ScrollViewer (already correct)
/// - DependencyObject → AvaloniaObject
/// - VisualTreeHelper → VisualTreeHelper (Avalonia has this)
/// - FlowDocument → N/A (use MarkdownFlowBuilder which returns a Panel)
/// - Run → N/A (replaced by TextBlock)
/// - Paragraph → N/A (replaced by TextBlock)
/// - Brush → IBrush
/// - FontWeights → FontWeights (Avalonia has this)
/// - Thickness → Thickness (Avalonia has this)
/// 
/// The ContentViewer control is expected to be a DockPanel or similar container
/// that hosts the markdown-rendered content from MarkdownFlowBuilder.Build().
/// </summary>
public partial class HelpWindow : Window
{
    private string? _currentTopicId;
    private readonly Dictionary<string, Button> _sidebarButtons = new();

    public HelpWindow()
    {
        InitializeComponent();
    }

    public HelpWindow(string? topicId) : this()
    {
        _currentTopicId = topicId;
    }

    private void Window_SourceInitialized(object sender, EventArgs e)
    {
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
        BuildSidebar();
        DisplayTopic(_currentTopicId ?? HelpService.IndexTopic.TopicId);
        ThemeManager.ThemeChanged    += OnThemeChanged;
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
    }

    private void Window_Closing(object sender, CancelEventArgs e)
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
            // TODO: Port to Avalonia - FontWeights not available
            // SidebarPanel.Children.Add(new TextBlock
            // {
            //     Text       = Loc.Instance[group.GroupTitle],
            //     Foreground = (IBrush)Application.Current.Resources["SubtextBrush"],
            //     FontWeight = FontWeights.SemiBold,
            //     FontSize   = 11,
            //     Margin     = new Thickness(6, 10, 6, 2),
            // });

            foreach (var topic in group.Topics)
                SidebarPanel.Children.Add(MakeSidebarButton(topic));
        }
    }

    private Button MakeSidebarButton(HelpTopic topic)
    {
        var btn = new Button
        {
            Content             = Loc.Instance["help_topic_" + topic.TopicId],
            Tag                 = topic.TopicId,
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Stretch,
            Margin              = new Thickness(0, 1, 0, 1),
        };
        btn.Click += (_, _) => DisplayTopic(topic.TopicId);
        _sidebarButtons[topic.TopicId] = btn;
        return btn;
    }

    // ── Topic display ─────────────────────────────────────────────────────────

    public void DisplayTopic(string topicId)
    {
        var topic = HelpService.FindById(topicId) ?? HelpService.IndexTopic;
        _currentTopicId = topic.TopicId;

        try
        {
            string markdown = HelpService.LoadMarkdown(topic);
            // MarkdownFlowBuilder.Build() now returns a Panel instead of FlowDocument
            var contentPanel = MarkdownFlowBuilder.Build(markdown, OnLinkNavigate);
            // Add the panel to ContentPanel
            ContentPanel.Children.Clear();
            ContentPanel.Children.Add(contentPanel);
        }
        catch (Exception ex)
        {
            // Show error message in ContentPanel
            ContentPanel.Children.Clear();
            ContentPanel.Children.Add(new TextBlock
                {
                    Text = $"Error loading help topic: {ex.Message}",
                    TextWrapping = TextWrapping.Wrap
                });
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
        var overlayBrush = (IBrush)Application.Current!.Resources["OverlayBrush"]!;
        var accentBrush  = (IBrush)Application.Current.Resources["AccentBrush"]!;
        var textBrush    = (IBrush)Application.Current.Resources["TextBrush"]!;

        foreach (var (id, btn) in _sidebarButtons)
        {
            bool selected  = id == topicId;
            btn.Background = selected ? overlayBrush : Brushes.Transparent;
            btn.Foreground = selected ? accentBrush  : textBrush;
        }
    }

     // ── Scroll speed ──────────────────────────────────────────────────────────

    // TODO: Port to Avalonia - requires different event handling and ScrollViewer API
    // private ScrollViewer? _contentScrollViewer;

    // private void ContentViewer_PreviewMouseWheel(object sender, WheelEventArgs e)
    // {
    //     _contentScrollViewer ??= FindDescendant<ScrollViewer>(ContentViewer);
    //     if (_contentScrollViewer is null) return;
    //     e.Handled = true;
    // }

    // private static T? FindDescendant<T>(AvaloniaObject parent) where T : AvaloniaObject
    // {
    //     for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
    //     {
    //         var child = VisualTreeHelper.GetChild(parent, i);
    //         if (child is T match) return match;
    //         var result = FindDescendant<T>(child);
    //         if (result is not null) return result;
    //     }
    //     return null;
    // }
}
