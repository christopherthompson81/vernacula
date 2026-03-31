using System.Collections.Generic;
using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Interactivity;
using Avalonia.Media;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views.Dialogs;

/// <summary>
/// Lets the user pick a token boundary at which to split a segment.
/// Pass pre-decoded token texts; the dialog returns the chosen split index.
/// </summary>
public partial class SplitSegmentDialog : Window
{
    private readonly IReadOnlyList<string> _tokenTexts;
    private readonly List<Button>          _tokenBtns = [];

    /// <summary>Index of the first token that belongs to the second (new) segment.</summary>
    public int SplitTokenIndex { get; private set; } = -1;

    /// <param name="tokenTexts">One decoded string per token, in order.</param>
    public SplitSegmentDialog(IReadOnlyList<string> tokenTexts)
    {
        InitializeComponent();

        _tokenTexts = tokenTexts;
        Loaded += (_, _) => 
        {
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
            BuildTokenButtons();
        };
    }

    private void BuildTokenButtons()
    {
        var surfaceBrush = (IBrush)Resources["SurfaceBrush"]!;
        var textBrush    = (IBrush)Resources["TextBrush"]!;

        for (int i = 0; i < _tokenTexts.Count; i++)
        {
            int    capturedIndex = i;
            string display       = _tokenTexts[i].Replace(" ", "·");

            var btn = new Button
            {
                Content         = display,
                Margin          = new Thickness(1),
                Padding         = new Thickness(4, 2, 4, 2),
                Background      = surfaceBrush,
                Foreground      = textBrush,
                BorderThickness = new Thickness(1),
                FontSize        = 12,
            };
            btn.SetValue(ToolTip.TipProperty, $"Token {i}: \"{_tokenTexts[i]}\"  — click to split before this token");

            // Token 0 cannot be the start of the second half (would leave empty first half)
            if (i == 0)
            {
                btn.IsEnabled = false;
                btn.Opacity   = 0.4;
            }
            else
            {
                btn.Click += (_, _) => SelectSplitPoint(capturedIndex);
            }

            _tokenBtns.Add(btn);
            TokenPanel.Children.Add(btn);
        }
    }

    private void SelectSplitPoint(int index)
    {
        SplitTokenIndex = index;
        OkBtn.IsEnabled = true;

        var accentColor  = ((SolidColorBrush)Resources["AccentBrush"]!).Color;
        var surfaceBrush = (IBrush)Resources["SurfaceBrush"]!;
        var textBrush    = (IBrush)Resources["TextBrush"]!;
        var subtextBrush = (IBrush)Resources["SubtextBrush"]!;

        for (int i = 0; i < _tokenBtns.Count; i++)
        {
            bool isSecond = i >= index;
            _tokenBtns[i].Background = isSecond
                ? new SolidColorBrush(Color.FromArgb(80, accentColor.R, accentColor.G, accentColor.B))
                : surfaceBrush;
            _tokenBtns[i].Foreground = isSecond ? textBrush : subtextBrush;
        }

        FirstPreview.Text  = string.Concat(_tokenTexts.Take(index)).Trim();
        SecondPreview.Text = string.Concat(_tokenTexts.Skip(index)).Trim();
    }

    private void OkBtn_Click(object sender, RoutedEventArgs e)
        => Close();

    private void CancelBtn_Click(object sender, RoutedEventArgs e)
        => Close();
}
