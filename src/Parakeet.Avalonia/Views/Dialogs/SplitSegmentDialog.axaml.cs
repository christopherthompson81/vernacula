using System.Collections.Generic;
using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
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
    private IReadOnlyList<string>? _tokenTexts;
    private readonly List<Button> _tokenBtns = [];

    /// <summary>Index of the first token that belongs to the second (new) segment.</summary>
    public int SplitTokenIndex { get; private set; } = -1;
    public bool DialogResult { get; private set; }

    public SplitSegmentDialog()
    {
        InitializeComponent();
        KeyDown += OnDialogKeyDown;
        Closed += (_, _) =>
        {
            if (!DialogResult)
                SplitTokenIndex = -1;
        };
    }

    /// <param name="tokenTexts">One decoded string per token, in order.</param>
    public SplitSegmentDialog(IReadOnlyList<string> tokenTexts) : this()
    {
        _tokenTexts = tokenTexts;
        Loaded += (_, _) =>
        {
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
            BuildTokenButtons();
        };
    }

    private void BuildTokenButtons()
    {
        if (_tokenTexts is null) return;

        TokenPanel.Children.Clear();
        _tokenBtns.Clear();

        for (int i = 0; i < _tokenTexts.Count; i++)
        {
            int    capturedIndex = i;
            string display       = _tokenTexts[i].Replace(" ", "·");

            var btn = new Button
            {
                Content         = display,
                Classes         = { "token-button" },
                Margin          = new Thickness(1),
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

        for (int i = 0; i < _tokenBtns.Count; i++)
        {
            bool isSecond = i >= index;
            _tokenBtns[i].Classes.Set("second-half", isSecond);
        }

        FirstPreview.Text  = string.Concat(_tokenTexts!.Take(index)).Trim();
        SecondPreview.Text = string.Concat(_tokenTexts!.Skip(index)).Trim();
    }

    private void OkBtn_Click(object sender, RoutedEventArgs e)
    {
        if (SplitTokenIndex <= 0)
            return;

        DialogResult = true;
        Close();
    }

    private void CancelBtn_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        SplitTokenIndex = -1;
        Close();
    }

    private void OnDialogKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && OkBtn.IsEnabled)
        {
            OkBtn_Click(OkBtn, new RoutedEventArgs());
            e.Handled = true;
        }
        else if (e.Key == Key.Escape)
        {
            CancelBtn_Click(CancelBtn, new RoutedEventArgs());
            e.Handled = true;
        }
    }
}
