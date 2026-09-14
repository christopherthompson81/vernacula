using System.ComponentModel;
using Avalonia.Controls;

namespace Vernacula.App.Views;

public partial class TtsReaderView : UserControl
{
    public TtsReaderView()
    {
        InitializeComponent();
        ApplyLocalizedText();
        Loaded += (_, _) => Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
        Unloaded += (_, _) => Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
    }

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;
        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        RawMarkdownCheck.Content    = Loc.Instance["tts_raw_markdown"];
        ExportButton.Content        = Loc.Instance["btn_export_tts"];
        IpaAnnotationCheck.Content  = Loc.Instance["tts_ipa_annotation"];
        CancelJobButton.Content     = Loc.Instance["btn_cancel_job"];
        BackButton.Content          = Loc.Instance["tts_reader_back"];
    }

    /// <summary>Blur commits the card — the ordinary way to finish editing one and move on.</summary>
    private void BlockEditor_LostFocus(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if ((sender as Avalonia.Controls.Control)?.DataContext is ViewModels.BlockItemViewModel b)
            b.CommitEditCommand.Execute(null);
    }

    /// <summary>Escape abandons the card's edit; Ctrl+Enter commits without reaching for the mouse.
    /// ⚠ Plain Enter must NOT commit — a paragraph can contain line breaks.</summary>
    private void BlockEditor_KeyDown(object? sender, Avalonia.Input.KeyEventArgs e)
    {
        if ((sender as Avalonia.Controls.Control)?.DataContext is not ViewModels.BlockItemViewModel b) return;
        if (e.Key == Avalonia.Input.Key.Escape)
        {
            b.CancelEditCommand.Execute(null);
            e.Handled = true;
        }
        else if (e.Key == Avalonia.Input.Key.Enter && e.KeyModifiers.HasFlag(Avalonia.Input.KeyModifiers.Control))
        {
            b.CommitEditCommand.Execute(null);
            e.Handled = true;
        }
    }
}
