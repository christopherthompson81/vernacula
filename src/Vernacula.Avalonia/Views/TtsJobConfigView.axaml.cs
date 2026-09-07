using System.ComponentModel;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.VisualTree;

namespace Vernacula.App.Views;

public partial class TtsJobConfigView : UserControl
{
    public TtsJobConfigView()
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
        ConfigHeadingText.Text        = Loc.Instance["tts_config_heading"];
        DocumentLabelText.Text        = Loc.Instance["label_document"];
        JobNameLabelText.Text         = Loc.Instance["label_job_name"];
        EngineLabelText.Text          = Loc.Instance["label_engine"];
        ReferenceClipLabelText.Text   = Loc.Instance["label_reference_clip"];
        BrowseButton.Content          = Loc.Instance["btn_browse"];
        BackButton.Content            = Loc.Instance["btn_back"];
        StartButton.Content           = Loc.Instance["btn_start_tts"];
    }

    // The type-ahead pickers: focusing one opens its full list and selects its text, so the
    // next keystroke starts a fresh search instead of appending to the current choice's name.
    private void OnPickerGotFocus(object? sender, GotFocusEventArgs e)
    {
        if (sender is not AutoCompleteBox box) return;
        box.FindDescendantOfType<TextBox>()?.SelectAll();
        box.IsDropDownOpen = true;
    }
}
