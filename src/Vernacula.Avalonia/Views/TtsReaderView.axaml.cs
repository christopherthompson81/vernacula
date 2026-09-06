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
        RenderMarkdownCheck.Content = Loc.Instance["tts_render_markdown"];
        IpaAnnotationCheck.Content  = Loc.Instance["tts_ipa_annotation"];
        CancelJobButton.Content     = Loc.Instance["btn_cancel_job"];
        BackButton.Content          = Loc.Instance["tts_reader_back"];
    }
}
