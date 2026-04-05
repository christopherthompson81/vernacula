using Avalonia.Controls;
using System.ComponentModel;

namespace Vernacula.App.Views;

public partial class ConfigView : UserControl
{
    public ConfigView()
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
        ConfigHeadingText.Text = Loc.Instance["config_heading"];
        AudioFileLabelText.Text = Loc.Instance["label_audio_file"];
        BrowseButton.Content = Loc.Instance["btn_browse"];
        JobNameLabelText.Text = Loc.Instance["label_job_name"];
        BackButton.Content = Loc.Instance["btn_back"];
        StartButton.Content = Loc.Instance["btn_start"];
    }
}
