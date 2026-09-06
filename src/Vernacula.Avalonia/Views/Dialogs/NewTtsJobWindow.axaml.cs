using Avalonia.Controls;
using Vernacula.App.Models;
using System.ComponentModel;

namespace Vernacula.App.Views.Dialogs;

public partial class NewTtsJobWindow : Window
{
    public NewTtsJobWindow()
    {
        InitializeComponent();
        ApplyLocalizedText();
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
        Loaded += (_, _) =>
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
    }

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;
        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        Title = Loc.Instance["menu_new_tts_job"];
    }

    protected override void OnClosed(EventArgs e)
    {
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
        base.OnClosed(e);
    }
}
