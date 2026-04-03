using Avalonia.Controls;
using Avalonia.Interactivity;
using ParakeetCSharp.Models;
using System.ComponentModel;

namespace ParakeetCSharp.Views.Dialogs;

public partial class AboutDialog : Window
{
    public AboutDialog()
    {
        InitializeComponent();
        ApplyLocalizedText();
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
        Loaded += (_, _) =>
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
    }

    private void OK_Click(object sender, RoutedEventArgs e) => Close();

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;

        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        Title = Loc.Instance["about_title"];
        AboutHeadingText.Text = Loc.Instance["about_heading"];
        AboutDescriptionText.Text = Loc.Instance["about_description"];
        AboutTechText.Text = Loc.Instance["about_tech"];
        AboutCreditText.Text = Loc.Instance["about_credit"];
        AboutOkButton.Content = Loc.Instance["btn_ok"];
    }

    protected override void OnClosed(EventArgs e)
    {
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
        base.OnClosed(e);
    }
}
