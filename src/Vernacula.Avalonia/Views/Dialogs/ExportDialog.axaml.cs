using Avalonia.Controls;
using Avalonia.Interactivity;
using Vernacula.App.Models;
using System.ComponentModel;

namespace Vernacula.App.Views.Dialogs;

public enum ExportFormat { Xlsx, Csv, Json, Srt, Md, Docx, Db }

public partial class ExportDialog : Window
{
    public ExportFormat SelectedFormat { get; private set; } = ExportFormat.Xlsx;
    public bool DialogResult { get; private set; }

    public ExportDialog()
    {
        InitializeComponent();
        ApplyLocalizedText();
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
        Loaded += (_, _) =>
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
    }

    private void Save_Click(object sender, RoutedEventArgs e)
    {
        SelectedFormat = FormatCombo.SelectedIndex switch
        {
            0 => ExportFormat.Xlsx,
            1 => ExportFormat.Csv,
            2 => ExportFormat.Json,
            3 => ExportFormat.Srt,
            4 => ExportFormat.Md,
            5 => ExportFormat.Docx,
            6 => ExportFormat.Db,
            _ => ExportFormat.Xlsx,
        };
        DialogResult = true;
        Close();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;

        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        Title = Loc.Instance["modal_export_heading"];
        FormatLabelText.Text = Loc.Instance["label_format"];
        FormatExcelItem.Content = Loc.Instance["format_excel"];
        FormatCsvItem.Content = Loc.Instance["format_csv"];
        FormatJsonItem.Content = Loc.Instance["format_json"];
        FormatSrtItem.Content = Loc.Instance["format_srt"];
        FormatMdItem.Content = Loc.Instance["format_md"];
        FormatDocxItem.Content = Loc.Instance["format_docx"];
        FormatDbItem.Content = Loc.Instance["format_db"];
        SaveButton.Content = Loc.Instance["btn_save"];
        CancelButton.Content = Loc.Instance["btn_cancel"];
    }

    protected override void OnClosed(EventArgs e)
    {
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
        base.OnClosed(e);
    }
}
