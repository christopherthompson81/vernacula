using Avalonia.Controls;
using Avalonia.Interactivity;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views.Dialogs;

public enum ExportFormat { Xlsx, Csv, Json, Srt, Md, Docx, Db }

public partial class ExportDialog : Window
{
    public ExportFormat SelectedFormat { get; private set; } = ExportFormat.Xlsx;
    public bool DialogResult { get; private set; }

    public ExportDialog()
    {
        InitializeComponent();
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
}
