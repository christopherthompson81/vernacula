using Avalonia.Controls;
using Avalonia.Interactivity;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views.Dialogs;

public partial class AdjustTimesDialog : Window
{
    public double NewStartTime { get; private set; }
    public double NewEndTime   { get; private set; }

    public AdjustTimesDialog(double currentStart, double currentEnd)
    {
        InitializeComponent();
        Loaded += (_, _) =>
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);

        StartBox.Text = currentStart.ToString("F3");
        EndBox.Text   = currentEnd.ToString("F3");

        StartBox.PreviewMouseWheel += (_, e) =>
        {
            if (double.TryParse(StartBox.Text, out double v))
                StartBox.Text = Math.Max(0, v + (e.Delta > 0 ? 0.1 : -0.1)).ToString("F3");
            e.Handled = true;
        };
        EndBox.PreviewMouseWheel += (_, e) =>
        {
            if (double.TryParse(EndBox.Text, out double v))
                EndBox.Text = (v + (e.Delta > 0 ? 0.1 : -0.1)).ToString("F3");
            e.Handled = true;
        };
    }

    private void OkBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!double.TryParse(StartBox.Text, out double start) || start < 0)
        {
            ShowError("Start time must be a non-negative number.");
            return;
        }
        if (!double.TryParse(EndBox.Text, out double end) || end <= start)
        {
            ShowError("End time must be greater than start time.");
            return;
        }
        NewStartTime = start;
        NewEndTime   = end;
        Close();
    }

    private void CancelBtn_Click(object sender, RoutedEventArgs e)
        => Close();

    private void ShowError(string msg)
    {
        ErrorLabel.Text       = msg;
        ErrorLabel.Visibility = Visibility.Visible;
    }
}
