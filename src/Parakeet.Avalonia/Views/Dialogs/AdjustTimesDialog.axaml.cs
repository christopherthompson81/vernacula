using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views.Dialogs;

public partial class AdjustTimesDialog : Window
{
    public double NewStartTime { get; private set; }
    public double NewEndTime   { get; private set; }
    public bool DialogResult   { get; private set; }

    public AdjustTimesDialog()
    {
        InitializeComponent();
    }

    public AdjustTimesDialog(double currentStart, double currentEnd) : this()
    {
        Loaded += (_, _) =>
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);

        StartBox.Text = currentStart.ToString("F3");
        EndBox.Text   = currentEnd.ToString("F3");

        // TODO: Port mouse wheel handling for Avalonia
        // StartBox.MouseWheel += OnStartBoxMouseWheel;
        // EndBox.MouseWheel += OnEndBoxMouseWheel;
    }

    // TODO: Port to Avalonia - requires different event handling
    // private void OnStartBoxMouseWheel(object? sender, WheelChangedEventArgs e)
    // {
    //     if (double.TryParse(StartBox.Text, out double v))
    //         StartBox.Text = Math.Max(0, v + (e.Delta.Y > 0 ? 0.1 : -0.1)).ToString("F3");
    //     e.Handled = true;
    // }

    // private void OnEndBoxMouseWheel(object? sender, WheelChangedEventArgs e)
    // {
    //     if (double.TryParse(EndBox.Text, out double v))
    //         EndBox.Text = (v + (e.Delta.Y > 0 ? 0.1 : -0.1)).ToString("F3");
    //     e.Handled = true;
    // }

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

        ErrorLabel.IsVisible = false;
        NewStartTime = start;
        NewEndTime   = end;
        DialogResult = true;
        Close();
    }

    private void CancelBtn_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }

    private void ShowError(string msg)
    {
        ErrorLabel.Text = msg;
        ErrorLabel.IsVisible = true;
    }
}
