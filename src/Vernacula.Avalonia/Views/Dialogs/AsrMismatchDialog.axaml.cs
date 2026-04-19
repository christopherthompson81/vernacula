using Avalonia.Controls;
using Avalonia.Interactivity;
using Vernacula.App.Models;

namespace Vernacula.App.Views.Dialogs;

/// <summary>
/// Result returned from <see cref="AsrMismatchDialog"/>. Null from
/// <c>ShowDialog</c> means the user closed the window without picking
/// a button (treated as "cancel job" by callers).
/// </summary>
public enum AsrMismatchChoice
{
    SwitchBackend,
    KeepCurrent,
    CancelJob,
    /// <summary>
    /// User asserted "I know better" and picked a related language the
    /// current backend can transcribe (e.g. Croatian for Serbian).
    /// <see cref="AsrMismatchResult.ForcedIso"/> carries the picked code.
    /// </summary>
    ForceLanguage,
}

/// <summary>
/// Bundles the user's choice plus, for <see cref="AsrMismatchChoice.ForceLanguage"/>,
/// the ISO 639-1 code they picked from the current backend's supported set.
/// </summary>
public sealed record AsrMismatchResult(AsrMismatchChoice Choice, string? ForcedIso = null);

public partial class AsrMismatchDialog : Window
{
    public AsrMismatchDialog()
    {
        InitializeComponent();
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
    }

    /// <summary>
    /// Populate the body text and button labels based on the detected
    /// language + current / suggested backend.
    /// </summary>
    public void Configure(
        string detectedIso,
        string detectedName,
        float  detectedProbability,
        AsrBackend currentBackend,
        AsrBackend? suggestedBackend)
    {
        BodyText.Text =
            $"The audio appears to be {detectedName} ({detectedIso}) with " +
            $"{detectedProbability:P0} confidence, but your current ASR backend " +
            $"({AsrLanguageSupport.DisplayName(currentBackend)}) doesn't support {detectedName}.";

        if (suggestedBackend is { } alt)
        {
            SubtleText.Text =
                $"{AsrLanguageSupport.DisplayName(alt)} supports {detectedName}. Click " +
                $"“Switch ASR” to use it for this job.";
            SwitchBackendButton.Content = $"Switch to {AsrLanguageSupport.DisplayName(alt)}";
            SwitchBackendButton.IsEnabled = true;
            ForceLanguagePanel.IsVisible = false;
        }
        else
        {
            SubtleText.Text =
                $"No installed ASR backend supports {detectedName}. " +
                $"You can keep the current backend (the transcription will " +
                $"likely be wrong), pick a related language below, or cancel " +
                $"the job.";
            SwitchBackendButton.Content = "No alternative available";
            SwitchBackendButton.IsEnabled = false;

            // Conservative scope: list only languages the *current* backend
            // can actually transcribe. The other branch (suggested backend)
            // already covers cases where some installed backend supports the
            // detected ISO directly, so a union picker would add nothing here.
            ForceLanguageBox.ItemsSource = AsrLanguageSupport.LanguageOptions(currentBackend);
            ForceLanguageBox.SelectedIndex = -1;
            ForceLanguageButton.IsEnabled = false;
            ForceLanguagePanel.IsVisible = true;
        }
    }

    private void ForceLanguageBox_SelectionChanged(object? sender, SelectionChangedEventArgs e)
        => ForceLanguageButton.IsEnabled = ForceLanguageBox.SelectedItem is AsrLanguageOption;

    private void ForceLanguage_Click(object? sender, RoutedEventArgs e)
    {
        if (ForceLanguageBox.SelectedItem is AsrLanguageOption opt)
            Close(new AsrMismatchResult(AsrMismatchChoice.ForceLanguage, opt.Code));
    }

    private void SwitchBackend_Click(object? sender, RoutedEventArgs e)
        => Close(new AsrMismatchResult(AsrMismatchChoice.SwitchBackend));

    private void KeepCurrent_Click(object? sender, RoutedEventArgs e)
        => Close(new AsrMismatchResult(AsrMismatchChoice.KeepCurrent));

    private void CancelJob_Click(object? sender, RoutedEventArgs e)
        => Close(new AsrMismatchResult(AsrMismatchChoice.CancelJob));
}
