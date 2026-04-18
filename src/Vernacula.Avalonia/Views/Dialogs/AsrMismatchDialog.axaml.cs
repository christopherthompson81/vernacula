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
}

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
            $"({HumanName(currentBackend)}) doesn't support {detectedName}.";

        if (suggestedBackend is { } alt)
        {
            SubtleText.Text =
                $"{HumanName(alt)} supports {detectedName}. Click " +
                $"“Switch ASR” to use it for this job.";
            SwitchBackendButton.Content = $"Switch to {HumanName(alt)}";
            SwitchBackendButton.IsEnabled = true;
        }
        else
        {
            SubtleText.Text =
                $"No available ASR backend supports {detectedName}. " +
                $"You can keep the current backend (the transcription will " +
                $"likely be wrong), or cancel the job and pick a different " +
                $"audio file.";
            SwitchBackendButton.Content = "No alternative available";
            SwitchBackendButton.IsEnabled = false;
        }
    }

    private static string HumanName(AsrBackend b) => b switch
    {
        AsrBackend.Parakeet  => "Parakeet",
        AsrBackend.Cohere    => "Cohere Transcribe",
        AsrBackend.Qwen3Asr  => "Qwen3-ASR",
        AsrBackend.VibeVoice => "VibeVoice-ASR",
        _                    => b.ToString(),
    };

    private void SwitchBackend_Click(object? sender, RoutedEventArgs e)
        => Close(AsrMismatchChoice.SwitchBackend);

    private void KeepCurrent_Click(object? sender, RoutedEventArgs e)
        => Close(AsrMismatchChoice.KeepCurrent);

    private void CancelJob_Click(object? sender, RoutedEventArgs e)
        => Close(AsrMismatchChoice.CancelJob);
}
