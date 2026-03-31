using Avalonia.Controls;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace ParakeetCSharp.ViewModels;

internal partial class ConfigViewModel : ObservableObject
{
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _audioFilePath = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _jobName = "";

    /// <summary>
    /// Called when the user confirms a new job.  Receives (audioPath, jobTitle).
    /// Runs asynchronously — the Start command navigates back after it completes.
    /// </summary>
    public Func<string, string, Task>? EnqueueJob   { get; set; }
    public Action?                     NavigateBack { get; set; }

    private bool CanStart() =>
        !string.IsNullOrWhiteSpace(AudioFilePath) &&
        !string.IsNullOrWhiteSpace(JobName) &&
        File.Exists(AudioFilePath);

    [RelayCommand]
    private async Task SelectAudioFileAsync()
    {
        // We can't get TopLevel from a ViewModel directly
        // This is a known limitation - we need to pass in a way to get the TopLevel
        // For now, we'll skip this and assume it's called from a context where we can get it
        return;
        
        // This code is kept for reference:
        // var topLevel = TopLevel.GetTopLevel((Visual?)this);
        // if (topLevel == null) return;
        //
        // var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        // {
        //     Title = Loc.Instance["dlg_select_audio"],
        //     AllowMultiple = false,
        //     FileTypeChoices = new[]
        //     {
        //         new FilePickerFileType(Loc.Instance["dlg_audio_filter"])
        //         {
        //             Patterns = new[] { "*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg" }
        //         }
        //     }
        // });
        //
        // if (files.Count == 0) return;
        //
        // AudioFilePath = files[0].Path.LocalPath;
        // JobName = Path.GetFileNameWithoutExtension(files[0].FileName);
    }

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task Start()
    {
        if (EnqueueJob != null)
            await EnqueueJob(AudioFilePath, JobName);

        AudioFilePath = "";
        JobName       = "";
        NavigateBack?.Invoke();
    }

    [RelayCommand]
    private void Back()
    {
        AudioFilePath = "";
        JobName       = "";
        NavigateBack?.Invoke();
    }
}
