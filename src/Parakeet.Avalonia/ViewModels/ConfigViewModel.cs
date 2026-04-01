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
    public Func<string, string, Task>? EnqueueJob     { get; set; }
    public Action?                     NavigateBack   { get; set; }
    /// <summary>Opens an audio file picker and returns the chosen path, or null if cancelled.</summary>
    public Func<Task<string?>>?        PickAudioFile  { get; set; }

    private bool CanStart() =>
        !string.IsNullOrWhiteSpace(AudioFilePath) &&
        !string.IsNullOrWhiteSpace(JobName) &&
        File.Exists(AudioFilePath);

    [RelayCommand]
    private async Task SelectAudioFileAsync()
    {
        if (PickAudioFile is null) return;
        var path = await PickAudioFile();
        if (path is null) return;
        AudioFilePath = path;
        JobName       = Path.GetFileNameWithoutExtension(path);
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
