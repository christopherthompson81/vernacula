using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Vernacula.Avalonia.ViewModels;

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
        Console.WriteLine($"[ConfigVM] SelectAudioFileAsync() — PickAudioFile is null={PickAudioFile is null}");
        if (PickAudioFile is null) return;
        var path = await PickAudioFile();
        Console.WriteLine($"[ConfigVM] PickAudioFile returned: '{path}'");
        if (path is null) return;
        AudioFilePath = path;
        JobName       = Path.GetFileNameWithoutExtension(path);
        Console.WriteLine($"[ConfigVM] Set AudioFilePath='{AudioFilePath}', JobName='{JobName}', File.Exists={File.Exists(path)}");
    }

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task Start()
    {
        Console.WriteLine($"[ConfigVM] Start() called — AudioFilePath='{AudioFilePath}', JobName='{JobName}', EnqueueJob is null={EnqueueJob is null}");
        try
        {
            if (EnqueueJob != null)
            {
                Console.WriteLine("[ConfigVM] Calling EnqueueJob...");
                await EnqueueJob(AudioFilePath, JobName);
            }
            else
            {
                Console.WriteLine("[ConfigVM] EnqueueJob is NULL — job will NOT be added!");
            }

            AudioFilePath = "";
            JobName       = "";
            Console.WriteLine("[ConfigVM] Calling NavigateBack...");
            NavigateBack?.Invoke();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ConfigVM] Start() EXCEPTION: {ex}");
        }
    }

    [RelayCommand]
    private void Back()
    {
        AudioFilePath = "";
        JobName       = "";
        NavigateBack?.Invoke();
    }
}
