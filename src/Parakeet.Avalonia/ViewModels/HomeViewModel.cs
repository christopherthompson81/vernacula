using System.Collections.ObjectModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ParakeetCSharp.Models;
using ParakeetCSharp.Services;

namespace ParakeetCSharp.ViewModels;

internal partial class HomeViewModel : ObservableObject
{
    private readonly ModelManagerService _modelMgr;
    private readonly ControlDb           _controlDb;

    [ObservableProperty] private string _modelStatusText  = "";
    [ObservableProperty] private IBrush _modelStatusBrush = Brushes.Gray;
    [ObservableProperty] private bool   _modelsReady      = false;
    [ObservableProperty] private bool   _modelsWarning    = false;
    [ObservableProperty] private bool   _updateAvailable  = false;

    public ObservableCollection<JobRecord> Jobs { get; } = new();

    public Action?               NavigateToConfig  { get; set; }
    public Action<JobRecord>?    RequeueJob        { get; set; }
    public Action<int>?          CancelJob         { get; set; }
    public Action<JobRecord>?    LoadJobToResults  { get; set; }
    public Action<JobRecord>?    MonitorJob        { get; set; }
    public Func<string[], Task>? BulkEnqueueFiles  { get; set; }
    public Action?               OpenSettings      { get; set; }

    public HomeViewModel(ModelManagerService modelMgr, ControlDb controlDb)
    {
        _modelMgr  = modelMgr;
        _controlDb = controlDb;
        ModelStatusText = Loc.Instance["model_status_checking"];

        Loc.Instance.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == "Item[]")
                UpdateStatusText();
        };
    }

    private void UpdateStatusText()
    {
        if (ModelsReady)
            ModelStatusText = Loc.Instance.T("model_status_ok", new() { ["count"] = _modelMgr.GetPresentFiles().Count.ToString() });
        else if (ModelsWarning)
            ModelStatusText = Loc.Instance["settings_model_warning"];
    }

    public async Task InitializeAsync()
    {
        RefreshJobs();

        foreach (var job in Jobs.ToList())
        {
            if (job.Status == JobStatus.Running)
            {
                _controlDb.UpdateJobStatus(job.JobId, JobStatus.Failed);
                job.Status = JobStatus.Failed;
            }
            else if (job.Status == JobStatus.Queued)
            {
                _controlDb.UpdateJobStatus(job.JobId, JobStatus.Cancelled);
                job.Status = JobStatus.Cancelled;
            }
        }

        await CheckModelsAsync();
    }

    public void RefreshJobs()
    {
        Jobs.Clear();
        foreach (var job in _controlDb.GetJobs())
            Jobs.Add(job);
    }

    [RelayCommand]
    internal async Task CheckModelsAsync()
    {
        ModelStatusText  = Loc.Instance["model_status_checking"];
        ModelStatusBrush = Application.Current!.Resources["SubtextBrush"] as IBrush ?? Brushes.Gray;

        IReadOnlyList<string> missing = [];
        await Task.Run(() => missing = _modelMgr.GetMissingFiles());

        ModelsReady   = missing.Count == 0;
        ModelsWarning = missing.Count > 0;
        UpdateStatusText();

        ModelStatusBrush = Application.Current.Resources[ModelsReady ? "GreenBrush" : "YellowBrush"] as IBrush
                           ?? (ModelsReady ? Brushes.LimeGreen : Brushes.Goldenrod);
    }

    [RelayCommand]
    private void OpenSettingsWindow() => OpenSettings?.Invoke();

    [RelayCommand]
    private void DismissUpdateBanner() => UpdateAvailable = false;

    [RelayCommand]
    private void NewTranscription() => NavigateToConfig?.Invoke();

    [RelayCommand]
    private async Task BulkAddJobs()
    {
        // ViewModels can't directly access TopLevel
        // This needs to be handled by the view or through a service
        // For now, this is a placeholder
        return;
    }

    [RelayCommand]
    private void RenameJob(JobRecord job)
    {
        if (!string.IsNullOrWhiteSpace(job.JobTitle))
            _controlDb.UpdateJobTitle(job.JobId, job.JobTitle);
    }

    [RelayCommand] private void ResumeJob(JobRecord job)        => RequeueJob?.Invoke(job);
    [RelayCommand] private void CancelQueuedJob(JobRecord job)  => CancelJob?.Invoke(job.JobId);
    [RelayCommand] private void MonitorQueuedJob(JobRecord job) => MonitorJob?.Invoke(job);

    [RelayCommand]
    private void LoadJob(JobRecord job)
    {
        if (job.Status == JobStatus.Complete)
            LoadJobToResults?.Invoke(job);
    }

    [RelayCommand]
    private void DeleteJob(JobRecord job)
    {
        try
        {
            if (File.Exists(job.ResultsFile))
                File.Delete(job.ResultsFile);
            _controlDb.DeleteJob(job.JobId);
            Jobs.Remove(job);
        }
          catch (Exception ex)
        {
            // TODO: Show proper error dialog
            // For now, log to debug output
            System.Diagnostics.Debug.WriteLine($"Error deleting job: {ex.Message}");
        }
    }
}
