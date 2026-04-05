namespace Vernacula.Avalonia.Models;

public enum TranscriptionPhase
{
    LoadingAudio,
    Diarizing,
    Recognizing,
    Done,
    Failed,
    Cancelled
}

public record TranscriptionProgress(
    TranscriptionPhase Phase,
    int    CurrentStep,
    int    TotalSteps,
    string StatusMessage,
    int?    CompletedSegmentId = null,
    string? SegmentText        = null,
    double? OverridePercent    = null)
{
    public double Percent =>
        OverridePercent ?? (TotalSteps > 0 ? (double)CurrentStep / TotalSteps * 100 : 0);
}
