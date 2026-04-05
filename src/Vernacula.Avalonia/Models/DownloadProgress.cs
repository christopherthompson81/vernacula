namespace Vernacula.Avalonia.Models;

public record DownloadProgress(
    string FileName,
    int    FileIndex,
    int    TotalFiles,
    long   BytesDownloaded,
    long   TotalBytes,
    long   GrandTotalBytes,
    long   PrevBytesTotal)
{
    public double FilePercent =>
        TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes * 100 : 0;

    public double OverallPercent =>
        GrandTotalBytes > 0
            ? (double)(PrevBytesTotal + BytesDownloaded) / GrandTotalBytes * 100
            : (FileIndex + FilePercent / 100.0) / TotalFiles * 100;

    public string SizeText => TotalBytes > 0
        ? $"{BytesDownloaded / 1_048_576.0:F1} / {TotalBytes / 1_048_576.0:F1} MB"
        : $"{BytesDownloaded / 1_048_576.0:F1} MB downloaded";

    public string OverallSizeText => GrandTotalBytes > 0
        ? $"{(PrevBytesTotal + BytesDownloaded) / 1_048_576.0:F1} / {GrandTotalBytes / 1_048_576.0:F1} MB"
        : "";
}
