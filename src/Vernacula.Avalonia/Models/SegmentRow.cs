using CommunityToolkit.Mvvm.ComponentModel;

namespace Vernacula.App.Models;

public partial class SegmentRow : ObservableObject
{
    public int    SegmentId { get; init; }
    public string SpeakerTag { get; init; } = "";
    public double StartTime  { get; init; }
    public double EndTime    { get; init; }

    [ObservableProperty]
    private string _speakerDisplayName = "";

    [ObservableProperty]
    private string _text = "";
}
