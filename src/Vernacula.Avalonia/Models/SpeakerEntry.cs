using CommunityToolkit.Mvvm.ComponentModel;

namespace Vernacula.Avalonia.Models;

public partial class SpeakerEntry : ObservableObject
{
    public string SpeakerTag { get; init; } = "";

    [ObservableProperty]
    private string _name = "";
}
