using System.Collections.Generic;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Vernacula.App.Models;

internal partial class EditorSegment : ObservableObject
{
    public int    CardId    { get; set; }
    public int    SpeakerId { get; set; }

    [ObservableProperty]
    private string _speakerDisplayName = "";

    [ObservableProperty]
    private double _playStart;

    [ObservableProperty]
    private double _playEnd;

    public string AsrContent { get; set; } = "";
    public string? Language  { get; set; }
    public string? LidLanguage { get; set; }

    [ObservableProperty]
    private string _content = "";

    public IReadOnlyList<int>        Tokens     { get; set; } = [];
    public IReadOnlyList<int>        Timestamps { get; set; } = [];
    public IReadOnlyList<float>      Logprobs   { get; set; } = [];
    public IReadOnlyList<CardSource> Sources    { get; set; } = [];

    [ObservableProperty]
    private bool _verified;

    [ObservableProperty]
    private bool _isSuppressed;

    public bool HasUserEdits => Content != AsrContent;
}
