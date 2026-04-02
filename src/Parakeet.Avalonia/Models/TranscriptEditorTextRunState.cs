using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;

namespace ParakeetCSharp.Models;

internal sealed partial class TranscriptEditorTextRunState : ObservableObject
{
    public TranscriptEditorTextRunState(string text, IBrush background)
    {
        Text = text;
        _background = background;
    }

    public string Text { get; }

    [ObservableProperty] private IBrush _background;
}
