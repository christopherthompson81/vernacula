using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Vernacula.App.Models;

internal sealed partial class TranscriptEditorTextRunState : ObservableObject
{
    public TranscriptEditorTextRunState(string text, IBrush background, IBrush foreground)
    {
        Text = text;
        _background = background;
        _foreground = foreground;
    }

    public string Text { get; }

    [ObservableProperty] private IBrush _background;
    [ObservableProperty] private IBrush _foreground;
}
