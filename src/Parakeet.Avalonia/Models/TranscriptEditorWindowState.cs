using System.Collections.ObjectModel;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using ParakeetCSharp.ViewModels;

namespace ParakeetCSharp.Models;

internal sealed partial class TranscriptEditorWindowState : ObservableObject
{
    [ObservableProperty] private string _headerTitle = "";
    [ObservableProperty] private string _headerSubtext = "";
    [ObservableProperty] private bool _isLoading;
    [ObservableProperty] private string _playPauseGlyph = "▶";
    [ObservableProperty] private bool _isPlaying;
    [ObservableProperty] private double _seekValue;
    [ObservableProperty] private double _playbackSpeed = 1.0;
    [ObservableProperty] private string _speedLabel = "1.0x";
    [ObservableProperty] private int _selectedPlaybackModeIndex;
    [ObservableProperty] private bool _canPlayPause;
    [ObservableProperty] private bool _canSeek;
    [ObservableProperty] private bool _canChangePlaybackSpeed;
    [ObservableProperty] private bool _canChangePlaybackMode;
    [ObservableProperty] private bool _canGoPrev;
    [ObservableProperty] private bool _canGoNext;
    [ObservableProperty] private Bitmap? _playIconImage;
    [ObservableProperty] private Bitmap? _pauseIconImage;
    [ObservableProperty] private Bitmap? _prevIconImage;
    [ObservableProperty] private Bitmap? _nextIconImage;
    [ObservableProperty] private TranscriptEditorCardState? _selectedCard;

    public ObservableCollection<TranscriptEditorCardState> Cards { get; } = [];
    public ObservableCollection<string> PlaybackModes { get; } = [];

    public bool CanInteract => !IsLoading;

    partial void OnIsLoadingChanged(bool value) => OnPropertyChanged(nameof(CanInteract));

    public void SetHeader(string title, string subtext)
    {
        HeaderTitle = title;
        HeaderSubtext = subtext;
    }

    public void SetPlaybackModes(IEnumerable<string> modes, int selectedIndex)
    {
        PlaybackModes.Clear();
        foreach (var mode in modes)
            PlaybackModes.Add(mode);
        SelectedPlaybackModeIndex = selectedIndex;
    }

    public void RefreshPlayback(TranscriptEditorViewModel viewModel, bool seekDragging, bool isLoading)
    {
        IsLoading = isLoading;
        IsPlaying = viewModel.IsPlaying;
        PlayPauseGlyph = viewModel.IsPlaying ? "⏸" : "▶";
        if (!seekDragging)
            SeekValue = viewModel.PlaybackPosition;

        PlaybackSpeed = viewModel.PlaybackSpeed;
        SpeedLabel = $"{viewModel.PlaybackSpeed:0.0}x";
        SelectedPlaybackModeIndex = (int)viewModel.PlaybackMode;

        bool playbackSupported = TranscriptEditorViewModel.SupportsAudioPlayback;
        CanPlayPause = !isLoading && playbackSupported && (viewModel.IsPlaying || viewModel.PlayCommand.CanExecute(null));
        CanSeek = !isLoading && playbackSupported;
        CanChangePlaybackSpeed = !isLoading && playbackSupported;
        CanChangePlaybackMode = !isLoading && playbackSupported;
        CanGoPrev = !isLoading && viewModel.PrevSegmentCommand.CanExecute(null);
        CanGoNext = !isLoading && viewModel.NextSegmentCommand.CanExecute(null);
    }

    public void ApplyFocusedIndex(int focusedIndex, bool force = false)
    {
        if (focusedIndex < 0 || focusedIndex >= Cards.Count)
        {
            SelectedCard = null;
            return;
        }

        for (int i = 0; i < Cards.Count; i++)
        {
            bool isFocused = i == focusedIndex;
            if (force || Cards[i].IsFocused != isFocused)
                Cards[i].IsFocused = isFocused;
        }

        if (force || SelectedCard != Cards[focusedIndex])
            SelectedCard = Cards[focusedIndex];
    }
}
