using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Parakeet.Base;
using ParakeetCSharp.Models;
using ParakeetCSharp.Services;
using ParakeetCSharp.ViewModels;
using ParakeetCSharp.Views.Dialogs;

namespace ParakeetCSharp.Views;

public partial class TranscriptEditorWindow : Window
{
    private const int RedoSpinnerFrameCount = 20;
    private static readonly Bitmap SuppressDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/suppress_dark.png");
    private static readonly Bitmap SuppressLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/suppress_light.png");
    private static readonly Bitmap SuppressRedBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/suppress_red.png");
    private static readonly Bitmap AdjustTimesDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/adjust_times_dark.png");
    private static readonly Bitmap AdjustTimesLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/adjust_times_light.png");
    private static readonly Bitmap ReassignSpeakerDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/reassign_dark.png");
    private static readonly Bitmap ReassignSpeakerLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/reassign_light.png");
    private static readonly Bitmap AddSpeakerDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/add_speaker_dark.png");
    private static readonly Bitmap AddSpeakerLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/add_speaker_light.png");
    private static readonly Bitmap MergePrevDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/merge_prev_dark.png");
    private static readonly Bitmap MergePrevLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/merge_prev_light.png");
    private static readonly Bitmap MergeNextDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/merge_next_dark.png");
    private static readonly Bitmap MergeNextLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/merge_next_light.png");
    private static readonly Bitmap SplitDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/split_dark.png");
    private static readonly Bitmap SplitLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/split_light.png");
    private static readonly Bitmap RedoDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/redo_dark.png");
    private static readonly Bitmap RedoLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/redo_light.png");
    private static readonly Bitmap PlayDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/play_dark.png");
    private static readonly Bitmap PlayLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/play_light.png");
    private static readonly Bitmap PauseDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/pause_dark.png");
    private static readonly Bitmap PauseLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/pause_light.png");
    private static readonly Bitmap PrevDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/prev_dark.png");
    private static readonly Bitmap PrevLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/prev_light.png");
    private static readonly Bitmap NextDarkBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/next_dark.png");
    private static readonly Bitmap NextLightBitmap = LoadBitmap("avares://Parakeet.Avalonia/Assets/toolbar_icons/next_light.png");
    private static readonly Bitmap[] RedoSpinnerDarkFrames = LoadSpinnerFrames("avares://Parakeet.Avalonia/Assets/redo_spinner_dark_frames");
    private static readonly Bitmap[] RedoSpinnerLightFrames = LoadSpinnerFrames("avares://Parakeet.Avalonia/Assets/redo_spinner_light_frames");

    private readonly TranscriptEditorViewModel _vm = null!;
    private readonly TranscriptEditorWindowState _state = new();
    private readonly VocabService? _vocab;
    private readonly string _dbPath = "";
    private readonly string _audioBaseName = "";
    private readonly bool _asrModelsAvailable;
    private bool _isUpdatingUi;
    private bool _isLoading;
    private bool _suppressSegmentCollectionChanged;
    private bool _seekDragging;
    private bool _redoAsrRunning;
    private int _redoAsrCardIndex = -1;
    private int _redoAsrSpinnerFrameIndex;
    private bool _suppressSelectionChanged;
    private DispatcherTimer? _redoAsrSpinnerTimer;

    public event Action? DataChanged;

    public TranscriptEditorWindow()
    {
        InitializeComponent();
        DataContext = _state;
    }

    public TranscriptEditorWindow(string dbPath, string audioBaseName) : this()
    {
        _dbPath = dbPath;
        _audioBaseName = audioBaseName;
        _vm = new TranscriptEditorViewModel();
        _vm.PlaybackMode = App.Current.Settings.Current.EditorPlaybackMode;
        _vm.PropertyChanged += OnViewModelPropertyChanged;
        _vm.FocusedIndexChanging += OnFocusedIndexChanging;
        _vm.Segments.CollectionChanged += OnSegmentsCollectionChanged;
        _vm.PlayCommand.CanExecuteChanged += OnPlayCommandCanExecuteChanged;

        string modelsDir = App.Current.Settings.GetModelsDir();
        string vocabPath = Path.Combine(modelsDir, Config.VocabFile);
        if (File.Exists(vocabPath))
        {
            _vocab = new VocabService(modelsDir);
        }

        var (encoderFile, _) = Config.GetAsrFiles(App.Current.Settings.Current.Precision);
        _asrModelsAvailable = File.Exists(Path.Combine(modelsDir, encoderFile));

        Loaded += OnLoaded;
        Closed += OnClosed;

        SeekSlider.AddHandler(InputElement.PointerPressedEvent, OnSeekSliderPointerPressed, RoutingStrategies.Tunnel | RoutingStrategies.Bubble, handledEventsToo: true);
        SeekSlider.AddHandler(InputElement.PointerMovedEvent, OnSeekSliderPointerMoved, RoutingStrategies.Tunnel | RoutingStrategies.Bubble, handledEventsToo: true);
        SeekSlider.AddHandler(InputElement.PointerReleasedEvent, OnSeekSliderPointerReleased, RoutingStrategies.Tunnel | RoutingStrategies.Bubble, handledEventsToo: true);
        SeekSlider.AddHandler(InputElement.PointerExitedEvent, OnSeekSliderPointerLeave, RoutingStrategies.Tunnel | RoutingStrategies.Bubble, handledEventsToo: true);
        SpeedSlider.PropertyChanged += OnSpeedSliderPropertyChanged;
        SpeedSlider.PointerWheelChanged += OnSpeedSliderPointerWheelChanged;
    }

    private async void OnLoaded(object? sender, RoutedEventArgs e)
    {
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
        ThemeManager.ThemeChanged += OnThemeChanged;
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;

        RefreshPlaybackButtonImages();
        RefreshPlaybackModeCombo();
        SetLoadingState(true);
        RefreshHeader();
        RefreshPlaybackUi();
        await Task.Yield();

        var snapshot = await Task.Run(() => TranscriptEditorViewModel.LoadSnapshotData(_dbPath));

        _suppressSegmentCollectionChanged = true;
        try
        {
            _vm.ApplyLoadedData(_dbPath, _audioBaseName, snapshot);
        }
        finally
        {
            _suppressSegmentCollectionChanged = false;
        }

        RebuildCards();
        RefreshHeader();
        RefreshPlaybackUi();
        ApplyFocusedIndex(_vm.FocusedIndex, force: true);
        SetLoadingState(false);
    }

    private void OnClosed(object? sender, EventArgs e)
    {
        StopRedoAsrSpinner();
        PersistAllDrafts();
        ThemeManager.ThemeChanged -= OnThemeChanged;
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
        _vm.PropertyChanged -= OnViewModelPropertyChanged;
        _vm.FocusedIndexChanging -= OnFocusedIndexChanging;
        _vm.Segments.CollectionChanged -= OnSegmentsCollectionChanged;
        _vm.PlayCommand.CanExecuteChanged -= OnPlayCommandCanExecuteChanged;
        _vm.Dispose();
    }

    private void OnThemeChanged(AppTheme _)
    {
        Dispatcher.UIThread.Post(() =>
        {
            RefreshActionButtonImages();
            RefreshPlaybackButtonImages();
            RefreshRedoAsrSpinnerImages();
            RefreshHeader();
            RefreshPlaybackUi();
            RefreshAllCardState();
            RefreshFocusedCardAsrHighlighting();
        }, DispatcherPriority.Background);
    }

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage))
        {
            return;
        }

        Dispatcher.UIThread.Post(() =>
        {
            RefreshPlaybackModeCombo();
            RefreshHeader();
            RefreshAllCardState();
            RefreshPlaybackUi();
        }, DispatcherPriority.Background);
    }

    private void OnFocusedIndexChanging(int newIndex)
    {
        int oldIndex = _vm.FocusedIndex;
        if (oldIndex >= 0 && oldIndex < _state.Cards.Count)
        {
            SaveDraft(_state.Cards[oldIndex], refreshUi: false);
        }
    }

    private void OnPlayCommandCanExecuteChanged(object? sender, EventArgs e)
    {
        Dispatcher.UIThread.Post(() =>
        {
            RefreshPlaybackUi();
            RefreshAllCardState();
        }, DispatcherPriority.Background);
    }

    private void OnSegmentsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (_suppressSegmentCollectionChanged || _isLoading)
        {
            return;
        }

        Dispatcher.UIThread.Post(() =>
        {
            RebuildCards();
            RefreshHeader();
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
        }, DispatcherPriority.Background);
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        Dispatcher.UIThread.Post(() =>
        {
            switch (e.PropertyName)
            {
                case nameof(TranscriptEditorViewModel.FocusedIndex):
                    ApplyFocusedIndex(_vm.FocusedIndex);
                    RefreshPlaybackUi();
                    break;
                case nameof(TranscriptEditorViewModel.HighlightedToken):
                    UpdateFocusedCardTokenHighlight(_vm.HighlightedToken);
                    break;
                case nameof(TranscriptEditorViewModel.PlaybackPosition):
                    if (!_seekDragging)
                    {
                        _isUpdatingUi = true;
                        SeekSlider.Value = _vm.PlaybackPosition;
                        _isUpdatingUi = false;
                    }
                    break;
                case nameof(TranscriptEditorViewModel.IsPlaying):
                case nameof(TranscriptEditorViewModel.PlaybackMode):
                case nameof(TranscriptEditorViewModel.PlaybackSpeed):
                    RefreshPlaybackUi();
                    break;
            }
        }, DispatcherPriority.Background);
    }

    private void RebuildCards()
    {
        _state.Cards.Clear();

        for (int i = 0; i < _vm.Segments.Count; i++)
        {
            var state = new TranscriptEditorCardState(_vm.Segments[i], i);
            AssignActionButtonImages(state);
            state.RedoAsrSpinnerImage = GetRedoAsrSpinnerImage();
            RefreshCardState(state, preserveDrafts: false);
            _state.Cards.Add(state);
        }
    }

    private void RefreshActionButtonImages()
    {
        foreach (var card in _state.Cards)
            AssignActionButtonImages(card);
    }

    private void RefreshPlaybackButtonImages()
    {
        bool dark = App.Current.Settings.Current.Theme == AppTheme.Dark;
        _state.PlayIconImage = dark ? PlayDarkBitmap : PlayLightBitmap;
        _state.PauseIconImage = dark ? PauseDarkBitmap : PauseLightBitmap;
        _state.PrevIconImage = dark ? PrevDarkBitmap : PrevLightBitmap;
        _state.NextIconImage = dark ? NextDarkBitmap : NextLightBitmap;
    }

    private static void AssignActionButtonImages(TranscriptEditorCardState card)
    {
        bool dark = App.Current.Settings.Current.Theme == AppTheme.Dark;
        card.SuppressIconImage = dark ? SuppressDarkBitmap : SuppressLightBitmap;
        card.SuppressAlertIconImage = SuppressRedBitmap;
        card.AdjustTimesIconImage = dark ? AdjustTimesDarkBitmap : AdjustTimesLightBitmap;
        card.ReassignSpeakerIconImage = dark ? ReassignSpeakerDarkBitmap : ReassignSpeakerLightBitmap;
        card.AddSpeakerIconImage = dark ? AddSpeakerDarkBitmap : AddSpeakerLightBitmap;
        card.MergePrevIconImage = dark ? MergePrevDarkBitmap : MergePrevLightBitmap;
        card.MergeNextIconImage = dark ? MergeNextDarkBitmap : MergeNextLightBitmap;
        card.SplitIconImage = dark ? SplitDarkBitmap : SplitLightBitmap;
        card.RedoAsrIconImage = dark ? RedoDarkBitmap : RedoLightBitmap;
    }

    private void RefreshRedoAsrSpinnerImages()
    {
        Bitmap image = GetRedoAsrSpinnerImage();
        foreach (var card in _state.Cards)
            card.RedoAsrSpinnerImage = image;
    }

    private static Bitmap GetRedoAsrSpinnerImage()
        => GetRedoAsrSpinnerFrames()[0];

    private static Bitmap[] GetRedoAsrSpinnerFrames()
        => App.Current.Settings.Current.Theme == AppTheme.Dark
            ? RedoSpinnerDarkFrames
            : RedoSpinnerLightFrames;

    private static Bitmap[] LoadSpinnerFrames(string baseUri)
    {
        var frames = new Bitmap[RedoSpinnerFrameCount];
        for (int i = 0; i < RedoSpinnerFrameCount; i++)
        {
            string uri = $"{baseUri}/frame_{i:00}.png";
            frames[i] = new Bitmap(AssetLoader.Open(new Uri(uri)));
        }

        return frames;
    }

    private static Bitmap LoadBitmap(string uri)
        => new(AssetLoader.Open(new Uri(uri)));

    private void RefreshAllCardState()
    {
        for (int i = 0; i < _state.Cards.Count; i++)
        {
            _state.Cards[i].Index = i;
            RefreshCardState(_state.Cards[i], preserveDrafts: true);
        }

        RefreshFocusedNeighborhood();
    }

    private void RefreshCardState(TranscriptEditorCardState card, bool preserveDrafts)
    {
        _vm.RefreshCardState(card, preserveDrafts, _state.Cards.Count, _redoAsrRunning, _asrModelsAvailable, _vocab != null);
        _vm.RefreshFocusedCardAppearance(
            card,
            GetThemeBrush("SurfaceBrush"),
            GetThemeBrush("AccentBrush"),
            GetThemeBrush("GreenBrush"),
            GetThemeBrush("RedBrush"),
            GetThemeColor("GreenColor"),
            GetThemeColor("RedColor"),
            GetThemeBrush("TextBrush"));
        _vm.RefreshAdjacentCardHighlighting(
            card,
            _vocab,
            GetThemeColor("ConfidenceLowBrush"),
            GetThemeColor("GreenColor"),
            GetThemeColor("RedColor"),
            GetThemeBrush("SubtextBrush"),
            GetThemeBrush("TextBrush"));
    }

    private void ApplyFocusedIndex(int focusedIndex, bool force = false)
    {
        if (focusedIndex < 0 || focusedIndex >= _state.Cards.Count)
        {
            _suppressSelectionChanged = true;
            try
            {
                _state.ApplyFocusedIndex(-1, force);
            }
            finally
            {
                _suppressSelectionChanged = false;
            }
            return;
        }

        _isUpdatingUi = true;
        _suppressSelectionChanged = true;
        try
        {
            _state.ApplyFocusedIndex(focusedIndex, force);
        }
        finally
        {
            _suppressSelectionChanged = false;
            _isUpdatingUi = false;
        }

        for (int i = 0; i < _state.Cards.Count; i++)
            if (force || IsWarmCardIndex(i, focusedIndex))
                RefreshCardState(_state.Cards[i], preserveDrafts: true);

        RefreshFocusedNeighborhood();
        SegmentList.ScrollIntoView(_state.Cards[focusedIndex]);
    }

    private void RefreshFocusedNeighborhood()
    {
        if (_vm.FocusedIndex < 0 || _vm.FocusedIndex >= _state.Cards.Count)
            return;

        for (int i = 0; i < _state.Cards.Count; i++)
        {
            if (!IsWarmCardIndex(i, _vm.FocusedIndex))
                continue;

            _vm.RefreshFocusedCardAsrHighlighting(
                _state.Cards[i],
                _vocab,
                GetThemeColor("ConfidenceLowBrush"),
                GetThemeColor("AccentBrush"),
                i == _vm.FocusedIndex ? _vm.HighlightedToken : -1);
        }
    }

    private static bool IsWarmCardIndex(int cardIndex, int focusedIndex)
        => Math.Abs(cardIndex - focusedIndex) <= 1;

    private void SetFocusedIndex(int index, bool force = false)
    {
        if (index < 0 || index >= _state.Cards.Count)
        {
            return;
        }

        if (_vm.FocusedIndex != index)
        {
            _vm.NavigateTo(index);
        }
        else
        {
            ApplyFocusedIndex(index, force);
        }
    }

    private void RefreshHeader()
    {
        _state.SetHeader(_audioBaseName, $"{_vm.Segments.Count} {Loc.Instance["results_segments_label"]}");
    }

    private void RefreshPlaybackModeCombo()
    {
        _state.SetPlaybackModes(new[]
        {
            Loc.Instance["editor_mode_single"],
            Loc.Instance["editor_auto_advance"],
            Loc.Instance["editor_mode_continuous"],
        }, (int)_vm.PlaybackMode);
    }

    private void RefreshPlaybackUi()
    {
        _isUpdatingUi = true;
        _state.RefreshPlayback(_vm, _seekDragging, _isLoading);
        _isUpdatingUi = false;
    }

    private void RefreshFocusedCardAsrHighlighting()
    {
        RefreshFocusedNeighborhood();
    }

    private void UpdateFocusedCardTokenHighlight(int tokenIndex)
    {
        if (_vm.FocusedIndex < 0 || _vm.FocusedIndex >= _state.Cards.Count)
            return;

        _vm.UpdateFocusedCardTokenHighlight(
            _state.Cards[_vm.FocusedIndex],
            GetThemeColor("AccentBrush"),
            tokenIndex);
    }

    private Color GetThemeColor(string brushKey)
    {
        if (Application.Current?.Resources.TryGetResource(brushKey, null, out var value) == true)
        {
            if (value is Color color)
                return color;
            if (value is ISolidColorBrush brush)
                return brush.Color;
        }

        return Colors.Transparent;
    }

    private IBrush GetThemeBrush(string brushKey)
    {
        if (Application.Current?.Resources.TryGetResource(brushKey, null, out var value) == true
            && value is IBrush brush)
            return brush;

        return Brushes.Transparent;
    }

    private void SetLoadingState(bool isLoading)
    {
        _isLoading = isLoading;
        _state.IsLoading = isLoading;
        RefreshPlaybackUi();
    }

    private static TranscriptEditorCardState? CardFromSender(object? sender)
        => (sender as StyledElement)?.DataContext as TranscriptEditorCardState;

    private void SegmentList_SelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (_isUpdatingUi || _suppressSelectionChanged)
        {
            return;
        }

        if (_redoAsrRunning)
        {
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            return;
        }

        if (_vm.IsPlaying && _vm.PlaybackMode == PlaybackMode.Continuous)
        {
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            return;
        }

        if (_state.SelectedCard is { } card && card.Index != _vm.FocusedIndex)
        {
            _vm.NavigateTo(card.Index);
        }
    }

    private void SegmentCard_PointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        if (_redoAsrRunning)
        {
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            return;
        }

        if (card.Index != _vm.FocusedIndex)
        {
            _vm.NavigateTo(card.Index);
        }

        ApplyFocusedIndex(_vm.FocusedIndex, force: true);
        RefreshPlaybackUi();
    }

    private void SpeakerNameBox_LostFocus(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is { } card)
        {
            SaveDraft(card);
        }
    }

    private void EditTextBox_LostFocus(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is { } card)
        {
            SaveDraft(card);
        }
    }

    private void SaveDraft(TranscriptEditorCardState card, bool refreshUi = true)
    {
        bool changed = _vm.SaveDraft(card);
        if (changed)
        {
            DataChanged?.Invoke();
            if (refreshUi)
            {
                RefreshAllCardState();
                ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            }
        }
    }

    private void PersistAllDrafts()
    {
        for (int i = 0; i < _state.Cards.Count; i++)
        {
            SaveDraft(_state.Cards[i]);
        }
    }

    private void ApplyStructuralChange(int newFocusIndex, string statusMessage)
    {
        DataChanged?.Invoke();
        RebuildCards();
        RefreshHeader();
        SetFocusedIndex(newFocusIndex, force: true);
        if (_vm.FocusedIndex >= 0 && _vm.FocusedIndex < _state.Cards.Count)
            SetCardStatus(_state.Cards[_vm.FocusedIndex], statusMessage);
    }

    private void SetCardStatus(TranscriptEditorCardState card, string text)
    {
        card.StatusMessage = text;
    }

    private void PlayPauseButton_Click(object? sender, RoutedEventArgs e)
    {
        if (!TranscriptEditorViewModel.SupportsAudioPlayback)
        {
            if (_vm.FocusedIndex >= 0 && _vm.FocusedIndex < _state.Cards.Count)
            {
                SetCardStatus(_state.Cards[_vm.FocusedIndex], TranscriptEditorViewModel.PlaybackUnavailableReason);
            }
            return;
        }

        if (_vm.IsPlaying)
        {
            _vm.PauseCommand.Execute(null);
        }
        else
        {
            _vm.PlayCommand.Execute(null);
        }
    }

    private void PrevButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_redoAsrRunning)
        {
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            return;
        }

        if (!_vm.PrevSegmentCommand.CanExecute(null))
        {
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            RefreshPlaybackUi();
            return;
        }

        _vm.PrevSegmentCommand.Execute(null);
        ApplyFocusedIndex(_vm.FocusedIndex, force: true);
        RefreshPlaybackUi();
    }

    private void NextButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_redoAsrRunning)
        {
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            return;
        }

        if (!_vm.NextSegmentCommand.CanExecute(null))
        {
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            RefreshPlaybackUi();
            return;
        }

        _vm.NextSegmentCommand.Execute(null);
        ApplyFocusedIndex(_vm.FocusedIndex, force: true);
        RefreshPlaybackUi();
    }

    private void PlayModeCombo_SelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (_isUpdatingUi || PlayModeCombo.SelectedIndex < 0)
        {
            return;
        }

        var mode = (PlaybackMode)PlayModeCombo.SelectedIndex;
        _vm.PlaybackMode = mode;
        App.Current.Settings.Current.EditorPlaybackMode = mode;
        App.Current.Settings.Save();
        RefreshPlaybackUi();
    }

    private void OnSeekSliderPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        _seekDragging = true;
        OnSeekSliderPointerMoved(sender, e);
    }

    private void OnSeekSliderPointerReleased(object? sender, PointerReleasedEventArgs e)
    {
        _seekDragging = false;
        HideSeekBubble();
        if (_isUpdatingUi)
        {
            return;
        }

        Dispatcher.UIThread.Post(() =>
        {
            double seekValue = SeekSlider.Value;
            if (_vm.PlaybackMode == PlaybackMode.Continuous)
            {
                _vm.SeekContinuous(seekValue);
                ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            }
            else
            {
                _vm.Seek(seekValue);
                ApplyFocusedIndex(_vm.FocusedIndex, force: true);
            }
            RefreshPlaybackUi();
        }, DispatcherPriority.Background);
    }

    private void OnSeekSliderPointerMoved(object? sender, PointerEventArgs e)
    {
        if (_vm.PlaybackMode != PlaybackMode.Continuous
            || _vm.TotalAudioSeconds is not { } totalSec
            || totalSec <= 0
            || SeekSlider.Bounds.Width <= 0)
        {
            HideSeekBubble();
            return;
        }

        Point pointer = e.GetPosition(SeekSlider);
        double ratio = Math.Clamp(pointer.X / SeekSlider.Bounds.Width, 0.0, 1.0);
        SeekBubbleText.Text = FormatTime(ratio * totalSec);

        Point bubbleAnchor = SeekSlider.TranslatePoint(new Point(pointer.X, 0), SeekBubbleLayer)
            ?? default;

        double bubbleWidth = SeekBubbleHost.Bounds.Width > 0 ? SeekBubbleHost.Bounds.Width : 72;
        Canvas.SetLeft(SeekBubbleHost, bubbleAnchor.X - (bubbleWidth / 2));
        Canvas.SetTop(SeekBubbleHost, bubbleAnchor.Y - 46);
        SeekBubbleHost.IsVisible = true;
    }

    private void OnSeekSliderPointerLeave(object? sender, PointerEventArgs e)
    {
        if (!_seekDragging)
        {
            HideSeekBubble();
        }
    }

    private void OnSpeedSliderPropertyChanged(object? sender, AvaloniaPropertyChangedEventArgs e)
    {
        if (e.Property != RangeBase.ValueProperty || _isUpdatingUi)
        {
            return;
        }

        _vm.PlaybackSpeed = SpeedSlider.Value;
        RefreshPlaybackUi();
    }

    private void OnSpeedSliderPointerWheelChanged(object? sender, PointerWheelEventArgs e)
    {
        if (!SpeedSlider.IsEnabled)
        {
            return;
        }

        double step = e.Delta.Y > 0 ? 0.1 : -0.1;
        SpeedSlider.Value = Math.Clamp(
            SpeedSlider.Value + step,
            SpeedSlider.Minimum,
            SpeedSlider.Maximum);
        e.Handled = true;
    }

    private void HideSeekBubble()
    {
        SeekBubbleHost.IsVisible = false;
    }

    private void RenameSpeaker_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        if (_vm.RenameSpeaker(card))
        {
            DataChanged?.Invoke();
            RefreshAllCardState();
            SetCardStatus(card, "Speaker name updated.");
        }
    }

    private void ReassignSpeaker_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card || card.SelectedSpeaker is null)
        {
            return;
        }

        if (_vm.ReassignSpeaker(card))
        {
            DataChanged?.Invoke();
            RefreshAllCardState();
            SetCardStatus(card, "Segment reassigned.");
        }
    }

    private async void AddSpeaker_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        var dialog = new AddSpeakerDialog();
        await dialog.ShowDialog(this);
        if (string.IsNullOrWhiteSpace(dialog.SpeakerName))
        {
            return;
        }

        int newSpeakerId = _vm.AddSpeaker(dialog.SpeakerName);
        if (newSpeakerId < 0)
        {
            return;
        }

        DataChanged?.Invoke();
        RefreshAllCardState();
        card.SelectedSpeaker = card.SpeakerChoices.FirstOrDefault(c => c.SpeakerId == newSpeakerId);
        card.SyncDraftsFromSegment();
        SetCardStatus(card, "Speaker added.");
    }

    private void VerifiedCheckBox_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card || card.Index != _vm.FocusedIndex)
        {
            return;
        }

        _vm.ToggleVerifiedCommand.Execute(null);
        DataChanged?.Invoke();
        RefreshCardState(card, preserveDrafts: true);
    }

    private void SuppressButton_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        if (_vm.ToggleSuppressed(card))
        {
            DataChanged?.Invoke();
            RefreshCardState(card, preserveDrafts: true);
            SetCardStatus(card, card.Segment.IsSuppressed ? "Segment suppressed." : "Segment restored.");
        }
    }

    private async void AdjustTimes_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        var seg = card.Segment;
        var dialog = new AdjustTimesDialog(seg.PlayStart, seg.PlayEnd);
        await dialog.ShowDialog(this);
        if (!dialog.DialogResult)
        {
            return;
        }

        if (_vm.AdjustSegmentTimes(card, dialog.NewStartTime, dialog.NewEndTime))
        {
            DataChanged?.Invoke();
            RefreshCardState(card, preserveDrafts: true);
            SetCardStatus(card, "Segment times updated.");
        }
    }

    private async void MergePrev_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        SaveDraft(card);
        if (_vm.MergeWithPrev(card.Index))
        {
            ApplyStructuralChange(Math.Max(0, card.Index - 1), "Segments merged.");
            await RunRedoAsrForFocusedSegmentAsync();
        }
    }

    private async void MergeNext_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        SaveDraft(card);
        if (_vm.MergeWithNext(card.Index))
        {
            ApplyStructuralChange(Math.Min(card.Index, _state.Cards.Count - 1), "Segments merged.");
            await RunRedoAsrForFocusedSegmentAsync();
        }
    }

    private async void Split_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        var seg = card.Segment;
        if (seg.Tokens.Count < 2)
        {
            return;
        }

        IReadOnlyList<string> tokenTexts = _vocab != null
            ? _vocab.GetTokenRuns(seg.Tokens, seg.Logprobs)
                .Select(r => string.IsNullOrWhiteSpace(r.text) ? "[]" : r.text)
                .ToList()
            : seg.Tokens.Select(t => $"[{t}]").ToList();

        var dialog = new SplitSegmentDialog(tokenTexts);
        await dialog.ShowDialog(this);
        if (!dialog.DialogResult || dialog.SplitTokenIndex <= 0)
        {
            return;
        }

        SaveDraft(card);
        if (_vm.SplitSegment(card.Index, dialog.SplitTokenIndex, _vocab))
        {
            ApplyStructuralChange(Math.Min(card.Index, _state.Cards.Count - 1), "Segment split.");
        }
    }

    private async void RedoAsr_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        await RunRedoAsrAsync(card);
    }

    private async Task RunRedoAsrForFocusedSegmentAsync()
    {
        if (_vm.FocusedIndex < 0 || _vm.FocusedIndex >= _state.Cards.Count)
        {
            return;
        }

        await RunRedoAsrAsync(_state.Cards[_vm.FocusedIndex]);
    }

    private async Task RunRedoAsrAsync(TranscriptEditorCardState card)
    {
        ApplyFocusedIndex(_vm.FocusedIndex, force: true);

        if (_redoAsrRunning || !_asrModelsAvailable || !_vm.HasAudio)
        {
            return;
        }

        _redoAsrRunning = true;
        StartRedoAsrSpinner(card.Index);
        RefreshAllCardState();
        SetCardStatus(card, Loc.Instance["editor_redo_asr_running"]);

        try
        {
            string modelsDir = App.Current.Settings.GetModelsDir();
            var (encoderFile, decoderJointFile) = Config.GetAsrFiles(App.Current.Settings.Current.Precision);

            var result = await Task.Run(() => _vm.PerformRedoAsr(card.Index, modelsDir, encoderFile, decoderJointFile));
            if (result is null)
            {
                SetCardStatus(card, "Redo ASR did not produce a result.");
                return;
            }

            _vm.ApplyRedoAsr(card.Index, result.Value.newResultId, result.Value.asrContent,
                result.Value.tokens, result.Value.timestamps, result.Value.logprobs);

            card.DraftContent = card.Segment.Content;
            DataChanged?.Invoke();
            RefreshCardState(card, preserveDrafts: true);
            RefreshFocusedCardAsrHighlighting();
            SetCardStatus(card, "ASR regenerated.");
        }
        catch (Exception ex)
        {
            SetCardStatus(card, $"Redo ASR failed: {ex.Message}");
        }
        finally
        {
            _redoAsrRunning = false;
            StopRedoAsrSpinner();
            RefreshAllCardState();
        }
    }

    private void StartRedoAsrSpinner(int cardIndex)
    {
        _redoAsrCardIndex = cardIndex;
        _redoAsrSpinnerFrameIndex = 0;
        _redoAsrSpinnerTimer ??= new DispatcherTimer(TimeSpan.FromMilliseconds(50), DispatcherPriority.Background, OnRedoAsrSpinnerTick);

        if (cardIndex >= 0 && cardIndex < _state.Cards.Count)
        {
            _state.Cards[cardIndex].IsRedoAsrSpinning = true;
            _state.Cards[cardIndex].RedoAsrSpinnerImage = GetRedoAsrSpinnerFrames()[0];
        }

        _redoAsrSpinnerTimer.Start();
    }

    private void StopRedoAsrSpinner()
    {
        _redoAsrSpinnerTimer?.Stop();

        if (_redoAsrCardIndex >= 0 && _redoAsrCardIndex < _state.Cards.Count)
        {
            _state.Cards[_redoAsrCardIndex].IsRedoAsrSpinning = false;
            _state.Cards[_redoAsrCardIndex].RedoAsrSpinnerImage = GetRedoAsrSpinnerFrames()[0];
        }

        _redoAsrCardIndex = -1;
    }

    private void OnRedoAsrSpinnerTick(object? sender, EventArgs e)
    {
        if (_redoAsrCardIndex < 0 || _redoAsrCardIndex >= _state.Cards.Count)
        {
            return;
        }

        var card = _state.Cards[_redoAsrCardIndex];
        var frames = GetRedoAsrSpinnerFrames();
        _redoAsrSpinnerFrameIndex = (_redoAsrSpinnerFrameIndex + 1) % frames.Length;
        card.RedoAsrSpinnerImage = frames[_redoAsrSpinnerFrameIndex];
    }

    private static string FormatTime(double s)
    {
        int h = (int)(s / 3600);
        int m = (int)(s % 3600 / 60);
        double sec = s % 60;
        return h > 0 ? $"{h}:{m:D2}:{sec:00.0}" : $"{m:D2}:{sec:00.0}";
    }
}
