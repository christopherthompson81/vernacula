using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Linq;
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
using Vernacula.Base;
using Vernacula.Avalonia.Models;
using Vernacula.Avalonia.Services;
using Vernacula.Avalonia.ViewModels;
using Vernacula.Avalonia.Views.Dialogs;

namespace Vernacula.Avalonia.Views;

public partial class TranscriptEditorWindow : Window
{
    private const int RedoSpinnerFrameCount = 20;
    private static readonly TimeSpan SpeakerWheelCooldown = TimeSpan.FromMilliseconds(180);
    private static readonly Bitmap SuppressBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/suppress.png");
    private static readonly Bitmap SuppressRedBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/suppress_red.png");
    private static readonly Bitmap AdjustTimesBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/adjust_times.png");
    private static readonly Bitmap ReassignSpeakerBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/reassign.png");
    private static readonly Bitmap AddSpeakerBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/add_speaker.png");
    private static readonly Bitmap MergePrevBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/merge_prev.png");
    private static readonly Bitmap MergeNextBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/merge_next.png");
    private static readonly Bitmap SplitBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/split.png");
    private static readonly Bitmap RedoBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/redo.png");
    private static readonly Bitmap PlayBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/play.png");
    private static readonly Bitmap PauseBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/pause.png");
    private static readonly Bitmap PrevBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/prev.png");
    private static readonly Bitmap NextBitmap = LoadBitmap("avares://Vernacula.Avalonia/Assets/toolbar_icons/next.png");
    private static readonly Bitmap[] RedoSpinnerFrames = LoadSpinnerFrames("avares://Vernacula.Avalonia/Assets/redo_spinner_frames");

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
    private int _redoAsrCardId = -1;
    private int _redoAsrSpinnerFrameIndex;
    private bool _suppressSelectionChanged;
    private int _lastSpeakerWheelCardIndex = -1;
    private DateTime _lastSpeakerWheelUtc = DateTime.MinValue;
    private DispatcherTimer? _redoAsrSpinnerTimer;
    private int _focusRefreshVersion;

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
        AddHandler(InputElement.PointerMovedEvent, OnWindowPointerMoved, RoutingStrategies.Tunnel | RoutingStrategies.Bubble, handledEventsToo: true);
        AddHandler(InputElement.PointerPressedEvent, OnWindowPointerPressed, RoutingStrategies.Tunnel | RoutingStrategies.Bubble, handledEventsToo: true);
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
            SyncRedoAsrSpinnerState(state);
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
        _state.PlayIconImage = PlayBitmap;
        _state.PauseIconImage = PauseBitmap;
        _state.PrevIconImage = PrevBitmap;
        _state.NextIconImage = NextBitmap;
    }

    private static void AssignActionButtonImages(TranscriptEditorCardState card)
    {
        card.SuppressIconImage = SuppressBitmap;
        card.SuppressAlertIconImage = SuppressRedBitmap;
        card.AdjustTimesIconImage = AdjustTimesBitmap;
        card.ReassignSpeakerIconImage = ReassignSpeakerBitmap;
        card.AddSpeakerIconImage = AddSpeakerBitmap;
        card.MergePrevIconImage = MergePrevBitmap;
        card.MergeNextIconImage = MergeNextBitmap;
        card.SplitIconImage = SplitBitmap;
        card.RedoAsrIconImage = RedoBitmap;
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
        => RedoSpinnerFrames;

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
            _state.Cards[i].RefreshLocalizedText();
            RefreshCardState(_state.Cards[i], preserveDrafts: true);
        }

        RefreshFocusedNeighborhood();
    }

    private void RefreshCardState(TranscriptEditorCardState card, bool preserveDrafts)
    {
        _vm.RefreshCardState(card, preserveDrafts, _state.Cards.Count, _redoAsrRunning, _asrModelsAvailable, _vocab != null);
        IBrush surfaceBrush = GetThemeBrush("SurfaceBrush");
        IBrush accentBrush = GetThemeBrush("AccentBrush");
        IBrush greenBrush = GetThemeBrush("GreenBrush");
        IBrush redBrush = GetThemeBrush("RedBrush");
        Color greenColor = GetThemeColor("GreenColor");
        Color redColor = GetThemeColor("RedColor");
        IBrush textBrush = GetThemeBrush("TextBrush");
        _vm.RefreshAdjacentCardHighlighting(
            card,
            _vocab,
            GetThemeColor("ConfidenceLowBrush"),
            greenColor,
            redColor,
            GetThemeBrush("SubtextBrush"),
            textBrush);
        card.SetUnfocusedAppearance(card.AdjacentBackground, GetThemeBrush("OverlayBrush"));
        _vm.RefreshFocusedCardAppearance(
            card,
            surfaceBrush,
            accentBrush,
            greenBrush,
            redBrush,
            greenColor,
            redColor,
            textBrush);
        SyncRedoAsrSpinnerState(card);
    }

    private void ApplyFocusedIndex(int focusedIndex, bool force = false)
    {
        int previousFocusedIndex = _state.SelectedCard?.Index ?? -1;
        int refreshVersion = ++_focusRefreshVersion;

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
            if (force || ShouldRefreshCardIndex(i, focusedIndex, previousFocusedIndex))
                RefreshCardState(_state.Cards[i], preserveDrafts: true);

        RefreshFocusedNeighborhood();
        SegmentList.InvalidateMeasure();
        SegmentList.InvalidateArrange();
        SegmentList.ScrollIntoView(_state.Cards[focusedIndex]);
        QueueDeferredCardRefresh(focusedIndex, previousFocusedIndex, refreshVersion);
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
                GetThemeBrush("TextBrush"),
                i == _vm.FocusedIndex ? _vm.HighlightedToken : -1);
        }
    }

    private static bool IsWarmCardIndex(int cardIndex, int focusedIndex)
        => Math.Abs(cardIndex - focusedIndex) <= 1;

    private static bool ShouldRefreshCardIndex(int cardIndex, int focusedIndex, int previousFocusedIndex)
        => IsWarmCardIndex(cardIndex, focusedIndex)
           || (previousFocusedIndex >= 0 && IsWarmCardIndex(cardIndex, previousFocusedIndex));

    private void QueueDeferredCardRefresh(int focusedIndex, int previousFocusedIndex, int refreshVersion)
    {
        var pendingIndexes = Enumerable.Range(0, _state.Cards.Count)
            .Where(i => !ShouldRefreshCardIndex(i, focusedIndex, previousFocusedIndex))
            .OrderBy(i => Math.Abs(i - focusedIndex))
            .ToArray();

        if (pendingIndexes.Length == 0)
        {
            return;
        }

        const int batchSize = 8;

        void ProcessBatch(int start)
        {
            if (refreshVersion != _focusRefreshVersion)
            {
                return;
            }

            int end = Math.Min(start + batchSize, pendingIndexes.Length);
            for (int i = start; i < end; i++)
            {
                RefreshCardState(_state.Cards[pendingIndexes[i]], preserveDrafts: true);
            }

            if (end < pendingIndexes.Length)
            {
                Dispatcher.UIThread.Post(
                    () => ProcessBatch(end),
                    DispatcherPriority.Background);
            }
        }

        Dispatcher.UIThread.Post(
            () => ProcessBatch(0),
            DispatcherPriority.Background);
    }

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
            SaveSpeakerName(card, statusMessage: "Speaker name updated.");
        }
    }

    private void SpeakerNameBox_KeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key is not (Key.Enter or Key.Tab))
        {
            return;
        }

        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        SaveSpeakerName(card, statusMessage: "Speaker name updated.");
        FocusSegmentContent(sender);
        e.Handled = true;
    }

    private void SpeakerNameBox_PointerWheelChanged(object? sender, PointerWheelEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        var now = DateTime.UtcNow;
        if (card.Index == _lastSpeakerWheelCardIndex
            && now - _lastSpeakerWheelUtc < SpeakerWheelCooldown)
        {
            e.Handled = true;
            return;
        }

        var speakers = _vm.AllSpeakers;
        if (speakers.Count < 2)
        {
            e.Handled = true;
            return;
        }

        int currentIndex = speakers.FindIndex(s => s.SpeakerId == _vm.Segments[card.Index].SpeakerId);
        if (currentIndex < 0)
        {
            e.Handled = true;
            return;
        }

        int nextIndex = (currentIndex + (e.Delta.Y > 0 ? -1 : 1) + speakers.Count) % speakers.Count;
        _lastSpeakerWheelCardIndex = card.Index;
        _lastSpeakerWheelUtc = now;
        ApplySpeakerReassignment(card, speakers[nextIndex].SpeakerId, "Segment reassigned.");
        e.Handled = true;
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

    private void SaveSpeakerName(TranscriptEditorCardState card, string statusMessage)
    {
        string speakerName = (card.DraftSpeakerName ?? string.Empty).Trim();
        if (string.IsNullOrEmpty(speakerName))
        {
            card.SyncDraftsFromSegment();
            return;
        }

        bool changed = _vm.SaveSpeakerName(card.Index, speakerName);
        card.SyncDraftsFromSegment();
        if (!changed)
        {
            return;
        }

        DataChanged?.Invoke();
        RefreshAllCardState();
        SetCardStatus(card, statusMessage);
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
                bool resumePlayback = _vm.SeekContinuous(seekValue, resumePlayback: false);
                ApplyFocusedIndex(_vm.FocusedIndex, force: true);
                RefreshPlaybackUi();

                if (resumePlayback)
                {
                    Dispatcher.UIThread.Post(() =>
                    {
                        if (!_seekDragging && !_isLoading && !_redoAsrRunning && _vm.PlayCommand.CanExecute(null))
                        {
                            _vm.PlayCommand.Execute(null);
                        }
                    }, DispatcherPriority.Background);
                }
            }
            else
            {
                _vm.Seek(seekValue);
                ApplyFocusedIndex(_vm.FocusedIndex, force: true);
                RefreshPlaybackUi();
            }
        }, DispatcherPriority.Background);
    }

    private void OnSeekSliderPointerMoved(object? sender, PointerEventArgs e)
    {
        if (_vm.PlaybackMode != PlaybackMode.Continuous
            || _vm.TotalAudioSeconds is not { } totalSec
            || totalSec <= 0
            || SeekSlider.Bounds.Width <= 0
            || (!_seekDragging && !SeekSlider.IsPointerOver))
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

    private void OnWindowPointerMoved(object? sender, PointerEventArgs e)
    {
        if (_seekDragging || !SeekBubbleHost.IsVisible)
        {
            return;
        }

        if (!IsPointerOverSeekSlider(e))
        {
            HideSeekBubble();
        }
    }

    private void OnWindowPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (_seekDragging)
        {
            return;
        }

        if (!IsPointerOverSeekSlider(e))
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

    private bool IsPointerOverSeekSlider(PointerEventArgs e)
    {
        var visual = e.Source as Visual;
        if (visual is null)
        {
            return SeekSlider.IsPointerOver;
        }

        return visual == SeekSlider || SeekSlider.IsVisualAncestorOf(visual);
    }

    private void SpeakerMenu_Click(object? sender, RoutedEventArgs e)
    {
        if (sender is not Button anchor || CardFromSender(sender) is not { } card)
        {
            return;
        }

        ShowSpeakerMenu(anchor, card.Index);
    }

    private void ShowSpeakerMenu(Button anchor, int segmentIndex)
    {
        int currentSpeakerId = _vm.Segments[segmentIndex].SpeakerId;
        var menuItems = new List<object>();

        foreach (var (speakerId, name) in _vm.AllSpeakers)
        {
            int id = speakerId;
            bool isCurrent = speakerId == currentSpeakerId;
            string label = string.IsNullOrWhiteSpace(name) ? $"speaker_{speakerId - 1}" : name;
            var item = new MenuItem
            {
                Header = isCurrent ? $"✓ {label}" : label,
                IsEnabled = !isCurrent
            };
            item.Click += (_, _) => ApplySpeakerReassignment(segmentIndex, id, "Segment reassigned.");
            menuItems.Add(item);
        }

        menuItems.Add(new Separator());

        var addItem = new MenuItem
        {
            Header = Loc.Instance["editor_add_speaker"]
        };
        addItem.Click += async (_, _) => await PromptAddAndReassignAsync(segmentIndex);
        menuItems.Add(addItem);

        var menu = new ContextMenu
        {
            PlacementTarget = anchor,
            ItemsSource = menuItems
        };

        anchor.ContextMenu = menu;
        menu.Open(anchor);
    }

    private void ApplySpeakerReassignment(TranscriptEditorCardState card, int speakerId, string statusMessage)
    {
        if (!_vm.ReassignSegment(card.Index, speakerId))
        {
            return;
        }

        DataChanged?.Invoke();
        RefreshAllCardState();
        SetCardStatus(card, statusMessage);
    }

    private void ApplySpeakerReassignment(int segmentIndex, int speakerId, string statusMessage)
    {
        if (segmentIndex < 0 || segmentIndex >= _state.Cards.Count)
        {
            return;
        }

        ApplySpeakerReassignment(_state.Cards[segmentIndex], speakerId, statusMessage);
    }

    private async Task PromptAddAndReassignAsync(int segmentIndex)
    {
        if (segmentIndex < 0 || segmentIndex >= _state.Cards.Count)
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

        ApplySpeakerReassignment(segmentIndex, newSpeakerId, "Speaker added.");
    }

    private static void FocusSegmentContent(object? sender)
    {
        if (sender is not Control control)
        {
            return;
        }

        var container = control.GetVisualAncestors()
            .OfType<Control>()
            .FirstOrDefault(ancestor => ancestor.FindControl<TextBox>("SegmentContentTextBox") is not null);
        if (container?.FindControl<TextBox>("SegmentContentTextBox") is { } contentBox)
        {
            contentBox.Focus();
        }
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
        _redoAsrSpinnerFrameIndex = 0;
        _redoAsrSpinnerTimer ??= new DispatcherTimer(TimeSpan.FromMilliseconds(50), DispatcherPriority.Background, OnRedoAsrSpinnerTick);

        if (cardIndex >= 0 && cardIndex < _state.Cards.Count)
        {
            _redoAsrCardId = _state.Cards[cardIndex].Segment.CardId;
            var card = _state.Cards[cardIndex];
            RefreshCardState(card, preserveDrafts: true);

            if (cardIndex == _vm.FocusedIndex)
            {
                ApplyFocusedIndex(cardIndex, force: true);
            }
        }

        _redoAsrSpinnerTimer.Start();
    }

    private void StopRedoAsrSpinner()
    {
        _redoAsrSpinnerTimer?.Stop();

        foreach (var card in _state.Cards)
            SyncRedoAsrSpinnerState(card);

        _redoAsrCardId = -1;
    }

    private void OnRedoAsrSpinnerTick(object? sender, EventArgs e)
    {
        if (_redoAsrCardId < 0)
        {
            return;
        }

        var frames = GetRedoAsrSpinnerFrames();
        _redoAsrSpinnerFrameIndex = (_redoAsrSpinnerFrameIndex + 1) % frames.Length;
        foreach (var card in _state.Cards)
        {
            if (card.Segment.CardId == _redoAsrCardId)
                card.RedoAsrSpinnerImage = frames[_redoAsrSpinnerFrameIndex];
        }
    }

    private void SyncRedoAsrSpinnerState(TranscriptEditorCardState card)
    {
        bool isSpinning = _redoAsrRunning && card.Segment.CardId == _redoAsrCardId;
        card.IsRedoAsrSpinning = isSpinning;
        card.RedoAsrSpinnerImage = GetRedoAsrSpinnerFrames()[
            isSpinning ? Math.Clamp(_redoAsrSpinnerFrameIndex, 0, GetRedoAsrSpinnerFrames().Length - 1) : 0];
    }

    private static string FormatTime(double s)
    {
        int h = (int)(s / 3600);
        int m = (int)(s % 3600 / 60);
        double sec = s % 60;
        return h > 0 ? $"{h}:{m:D2}:{sec:00.0}" : $"{m:D2}:{sec:00.0}";
    }
}
