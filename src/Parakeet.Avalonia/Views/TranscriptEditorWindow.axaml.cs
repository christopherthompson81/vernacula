using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Threading;
using Parakeet.Base;
using ParakeetCSharp.Models;
using ParakeetCSharp.Services;
using ParakeetCSharp.ViewModels;
using ParakeetCSharp.Views.Dialogs;

namespace ParakeetCSharp.Views;

public partial class TranscriptEditorWindow : Window
{
    private readonly TranscriptEditorViewModel _vm = null!;
    private readonly VocabService? _vocab;
    private readonly string _dbPath = "";
    private readonly string _audioBaseName = "";
    private readonly bool _asrModelsAvailable;
    private bool _isUpdatingUi;
    private bool _seekDragging;
    private bool _redoAsrRunning;

    private ObservableCollection<TranscriptEditorCardState> Cards { get; } = [];

    public event Action? DataChanged;

    public TranscriptEditorWindow()
    {
        InitializeComponent();
        DataContext = this;
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

        SeekSlider.PointerPressed += (_, _) => _seekDragging = true;
        SeekSlider.PointerReleased += OnSeekSliderPointerReleased;
        SpeedSlider.PropertyChanged += OnSpeedSliderPropertyChanged;
    }

    private void OnLoaded(object? sender, RoutedEventArgs e)
    {
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
        ThemeManager.ThemeChanged += OnThemeChanged;
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;

        HeaderTitle.Text = _audioBaseName;
        RefreshPlaybackModeCombo();

        _vm.Load(_dbPath, _audioBaseName);
        RebuildCards();
        RefreshHeader();
        RefreshPlaybackUi();
        ApplyFocusedIndex(_vm.FocusedIndex, force: true);
    }

    private void OnClosed(object? sender, EventArgs e)
    {
        PersistAllDrafts();
        ThemeManager.ThemeChanged -= OnThemeChanged;
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
        _vm.PropertyChanged -= OnViewModelPropertyChanged;
        _vm.FocusedIndexChanging -= OnFocusedIndexChanging;
        _vm.Segments.CollectionChanged -= OnSegmentsCollectionChanged;
        _vm.Dispose();
    }

    private void OnThemeChanged(AppTheme _)
    {
        Dispatcher.UIThread.Post(() =>
        {
            RefreshHeader();
            RefreshPlaybackUi();
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
        if (oldIndex >= 0 && oldIndex < Cards.Count)
        {
            SyncDraftToSegment(Cards[oldIndex]);
        }
    }

    private void OnSegmentsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
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
        Cards.Clear();

        for (int i = 0; i < _vm.Segments.Count; i++)
        {
            var state = new TranscriptEditorCardState(_vm.Segments[i], i);
            RefreshCardState(state, preserveDrafts: false);
            Cards.Add(state);
        }

        SegmentList.ItemsSource = Cards;
    }

    private void RefreshAllCardState()
    {
        foreach (var card in Cards)
        {
            card.Index = Cards.IndexOf(card);
            RefreshCardState(card, preserveDrafts: true);
        }
    }

    private void RefreshCardState(TranscriptEditorCardState card, bool preserveDrafts)
    {
        var seg = card.Segment;
        if (!preserveDrafts)
        {
            card.DraftSpeakerName = seg.SpeakerDisplayName;
            card.DraftContent = seg.Content;
        }

        card.RefreshDerived();
        card.SpeakerChoices.Clear();
        foreach (var choice in CreateSpeakerChoices(seg.SpeakerId))
        {
            card.SpeakerChoices.Add(choice);
        }

        card.SelectedSpeaker = card.SpeakerChoices.FirstOrDefault(c => c.SpeakerId == seg.SpeakerId);
        card.SuppressButtonText = seg.IsSuppressed ? Loc.Instance["editor_unsuppress"] : Loc.Instance["editor_suppress"];
        card.SuppressGlyph = seg.IsSuppressed ? "↺" : "⊘";
        card.CanMergePrev = !_redoAsrRunning && card.Index > 0;
        card.CanMergeNext = !_redoAsrRunning && card.Index < Cards.Count - 1;
        card.CanSplit = !_redoAsrRunning && seg.Tokens.Count > 1;
        card.CanRedoAsr = !_redoAsrRunning && _asrModelsAvailable && _vm.HasAudio;
        card.CanAdjustTimes = !_redoAsrRunning;
    }

    private void ApplyFocusedIndex(int focusedIndex, bool force = false)
    {
        if (focusedIndex < 0 || focusedIndex >= Cards.Count)
        {
            SegmentList.SelectedIndex = -1;
            return;
        }

        if (SegmentList.SelectedIndex != focusedIndex || force)
        {
            _isUpdatingUi = true;
            SegmentList.SelectedIndex = focusedIndex;
            _isUpdatingUi = false;
        }

        for (int i = 0; i < Cards.Count; i++)
        {
            Cards[i].IsFocused = i == focusedIndex;
            if (i == focusedIndex || force)
            {
                RefreshCardState(Cards[i], preserveDrafts: true);
            }
        }

        PrevButton.IsEnabled = _vm.PrevSegmentCommand.CanExecute(null);
        NextButton.IsEnabled = _vm.NextSegmentCommand.CanExecute(null);
        SegmentList.ScrollIntoView(Cards[focusedIndex]);
    }

    private void SetFocusedIndex(int index, bool force = false)
    {
        if (index < 0 || index >= Cards.Count)
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
        HeaderSubtext.Text = $"{_vm.Segments.Count} {Loc.Instance["results_segments_label"]}";
    }

    private void RefreshPlaybackModeCombo()
    {
        PlayModeCombo.ItemsSource = new[]
        {
            Loc.Instance["editor_mode_single"],
            Loc.Instance["editor_auto_advance"],
            Loc.Instance["editor_mode_continuous"],
        };
        PlayModeCombo.SelectedIndex = (int)_vm.PlaybackMode;
    }

    private void RefreshPlaybackUi()
    {
        _isUpdatingUi = true;
        PlayPauseButton.Content = _vm.IsPlaying ? "⏸" : "▶";
        if (!_seekDragging)
        {
            SeekSlider.Value = _vm.PlaybackPosition;
        }
        SpeedSlider.Value = _vm.PlaybackSpeed;
        SpeedLabel.Text = $"{_vm.PlaybackSpeed:0.0}x";
        PlayModeCombo.SelectedIndex = (int)_vm.PlaybackMode;

        bool playbackSupported = TranscriptEditorViewModel.SupportsAudioPlayback;
        PlayPauseButton.IsEnabled = playbackSupported && (_vm.IsPlaying || _vm.PlayCommand.CanExecute(null));
        SeekSlider.IsEnabled = playbackSupported;
        SpeedSlider.IsEnabled = playbackSupported;
        PlayModeCombo.IsEnabled = playbackSupported;
        PrevButton.IsEnabled = _vm.PrevSegmentCommand.CanExecute(null);
        NextButton.IsEnabled = _vm.NextSegmentCommand.CanExecute(null);
        _isUpdatingUi = false;
    }

    private static TranscriptEditorCardState? CardFromSender(object? sender)
        => (sender as StyledElement)?.DataContext as TranscriptEditorCardState;

    private void SegmentList_SelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (_isUpdatingUi)
        {
            return;
        }

        if (SegmentList.SelectedItem is TranscriptEditorCardState card && card.Index != _vm.FocusedIndex)
        {
            _vm.NavigateTo(card.Index);
        }
    }

    private void SpeakerNameBox_LostFocus(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is { } card)
        {
            SyncDraftToSegment(card);
        }
    }

    private void EditTextBox_LostFocus(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is { } card)
        {
            SyncDraftToSegment(card);
        }
    }

    private void SyncDraftToSegment(TranscriptEditorCardState card)
    {
        card.Segment.Content = card.DraftContent ?? string.Empty;
        if (!string.IsNullOrWhiteSpace(card.DraftSpeakerName))
        {
            card.Segment.SpeakerDisplayName = card.DraftSpeakerName.Trim();
        }
    }

    private void SaveDraft(TranscriptEditorCardState card)
    {
        SyncDraftToSegment(card);

        bool changed = false;
        if (_vm.SaveContent(card.Index, card.DraftContent ?? string.Empty))
        {
            changed = true;
        }

        string speakerName = (card.DraftSpeakerName ?? string.Empty).Trim();
        if (!string.IsNullOrEmpty(speakerName) && _vm.SaveSpeakerName(card.Index, speakerName))
        {
            changed = true;
        }

        if (changed)
        {
            DataChanged?.Invoke();
            RefreshAllCardState();
            ApplyFocusedIndex(_vm.FocusedIndex, force: true);
        }
    }

    private void PersistAllDrafts()
    {
        for (int i = 0; i < Cards.Count; i++)
        {
            SaveDraft(Cards[i]);
        }
    }

    private List<TranscriptEditorCardState.SpeakerChoice> CreateSpeakerChoices(int selectedSpeakerId)
    {
        return _vm.AllSpeakers
            .Select(s => new TranscriptEditorCardState.SpeakerChoice(
                s.SpeakerId,
                string.IsNullOrWhiteSpace(s.Name) ? $"speaker_{s.SpeakerId - 1}" : s.Name))
            .OrderByDescending(s => s.SpeakerId == selectedSpeakerId)
            .ThenBy(s => s.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private void SetCardStatus(TranscriptEditorCardState card, string text)
    {
        card.StatusMessage = text;
    }

    private void PlayPauseButton_Click(object? sender, RoutedEventArgs e)
    {
        if (!TranscriptEditorViewModel.SupportsAudioPlayback)
        {
            if (_vm.FocusedIndex >= 0 && _vm.FocusedIndex < Cards.Count)
            {
                SetCardStatus(Cards[_vm.FocusedIndex], "Playback is currently only available on Windows in this Avalonia port.");
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

    private void PrevButton_Click(object? sender, RoutedEventArgs e) => _vm.PrevSegmentCommand.Execute(null);

    private void NextButton_Click(object? sender, RoutedEventArgs e) => _vm.NextSegmentCommand.Execute(null);

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

    private void OnSeekSliderPointerReleased(object? sender, PointerReleasedEventArgs e)
    {
        _seekDragging = false;
        if (!_isUpdatingUi)
        {
            _vm.Seek(SeekSlider.Value);
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

    private void RenameSpeaker_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        string name = (card.DraftSpeakerName ?? string.Empty).Trim();
        if (string.IsNullOrEmpty(name))
        {
            return;
        }

        if (_vm.SaveSpeakerName(card.Index, name))
        {
            card.DraftSpeakerName = name;
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

        if (_vm.ReassignSegment(card.Index, card.SelectedSpeaker.SpeakerId))
        {
            card.DraftSpeakerName = _vm.Segments[card.Index].SpeakerDisplayName;
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

        if (_vm.ToggleSuppressed(card.Index))
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

        if (_vm.AdjustSegmentTimes(card.Index, dialog.NewStartTime, dialog.NewEndTime))
        {
            DataChanged?.Invoke();
            RefreshCardState(card, preserveDrafts: true);
            SetCardStatus(card, "Segment times updated.");
        }
    }

    private void MergePrev_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        SaveDraft(card);
        if (_vm.MergeWithPrev(card.Index))
        {
            DataChanged?.Invoke();
            RebuildCards();
            RefreshHeader();
            SetFocusedIndex(Math.Max(0, card.Index - 1), force: true);
            if (_vm.FocusedIndex >= 0 && _vm.FocusedIndex < Cards.Count)
            {
                SetCardStatus(Cards[_vm.FocusedIndex], "Segments merged.");
            }
        }
    }

    private void MergeNext_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        SaveDraft(card);
        if (_vm.MergeWithNext(card.Index))
        {
            DataChanged?.Invoke();
            RebuildCards();
            RefreshHeader();
            SetFocusedIndex(Math.Min(card.Index, Cards.Count - 1), force: true);
            if (_vm.FocusedIndex >= 0 && _vm.FocusedIndex < Cards.Count)
            {
                SetCardStatus(Cards[_vm.FocusedIndex], "Segments merged.");
            }
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
        if (dialog.SplitTokenIndex <= 0)
        {
            return;
        }

        SaveDraft(card);
        if (_vm.SplitSegment(card.Index, dialog.SplitTokenIndex, _vocab))
        {
            DataChanged?.Invoke();
            RebuildCards();
            RefreshHeader();
            SetFocusedIndex(Math.Min(card.Index, Cards.Count - 1), force: true);
            if (_vm.FocusedIndex >= 0 && _vm.FocusedIndex < Cards.Count)
            {
                SetCardStatus(Cards[_vm.FocusedIndex], "Segment split.");
            }
        }
    }

    private async void RedoAsr_Click(object? sender, RoutedEventArgs e)
    {
        if (CardFromSender(sender) is not { } card)
        {
            return;
        }

        if (_redoAsrRunning || !_asrModelsAvailable || !_vm.HasAudio)
        {
            return;
        }

        _redoAsrRunning = true;
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
            SetCardStatus(card, "ASR regenerated.");
        }
        catch (Exception ex)
        {
            SetCardStatus(card, $"Redo ASR failed: {ex.Message}");
        }
        finally
        {
            _redoAsrRunning = false;
            RefreshAllCardState();
        }
    }
}
