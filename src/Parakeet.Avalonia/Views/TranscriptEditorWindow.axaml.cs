using System.Collections.Generic;
using System.Text;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Threading;
using Avalonia.Animation;
using Parakeet.Base;
using Parakeet.Base.Models;
using ParakeetCSharp.Models;
using ParakeetCSharp.Services;
using ParakeetCSharp.ViewModels;
using ParakeetCSharp.Views.Dialogs;

namespace ParakeetCSharp.Views;

/// <summary>
/// ⚠️ AVALONIA MIGRATION REQUIRED ⚠️
/// 
/// This file uses several WPF-specific types that need to be replaced:
/// 
/// Types already converted:
/// - Window → Avalonia.Controls.Window (already correct)
/// - DispatcherTimer → Avalonia.Threading.DispatcherTimer (already correct)
/// 
/// WPF types requiring replacement:
/// - RichTextBox → No direct equivalent in Avalonia
///   - Use TextBox with multiline support for simple text editing
///   - For rich text editing, consider: AvaloniaEdit, RichTextBox from community libraries
///   - Or build custom control using TextBlock with inline styling
/// 
/// - Run → Avalonia.Controls.TextBlock.Inlines (TextSpan)
/// - FlowDocument → N/A (not used directly, but referenced in comments)
/// - TextBlock → Avalonia.Controls.TextBlock (already correct)
/// - Border → Avalonia.Controls.Border (already correct)
/// - Button → Avalonia.Controls.Button (already correct)
/// - TextBox → Avalonia.Controls.TextBox (already correct)
/// - StackPanel → Avalonia.Controls.StackPanel (already correct)
/// - Grid → Avalonia.Controls.Grid (already correct)
/// - ColumnDefinition → Avalonia.Controls.ColumnDefinition (already correct)
/// - ScrollViewer → Avalonia.Controls.ScrollViewer (already correct)
/// - Popup → Avalonia.Controls.Popup (already correct)
/// - VisualTreeHelper → Avalonia.Media.VisualTreeHelper (already available)
/// - DependencyObject → AvaloniaObject
/// - FrameworkElement → AvaloniaObject or Control
/// - Cursor, Cursors → Avalonia.Input.Cursors (already available)
/// - Thickness → Avalonia.Thickness (already available)
/// - CornerRadius → Avalonia.CornerRadius (already available)
/// - FontWeights → Avalonia.Media.FontWeights (already available)
/// - FontStyles → Avalonia.Media.FontStyles (already available)
/// - Brush → IBrush
/// - Color → Color (Avalonia.Media.Color, already available)
/// - SolidColorBrush → SolidColorBrush (Avalonia.Media.SolidColorBrush, already available)
/// - GridLength, GridUnitType → Avalonia.Layout.GridLength, GridUnitType (already available)
/// - Visibility → Avalonia.Media.Visibility (already available)
/// - HorizontalAlignment, VerticalAlignment → Avalonia.Controls (already available)
/// - Orientation → Avalonia.Controls.Orientation (already available)
/// - ScrollBarVisibility → Avalonia.Controls.ScrollBarVisibility (already available)
/// - Dock, DockPanel → Avalonia.Controls.DockPanel (already available)
/// - Key, KeyEventArgs → Avalonia.Input.Key, KeyEventArgs (already available)
/// - Keyboard → Avalonia.Input.Keyboard (already available)
/// - ModifierKeys → Avalonia.Input.KeyModifiers (different API)
/// - MouseButtonEventArgs → Avalonia.Input.MouseButtonEventArgs (already available)
/// - Clipboard → Avalonia.Clipboard (different API)
/// 
/// RichTextBox-specific methods that need replacement:
/// - SetRichTextBoxText() → Set text content in alternative control
/// - GetRichTextBoxText() → Get text content from alternative control
/// - RichTextBox.Document → Depends on replacement control
/// - FlowDocument, Paragraph, Run → Not applicable in Avalonia
/// 
/// Text manipulation in the editor:
/// - The editor uses RichTextBox for side-by-side ASR vs edited text comparison
/// - Diff highlighting uses Run elements with different background colors
/// - In Avalonia, consider:
///   1. Using two TextBlocks (ASR on left, edited on right)
///   2. Using AvaloniaEdit for syntax-highlighting-like diff display
///   3. Building custom control with TextBlock.Inlines for colored spans
/// </summary>
public partial class TranscriptEditorWindow : Window
{
    private readonly TranscriptEditorViewModel _vm;
    private readonly VocabService?             _vocab;
    private readonly List<CardInfo>            _cards = [];
    private          DispatcherTimer?          _diffTimer;
    private          bool                      _suppressDiff;
    private          int                       _prevFocusedIndex = -1;
    private          bool                      _seekDragging;
    private          EditorSegment?            _subscribedSegment;
    private          bool                      _asrModelsAvailable;
    private          bool                      _asrRedoInProgress;
    private          Popup?                    _seekPopup;
    private          TextBlock?                _seekTimeText;

    /// <summary>Raised whenever segment content or speaker names are persisted.</summary>
    public event Action? DataChanged;

    // ── Card metadata ─────────────────────────────────────────────────────────

    private sealed class CardInfo
      {
        public Border          Root             = null!;
        public Border          FocusedContent   = null!;
        public Border          AdjacentContent  = null!;
        public TextBlock       AsrTextBlock     = null!;
        public List<object>  AsrRuns          = [];
        public TextBlock       AdjTextBlock     = null!;
        public List<object>  AdjRuns          = [];
        public TextBox         EditBox          = null!;  // Changed from RichTextBox to TextBox
        public TextBox         SpeakerEdit      = null!;
        public TextBlock       SpeakerLabel     = null!;
        public TextBlock       FocusedTimeLabel = null!;
        public TextBlock       AdjTimeLabel     = null!;
        public Button          SuppressBtn      = null!;
        public TextBlock       SuppressedBadge  = null!;
        public Button          ReassignBtn      = null!;
        public Button          AdjustTimesBtn   = null!;
        public Button          MergePrevBtn     = null!;
        public Button          MergeNextBtn     = null!;
        public Button          SplitBtn         = null!;
        public Button          RedoAsrBtn       = null!;
        public TextBlock       AsrLabel         = null!;
        public TextBlock       EditLabel        = null!;
        public bool            AdjRunsBuilt;
    }

    // ── Helper methods ───────────────────────────────────────────────────────

    /// <summary>Gets a resource brush by name from the application resources.</summary>
    private SolidColorBrush GetResourceBrush(string key)
    {
        var brush = Application.Current?.FindResource(key); 
            return brush is SolidColorBrush sb ? sb : (SolidColorBrush)Brushes.Black;
    }

    // ── Constructor ───────────────────────────────────────────────────────────

    public TranscriptEditorWindow(string dbPath, string audioBaseName)
    {
        InitializeComponent();

        SourceInitialized += (_, _) =>
        {
            DwmHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
            ThemeManager.ThemeChanged += OnThemeChanged;
            Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
        };

        _vm = new TranscriptEditorViewModel();
        _vm.PlaybackMode = App.Current.Settings.Current.EditorPlaybackMode;
        DataContext = _vm;

        string modelsDir = App.Current.Settings.GetModelsDir();
        string vocabPath = System.IO.Path.Combine(modelsDir, Config.VocabFile);
        if (System.IO.File.Exists(vocabPath))
            _vocab = new VocabService(modelsDir);

        var (encFile, _) = Config.GetAsrFiles(App.Current.Settings.Current.Precision);
        _asrModelsAvailable = System.IO.File.Exists(System.IO.Path.Combine(modelsDir, encFile));

        _vm.FocusedIndexChanging += OnFocusedIndexChanging;
        _vm.PropertyChanged += (_, e) =>
        {
            switch (e.PropertyName)
            {
                case nameof(TranscriptEditorViewModel.HighlightedToken):
                    UpdateTokenHighlight(_vm.HighlightedToken);
                    break;
                case nameof(TranscriptEditorViewModel.FocusedIndex):
                    OnFocusChanged(_vm.FocusedIndex);
                    break;
                case nameof(TranscriptEditorViewModel.IsPlaying):
                    PlayPauseBtn.Content = _vm.IsPlaying ? "⏸" : "▶";
                    break;
                case nameof(TranscriptEditorViewModel.PlaybackMode):
                    if (PlayModeCombo.SelectedIndex != (int)_vm.PlaybackMode)
                        PlayModeCombo.SelectedIndex = (int)_vm.PlaybackMode;
                    if (_seekPopup != null) _seekPopup.IsOpen = false;
                    break;
                case nameof(TranscriptEditorViewModel.PlaybackPosition):
                    if (!_seekDragging) SeekSlider.Value = _vm.PlaybackPosition;
                    break;
                case nameof(TranscriptEditorViewModel.FocusedSegment):
                    if (_subscribedSegment != null)
                        _subscribedSegment.PropertyChanged -= OnFocusedSegmentPropertyChanged;
                    _subscribedSegment = _vm.FocusedSegment;
                    if (_subscribedSegment != null)
                        _subscribedSegment.PropertyChanged += OnFocusedSegmentPropertyChanged;
                    break;
            }
        };

        SetupAudioBar();

        Loaded += (_, _) =>
        {
            _vm.Load(dbPath, audioBaseName);
            HeaderTitle.Text   = audioBaseName;
            HeaderSubtext.Text = $"{_vm.Segments.Count} {Loc.Instance["results_segments_label"]}";
            BuildAllCards();
            if (_vm.Segments.Count > 0)
            {
                OnFocusChanged(0);
                // FocusedIndex doesn't change during Load (stays 0), so OnFocusedIndexChanged
                // never fires and _subscribedSegment is never wired. Set it up here.
                _subscribedSegment = _vm.Segments[0];
                _subscribedSegment.PropertyChanged += OnFocusedSegmentPropertyChanged;
            }
        };

        PreviewKeyDown += Window_PreviewKeyDown;
        Closed += (_, _) =>
        {
            ThemeManager.ThemeChanged -= OnThemeChanged;
            Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
            if (_subscribedSegment != null)
                _subscribedSegment.PropertyChanged -= OnFocusedSegmentPropertyChanged;
            _vm.Dispose();
            Owner?.Activate();
        };
    }

    // ── Audio bar setup ───────────────────────────────────────────────────────

    private void SetupAudioBar()
    {
        RefreshPlayModeCombo();
        PlayModeCombo.SelectionChanged += (_, _) =>
        {
            var mode = (PlaybackMode)PlayModeCombo.SelectedIndex;
            _vm.PlaybackMode = mode;
            App.Current.Settings.Current.EditorPlaybackMode = mode;
            App.Current.Settings.Save();
        };

        VerifiedChk.Content = Loc.Instance["editor_verified"];
        VerifiedChk.ToolTip = Loc.Instance["editor_verified_tooltip"];

        PlayPauseBtn.Click += (_, _) =>
        {
            if (_vm.IsPlaying) _vm.PauseCommand.Execute(null);
            else               _vm.PlayCommand.Execute(null);
        };

        PrevBtn.Command = _vm.PrevSegmentCommand;
        NextBtn.Command = _vm.NextSegmentCommand;

        // When audio finishes loading the play command becomes available; sync redo ASR buttons too.
        _vm.PlayCommand.CanExecuteChanged += (_, _) =>
        {
            bool enabled = _asrModelsAvailable && _vm.HasAudio;
            foreach (var c in _cards)
                c.RedoAsrBtn.IsEnabled = enabled;
        };

        VerifiedChk.Click += (_, _) => _vm.ToggleVerifiedCommand.Execute(null);

        // Seek slider
        SeekSlider.PreviewMouseDown += (_, _) => _seekDragging = true;
        SeekSlider.PreviewMouseUp   += (_, _) =>
        {
            _seekDragging = false;
            _vm.Seek(SeekSlider.Value);
        };

        // Speed slider
        SpeedSlider.Value = _vm.PlaybackSpeed;
        UpdateSpeedLabel();
        SpeedSlider.ValueChanged += (_, _) =>
        {
            _vm.PlaybackSpeed = SpeedSlider.Value;
            UpdateSpeedLabel();
        };
        SpeedSlider.PreviewMouseWheel += (_, e) =>
        {
            SpeedSlider.Value = Math.Clamp(
                SpeedSlider.Value + (e.Delta > 0 ? 0.1 : -0.1),
                SpeedSlider.Minimum, SpeedSlider.Maximum);
            e.Handled = true;
        };

        // Seek-time bubble (continuous mode)
        BuildSeekPopup();
        SeekSlider.MouseMove  += OnSeekSliderMouseMove;
        SeekSlider.MouseLeave += (_, _) => { if (_seekPopup != null) _seekPopup.IsOpen = false; };
    }

    private void UpdateSpeedLabel() => SpeedLabel.Text = $"{_vm.PlaybackSpeed:0.0}x";

    private void RefreshPlayModeCombo()
    {
        int prev = PlayModeCombo.SelectedIndex;
        PlayModeCombo.Items.Clear();
        PlayModeCombo.Items.Add(Loc.Instance["editor_mode_single"]);
        PlayModeCombo.Items.Add(Loc.Instance["editor_auto_advance"]);
        PlayModeCombo.Items.Add(Loc.Instance["editor_mode_continuous"]);
        PlayModeCombo.SelectedIndex = prev < 0 ? (int)_vm.PlaybackMode : prev;
    }

    // ── Keyboard navigation ───────────────────────────────────────────────────

    private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.F1)
        {
            e.Handled = true;
            _vm.ToggleVerifiedCommand.Execute(null);
            return;
        }

        bool editHasFocus = FocusedCard()?.EditBox.IsKeyboardFocusWithin ?? false;
        if (editHasFocus) return;
        if (_asrRedoInProgress) { e.Handled = true; return; }

        if (e.Key == Key.Tab && Keyboard.Modifiers == KeyModifiers.None)
        {
            e.Handled = true;
            _vm.NextSegmentCommand.Execute(null);
        }
        else if (e.Key == Key.Tab && (Keyboard.Modifiers & KeyModifiers.Shift) != 0)
        {
            e.Handled = true;
            _vm.PrevSegmentCommand.Execute(null);
        }
    }

    // ── Card building ─────────────────────────────────────────────────────────

    private void BuildAllCards()
    {
        SegmentStack.Children.Clear();
        _cards.Clear();

        for (int i = 0; i < _vm.Segments.Count; i++)
        {
            var seg  = _vm.Segments[i];
            var card = BuildCard(seg, i);
            _cards.Add(card);
            SegmentStack.Children.Add(card.Root);
        }

        // Eagerly populate all adjacent cards so content is visible on first load.
        // After this, OnFocusChanged keeps only neighbors updated (lazy).
        int focused = _vm.FocusedIndex;
        for (int i = 0; i < _cards.Count; i++)
        {
            if (i != focused)
                UpdateAdjacentCard(_cards[i], _vm.Segments[i]);
        }
    }

    private CardInfo BuildCard(EditorSegment seg, int index)
    {
        var card = new CardInfo();

        // ── Adjacent view (shown when not focused) ────────────────────────────
        var adjTextBlock = new TextBlock
        {
            FontSize     = 12,
            Foreground   = GetResourceBrush("SubtextBrush"),
            TextWrapping = TextWrapping.Wrap,
            Padding      = new Thickness(4, 2, 4, 2),
        };
        card.AdjTextBlock = adjTextBlock;

        var adjTimeLabel = new TextBlock
        {
            FontSize   = 11,
            Foreground = GetResourceBrush("SubtextBrush"),
            Text       = FormatTimeRange(seg.PlayStart, seg.PlayEnd),
        };
        card.AdjTimeLabel = adjTimeLabel;

        var suppressedBadge = new TextBlock
        {
            Text              = Loc.Instance["editor_suppressed_badge"],
            Foreground        = GetResourceBrush("RedBrush"),
            FontWeight        = FontWeights.Bold,
            FontSize          = 11,
            Margin            = new Thickness(0, 0, 8, 0),
            VerticalAlignment = VerticalAlignment.Center,
            Visibility        = seg.IsSuppressed ? Visibility.Visible : Visibility.Collapsed,
        };
        card.SuppressedBadge = suppressedBadge;

        var adjHeader = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 4) };
        adjHeader.Children.Add(suppressedBadge);
        var adjSpeakerLabel = new TextBlock
        {
            Foreground = GetResourceBrush("SubtextBrush"),
            FontSize   = 12,
            FontWeight = FontWeights.SemiBold,
            Text       = seg.SpeakerDisplayName,
            Margin     = new Thickness(0, 0, 8, 0),
        };
        adjHeader.Children.Add(adjSpeakerLabel);
        adjHeader.Children.Add(adjTimeLabel);

        var adjGrid = new StackPanel { Margin = new Thickness(8) };
        adjGrid.Children.Add(adjHeader);
        adjGrid.Children.Add(adjTextBlock);

        card.AdjacentContent = new Border
        {
            Background   = Brushes.Transparent,
            CornerRadius = new CornerRadius(6),
            Child        = adjGrid,
            Visibility   = Visibility.Visible,
        };

        card.SpeakerLabel = adjSpeakerLabel;

        // ── Focused view ──────────────────────────────────────────────────────

        card.SpeakerEdit = new TextBox
        {
            Text       = seg.SpeakerDisplayName,
            FontWeight = FontWeights.SemiBold,
            Margin     = new Thickness(0, 0, 8, 0),
            MinWidth   = 80,
            MaxWidth   = 200,
        };
        int capturedIndex = index;
        card.SpeakerEdit.LostFocus += (_, _) =>
        {
            if (_vm.SaveSpeakerName(capturedIndex, card.SpeakerEdit.Text))
            {
                RefreshAllSpeakerLabels();
                DataChanged?.Invoke();
            }
        };
        card.SpeakerEdit.KeyDown += (_, e) =>
        {
            if (e.Key is Key.Enter or Key.Tab)
            {
                if (_vm.SaveSpeakerName(capturedIndex, card.SpeakerEdit.Text))
                {
                    RefreshAllSpeakerLabels();
                    DataChanged?.Invoke();
                }
                card.EditBox.Focus();
                e.Handled = true;
            }
        };
        // Speaker scroll wheel: cycle through speakers
        card.SpeakerEdit.PreviewMouseWheel += (_, e) =>
        {
            var speakers = _vm.AllSpeakers;
            if (speakers.Count < 2) { e.Handled = true; return; }
            int ci = speakers.FindIndex(s => s.SpeakerId == _vm.Segments[capturedIndex].SpeakerId);
            int ni = (ci + (e.Delta > 0 ? -1 : 1) + speakers.Count) % speakers.Count;
            ApplyReassignment(capturedIndex, speakers[ni].SpeakerId);
            e.Handled = true;
        };

        var reassignBtn = new Button
        {
            Content           = MakeActionIcon("\uE77B"),  // Segoe MDL2: People/Contact
            Width             = 26,
            Height            = 26,
            Padding           = new Thickness(2),
            Margin            = new Thickness(4, 0, 8, 0),
            ToolTip           = Loc.Instance["editor_reassign_tooltip"],
            VerticalAlignment = VerticalAlignment.Center,
        };
        reassignBtn.Click += (_, _) => ShowReassignMenu(reassignBtn, capturedIndex);
        card.ReassignBtn = reassignBtn;

        var timeLabel = new TextBlock
        {
            Text              = FormatTimeRange(seg.PlayStart, seg.PlayEnd),
            Foreground        = GetResourceBrush("SubtextBrush"),
            VerticalAlignment = VerticalAlignment.Center,
        };
        card.FocusedTimeLabel = timeLabel;

        var adjustTimesBtnInline = new Button
        {
            Content  = MakeActionIcon("\uE916"),   // Segoe MDL2: Clock/Timer
            Width    = 26,
            Height   = 26,
            Padding  = new Thickness(2),
            Margin            = new Thickness(6, 0, 0, 0),
            ToolTip           = Loc.Instance["editor_adjust_times_tooltip"],
            VerticalAlignment = VerticalAlignment.Center,
        };
        adjustTimesBtnInline.Click += (_, _) => HandleAdjustTimes(capturedIndex, card);
        card.AdjustTimesBtn = adjustTimesBtnInline;

        var segHeader = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            Margin      = new Thickness(0, 0, 0, 8),
        };
        segHeader.Children.Add(card.SpeakerEdit);
        segHeader.Children.Add(reassignBtn);
        segHeader.Children.Add(timeLabel);
        segHeader.Children.Add(adjustTimesBtnInline);

        // ── Action toolbar ────────────────────────────────────────────────────

        var suppressBtn = new Button
        {
            Content  = MakeActionIcon("\uEA39"),   // Segoe MDL2: Blocked (⊘-style circle-slash)
            Width    = 30,
            Height   = 26,
            Padding  = new Thickness(2),
            Margin   = new Thickness(0, 0, 4, 0),
            ToolTip  = seg.IsSuppressed ? Loc.Instance["editor_unsuppress_tooltip"] : Loc.Instance["editor_suppress_tooltip"],
        };
        card.SuppressBtn = suppressBtn;
        suppressBtn.Click += (_, _) => HandleSuppress(capturedIndex, card);

        var mergePrevBtn = new Button
        {
            Content   = MakeActionIcon("\uE72B"),  // Segoe MDL2: Back (←)
            Width     = 30,
            Height    = 26,
            Padding   = new Thickness(2),
            Margin    = new Thickness(0, 0, 4, 0),
            IsEnabled = index > 0,
            ToolTip   = Loc.Instance["editor_merge_prev_tooltip"],
        };
        mergePrevBtn.Click += async (_, _) => await HandleMergePrev(capturedIndex);
        card.MergePrevBtn = mergePrevBtn;

        var mergeNextBtn = new Button
        {
            Content   = MakeActionIcon("\uE72A"),  // Segoe MDL2: Forward (→)
            Width     = 30,
            Height    = 26,
            Padding   = new Thickness(2),
            Margin    = new Thickness(0, 0, 4, 0),
            IsEnabled = index < _vm.Segments.Count - 1,
            ToolTip   = Loc.Instance["editor_merge_next_tooltip"],
        };
        mergeNextBtn.Click += async (_, _) => await HandleMergeNext(capturedIndex);
        card.MergeNextBtn = mergeNextBtn;

        var splitBtn = new Button
        {
            Content   = MakeActionIcon("\uE8C6"),  // Segoe MDL2: Cut (scissors)
            Width     = 30,
            Height    = 26,
            Padding   = new Thickness(2),
            Margin    = new Thickness(0, 0, 4, 0),
            IsEnabled = seg.Tokens.Count > 1 && _vocab != null && seg.Sources.Count == 1,
            ToolTip   = Loc.Instance["editor_split_tooltip"],
        };
        splitBtn.Click += (_, _) => HandleSplit(capturedIndex);
        card.SplitBtn = splitBtn;

        var redoAsrBtn = new Button
        {
            Content           = MakeRedoAsrIcon(),
            Width             = 30,
            Height            = 26,
            Padding           = new Thickness(2),
            IsEnabled         = _asrModelsAvailable && _vm.HasAudio,
            ToolTip           = Loc.Instance["editor_redo_asr_tooltip"],
        };
        redoAsrBtn.Click += async (_, _) => await HandleRedoAsr(capturedIndex, card);
        card.RedoAsrBtn = redoAsrBtn;

        var actionBar = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            Margin      = new Thickness(0, 0, 0, 8),
        };
        actionBar.Children.Add(suppressBtn);
        actionBar.Children.Add(mergePrevBtn);
        actionBar.Children.Add(mergeNextBtn);
        actionBar.Children.Add(splitBtn);
        actionBar.Children.Add(redoAsrBtn);

        // ASR confidence-colored panel (left)
        card.AsrTextBlock = new TextBlock
        {
            TextWrapping = TextWrapping.Wrap,
            Padding      = new Thickness(4),
            Foreground   = GetResourceBrush("TextBrush"),
        };

        var asrScrollViewer = new ScrollViewer
        {
            VerticalScrollBarVisibility   = ScrollBarVisibility.Auto,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
            Background = Brushes.Transparent,
            Content    = card.AsrTextBlock,
        };

        var asrLabel = new TextBlock
        {
            Text       = Loc.Instance["editor_asr_label"],
            FontSize   = 11,
            Foreground = GetResourceBrush("SubtextBrush"),
            Margin     = new Thickness(0, 0, 0, 4),
        };

        card.AsrLabel = asrLabel;

        var asrPanel = new DockPanel { Margin = new Thickness(0, 0, 4, 0) };
        DockPanel.SetDock(asrLabel, Dock.Top);
        asrPanel.Children.Add(asrLabel);
        asrPanel.Children.Add(asrScrollViewer);

        // ⚠️ TODO: RichTextBox doesn't exist in Avalonia. For now, using TextBox as placeholder.
        // The diff highlighting features will need to be reimplemented using TextBlock.Inlines
        // or a third-party library like AvaloniaEdit.
        card.EditBox = new TextBox
        {
            Background      = GetResourceBrush("SurfaceBrush"),
            Foreground      = GetResourceBrush("TextBrush"),
            BorderBrush     = GetResourceBrush("OverlayBrush"),
            BorderThickness = new Thickness(1),
            Padding         = new Thickness(4),
            AcceptsReturn   = true,
            TextWrapping    = TextWrapping.Wrap,
            VerticalScrollBarVisibility   = ScrollBarVisibility.Auto,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
        };

        // Set text content (simple version for TextBox)
        card.EditBox.Text = seg.Content;

        card.EditBox.TextChanged += (_, _) => ScheduleDiffUpdate(card, capturedIndex);
        card.EditBox.LostFocus   += (_, _) =>
        {
            if (_vm.SaveContent(capturedIndex, card.EditBox.Text))
                DataChanged?.Invoke();
            RebuildDiffHighlight(card, capturedIndex);
        };

        var editLabel = new TextBlock
        {
            Text       = Loc.Instance["editor_edit_label"],
            FontSize   = 11,
            Foreground = GetResourceBrush("SubtextBrush"),
            Margin     = new Thickness(0, 0, 0, 4),
        };

        card.EditLabel = editLabel;

        var editPanel = new DockPanel { Margin = new Thickness(4, 0, 0, 0) };
        DockPanel.SetDock(editLabel, Dock.Top);
        editPanel.Children.Add(editLabel);
        editPanel.Children.Add(card.EditBox);

        var compGrid = new Grid { Height = 180 };
        compGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        compGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        Grid.SetColumn(asrPanel,  0);
        Grid.SetColumn(editPanel, 1);
        compGrid.Children.Add(asrPanel);
        compGrid.Children.Add(editPanel);

        // Focused card body (no per-card audio controls — shared bar handles those)
        var focusedBody = new StackPanel { Margin = new Thickness(12) };
        focusedBody.Children.Add(segHeader);
        focusedBody.Children.Add(actionBar);
        focusedBody.Children.Add(compGrid);

        card.FocusedContent = new Border
        {
            Background      = GetResourceBrush("SurfaceBrush"),
            BorderBrush     = GetResourceBrush("AccentBrush"),
            BorderThickness = new Thickness(2),
            CornerRadius    = new CornerRadius(8),
            Child           = focusedBody,
            Visibility      = Visibility.Collapsed,
        };

        card.Root = new Border
        {
            Margin       = new Thickness(0, 4, 0, 4),
            CornerRadius = new CornerRadius(6),
            Cursor       = Cursors.Hand,
        };
        card.Root.MouseLeftButtonDown += (_, e) =>
        {
            if (_asrRedoInProgress) { e.Handled = true; return; }
            if (_vm.FocusedIndex != capturedIndex)
            {
                _vm.NavigateTo(capturedIndex);
                e.Handled = true;
            }
        };

        var rootStack = new StackPanel();
        rootStack.Children.Add(card.AdjacentContent);
        rootStack.Children.Add(card.FocusedContent);
        card.Root.Child = rootStack;

        return card;
    }

    // ── Focus management ──────────────────────────────────────────────────────

    private CardInfo? FocusedCard() =>
        _vm.FocusedIndex >= 0 && _vm.FocusedIndex < _cards.Count
            ? _cards[_vm.FocusedIndex] : null;

    private void OnFocusedIndexChanging(int newIndex)
    {
        int oldIdx = _vm.FocusedIndex;
        if (oldIdx >= 0 && oldIdx < _cards.Count)
        {
           var oldCard = _cards[oldIdx];
            if (_vm.SaveContent(oldIdx, oldCard.EditBox.Text))
                DataChanged?.Invoke();
            if (_vm.SaveSpeakerName(oldIdx, oldCard.SpeakerEdit.Text))
            {
                RefreshAllSpeakerLabels();
                DataChanged?.Invoke();
            }
        }
    }

    private void OnFocusChanged(int newIndex)
    {
        int prev = _prevFocusedIndex;
        _prevFocusedIndex = newIndex;

        // Only update cards whose display state actually changed.
        // Distant cards that weren't adjacent before and still aren't need no update.
        var toUpdate = new HashSet<int>();
        void AddIfValid(int idx) { if (idx >= 0 && idx < _cards.Count) toUpdate.Add(idx); }

        AddIfValid(newIndex);
        AddIfValid(newIndex - 1);
        AddIfValid(newIndex + 1);
        if (prev >= 0)
        {
            AddIfValid(prev);
            AddIfValid(prev - 1);
            AddIfValid(prev + 1);
        }

        foreach (int i in toUpdate)
            ApplyCardState(i, newIndex);

        AnimateScrollToCenter(newIndex);
    }

    private void ApplyCardState(int i, int focused)
    {
        if (i < 0 || i >= _cards.Count) return;
        var card = _cards[i];
        var seg  = _vm.Segments[i];
        int dist = Math.Abs(i - focused);

        if (dist == 0)
        {
            card.AdjacentContent.Visibility = Visibility.Collapsed;
            card.FocusedContent.Visibility  = Visibility.Visible;

            RebuildAsrRuns(card, seg);
            RebuildDiffHighlight(card, i);
            UpdateFocusedCardBackground(card, seg);
        }
        else
        {
            card.FocusedContent.Visibility  = Visibility.Collapsed;
            card.AdjacentContent.Visibility = Visibility.Visible;
            UpdateAdjacentCard(card, seg);
        }
    }

    private void RefreshAllSpeakerLabels()
    {
        for (int i = 0; i < _cards.Count && i < _vm.Segments.Count; i++)
        {
            var c = _cards[i];
            var s = _vm.Segments[i];
            if (c.SpeakerLabel != null)
                c.SpeakerLabel.Text = s.SpeakerDisplayName;
            if (c.SpeakerEdit != null && i != _vm.FocusedIndex)
                c.SpeakerEdit.Text = s.SpeakerDisplayName;
        }
    }

    // ── Speaker reassignment ──────────────────────────────────────────────────

    private void ShowReassignMenu(Button anchor, int segmentIndex)
    {
        var menu = new ContextMenu { PlacementTarget = anchor };
        int currentSpeakerId = _vm.Segments[segmentIndex].SpeakerId;

        foreach (var (spkId, name) in _vm.AllSpeakers)
        {
            int  id   = spkId;
            bool curr = spkId == currentSpeakerId;
            var item  = new MenuItem
            {
                Header     = curr ? $"✓ {name}" : name,
                IsEnabled  = !curr,
                FontWeight = curr ? FontWeights.SemiBold : FontWeights.Normal,
            };
            item.Click += (_, _) => ApplyReassignment(segmentIndex, id);
            menu.Items.Add(item);
        }

        menu.Items.Add(new Separator());

        var addItem = new MenuItem { Header = Loc.Instance["editor_add_speaker"] };
        addItem.Click += (_, _) => PromptAddAndReassign(segmentIndex);
        menu.Items.Add(addItem);

        menu.IsOpen = true;
    }

    private void ApplyReassignment(int segmentIndex, int newSpeakerId)
    {
        if (!_vm.ReassignSegment(segmentIndex, newSpeakerId)) return;
        var card = _cards[segmentIndex];
        card.SpeakerEdit.Text  = _vm.Segments[segmentIndex].SpeakerDisplayName;
        card.SpeakerLabel.Text = _vm.Segments[segmentIndex].SpeakerDisplayName;
        DataChanged?.Invoke();
    }

    private void PromptAddAndReassign(int segmentIndex)
    {
        var dialog = new AddSpeakerDialog { Owner = this };
        if (dialog.ShowDialog() != true) return;
        int newId = _vm.AddSpeaker(dialog.SpeakerName);
        if (newId > 0)
            ApplyReassignment(segmentIndex, newId);
    }

    // ── Segment operation handlers ────────────────────────────────────────────

    private void SaveCurrentEdit()
    {
        int fi = _vm.FocusedIndex;
        if (fi >= 0 && fi < _cards.Count)
            if (_vm.SaveContent(fi, _cards[fi].EditBox.Text))
                DataChanged?.Invoke();
    }

    private void HandleSuppress(int index, CardInfo card)
    {
        if (!_vm.ToggleSuppressed(index)) return;
        var seg = _vm.Segments[index];
        card.SuppressBtn.ToolTip = seg.IsSuppressed ? Loc.Instance["editor_unsuppress_tooltip"] : Loc.Instance["editor_suppress_tooltip"];
        UpdateAdjacentCard(card, seg);
        DataChanged?.Invoke();
    }

    private void HandleAdjustTimes(int index, CardInfo card)
    {
        var seg    = _vm.Segments[index];
        var dialog = new AdjustTimesDialog(seg.PlayStart, seg.PlayEnd) { Owner = this };
        if (dialog.ShowDialog() != true) return;
        if (!_vm.AdjustSegmentTimes(index, dialog.NewStartTime, dialog.NewEndTime)) return;

        string fmt = FormatTimeRange(seg.PlayStart, seg.PlayEnd);
        card.FocusedTimeLabel.Text = fmt;
        card.AdjTimeLabel.Text     = fmt;
        DataChanged?.Invoke();
    }

    private async Task HandleMergeNext(int index)
    {
        SaveCurrentEdit();
        if (!_vm.MergeWithNext(index)) return;
        int newFocus = Math.Min(index, _vm.Segments.Count - 1);
        RebuildAfterStructuralChange(newFocus);
        UpdateHeaderSubtext();
        DataChanged?.Invoke();
        await HandleRedoAsr(newFocus, _cards[newFocus]);
    }

    private async Task HandleMergePrev(int index)
    {
        if (index <= 0) return;
        SaveCurrentEdit();
        if (!_vm.MergeWithPrev(index)) return;
        int newFocus = index - 1;
        RebuildAfterStructuralChange(newFocus);
        UpdateHeaderSubtext();
        DataChanged?.Invoke();
        await HandleRedoAsr(newFocus, _cards[newFocus]);
    }

    private void HandleSplit(int index)
    {
        var seg = _vm.Segments[index];
        if (seg.Tokens.Count <= 1 || _vocab == null || seg.Sources.Count != 1) return;
        SaveCurrentEdit();

        var runs       = _vocab.GetTokenRuns(seg.Tokens, seg.Logprobs);
        var tokenTexts = runs.Select(r => r.text).ToList();

        var dialog = new SplitSegmentDialog(tokenTexts) { Owner = this };
        if (dialog.ShowDialog() != true) return;
        if (!_vm.SplitSegment(index, dialog.SplitTokenIndex, _vocab)) return;
        RebuildAfterStructuralChange(index);
        UpdateHeaderSubtext();
        DataChanged?.Invoke();
    }

    private async Task HandleRedoAsr(int index, CardInfo card)
    {
        if (index < 0 || index >= _vm.Segments.Count) return;
        if (!_asrModelsAvailable || !_vm.HasAudio) return;
        SaveCurrentEdit();

        string modelsDir = App.Current.Settings.GetModelsDir();
        var (encoderFile, decoderJointFile) = Config.GetAsrFiles(App.Current.Settings.Current.Precision);

        SetRedoInProgress(true);
        StartSpinner(card);

        bool rebuilt = false;
        try
        {
            var result = await Task.Run(() =>
                _vm.PerformRedoAsr(index, modelsDir, encoderFile, decoderJointFile));

            if (result is null) return;

            var (newResultId, text, tokens, timestamps, logprobs) = result.Value;
            _vm.ApplyRedoAsr(index, newResultId, text, tokens, timestamps, logprobs);
            RebuildAfterStructuralChange(index);
            rebuilt = true;
            DataChanged?.Invoke();
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.Message, "Redo ASR", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            if (!rebuilt) StopSpinner(card);
            SetRedoInProgress(false);
        }
    }

    private void RebuildAfterStructuralChange(int newFocusIndex)
    {
        _prevFocusedIndex = -1;
        BuildAllCards();
        if (_vm.FocusedIndex != newFocusIndex)
            _vm.FocusedIndex = newFocusIndex;
        else
            OnFocusChanged(newFocusIndex);
    }

    private void UpdateHeaderSubtext()
        => HeaderSubtext.Text = $"{_vm.Segments.Count} {Loc.Instance["results_segments_label"]}";

    // ── Seek-time popup ───────────────────────────────────────────────────────

    private void BuildSeekPopup()
    {
        _seekTimeText = new TextBlock
        {
            FontSize            = 12,
            Foreground          = GetResourceBrush("TextBrush"),
            Padding             = new Thickness(8, 3, 8, 3),
            MinWidth            = 72,
            TextAlignment       = TextAlignment.Center,
        };
        var bubble = new Border
        {
            Background      = GetResourceBrush("SurfaceBrush"),
            BorderBrush     = GetResourceBrush("OverlayBrush"),
            BorderThickness = new Thickness(1),
            CornerRadius    = new CornerRadius(4),
            Child           = _seekTimeText,
        };
        var arrow = new TextBlock
        {
            Text                = "▼",
            FontSize            = 8,
            Foreground          = GetResourceBrush("SubtextBrush"),
            HorizontalAlignment = HorizontalAlignment.Center,
            Margin              = new Thickness(0, -1, 0, 0),
        };
        var content = new StackPanel { IsHitTestVisible = false };
        content.Children.Add(bubble);
        content.Children.Add(arrow);

        _seekPopup = new Popup
        {
            Child              = content,
            PlacementTarget    = SeekSlider,
            Placement          = PlacementMode.RelativePoint,
            AllowsTransparency = true,
            IsHitTestVisible   = false,
            PlacementClamp     = PlacementClamp.None,
        };
    }

    private void OnSeekSliderMouseMove(object sender, object e)
    {
        if (_seekPopup is null || _seekTimeText is null ||
            _vm.PlaybackMode != PlaybackMode.Continuous)
        {
            if (_seekPopup != null) _seekPopup.IsOpen = false;
            return;
        }

        double totalSec = _vm.TotalAudioSeconds ?? 0;
        if (totalSec <= 0 || SeekSlider.Bounds.Width <= 0)
        {
            _seekPopup.IsOpen = false;
            return;
        }

        var    pos   = e.GetPosition(SeekSlider);
        double ratio = Math.Clamp(pos.X / SeekSlider.Bounds.Width, 0, 1);
        _seekTimeText.Text = FormatTime(ratio * totalSec);

        // Centre the bubble horizontally on the cursor; place it above the slider.
        // Avalonia Popup uses PlacementRectangle for positioning
        _seekPopup.HorizontalOffset = pos.X - 44;
        _seekPopup.VerticalOffset   = -46;
        _seekPopup.IsOpen = true;
    }

    // ── ASR redo spinner ──────────────────────────────────────────────────────

    private DispatcherTimer? _spinnerTimer;
    private double _spinnerAngle = 0;

    private void StartSpinner(CardInfo card)
    {
        _spinnerAngle = 0;
        _spinnerTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16) }; // ~60fps
        _spinnerTimer.Tick += (_, _) =>
        {
            _spinnerAngle += 6; // 6 degrees per frame
            if (_spinnerAngle >= 360) _spinnerAngle -= 360;
            
            var rt = new RotateTransform(_spinnerAngle);
            var tb = MakeRedoAsrIcon();
            tb.RenderTransform = rt;
            card.RedoAsrBtn.Content = tb;
        };
        _spinnerTimer.Start();
    }

    private void StopSpinner(CardInfo card)
    {
        _spinnerTimer?.Stop();
        _spinnerTimer = null;
        card.RedoAsrBtn.Content = MakeRedoAsrIcon();
    }

    private static TextBlock MakeRedoAsrIcon() => MakeActionIcon("\uE72C"); // Refresh

    /// <summary>Creates a Segoe MDL2 Assets icon TextBlock sized and centered for the action buttons.</summary>
    private static TextBlock MakeActionIcon(string glyph)
    {
        const double iconSize = 14;
        return new TextBlock
        {
            Text       = glyph,
            FontFamily = new FontFamily("Segoe MDL2 Assets"),
            FontSize   = iconSize,
            LineHeight = iconSize,
            Padding    = new Thickness(0),
        };
    }

    // ── ASR redo lock ─────────────────────────────────────────────────────────

    private void SetRedoInProgress(bool inProgress)
    {
        _asrRedoInProgress = inProgress;
        for (int i = 0; i < _cards.Count && i < _vm.Segments.Count; i++)
        {
            var card = _cards[i];
            var seg  = _vm.Segments[i];
            card.MergePrevBtn.IsEnabled   = !inProgress && i > 0;
            card.MergeNextBtn.IsEnabled   = !inProgress && i < _vm.Segments.Count - 1;
            card.SplitBtn.IsEnabled       = !inProgress && seg.Tokens.Count > 1
                                            && _vocab != null && seg.Sources.Count == 1;
            card.AdjustTimesBtn.IsEnabled = !inProgress;
            card.RedoAsrBtn.IsEnabled     = !inProgress && _asrModelsAvailable && _vm.HasAudio;
        }
    }

    private void UpdateFocusedCardBackground(CardInfo card, EditorSegment seg)
    {
        card.FocusedContent.BorderBrush = seg.IsSuppressed
            ? GetResourceBrush("RedBrush")
            : seg.Verified
                ? GetResourceBrush("GreenBrush")
                : GetResourceBrush("AccentBrush");
    }

    private void OnFocusedSegmentPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(EditorSegment.Verified) or nameof(EditorSegment.IsSuppressed))
        {
            var card = FocusedCard();
            var seg  = sender as EditorSegment;
            if (card != null && seg != null)
                UpdateFocusedCardBackground(card, seg);
        }
    }

    private void OnLocalePropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage)) return;

        // Static audio-bar controls
        RefreshPlayModeCombo();
        VerifiedChk.Content = Loc.Instance["editor_verified"];
        VerifiedChk.ToolTip = Loc.Instance["editor_verified_tooltip"];
        HeaderSubtext.Text  = $"{_vm.Segments.Count} {Loc.Instance["results_segments_label"]}";

        // Per-card controls
        for (int i = 0; i < _cards.Count; i++)
        {
            var card = _cards[i];
            var seg  = _vm.Segments[i];
            card.SuppressedBadge.Text        = Loc.Instance["editor_suppressed_badge"];
            card.ReassignBtn.ToolTip         = Loc.Instance["editor_reassign_tooltip"];
            card.SuppressBtn.ToolTip         = seg.IsSuppressed ? Loc.Instance["editor_unsuppress_tooltip"] : Loc.Instance["editor_suppress_tooltip"];
            card.AdjustTimesBtn.ToolTip      = Loc.Instance["editor_adjust_times_tooltip"];
            card.MergePrevBtn.ToolTip        = Loc.Instance["editor_merge_prev_tooltip"];
            card.MergeNextBtn.ToolTip        = Loc.Instance["editor_merge_next_tooltip"];
            card.SplitBtn.ToolTip            = Loc.Instance["editor_split_tooltip"];
            card.RedoAsrBtn.ToolTip          = Loc.Instance["editor_redo_asr_tooltip"];
            card.AsrLabel.Text               = Loc.Instance["editor_asr_label"];
            card.EditLabel.Text              = Loc.Instance["editor_edit_label"];
        }
    }

    private void OnThemeChanged(AppTheme _)
    {
        if (_cards.Count == 0) return;
        int fi = _vm.FocusedIndex;
        for (int i = 0; i < _cards.Count; i++)
        {
            var card = _cards[i];
            var seg  = _vm.Segments[i];
            if (i == fi)
            {
                card.FocusedContent.Background = GetResourceBrush("SurfaceBrush");
                UpdateFocusedCardBackground(card, seg);
                card.EditBox.Background  = GetResourceBrush("SurfaceBrush");
                card.EditBox.Foreground  = GetResourceBrush("TextBrush");
                card.EditBox.BorderBrush = GetResourceBrush("OverlayBrush");
                card.AsrTextBlock.Foreground = GetResourceBrush("TextBrush");
                RebuildAsrRuns(card, seg);
            }
            else
            {
                card.AdjRunsBuilt = false; // force re-render on next show
                UpdateAdjacentCard(card, seg);
            }
        }
    }

    private void UpdateAdjacentCard(CardInfo card, EditorSegment seg)
    {
        if (card.SpeakerLabel != null)
            card.SpeakerLabel.Text = seg.SpeakerDisplayName;
        if (card.AdjTimeLabel != null)
            card.AdjTimeLabel.Text = FormatTimeRange(seg.PlayStart, seg.PlayEnd);
        if (card.SuppressedBadge != null)
            card.SuppressedBadge.Visibility = seg.IsSuppressed ? Visibility.Visible : Visibility.Collapsed;

        if (seg.IsSuppressed)
        {
            card.AdjacentContent.Background = new SolidColorBrush(
                Color.FromArgb(30, 0xF3, 0x8B, 0xA8));
            card.AdjTextBlock.Inlines.Clear();
            card.AdjRuns.Clear();
            card.AdjRunsBuilt               = false;
            card.AdjTextBlock.Text          = seg.Content;
            card.AdjTextBlock.Foreground    = GetResourceBrush("SubtextBrush");
            card.AdjTextBlock.TextDecorations = TextDecorations.Strikethrough;
        }
        else if (seg.Verified)
        {
            card.AdjacentContent.Background = new SolidColorBrush(
                Color.FromArgb(40, 0xA6, 0xE3, 0xA1));
            card.AdjTextBlock.Inlines.Clear();
            card.AdjRuns.Clear();
            card.AdjRunsBuilt                  = false;
            card.AdjTextBlock.Text             = seg.Content;
            card.AdjTextBlock.Foreground       = GetResourceBrush("TextBrush");
            card.AdjTextBlock.TextDecorations  = null;
        }
        else if (seg.Content != seg.AsrContent)
        {
            card.AdjacentContent.Background   = Brushes.Transparent;
            card.AdjTextBlock.Inlines.Clear();
            card.AdjRuns.Clear();
            card.AdjRunsBuilt                 = false;
            card.AdjTextBlock.Text            = seg.Content;
            card.AdjTextBlock.Foreground      = GetResourceBrush("TextBrush");
            card.AdjTextBlock.TextDecorations = null;
        }
        else
        {
            card.AdjacentContent.Background    = Brushes.Transparent;
            card.AdjTextBlock.Foreground       = GetResourceBrush("SubtextBrush");
            card.AdjTextBlock.TextDecorations  = null;
            if (!card.AdjRunsBuilt)
            {
                RebuildAdjacentRuns(card, seg);
                card.AdjRunsBuilt = true;
            }
        }
    }

    private void RebuildAdjacentRuns(CardInfo card, EditorSegment seg)
    {
        card.AdjTextBlock.Inlines.Clear();
        card.AdjRuns.Clear();

        if (_vocab == null || seg.Tokens.Count == 0)
        {
            card.AdjTextBlock.Text = seg.Content;
            return;
        }

        var highlightColor = GetResourceBrush("ConfidenceLowBrush").Color;
        var runs           = _vocab.GetTokenRuns(seg.Tokens, seg.Logprobs);

        // TODO: Avalonia doesn't support inline backgrounds like WPF Run elements.
        // For now, just display the plain text.
        // To restore confidence highlighting, consider:
        // 1. Using a custom TextBlock with inline rendering
        // 2. Using AvaloniaEdit with custom highlighting
        // 3. Building a custom control with colored spans
        card.AdjTextBlock.Text = seg.Content;
    }

    // ── Confidence-colored ASR text ───────────────────────────────────────────

    private void RebuildAsrRuns(CardInfo card, EditorSegment seg)
    {
        card.AsrTextBlock.Inlines.Clear();
        card.AsrRuns.Clear();

        if (_vocab == null || seg.Tokens.Count == 0)
        {
            card.AsrTextBlock.Text = seg.AsrContent;
            return;
        }

        // TODO: Avalonia doesn't support inline backgrounds like WPF Run elements.
        // For now, just display the plain text.
        // To restore confidence highlighting, consider:
        // 1. Using a custom TextBlock with inline rendering
        // 2. Using AvaloniaEdit with custom highlighting
        // 3. Building a custom control with colored spans
        card.AsrTextBlock.Text = seg.AsrContent;
    }

    // ── Token highlight during playback ──────────────────────────────────────

    private void UpdateTokenHighlight(int tokenIndex)
    {
        // TODO: Token highlighting requires inline text rendering with backgrounds,
        // which Avalonia doesn't support natively like WPF.
        // This feature is temporarily disabled.
        // To restore, consider:
        // 1. Using AvaloniaEdit with custom highlighting
        // 2. Building a custom TextBlock with inline rendering
        // 3. Using a WebView to render HTML with highlighting
    }

    // ── Diff highlighting in edit box ─────────────────────────────────────────

    private void ScheduleDiffUpdate(CardInfo card, int index)
    {
        if (_suppressDiff) return;
        _diffTimer?.Stop();
        _diffTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _diffTimer.Tick += (_, _) =>
        {
            _diffTimer?.Stop();
            RebuildDiffHighlight(card, index);
        };
        _diffTimer.Start();
    }

    private void RebuildDiffHighlight(CardInfo card, int index)
    {
        if (index < 0 || index >= _vm.Segments.Count) return;
        // ⚠️ TODO: Diff highlighting requires RichTextBox which doesn't exist in Avalonia.
        // To restore this feature, consider:
        // 1. Using AvaloniaEdit for syntax highlighting-like diff display
        // 2. Building custom control using TextBlock.Inlines with colored spans
        // 3. Using a WebView to render HTML with diff highlighting
    }

    private static void ApplyWordDiffHighlight(TextBox tb, string original, string edited)
    {
        // ⚠️ TODO: This method requires RichTextBox and WPF FlowDocument APIs
        // which don't exist in Avalonia. Placeholder implementation.
    }

    private static string GetRichTextBoxText(TextBox tb)
    {
        return tb.Text;
    }

    // ── Simple word diff helpers (commented out - not used without RichTextBox) ──

    // These methods are kept for future implementation if diff highlighting is restored
    // using a different approach (AvaloniaEdit or custom control)

    /*
    private static List<(string word, int start, int length)> TokenizeWords(string text)
    {
        var result = new List<(string, int, int)>();
        int i = 0;
        while (i < text.Length)
        {
            while (i < text.Length && text[i] == ' ') i++;
            if (i >= text.Length) break;
            int start = i;
            while (i < text.Length && text[i] != ' ') i++;
            result.Add((text[start..i], start, i - start));
        }
        return result;
    }

    private static List<string> ComputeLcs(
        List<(string word, int start, int length)> a,
        List<(string word, int start, int length)> b)
    {
        int m = a.Count, n = b.Count;
        var dp = new int[m + 1, n + 1];
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= n; j++)
                dp[i, j] = a[i - 1].word == b[j - 1].word
                    ? dp[i - 1, j - 1] + 1
                    : Math.Max(dp[i - 1, j], dp[i, j - 1]);

        var lcs = new List<string>();
        int ii = m, jj = n;
        while (ii > 0 && jj > 0)
        {
            if (a[ii - 1].word == b[jj - 1].word)
            { lcs.Insert(0, a[ii - 1].word); ii--; jj--; }
            else if (dp[ii - 1, jj] > dp[ii, jj - 1]) ii--;
            else jj--;
        }
        return lcs;
    }

    private static List<(int startChar, int length)> FindChangedRanges(
        List<(string word, int start, int length)> editedWords,
        List<string> lcs)
    {
        var changed = new List<(int, int)>();
        int lcsIdx = 0;
        foreach (var (word, start, len) in editedWords)
        {
            if (lcsIdx < lcs.Count && lcs[lcsIdx] == word)
                lcsIdx++;
            else
                changed.Add((start, len));
        }
        return changed;
    }
    */

    // ── Scroll animation ──────────────────────────────────────────────────────

    private DispatcherTimer? _scrollTimer;
    private double _scrollStart, _scrollTarget, _scrollProgress;

    private void AnimateScrollToCenter(int targetIndex)
    {
        if (targetIndex < 0 || targetIndex >= _cards.Count) return;

        Dispatcher.UIThread.InvokeAsync(() =>
        {
            double offset = CalculateCenterOffset(targetIndex);
            AnimateScrollOffset(Scroller, offset, TimeSpan.FromMilliseconds(280));
            return Task.CompletedTask;
        }, DispatcherPriority.Loaded);
    }

    private double CalculateCenterOffset(int targetIndex)
    {
        double accum = 0;
        for (int i = 0; i < targetIndex && i < _cards.Count; i++)
            accum += _cards[i].Root.Bounds.Height + _cards[i].Root.Margin.Top + _cards[i].Root.Margin.Bottom;

        double cardHeight = _cards[targetIndex].Root.Bounds.Height;
        double viewHeight = Scroller.Bounds.Height;
        return Math.Max(0, accum - (viewHeight - cardHeight) / 2.0);
    }

    private void AnimateScrollOffset(ScrollViewer scroller, double target, TimeSpan duration)
    {
        // TODO: Implement smooth scroll animation using Avalonia animation API
        // For now, use direct scroll without animation
        scroller.ScrollTo(0, target);
    }

    // ── TextBox helpers (replacing RichTextBox) ─────────────────────────────────
    //
    // ⚠️ NOTE: These methods are placeholders. The original RichTextBox-based methods
    // are not compatible with Avalonia. The diff highlighting features that relied on
    // FlowDocument, Paragraph, Run, and TextRange have been disabled.
    //
    // To restore full functionality, consider:
    // 1. Using AvaloniaEdit for syntax highlighting-like diff display
    // 2. Building custom control using TextBlock.Inlines with TextSpan elements
    // 3. Using a WebView to render HTML with diff highlighting
    //

    private static string GetTextBoxText(TextBox tb)
    {
        return tb.Text;
    }

    private static void SetTextBoxText(TextBox tb, string text)
    {
        tb.Text = text;
    }

    private static string FormatTime(double s)
    {
        int    h   = (int)(s / 3600);
        int    m   = (int)(s % 3600 / 60);
        double sec = s % 60;
        return h > 0 ? $"{h}:{m:D2}:{sec:00.0}" : $"{m:D2}:{sec:00.0}";
    }

    private static string FormatTimeRange(double start, double end)
        => $"{FormatTime(start)} – {FormatTime(end)}";
}
