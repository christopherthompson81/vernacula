using System.ComponentModel;
using Avalonia.Controls;
using Avalonia.VisualTree;

namespace Vernacula.App.Views;

public partial class TtsReaderView : UserControl
{
    public TtsReaderView()
    {
        InitializeComponent();
        ApplyLocalizedText();
        // Tunnelling, so the press is seen before whatever it landed on handles it — a word button
        // in Listening, or a card overlay about to open a different card, both mark it handled.
        AddHandler(Avalonia.Input.InputElement.PointerPressedEvent, Root_PointerPressed,
                   Avalonia.Interactivity.RoutingStrategies.Tunnel);
        Loaded += (_, _) => Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
        Unloaded += (_, _) => Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
    }

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;
        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        RawMarkdownCheck.Content    = Loc.Instance["tts_raw_markdown"];
        ExportButton.Content        = Loc.Instance["btn_export_tts"];
        IpaAnnotationCheck.Content  = Loc.Instance["tts_ipa_annotation"];
        CancelJobButton.Content     = Loc.Instance["btn_cancel_job"];
        BackButton.Content          = Loc.Instance["tts_reader_back"];
    }

    /// <summary>
    /// A press anywhere that is not inside a text box closes the open card.
    ///
    /// ⚠ THIS IS NOT REDUNDANT WITH <see cref="BlockEditor_LostFocus"/>. Most of this view is inert —
    /// the card's caption, the space between cards, the scroll area — and clicking an inert surface
    /// moves focus nowhere at all, so the editor keeps focus, never raises LostFocus, and the card
    /// looks stuck open. The test is "is the press inside SOME text box", not "inside THIS one", so
    /// clicking from a card into the raw-markdown editor does not fight over the caret.
    /// </summary>
    private void Root_PointerPressed(object? sender, Avalonia.Input.PointerPressedEventArgs e)
    {
        if (DataContext is not ViewModels.TtsReaderViewModel vm) return;
        if (e.Source is Avalonia.Visual v &&
            v.FindAncestorOfType<TextBox>(includeSelf: true) is not null) return;
        vm.CommitOpenCard();
    }

    /// <summary>Blur commits the card — the ordinary way to finish editing one and move on.</summary>
    private void BlockEditor_LostFocus(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if ((sender as Avalonia.Controls.Control)?.DataContext is ViewModels.BlockItemViewModel b)
            b.CommitEditCommand.Execute(null);
    }

    /// <summary>Escape abandons the card's edit; Ctrl+Enter commits without reaching for the mouse.
    /// ⚠ Plain Enter must NOT commit — a paragraph can contain line breaks.</summary>
    private void BlockEditor_KeyDown(object? sender, Avalonia.Input.KeyEventArgs e)
    {
        if ((sender as Avalonia.Controls.Control)?.DataContext is not ViewModels.BlockItemViewModel b) return;
        if (e.Key == Avalonia.Input.Key.Escape)
        {
            b.CancelEditCommand.Execute(null);
            e.Handled = true;
        }
        else if (e.Key == Avalonia.Input.Key.Enter && e.KeyModifiers.HasFlag(Avalonia.Input.KeyModifiers.Control))
        {
            b.CommitEditCommand.Execute(null);
            e.Handled = true;
        }
    }

    /// <summary>
    /// ⚠ OPENING A CARD MUST PUT THE CARET IN IT. Without this the card turns into a text box and
    /// then waits for a SECOND click before it will accept a keystroke, which reads as the editor
    /// being broken. The box is always in the tree and only toggles IsVisible, so there is no Loaded
    /// event to hang this on — it watches its own visibility instead.
    /// </summary>
    private void BlockEditor_Initialized(object? sender, System.EventArgs e)
    {
        if (sender is not Avalonia.Controls.TextBox box) return;
        box.PropertyChanged += (_, args) =>
        {
            if (args.Property != Avalonia.Visual.IsVisibleProperty || !box.IsVisible) return;
            // Posted: the box is not yet laid out at the moment visibility flips.
            Avalonia.Threading.Dispatcher.UIThread.Post(() =>
            {
                box.Focus();
                box.CaretIndex = box.Text?.Length ?? 0;
            });
        };
    }
}
