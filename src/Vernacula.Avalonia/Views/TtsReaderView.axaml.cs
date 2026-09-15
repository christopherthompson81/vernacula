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

    /// <summary>The view model whose cards are currently hooked, so the hook can be moved cleanly.</summary>
    private ViewModels.TtsReaderViewModel? _hooked;

    /// <summary>
    /// ⚠ REBUILDING THE CARDS SCROLLS THE DOCUMENT BACK TO THE TOP, and that is now a thing that
    /// happens while the user is working rather than only on load: committing a marker edit rebuilds
    /// the list so the card can be redrawn as its new kind. Editing a heading two thirds of the way
    /// down a document and being thrown back to the title is its own bug. Clearing the collection
    /// collapses the extent and clamps the offset to zero, so the offset is saved on the Reset and
    /// put back once the new items have been laid out.
    /// </summary>
    protected override void OnDataContextChanged(EventArgs e)
    {
        base.OnDataContextChanged(e);
        if (_hooked is not null) _hooked.DisplayBlocks.CollectionChanged -= Blocks_CollectionChanged;
        _hooked = DataContext as ViewModels.TtsReaderViewModel;
        if (_hooked is not null) _hooked.DisplayBlocks.CollectionChanged += Blocks_CollectionChanged;
    }

    private void Blocks_CollectionChanged(object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
    {
        if (e.Action != System.Collections.Specialized.NotifyCollectionChangedAction.Reset) return;
        double y = CardScroller.Offset.Y;
        if (y <= 0) return;
        Avalonia.Threading.Dispatcher.UIThread.Post(() =>
        {
            // Clamped: the document may now be shorter than it was.
            double max = Math.Max(0, CardScroller.Extent.Height - CardScroller.Viewport.Height);
            CardScroller.Offset = new Avalonia.Vector(CardScroller.Offset.X, Math.Min(y, max));
        }, Avalonia.Threading.DispatcherPriority.Loaded);
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
        if (e.Source is not Avalonia.Visual v) { vm.CommitOpenCard(); return; }
        if (v.FindAncestorOfType<TextBox>(includeSelf: true) is not null) return;
        // ⚠ A PRESS ON ANOTHER CARD IS LEFT TO THAT CARD'S OWN COMMAND, which commits the open one
        // before it opens itself. Committing here instead can rebuild the cards — that is what a
        // marker edit does — and the button this press is travelling to is then detached before it
        // ever raises Click, so the card the user aimed at silently fails to open and they have to
        // click again. RequestEdit re-resolves its block by index for exactly this reason.
        if (v.FindAncestorOfType<Button>(includeSelf: true) is { } button &&
            button.Classes.Contains("card-overlay")) return;
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
