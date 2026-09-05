using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.Tts.App.ViewModels;

/// <summary>
/// One markdown block (heading / paragraph / list item / quote) in the structured karaoke
/// view, holding the word view models it contains. The AXAML switches layout on the
/// <see cref="IsHeading"/>/<see cref="IsListItem"/>/<see cref="IsQuote"/> helpers (bullet,
/// indent, spacing); per-word font/size/style live on <see cref="WordItemViewModel"/>.
/// </summary>
public sealed partial class BlockItemViewModel : ObservableObject
{
    public BlockItemViewModel(BlockKind kind, int level)
    {
        Kind = kind;
        Level = level;
    }

    public BlockKind Kind { get; }
    public int Level { get; }
    /// <summary>The block's words in LOGICAL order — the order they are spoken, which is what
    /// alignment attaches timing to.</summary>
    public ObservableCollection<WordItemViewModel> Words { get; } = new();

    /// <summary>
    /// The same words in the order the panel must place them to read correctly. For a block with
    /// one direction throughout this is just <see cref="Words"/>; where the two directions mix, the
    /// embedded run is reversed so that the panel's mirroring puts it back the right way round.
    /// </summary>
    public ObservableCollection<WordItemViewModel> Display { get; } = new();

    public bool IsHeading => Kind == BlockKind.Heading;
    public bool IsParagraph => Kind == BlockKind.Paragraph;
    public bool IsListItem => Kind == BlockKind.ListItem;
    public bool IsQuote => Kind == BlockKind.Quote;

    /// <summary>
    /// Which way this block's words are laid out. A browser reorders inline elements by the
    /// bidirectional algorithm; a layout panel does not, so an Arabic or Persian line would run
    /// left to right with its first word on the left -- backwards. Set once the block's words are
    /// in, from the text itself rather than the picked language, so a quoted RTL passage inside an
    /// English document is still laid out correctly.
    /// </summary>
    [ObservableProperty] private FlowDirection _flowDirection = FlowDirection.LeftToRight;

    /// <summary>
    /// Decide the block's direction from the words it now holds, and lay them out for it.
    ///
    /// ⚠ A PANEL MIRRORS EVERY CHILD, WHICH IS NOT WHAT BIDIRECTIONAL TEXT DOES. Mirroring an
    /// Arabic line is right; mirroring an English phrase embedded in it is not — "Text To Speech"
    /// inside a Persian sentence would read "Speech To Text". The bidirectional algorithm reverses
    /// runs, not words, so each embedded run is reversed here and the panel's mirroring undoes it.
    /// Words with no strong direction of their own (digits, punctuation) stay with the run they
    /// are in, exactly as they take their direction from their surroundings in text.
    /// </summary>
    /// <param name="languageIsRtl">Whether the language being read is right-to-left, which is what
    /// settles a block that contains both directions.</param>
    public void UpdateFlowDirection(bool? languageIsRtl = null)
    {
        var rtl = TextDirection.Resolve(string.Join(' ', Words.Select(w => w.Text)), languageIsRtl);
        FlowDirection = rtl ? FlowDirection.RightToLeft : FlowDirection.LeftToRight;

        Display.Clear();
        var run = new List<WordItemViewModel>();      // an embedded run, awaiting its reversal
        var pending = new List<WordItemViewModel>();  // neutrals whose side is not settled yet
        foreach (var w in Words)
        {
            var strong = TextDirection.StrongDirectionOf(w.Text);
            if (strong is null)
            {
                if (run.Count == 0) Display.Add(w);
                else if (TextDirection.IsNumberWord(w.Text))
                {
                    // A number keeps company with the word before it -- "iPhone 15" stays "iPhone
                    // 15" -- so it joins the run rather than waiting to see what follows.
                    run.AddRange(pending);
                    pending.Clear();
                    run.Add(w);
                }
                // Any other neutral takes its direction from what surrounds it, so it cannot be
                // placed until the next strong word says which side of the boundary it fell on.
                else pending.Add(w);
            }
            else if (strong == rtl)
            {
                // Back to the block's own direction: the run ends, and any neutrals waiting inside
                // it were trailing it -- a full stop after an English phrase still ends the
                // Persian line -- so they belong to the block.
                FlushRun();
                foreach (var n in pending) Display.Add(n);
                pending.Clear();
                Display.Add(w);
            }
            else
            {
                // The run continues, and it swallows the neutrals in the middle of it: "Text &
                // Speech" is one embedded phrase, not two with an ampersand between them.
                run.AddRange(pending);
                pending.Clear();
                run.Add(w);
            }
        }
        FlushRun();
        foreach (var n in pending) Display.Add(n);
        return;

        // Reversed, so that the panel mirroring the block puts the run back the right way round.
        void FlushRun()
        {
            for (var i = run.Count - 1; i >= 0; i--) Display.Add(run[i]);
            run.Clear();
        }
    }
}
