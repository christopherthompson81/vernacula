using System.Collections.ObjectModel;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.Tts.App.ViewModels;

/// <summary>
/// One markdown block (heading / paragraph / list item / quote) in the structured karaoke
/// view, holding the word view models it contains. The AXAML switches layout on the
/// <see cref="IsHeading"/>/<see cref="IsListItem"/>/<see cref="IsQuote"/> helpers (bullet,
/// indent, spacing); per-word font/size/style live on <see cref="WordItemViewModel"/>.
/// </summary>
public sealed class BlockItemViewModel
{
    public BlockItemViewModel(BlockKind kind, int level)
    {
        Kind = kind;
        Level = level;
    }

    public BlockKind Kind { get; }
    public int Level { get; }
    public ObservableCollection<WordItemViewModel> Words { get; } = new();

    public bool IsHeading => Kind == BlockKind.Heading;
    public bool IsParagraph => Kind == BlockKind.Paragraph;
    public bool IsListItem => Kind == BlockKind.ListItem;
    public bool IsQuote => Kind == BlockKind.Quote;
}
