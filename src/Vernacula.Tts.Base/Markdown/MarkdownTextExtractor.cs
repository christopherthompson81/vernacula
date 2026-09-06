using System.Text;
using Markdig;
using Markdig.Parsers;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;

namespace Vernacula.Tts.Base.Markdown;

/// <summary>
/// One contiguous span of extracted output text and the source range it
/// came from. The Source* fields are character offsets into the original
/// markdown string (the same units the file was read in).
///
/// <para><see cref="OutputLength"/> and <see cref="SourceLength"/> may differ:
/// inline code's source range covers the backticks (<c>`foo()`</c>, length 7)
/// while the output omits them (<c>foo()</c>, length 5). Consumers should
/// not assume the two slices are equal-length, only that the output
/// substring is a *subsequence* of the source substring.</para>
///
/// <para>Synthetic whitespace (paragraph separators, sentence-terminator periods
/// the extractor appends to headings) does NOT get an entry — those bytes
/// exist in <see cref="MarkdownExtractionResult.Text"/> but map to nothing
/// in the source. Word-highlighting downstream walks the ranges in order
/// and accepts that some output offsets fall in the gaps.</para>
/// </summary>
/// <summary>Inline formatting carried on a <see cref="TextRange"/> for structured display.</summary>
[Flags]
public enum InlineStyle
{
    None = 0,
    Bold = 1,
    Italic = 2,
    Code = 4,
    Link = 8,
}

/// <summary>Block kind of a <see cref="BlockSpan"/>.</summary>
public enum BlockKind
{
    Paragraph,
    Heading,
    ListItem,
    Quote,
}

public sealed record TextRange(int OutputStart, int OutputLength, int SourceStart, int SourceLength,
    InlineStyle Style = InlineStyle.None);

/// <summary>
/// One text-bearing block's span in the output text, for structured rendering.
/// <paramref name="Level"/> is the heading level for <see cref="BlockKind.Heading"/>, else 0.
/// Spans are in output order and non-overlapping; block separators fall in the gaps.
/// </summary>
public sealed record BlockSpan(BlockKind Kind, int Level, int OutputStart, int OutputLength);

/// <summary>
/// Output of <see cref="MarkdownTextExtractor.Extract"/>: the speakable
/// text and a parallel index of source positions. Index is sorted by
/// OutputStart and non-overlapping; gaps correspond to synthetic
/// whitespace the extractor inserted.
/// </summary>
public sealed record MarkdownExtractionResult(string Text, IReadOnlyList<TextRange> Ranges,
    IReadOnlyList<BlockSpan> Blocks);

/// <summary>
/// Markdown → speakable plain text, preserving a source-range index for
/// downstream forced alignment (Stage 1 step 9). Walks Markdig's AST and
/// emits text-bearing inlines while dropping markup. The defaults match
/// the user-confirmed decisions captured in PR <see href="https://github.com/christopherthompson81/vernacula/pull/68"/>:
///
/// <list type="bullet">
/// <item>Headings — emit text, append a period if it doesn't already end
///       in terminal punctuation (so the TTS pauses correctly).</item>
/// <item>Paragraphs — text with a double newline between (TTS interprets
///       as a longer pause).</item>
/// <item>List items — one per line, no enumeration prefix.</item>
/// <item>Block quotes — text only, the <c>&gt;</c> markup is dropped.</item>
/// <item>Inline code / bold / italic / strike — markup stripped, content kept.</item>
/// <item>Links — emit the link text, drop the URL.</item>
/// <item>Inline HTML — tags (<c>&lt;span&gt;</c>, <c>&lt;/span&gt;</c>) dropped,
///       but text nodes between tags survive as ordinary literals (Markdig
///       parses them as <c>LiteralInline</c> not <c>HtmlInline</c>). Block
///       HTML is filtered at block level.</item>
/// <item>Fenced code blocks, tables, images, horizontal rules,
///       footnotes — <b>skipped entirely.</b></item>
/// </list>
///
/// Pure-CPU, no ORT dependency. Lives in Vernacula.Tts.Base so both the CLI
/// and the eventual Avalonia app share it. Not thread-safe per-instance
/// because the extractor object holds a builder + ranges list; the
/// static <see cref="Extract(string)"/> facade creates a fresh instance
/// per call so concurrent callers are fine.
/// </summary>
public sealed class MarkdownTextExtractor
{
    private readonly StringBuilder _sb = new();
    private readonly List<TextRange> _ranges = new();
    private readonly List<BlockSpan> _blocks = new();
    // Current inline style while walking inlines, and the enclosing block context
    // (a paragraph inside a list item / quote takes that kind, not Paragraph).
    private InlineStyle _style = InlineStyle.None;
    private BlockKind _contextKind = BlockKind.Paragraph;

    // Pipeline is shared across extractions — Markdig pipelines are
    // thread-safe to call concurrently per their docs. Built with the
    // "advanced" extension set so tables/footnotes parse into typed
    // blocks (and then get skipped at walk time) instead of leaking
    // their syntax through as literal text.
    private static readonly MarkdownPipeline Pipeline =
        new MarkdownPipelineBuilder().UseAdvancedExtensions().Build();

    private MarkdownTextExtractor() { }

    /// <summary>Extract speakable text + source-range index from markdown.</summary>
    public static MarkdownExtractionResult Extract(string markdown)
    {
        var extractor = new MarkdownTextExtractor();
        return extractor.Run(markdown);
    }

    // The markdown being extracted, for bounds-checking recorded source spans.
    private string _source = string.Empty;

    // Document-coordinate start/length for an inline's source span, clamped to
    // the document. The span may legitimately be longer than the emitted text
    // (a backslash escape emits one character for two source ones), which the
    // TextRange contract allows: output is a subsequence of the source slice.
    private int SourceStartOf(SourceSpan span, int outLen, out int srcLen)
    {
        int start = span.Start;
        srcLen = span.End - span.Start + 1;
        if (start < 0 || start > _source.Length)
        {
            // Defensive: an inline with no usable position. Record an empty
            // span at a valid offset rather than one a consumer would throw
            // on when it slices the source.
            srcLen = 0;
            return Math.Clamp(start, 0, _source.Length);
        }
        if (srcLen < outLen) srcLen = outLen;
        if (srcLen > _source.Length - start) srcLen = _source.Length - start;
        return start;
    }

    // Drop or shorten range entries that extend past the final output text.
    // Ranges are appended in output order, so only the tail can be affected.
    private void TrimRangesToOutput(int outputLength)
    {
        for (int i = _ranges.Count - 1; i >= 0; i--)
        {
            var r = _ranges[i];
            if (r.OutputStart + r.OutputLength <= outputLength) break;
            if (r.OutputStart >= outputLength)
            {
                _ranges.RemoveAt(i);
                continue;
            }
            _ranges[i] = r with { OutputLength = outputLength - r.OutputStart };
        }
    }

    private MarkdownExtractionResult Run(string markdown)
    {
        _source = markdown;
        var doc = MarkdownParser.Parse(markdown, Pipeline);
        bool first = true;
        foreach (var block in doc)
        {
            if (TryEmitBlock(block, ref first))
                first = false;
        }
        // Trim trailing whitespace the block-separator logic may have appended.
        while (_sb.Length > 0 && char.IsWhiteSpace(_sb[^1])) _sb.Length--;
        // The trim can cut into the last range, and dropped trailing markup
        // (a README ending in a badge image, a stray inline tag) leaves one
        // whose output span already ran past the text: the literal "hello "
        // in "hello ![img](/x.png)" records 6 characters, but Text is
        // "hello". Clamp so every entry is sliceable against Text — a
        // consumer walking the index has no other way to know.
        TrimRangesToOutput(_sb.Length);
        return new MarkdownExtractionResult(_sb.ToString(), _ranges, _blocks);
    }

    // Returns true if the block actually produced any output (so callers
    // can suppress the leading block separator for the first non-empty block).
    private bool TryEmitBlock(Block block, ref bool first)
    {
        switch (block)
        {
            case HeadingBlock heading:
                EmitBlockSeparator(first);
                int beforeHeading = _sb.Length;
                EmitInlines(heading.Inline);
                AppendTerminalPeriod(beforeHeading);
                if (_sb.Length > beforeHeading)
                    _blocks.Add(new BlockSpan(BlockKind.Heading, heading.Level, beforeHeading, _sb.Length - beforeHeading));
                return _sb.Length > beforeHeading;

            case ParagraphBlock para:
                EmitBlockSeparator(first);
                int beforePara = _sb.Length;
                EmitInlines(para.Inline);
                if (_sb.Length > beforePara)
                    _blocks.Add(new BlockSpan(_contextKind, 0, beforePara, _sb.Length - beforePara));
                return _sb.Length > beforePara;

            case QuoteBlock quote:
                // Block quote: walk children with the same separator rules; child
                // paragraphs are tagged Quote via the context kind.
                var savedQuoteCtx = _contextKind;
                _contextKind = BlockKind.Quote;
                bool emittedAny = false;
                bool innerFirst = first;
                foreach (var child in quote)
                {
                    if (TryEmitBlock(child, ref innerFirst))
                    {
                        emittedAny = true;
                        innerFirst = false;
                    }
                }
                _contextKind = savedQuoteCtx;
                if (emittedAny) first = false;
                return emittedAny;

            case ListBlock list:
                EmitBlockSeparator(first);
                int beforeList = _sb.Length;
                var savedListCtx = _contextKind;
                _contextKind = BlockKind.ListItem;
                foreach (var item in list)
                {
                    if (item is not ListItemBlock listItem) continue;
                    int beforeItem = _sb.Length;
                    // Each item: walk its children blocks (typically a single
                    // ParagraphBlock, tagged ListItem). Use a newline-only separator
                    // between items so the TTS gets a short pause, not the longer
                    // paragraph break.
                    bool itemFirst = true;
                    foreach (var child in listItem)
                    {
                        if (TryEmitBlock(child, ref itemFirst))
                            itemFirst = false;
                    }
                    // Newline between items (after the last item we strip
                    // trailing whitespace at the document level).
                    if (_sb.Length > beforeItem) _sb.Append('\n');
                }
                _contextKind = savedListCtx;
                return _sb.Length > beforeList;

            // Skipped block kinds: fenced code, tables, HTML, thematic break,
            // footnote group, link reference definitions, and anything else
            // not enumerated above. The "return false" tells the caller this
            // block didn't move the cursor.
            default:
                return false;
        }
    }

    private void EmitBlockSeparator(bool first)
    {
        if (first) return;
        // Paragraph-style separator: blank line. The TTS doesn't actually
        // pause on whitespace, but downstream chunking (Stage 1 step 8)
        // uses blank lines as natural chunk boundaries.
        _sb.Append("\n\n");
    }

    private void AppendTerminalPeriod(int blockStart)
    {
        if (_sb.Length == blockStart) return;
        char last = _sb[^1];
        if (last == '.' || last == '!' || last == '?' || last == ':' || last == ';') return;
        _sb.Append('.');
    }

    private void EmitInlines(ContainerInline? container)
    {
        if (container is null) return;
        foreach (var inline in container)
            EmitInline(inline);
    }

    private void EmitInline(Inline inline)
    {
        switch (inline)
        {
            case LiteralInline lit:
                // Literal text: copy the slice verbatim AND record a range
                // entry pointing back to the source span.
                //
                // The source position comes from lit.Span, NOT from the
                // slice. LiteralInline.Content is a slice over whatever
                // buffer Markdig assembled the inline from, and for a
                // continuation line of a container — the second line of a
                // block quote, a lazily continued list item — that buffer is
                // the container's re-joined content, not the original
                // document. slice.Start is an offset into that buffer and
                // means nothing here; lit.Span is in document coordinates for
                // every shape observed (quotes, lazy list continuations,
                // nested quotes, escapes, entities, CRLF, multi-line
                // emphasis, alerts, tables, footnotes, BOM, CR-only). Not
                // unconditionally, though — a synthesized inline can carry a
                // degenerate span, which is why SourceStartOf bounds-checks
                // rather than trusting it.
                var slice = lit.Content;
                int outLen = slice.End - slice.Start + 1;
                if (outLen <= 0) return;
                int outStart = _sb.Length;
                _sb.Append(slice.AsSpan());
                int srcStart = SourceStartOf(lit.Span, outLen, out int srcLen);
                _ranges.Add(new TextRange(outStart, outLen, srcStart, srcLen, _style));
                break;

            case CodeInline code:
                // `inline code` → emit the text, drop the backticks.
                if (string.IsNullOrEmpty(code.Content)) return;
                int codeOutStart = _sb.Length;
                _sb.Append(code.Content);
                // Source span includes the backticks; the inner content
                // starts after the leading backticks. Markdig's Span
                // points at the full delimited extent, so record the
                // full delim range — close-enough for forced alignment
                // which works at word granularity.
                int codeSrcStart = SourceStartOf(code.Span, code.Content.Length, out int codeSrcLen);
                _ranges.Add(new TextRange(codeOutStart, code.Content.Length,
                    codeSrcStart, codeSrcLen, _style | InlineStyle.Code));
                break;

            case LinkInline link when !link.IsImage:
                // [text](url) → emit just the text, drop the URL.
                var savedLinkStyle = _style;
                _style |= InlineStyle.Link;
                EmitInlines(link);
                _style = savedLinkStyle;
                break;

            // Image is a LinkInline with IsImage=true; skip entirely (no alt
            // text in the speech stream).
            case LinkInline { IsImage: true }:
                break;

            case EmphasisInline emph:
                // *italic* / **bold** / ~~strike~~ → markup stripped, walk children;
                // track bold/italic for display ('*'/'_' delimiters; '~' strike → no style).
                var add = emph.DelimiterChar is '*' or '_'
                    ? (emph.DelimiterCount >= 2 ? InlineStyle.Bold : InlineStyle.Italic)
                    : InlineStyle.None;
                var savedEmphStyle = _style;
                _style |= add;
                EmitInlines(emph);
                _style = savedEmphStyle;
                break;

            case LineBreakInline:
                // Both hard and soft line breaks → space. TTS doesn't
                // distinguish; both are intra-paragraph whitespace.
                // Don't emit if the previous char is already whitespace.
                if (_sb.Length > 0 && !char.IsWhiteSpace(_sb[^1]))
                    _sb.Append(' ');
                break;

            case HtmlInline:
                // Inline HTML — skip. Block HTML is filtered out at block level.
                break;

            case ContainerInline container:
                // Any other container (e.g. AutolinkInline with children) — walk.
                EmitInlines(container);
                break;

            // Unhandled inlines (TaskList, Math, etc. from advanced extensions):
            // skip silently. Logging them would be noisy on real documents.
            default:
                break;
        }
    }
}
