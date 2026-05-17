using System.Text;
using Markdig;
using Markdig.Parsers;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;

namespace Chatterbox.Base.Markdown;

/// <summary>
/// One contiguous span of extracted output text and the source range it
/// came from. The Source* fields are character offsets into the original
/// markdown string (the same units the file was read in).
///
/// Synthetic whitespace (paragraph separators, sentence-terminator periods
/// the extractor appends to headings) does NOT get an entry — those bytes
/// exist in <see cref="MarkdownExtractionResult.Text"/> but map to nothing
/// in the source. Word-highlighting downstream walks the ranges in order
/// and accepts that some output offsets fall in the gaps.
/// </summary>
public sealed record TextRange(int OutputStart, int OutputLength, int SourceStart, int SourceLength);

/// <summary>
/// Output of <see cref="MarkdownTextExtractor.Extract"/>: the speakable
/// text and a parallel index of source positions. Index is sorted by
/// OutputStart and non-overlapping; gaps correspond to synthetic
/// whitespace the extractor inserted.
/// </summary>
public sealed record MarkdownExtractionResult(string Text, IReadOnlyList<TextRange> Ranges);

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
/// <item>Fenced code blocks, tables, images, HTML, horizontal rules,
///       footnotes — <b>skipped entirely.</b></item>
/// </list>
///
/// Pure-CPU, no ORT dependency. Lives in Chatterbox.Base so both the CLI
/// and the eventual Avalonia app share it. Not thread-safe per-instance
/// because the extractor object holds a builder + ranges list; the
/// static <see cref="Extract(string)"/> facade creates a fresh instance
/// per call so concurrent callers are fine.
/// </summary>
public sealed class MarkdownTextExtractor
{
    private readonly StringBuilder _sb = new();
    private readonly List<TextRange> _ranges = new();

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

    private MarkdownExtractionResult Run(string markdown)
    {
        var doc = MarkdownParser.Parse(markdown, Pipeline);
        bool first = true;
        foreach (var block in doc)
        {
            if (TryEmitBlock(block, ref first))
                first = false;
        }
        // Trim trailing whitespace the block-separator logic may have appended.
        while (_sb.Length > 0 && char.IsWhiteSpace(_sb[^1])) _sb.Length--;
        return new MarkdownExtractionResult(_sb.ToString(), _ranges);
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
                return _sb.Length > beforeHeading;

            case ParagraphBlock para:
                EmitBlockSeparator(first);
                int beforePara = _sb.Length;
                EmitInlines(para.Inline);
                return _sb.Length > beforePara;

            case QuoteBlock quote:
                // Block quote: walk children with the same separator rules.
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
                if (emittedAny) first = false;
                return emittedAny;

            case ListBlock list:
                EmitBlockSeparator(first);
                int beforeList = _sb.Length;
                foreach (var item in list)
                {
                    if (item is not ListItemBlock listItem) continue;
                    int beforeItem = _sb.Length;
                    // Each item: walk its children blocks (typically a single
                    // ParagraphBlock). Use a newline-only separator between
                    // items so the TTS gets a short pause, not the longer
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
                var slice = lit.Content;
                if (slice.Length <= 0) return;
                int srcStart = slice.Start;
                int srcLen = slice.End - slice.Start + 1;
                int outStart = _sb.Length;
                _sb.Append(slice.AsSpan());
                _ranges.Add(new TextRange(outStart, srcLen, srcStart, srcLen));
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
                _ranges.Add(new TextRange(codeOutStart, code.Content.Length,
                    code.Span.Start, code.Span.End - code.Span.Start + 1));
                break;

            case LinkInline link when !link.IsImage:
                // [text](url) → emit just the text, drop the URL.
                EmitInlines(link);
                break;

            // Image is a LinkInline with IsImage=true; skip entirely (no alt
            // text in the speech stream).
            case LinkInline { IsImage: true }:
                break;

            case EmphasisInline emph:
                // *italic* / **bold** / ~~strike~~ → markup stripped, walk children.
                EmitInlines(emph);
                break;

            case LineBreakInline lb:
                // Hard line break → space (soft line breaks usually parse the
                // same way; both are intra-paragraph whitespace for TTS).
                // Don't emit if the previous char is already whitespace.
                if (_sb.Length > 0 && !char.IsWhiteSpace(_sb[^1]))
                    _sb.Append(lb.IsHard ? ' ' : ' ');
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
