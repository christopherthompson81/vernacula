using System;
using System.Linq;
using Vernacula.Tts.Base.Markdown;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// Unit tests for <see cref="MarkdownTextExtractor"/>. The behavior
/// matrix it locks down comes from the PR-confirmed defaults:
/// headings/paragraphs/lists/quotes/emphasis/links → text only;
/// fenced code blocks, tables, images, HTML, horizontal rules,
/// footnotes → skipped entirely.
///
/// Each test asserts both the output text AND that source-range
/// entries point back to a substring of the original markdown that
/// actually contains the extracted text — that's the contract the
/// forced-aligner downstream depends on.
/// </summary>
public class MarkdownTextExtractorTests
{
    // ── Headings ──────────────────────────────────────────────────────

    [Fact]
    public void Heading_emits_text_with_appended_period()
    {
        var r = MarkdownTextExtractor.Extract("# Hello World");
        Assert.Equal("Hello World.", r.Text);
    }

    [Fact]
    public void Heading_keeps_existing_terminal_punctuation()
    {
        var r = MarkdownTextExtractor.Extract("## Is it really?");
        Assert.Equal("Is it really?", r.Text);
    }

    [Fact]
    public void Heading_followed_by_paragraph_separates_with_blank_line()
    {
        var r = MarkdownTextExtractor.Extract("# Title\n\nBody text.");
        Assert.Equal("Title.\n\nBody text.", r.Text);
    }

    // ── Paragraphs + line breaks ──────────────────────────────────────

    [Fact]
    public void Paragraph_emits_text_verbatim()
    {
        var r = MarkdownTextExtractor.Extract("Hello world, this is a test.");
        Assert.Equal("Hello world, this is a test.", r.Text);
    }

    [Fact]
    public void Two_paragraphs_separated_by_blank_line()
    {
        var r = MarkdownTextExtractor.Extract("First paragraph.\n\nSecond paragraph.");
        Assert.Equal("First paragraph.\n\nSecond paragraph.", r.Text);
    }

    // ── Inline formatting (markup stripped, text kept) ────────────────

    [Fact]
    public void Bold_emphasis_strips_markup()
    {
        var r = MarkdownTextExtractor.Extract("This is **important** text.");
        Assert.Equal("This is important text.", r.Text);
    }

    [Fact]
    public void Italic_emphasis_strips_markup()
    {
        var r = MarkdownTextExtractor.Extract("An *emphasized* word.");
        Assert.Equal("An emphasized word.", r.Text);
    }

    [Fact]
    public void Inline_code_strips_backticks()
    {
        var r = MarkdownTextExtractor.Extract("Run `foo()` to start.");
        Assert.Equal("Run foo() to start.", r.Text);
    }

    [Fact]
    public void Link_emits_text_drops_url()
    {
        var r = MarkdownTextExtractor.Extract("See [the docs](https://example.com/long/url) for details.");
        Assert.Equal("See the docs for details.", r.Text);
    }

    // ── Lists ─────────────────────────────────────────────────────────

    [Fact]
    public void Bullet_list_emits_items_newline_separated()
    {
        var r = MarkdownTextExtractor.Extract("- apple\n- banana\n- cherry");
        Assert.Equal("apple\nbanana\ncherry", r.Text);
    }

    [Fact]
    public void Numbered_list_drops_enumeration_prefix()
    {
        var r = MarkdownTextExtractor.Extract("1. first\n2. second\n3. third");
        Assert.Equal("first\nsecond\nthird", r.Text);
    }

    // ── Block quotes ──────────────────────────────────────────────────

    [Fact]
    public void Block_quote_drops_marker_keeps_text()
    {
        var r = MarkdownTextExtractor.Extract("> quoted text here");
        Assert.Equal("quoted text here", r.Text);
    }

    // GitHub alert blockquotes. Markdig 1.x recognizes these (AlertExtension,
    // pulled in by UseAdvancedExtensions) and consumes the "[!KIND]" marker
    // into the block instead of leaving it as a literal, so the marker no
    // longer reaches the synthesizer. Under 0.34.0 the TTS read it aloud.
    // These tests pin the stripping: it moves every following output offset,
    // and the karaoke view aligns against those offsets.

    [Theory]
    [InlineData("NOTE")]
    [InlineData("TIP")]
    [InlineData("IMPORTANT")]
    [InlineData("WARNING")]
    [InlineData("CAUTION")]
    public void Alert_quote_drops_the_kind_marker(string kind)
    {
        var r = MarkdownTextExtractor.Extract($"> [!{kind}]\n> Body of the alert.");
        Assert.Equal("Body of the alert.", r.Text);
    }

    [Fact]
    public void Alert_marker_is_dropped_for_an_unknown_kind_too()
    {
        var r = MarkdownTextExtractor.Extract("> [!FOO]\n> Body of the alert.");
        Assert.Equal("Body of the alert.", r.Text);
    }

    // AlertBlock subclasses QuoteBlock, so the `case QuoteBlock` arm in the
    // extractor still matches it. If a future Markdig reparents it the block
    // would fall through to `default: return false` and vanish from the output
    // entirely — this is the test that localizes that.
    [Fact]
    public void Alert_is_still_a_quote_block_spanning_exactly_the_body()
    {
        var r = MarkdownTextExtractor.Extract("> [!NOTE]\n> Body of the alert.");
        var block = Assert.Single(r.Blocks);
        Assert.Equal(BlockKind.Quote, block.Kind);
        Assert.Equal("Body of the alert.",
                     r.Text[block.OutputStart..(block.OutputStart + block.OutputLength)]);
    }

    // Markdig's AlertParser only fires on a top-level quote, so an alert
    // indented under a list item keeps its marker and is spoken. GitHub does
    // render these as alerts, so the two disagree; pinned rather than fixed,
    // because a future Markdig closing the gap would shift output offsets
    // again and this is what would notice.
    [Fact]
    public void Alert_nested_in_a_list_item_keeps_its_marker()
    {
        var r = MarkdownTextExtractor.Extract("- item one\n  > [!NOTE]\n  > Nested alert body.");
        Assert.Equal("item one\n\n[!NOTE] Nested alert body.", r.Text);
    }

    // An alert whose body is empty contributes no text and no BlockSpan,
    // where 0.34.0 gave one quote block containing the marker: the marker was
    // the only content, and it is now consumed into the block. Consumers
    // walking Blocks to mirror document structure see one fewer quote than
    // the source has. (An empty *plain* quote also yields no block, on both
    // versions — the contrast here is against 0.34.0 on this same input.)
    [Fact]
    public void Alert_with_an_empty_body_contributes_no_text_and_no_block()
    {
        var r = MarkdownTextExtractor.Extract("> [!NOTE]");
        Assert.Equal("", r.Text);
        Assert.Empty(r.Blocks);
    }

    [Fact]
    public void Text_after_an_alert_is_not_shifted_by_the_marker()
    {
        var r = MarkdownTextExtractor.Extract("> [!NOTE]\n> Heed this.\n\nAfter the alert.");
        Assert.Equal("Heed this.\n\nAfter the alert.", r.Text);
        Assert.Equal(r.Text.IndexOf("After", StringComparison.Ordinal),
                     r.Blocks[^1].OutputStart);
    }

    [Fact]
    public void Bracketed_text_that_is_not_an_alert_marker_is_kept()
    {
        var r = MarkdownTextExtractor.Extract("> [not an alert]\n> Body.");
        Assert.Equal("[not an alert] Body.", r.Text);
    }

    // ── Skipped constructs ────────────────────────────────────────────

    [Fact]
    public void Fenced_code_block_is_skipped_entirely()
    {
        var r = MarkdownTextExtractor.Extract("Before.\n\n```python\nprint('hi')\n```\n\nAfter.");
        Assert.Equal("Before.\n\nAfter.", r.Text);
    }

    [Fact]
    public void Image_is_skipped()
    {
        var r = MarkdownTextExtractor.Extract("Look: ![alt text](pic.png) at this.");
        Assert.Equal("Look:  at this.", r.Text);
    }

    [Fact]
    public void Table_is_skipped_entirely()
    {
        var md = "Intro.\n\n| col1 | col2 |\n|------|------|\n| a    | b    |\n\nOutro.";
        var r = MarkdownTextExtractor.Extract(md);
        Assert.Equal("Intro.\n\nOutro.", r.Text);
    }

    [Fact]
    public void Horizontal_rule_is_skipped()
    {
        var r = MarkdownTextExtractor.Extract("Above.\n\n---\n\nBelow.");
        Assert.Equal("Above.\n\nBelow.", r.Text);
    }

    [Fact]
    public void Inline_html_tags_are_stripped_inner_text_kept()
    {
        var r = MarkdownTextExtractor.Extract("Hello <span>tag</span> world.");
        // Markdig parses this as Literal("Hello ") + HtmlInline("<span>") +
        // Literal("tag") + HtmlInline("</span>") + Literal(" world."). We
        // drop the HtmlInline nodes (open/close tokens) but the text nodes
        // BETWEEN them are ordinary Literals and survive. Locked in to make
        // the behavior explicit; if this becomes a problem we can switch to
        // a stricter HTML stripper.
        Assert.Equal("Hello tag world.", r.Text);
    }

    // ── Source-range index ────────────────────────────────────────────

    // Every recorded range must be in bounds and its source slice must
    // actually contain the output it claims to represent. The older tests
    // spot-check one construct each; this walks a battery of documents,
    // including the container shapes where the index used to point at
    // unrelated text (see the LiteralInline case in the extractor).
    //
    // "Contains" rather than "equals" because the contract allows a source
    // slice wider than the output: a backslash escape emits one character
    // for two source ones, and inline code's span covers its backticks.
    [Theory]
    [InlineData("Hello world.")]
    [InlineData("> plain quote\n> second line")]
    [InlineData("- item one\ncontinued lazily")]
    [InlineData("This [is not][nope] a link.")]
    [InlineData("> level one\n> > level two\n> back to one")]
    [InlineData("- a\n  - nested\n    continued\n- b")]
    [InlineData("> quote with `code`\n> and **bold** text")]
    [InlineData("escape \\*not italic\\* done")]
    [InlineData("*emph across\nlines* end")]
    [InlineData("> [!NOTE]\n> Alert body here.")]
    [InlineData("1. step one\n   > [!WARNING]\n   > careful\n2. step two")]
    [InlineData("para one\n\n> quoted\n\n- listed\n\n# heading")]
    [InlineData("CRLF quote:\r\n> line one\r\n> line two")]
    // Trailing markup that emits nothing: the last literal's output span used
    // to run past Text once the trailing whitespace was trimmed.
    [InlineData("hello ![img](/x.png)")]
    [InlineData("text <span>")]
    [InlineData("> quoted ![badge](/b.svg)")]
    public void Every_range_maps_to_source_that_contains_its_output(string src)
    {
        var r = MarkdownTextExtractor.Extract(src);
        Assert.NotEmpty(r.Ranges);
        foreach (var range in r.Ranges)
        {
            // Output bounds first: a range that overruns Text would make the
            // Substring below throw instead of failing readably.
            Assert.InRange(range.OutputStart, 0, r.Text.Length);
            Assert.InRange(range.OutputStart + range.OutputLength, 0, r.Text.Length);
            Assert.InRange(range.SourceStart, 0, src.Length);
            Assert.InRange(range.SourceStart + range.SourceLength, 0, src.Length);
            string outSlice = r.Text.Substring(range.OutputStart, range.OutputLength);
            string srcSlice = src.Substring(range.SourceStart, range.SourceLength);
            Assert.Contains(outSlice, srcSlice, StringComparison.Ordinal);
        }
    }

    // The three shapes from the bug report, pinned at exact offsets. Before
    // the fix these read LiteralInline.Content.Start, an offset into the
    // buffer Markdig re-assembles for a container's continuation lines rather
    // than into the document, so each pointed at unrelated text.

    [Fact]
    public void Quote_continuation_line_maps_to_its_real_document_offset()
    {
        const string src = "> plain quote\n> second line";
        var r = MarkdownTextExtractor.Extract(src);
        var second = r.Ranges[^1];
        Assert.Equal("second line", r.Text.Substring(second.OutputStart, second.OutputLength));
        Assert.Equal(16, second.SourceStart);   // was 12, an offset into "plain quote\nsecond line"
        Assert.Equal("second line", src.Substring(second.SourceStart, second.SourceLength));
    }

    [Fact]
    public void Lazily_continued_list_item_maps_to_its_real_document_offset()
    {
        const string src = "- item one\ncontinued lazily";
        var r = MarkdownTextExtractor.Extract(src);
        Assert.Equal(2, r.Ranges[0].SourceStart);    // was 0, pointing at "- item o"
        Assert.Equal("item one", src.Substring(r.Ranges[0].SourceStart, r.Ranges[0].SourceLength));
        Assert.Equal("continued lazily",
                     src.Substring(r.Ranges[^1].SourceStart, r.Ranges[^1].SourceLength));
    }

    [Fact]
    public void Unresolved_reference_link_bracket_maps_to_its_real_document_offset()
    {
        const string src = "This [is not][nope] a link.";
        var r = MarkdownTextExtractor.Extract(src);
        var bracket = r.Ranges.Single(t => r.Text.Substring(t.OutputStart, t.OutputLength) == "[");
        Assert.Equal(13, bracket.SourceStart);   // was 0, pointing at "T"
        Assert.Equal("[", src.Substring(bracket.SourceStart, bracket.SourceLength));
    }

    // Trailing markup that contributes no text (an image, an inline tag) left
    // the preceding literal's output span running past Text, because the
    // trailing-whitespace trim shortens the text without shortening the range.
    // A consumer walking the index to slice Text got an out-of-range throw.
    [Fact]
    public void Range_is_clamped_when_trailing_markup_is_dropped()
    {
        const string src = "hello ![img](/x.png)";
        var r = MarkdownTextExtractor.Extract(src);
        Assert.Equal("hello", r.Text);
        var last = r.Ranges[^1];
        Assert.Equal(r.Text.Length, last.OutputStart + last.OutputLength);
        Assert.Equal("hello", r.Text.Substring(last.OutputStart, last.OutputLength));
    }

    // A backslash escape emits one character for two source ones, so the
    // source span is legitimately wider than the output. Pinned because the
    // fix changed SourceLength from "however long the output was" to the
    // span's real extent, and the documented contract is subsequence, not
    // equality.
    [Fact]
    public void Escaped_character_keeps_the_wider_source_span()
    {
        const string src = "escape \\*not italic\\* done";
        var r = MarkdownTextExtractor.Extract(src);
        var escaped = r.Ranges.Single(t =>
            r.Text.Substring(t.OutputStart, t.OutputLength).StartsWith("*not", StringComparison.Ordinal));
        Assert.Equal("*not italic", r.Text.Substring(escaped.OutputStart, escaped.OutputLength));
        Assert.Equal("\\*not italic", src.Substring(escaped.SourceStart, escaped.SourceLength));
        Assert.True(escaped.SourceLength > escaped.OutputLength);
    }

    [Fact]
    public void Plain_paragraph_range_points_back_to_source()
    {
        const string src = "Hello world.";
        var r = MarkdownTextExtractor.Extract(src);
        Assert.Single(r.Ranges);
        var range = r.Ranges[0];
        Assert.Equal(0, range.OutputStart);
        Assert.Equal(12, range.OutputLength);
        Assert.Equal(0, range.SourceStart);
        Assert.Equal(src.Substring(range.SourceStart, range.SourceLength),
                     r.Text.Substring(range.OutputStart, range.OutputLength));
    }

    [Fact]
    public void Bold_inline_range_points_to_inner_text_only()
    {
        const string src = "A **bold** word.";
        var r = MarkdownTextExtractor.Extract(src);
        // Expect three text-bearing ranges: "A ", "bold", " word."
        Assert.Equal("A bold word.", r.Text);
        // Walk the ranges and verify each one's source slice contains the
        // output substring it claims to represent.
        foreach (var range in r.Ranges)
        {
            string outSlice = r.Text.Substring(range.OutputStart, range.OutputLength);
            string srcSlice = src.Substring(range.SourceStart, range.SourceLength);
            Assert.Contains(outSlice, srcSlice);
        }
    }

    [Fact]
    public void Heading_period_is_synthetic_no_range_for_it()
    {
        const string src = "# Title";
        var r = MarkdownTextExtractor.Extract(src);
        Assert.Equal("Title.", r.Text);
        // The "." at position 5 is synthetic — no range entry should cover it.
        // The single range entry covers "Title" at output [0..5].
        Assert.Single(r.Ranges);
        var range = r.Ranges[0];
        Assert.Equal(0, range.OutputStart);
        Assert.Equal(5, range.OutputLength);
        Assert.Equal("Title", r.Text.Substring(range.OutputStart, range.OutputLength));
    }

    // ── Edge cases ────────────────────────────────────────────────────

    [Fact]
    public void Empty_input_produces_empty_output()
    {
        var r = MarkdownTextExtractor.Extract("");
        Assert.Equal("", r.Text);
        Assert.Empty(r.Ranges);
    }

    [Fact]
    public void Whitespace_only_input_produces_empty_output()
    {
        var r = MarkdownTextExtractor.Extract("   \n\n   \n");
        Assert.Equal("", r.Text);
    }

    [Fact]
    public void Document_with_only_skipped_blocks_produces_empty_output()
    {
        var r = MarkdownTextExtractor.Extract("```\ncode only\n```\n\n---\n\n![img](p.png)");
        Assert.Equal("", r.Text);
    }

    [Fact]
    public void Real_world_doc_excerpt_round_trip()
    {
        // Sampled the kind of mixed-construct paragraph found in our own
        // README files. Just check it doesn't crash and produces something
        // sensible.
        const string md = """
            # Vernacula

            **Vernacula** is a TTS tool. See the [README](README.md) for usage.

            ## Features

            - Long-form synthesis
            - Voice cloning from a `reference.wav`
            - Markdown input

            > Note: GPU recommended.

            ```bash
            chatterbox --voice ref.wav --text-file in.md
            ```

            See the table below for benchmarks.

            | Lever | Win |
            |-------|-----|
            | Batching | 50% |
            """;
        var r = MarkdownTextExtractor.Extract(md);
        Assert.Contains("Vernacula", r.Text);
        Assert.Contains("Long-form synthesis", r.Text);
        Assert.Contains("Note: GPU recommended", r.Text);
        Assert.DoesNotContain("chatterbox --voice", r.Text);  // code block dropped
        Assert.DoesNotContain("Batching", r.Text);            // table dropped
        Assert.DoesNotContain("README.md", r.Text);           // URL dropped, text kept
        Assert.Contains("README", r.Text);
    }
}
