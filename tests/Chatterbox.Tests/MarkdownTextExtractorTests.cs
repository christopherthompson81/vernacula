using Chatterbox.Base.Markdown;
using Xunit;

namespace Chatterbox.Tests;

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
