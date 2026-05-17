using Vernacula.Base.Alignment;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// Unit tests for <see cref="NemoNfaAligner.GroupTokensIntoWords"/>. The
/// grouping is pure-data — takes per-token alignments and SP-prefixed
/// token strings, returns word-level timings — so testing it in isolation
/// (without loading the NeMo bundle) is the right cost/coverage trade.
/// </summary>
public class WordGroupingTests
{
    // ▁ = U+2581, the sentencepiece word-boundary marker. Tokens starting
    // with this character begin a new word; tokens without it continue
    // the current word.
    private const string SP = "▁";
    private const double FrameSec = 0.08;  // FastConformer encoder: 10 ms hop × 8 subsampling

    private static TokenAlignment TA(int pos, int start, int end) =>
        new(TokenId: 0, TokenPosition: pos, StartFrame: start, EndFrame: end);

    [Fact]
    public void Single_word_single_token()
    {
        var alignments = new[] { TA(0, 0, 4) };
        var strings = new[] { SP + "hello" };
        var words = NemoNfaAligner.GroupTokensIntoWords(alignments, strings, FrameSec);
        Assert.Single(words);
        Assert.Equal("hello", words[0].Text);
        Assert.Equal(0.0, words[0].StartSeconds, 5);
        Assert.Equal(5 * FrameSec, words[0].EndSeconds, 5);  // (endFrame=4)+1 frames * sec/frame
    }

    [Fact]
    public void Single_word_multi_subword_tokens()
    {
        // "▁hello" + "world" (no ▁ on the 2nd → glued to the first)
        var alignments = new[] { TA(0, 0, 2), TA(1, 3, 5) };
        var strings = new[] { SP + "hello", "world" };
        var words = NemoNfaAligner.GroupTokensIntoWords(alignments, strings, FrameSec);
        Assert.Single(words);
        Assert.Equal("helloworld", words[0].Text);
        Assert.Equal(0.0, words[0].StartSeconds, 5);
        Assert.Equal(6 * FrameSec, words[0].EndSeconds, 5);
    }

    [Fact]
    public void Two_words_each_with_word_boundary_marker()
    {
        // "▁the" + "▁quick"
        var alignments = new[] { TA(0, 0, 2), TA(1, 4, 7) };
        var strings = new[] { SP + "the", SP + "quick" };
        var words = NemoNfaAligner.GroupTokensIntoWords(alignments, strings, FrameSec);
        Assert.Equal(2, words.Count);
        Assert.Equal("the", words[0].Text);
        Assert.Equal("quick", words[1].Text);
        Assert.Equal(0.0, words[0].StartSeconds, 5);
        Assert.Equal(3 * FrameSec, words[0].EndSeconds, 5);
        Assert.Equal(4 * FrameSec, words[1].StartSeconds, 5);
        Assert.Equal(8 * FrameSec, words[1].EndSeconds, 5);
    }

    [Fact]
    public void First_token_without_word_boundary_still_starts_a_word()
    {
        // SP's AddDummyPrefix=false case: the very first token of a
        // sentence may not have ▁. The grouper must still treat it as
        // a word start so the output isn't empty.
        var alignments = new[] { TA(0, 0, 3) };
        var strings = new[] { "Hello" };  // no leading ▁
        var words = NemoNfaAligner.GroupTokensIntoWords(alignments, strings, FrameSec);
        Assert.Single(words);
        Assert.Equal("Hello", words[0].Text);
    }

    [Fact]
    public void Three_words_with_multi_subword_middle_word()
    {
        // "▁the" + "▁en" + "vironment" + "▁is" — "environment" splits
        // across two subwords; middle word should still group correctly.
        var alignments = new[] {
            TA(0, 0, 1),
            TA(1, 2, 3),
            TA(2, 4, 7),
            TA(3, 9, 10),
        };
        var strings = new[] { SP + "the", SP + "en", "vironment", SP + "is" };
        var words = NemoNfaAligner.GroupTokensIntoWords(alignments, strings, FrameSec);
        Assert.Equal(3, words.Count);
        Assert.Equal("the", words[0].Text);
        Assert.Equal("environment", words[1].Text);
        Assert.Equal("is", words[2].Text);
        // middle word spans frames 2..7 → 2*FrameSec to 8*FrameSec
        Assert.Equal(2 * FrameSec, words[1].StartSeconds, 5);
        Assert.Equal(8 * FrameSec, words[1].EndSeconds, 5);
    }

    [Fact]
    public void Empty_token_list_returns_empty()
    {
        var words = NemoNfaAligner.GroupTokensIntoWords(
            Array.Empty<TokenAlignment>(), Array.Empty<string>(), FrameSec);
        Assert.Empty(words);
    }

    [Fact]
    public void Word_boundary_only_token_is_emitted_as_empty_and_skipped()
    {
        // Pathological: a token consisting of just ▁ (no following chars).
        // After stripping ▁, the joined word is empty — should be filtered.
        var alignments = new[] { TA(0, 0, 1), TA(1, 2, 3) };
        var strings = new[] { SP, SP + "hello" };
        var words = NemoNfaAligner.GroupTokensIntoWords(alignments, strings, FrameSec);
        Assert.Single(words);
        Assert.Equal("hello", words[0].Text);
    }

    [Fact]
    public void Word_boundary_on_first_token_does_not_emit_extra_empty_word()
    {
        // Regression: the grouper's `startsWord = first-token || piece.StartsWith(▁)`
        // would trigger Flush() on an empty buffer at the very first token
        // when the token starts with ▁ (the common case). Flush's early-return
        // at "currentPieces.Count == 0" must prevent an extra empty word.
        var alignments = new[] { TA(0, 0, 2), TA(1, 4, 5) };
        var strings = new[] { SP + "hello", SP + "world" };
        var words = NemoNfaAligner.GroupTokensIntoWords(alignments, strings, FrameSec);
        // Expect exactly 2 words; no leading empty word from the first-token Flush.
        Assert.Equal(2, words.Count);
        Assert.Equal("hello", words[0].Text);
        Assert.Equal("world", words[1].Text);
    }

    [Fact]
    public void Timings_use_frame_plus_one_convention_for_end()
    {
        // End time = (endFrame + 1) * sec/frame so a single-frame token
        // gets a non-zero duration (start = end_inclusive=N → start=N*F,
        // end=(N+1)*F). Important for click-to-position UX: zero-width
        // intervals don't highlight cleanly.
        var alignments = new[] { TA(0, 5, 5) };  // single frame 5
        var strings = new[] { SP + "x" };
        var words = NemoNfaAligner.GroupTokensIntoWords(alignments, strings, FrameSec);
        Assert.Single(words);
        Assert.Equal(5 * FrameSec, words[0].StartSeconds, 5);
        Assert.Equal(6 * FrameSec, words[0].EndSeconds, 5);
        Assert.True(words[0].EndSeconds > words[0].StartSeconds);
    }
}
