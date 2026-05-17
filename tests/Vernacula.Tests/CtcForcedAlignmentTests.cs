using Vernacula.Base.Alignment;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// Unit tests for <see cref="CtcForcedAlignment"/>. The Viterbi DP is the
/// algorithmic heart of NFA-style forced alignment — tests fix correctness
/// on hand-crafted log-prob matrices where the optimal path is known by
/// construction, plus edge cases (single-token, same-token-repeat
/// requiring inserted blank, target too long for frames).
///
/// Vocab convention in these tests: ids 0..V-1 are tokens, id V is blank.
/// vocabPlusBlank = V+1.
/// </summary>
public class CtcForcedAlignmentTests
{
    // Helper: build a log-probs row that picks a clear winner at one
    // column and pushes other columns to a tiny log-prob. Makes the
    // optimal path unambiguous.
    private static float[] OneHotFrame(int vocabPlusBlank, int hotCol, float hotLogp = -0.01f, float coldLogp = -10.0f)
    {
        var row = new float[vocabPlusBlank];
        for (int i = 0; i < vocabPlusBlank; i++) row[i] = coldLogp;
        row[hotCol] = hotLogp;
        return row;
    }

    private static float[] Concat(params float[][] rows)
    {
        int len = rows.Sum(r => r.Length);
        var flat = new float[len];
        int off = 0;
        foreach (var r in rows) { Array.Copy(r, 0, flat, off, r.Length); off += r.Length; }
        return flat;
    }

    // ── Basic correctness ────────────────────────────────────────────

    [Fact]
    public void Single_token_single_frame_assigns_the_frame()
    {
        // vocab: [tok0=0, blank=1]. Target [0]. T=1, frame strongly favors tok0.
        var logp = Concat(OneHotFrame(2, 0));
        var result = CtcForcedAlignment.Align(logp, numFrames: 1, vocabPlusBlank: 2,
            targetTokens: new[] { 0 }, blankId: 1);
        Assert.Single(result);
        Assert.Equal(0, result[0].StartFrame);
        Assert.Equal(0, result[0].EndFrame);
        Assert.Equal(0, result[0].TokenId);
        Assert.Equal(0, result[0].TokenPosition);
    }

    [Fact]
    public void Two_distinct_tokens_split_frames()
    {
        // Target [0, 1]. T=4. Frames 0-1 favor tok0; frames 2-3 favor tok1.
        var logp = Concat(
            OneHotFrame(3, 0), OneHotFrame(3, 0),
            OneHotFrame(3, 1), OneHotFrame(3, 1));
        var result = CtcForcedAlignment.Align(logp, numFrames: 4, vocabPlusBlank: 3,
            targetTokens: new[] { 0, 1 }, blankId: 2);
        Assert.Equal(2, result.Count);
        // tok0 owns frames 0-1, tok1 owns frames 2-3.
        Assert.Equal((0, 1), (result[0].StartFrame, result[0].EndFrame));
        Assert.Equal((2, 3), (result[1].StartFrame, result[1].EndFrame));
    }

    [Fact]
    public void Same_token_repeat_requires_blank_between()
    {
        // Target [0, 0]. T=3 (the minimum: tok-blank-tok).
        // Frame 0 favors tok0, frame 1 favors blank, frame 2 favors tok0.
        var logp = Concat(
            OneHotFrame(2, 0),
            OneHotFrame(2, 1),
            OneHotFrame(2, 0));
        var result = CtcForcedAlignment.Align(logp, numFrames: 3, vocabPlusBlank: 2,
            targetTokens: new[] { 0, 0 }, blankId: 1);
        Assert.Equal(2, result.Count);
        Assert.Equal((0, 0), (result[0].StartFrame, result[0].EndFrame));
        Assert.Equal((2, 2), (result[1].StartFrame, result[1].EndFrame));
    }

    // ── Error paths ──────────────────────────────────────────────────

    [Fact]
    public void Empty_target_throws()
    {
        var logp = OneHotFrame(2, 0);
        Assert.Throws<ArgumentException>(() =>
            CtcForcedAlignment.Align(logp, 1, 2, Array.Empty<int>(), 1));
    }

    [Fact]
    public void Target_too_long_for_frames_throws()
    {
        // Target [0, 1, 0]. Minimum T=3 (all distinct).
        var logp = Concat(OneHotFrame(3, 0), OneHotFrame(3, 1));  // T=2
        Assert.Throws<ArgumentException>(() =>
            CtcForcedAlignment.Align(logp, 2, 3, new[] { 0, 1, 0 }, 2));
    }

    [Fact]
    public void Same_token_pair_needs_blank_between_throws_when_T_too_small()
    {
        // Target [0, 0]. Minimum T=3 (tok-blank-tok). T=2 should fail.
        var logp = Concat(OneHotFrame(2, 0), OneHotFrame(2, 0));
        Assert.Throws<ArgumentException>(() =>
            CtcForcedAlignment.Align(logp, 2, 2, new[] { 0, 0 }, 1));
    }

    [Fact]
    public void Target_token_equals_blank_throws()
    {
        var logp = OneHotFrame(2, 0);
        Assert.Throws<ArgumentException>(() =>
            CtcForcedAlignment.Align(logp, 1, 2, new[] { 1 }, blankId: 1));
    }

    [Fact]
    public void Blank_id_out_of_range_throws()
    {
        var logp = OneHotFrame(2, 0);
        Assert.Throws<ArgumentException>(() =>
            CtcForcedAlignment.Align(logp, 1, 2, new[] { 0 }, blankId: 99));
    }

    [Fact]
    public void Target_token_out_of_range_throws()
    {
        var logp = OneHotFrame(2, 0);
        Assert.Throws<ArgumentException>(() =>
            CtcForcedAlignment.Align(logp, 1, 2, new[] { 99 }, blankId: 1));
    }

    // ── Realistic shape ──────────────────────────────────────────────

    [Fact]
    public void Trailing_blanks_after_last_token_are_handled()
    {
        // Target [0]. T=4. tok0 fires once at frame 0; the rest is blank.
        var logp = Concat(
            OneHotFrame(2, 0),
            OneHotFrame(2, 1),
            OneHotFrame(2, 1),
            OneHotFrame(2, 1));
        var result = CtcForcedAlignment.Align(logp, 4, 2, new[] { 0 }, 1);
        Assert.Single(result);
        // tok0 owns at least frame 0 (could extend if alpha[stay] beats blank).
        Assert.Equal(0, result[0].StartFrame);
        // End frame is wherever the Viterbi path last assigned the token,
        // typically frame 0 here since blank dominates frames 1-3.
        Assert.True(result[0].EndFrame >= 0);
    }

    [Fact]
    public void Leading_blanks_before_first_token_are_handled()
    {
        // Target [0]. T=4. Blanks first, tok0 fires at frame 3.
        var logp = Concat(
            OneHotFrame(2, 1),
            OneHotFrame(2, 1),
            OneHotFrame(2, 1),
            OneHotFrame(2, 0));
        var result = CtcForcedAlignment.Align(logp, 4, 2, new[] { 0 }, 1);
        Assert.Single(result);
        Assert.Equal(3, result[0].StartFrame);
        Assert.Equal(3, result[0].EndFrame);
    }

    [Fact]
    public void Token_position_field_matches_input_order()
    {
        var logp = Concat(
            OneHotFrame(4, 0), OneHotFrame(4, 1), OneHotFrame(4, 2));
        var result = CtcForcedAlignment.Align(logp, 3, 4, new[] { 0, 1, 2 }, 3);
        Assert.Equal(3, result.Count);
        for (int k = 0; k < 3; k++)
        {
            Assert.Equal(k, result[k].TokenPosition);
            Assert.Equal(k, result[k].TokenId);
        }
    }

    [Fact]
    public void Long_realistic_alignment_does_not_throw()
    {
        // 100 frames, 5-token target. Random-ish log-probs but with the
        // target tokens slightly favored in chronological order so the
        // optimal alignment is non-degenerate.
        const int T = 100;
        const int VPlusBlank = 6;  // 5 tokens + blank
        const int blank = 5;
        var target = new[] { 0, 1, 2, 3, 4 };
        var logp = new float[T * VPlusBlank];
        for (int t = 0; t < T; t++)
        {
            // Favor blank everywhere, slightly favor the chronologically-
            // appropriate target token in its window.
            for (int v = 0; v < VPlusBlank; v++) logp[t * VPlusBlank + v] = -2.0f;
            logp[t * VPlusBlank + blank] = -1.0f;
            int window = t * target.Length / T;
            logp[t * VPlusBlank + target[Math.Min(window, target.Length - 1)]] = -0.5f;
        }
        var result = CtcForcedAlignment.Align(logp, T, VPlusBlank, target, blank);
        Assert.Equal(5, result.Count);
        // Tokens should be in monotone-increasing frame order.
        for (int k = 1; k < 5; k++)
            Assert.True(result[k].StartFrame >= result[k - 1].StartFrame,
                $"token {k} starts before token {k - 1}");
    }
}
