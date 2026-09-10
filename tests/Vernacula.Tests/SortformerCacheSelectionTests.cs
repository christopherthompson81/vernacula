using System;
using System.Linq;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// Which frames survive speaker-cache compression, and in particular what happens when
/// their scores TIE.
///
/// Ties are not an edge case here: Sortformer's float32 sigmoid saturates to exactly 1.0
/// above ~16.6 logits, so a confidently single-speaker stream produces bit-identical preds
/// rows and therefore bit-identical scores. NeMo's choice among equals is whatever
/// `torch.topk` returns, which is a quickselect artifact -- a block from the middle of the
/// tied range, at an offset following no rule and not reproducing across torch builds -- so
/// there is no order for the port to match (#171). What the port owes instead is that its
/// own order is deterministic and not systematically biased, which is what these lock.
/// </summary>
public class SortformerCacheSelectionTests
{
    private const int Spk  = 4;
    private const int Keep = 188;

    private static (float score, int tIdx, int sIdx)[] Flat(float[,] scores)
    {
        int t = scores.GetLength(0), s = scores.GetLength(1);
        var flat = new (float, int, int)[t * s];
        for (int i = 0; i < t; i++)
            for (int j = 0; j < s; j++)
                flat[i * s + j] = (scores[i, j], i, j);
        return flat;
    }

    [Fact]
    public void HighestScoresWin_AndKeptRowsComeBackSpeakerMajor()
    {
        // Distinct scores decreasing with t, so the first Keep/Spk frames of every speaker
        // are the obvious picks and nothing depends on the tie rule.
        int frames = 100;
        var scores = new float[frames, Spk];
        for (int t = 0; t < frames; t++)
            for (int s = 0; s < Spk; s++)
                scores[t, s] = frames - t;

        var selected = SortformerStreamer.SelectCacheFrames(Flat(scores), Keep, frames, frames);

        Assert.Equal(Keep, selected.Length);
        Assert.All(selected, x => Assert.False(x.disabled));
        Assert.True(selected.Max(x => x.tIdx) < Keep / Spk + 1);

        // The kept rows are ordered by NeMo's speaker-major flattened index. That ordering
        // is torch.sort(topk_indices), not a tie rule, and is deliberately not what the
        // frame-major tie-break below changed.
        var order = selected.Select(x => x.sIdx * frames + x.tIdx).ToArray();
        Assert.Equal(order.OrderBy(v => v).ToArray(), order);
    }

    [Fact]
    public void AllScoresTied_FillsEverySpeakerEqually()
    {
        // The regression this exists for. Ties used to be broken on the speaker-major
        // flattened index, which orders EVERY speaker-0 entry ahead of EVERY speaker-1
        // entry -- so a fully tied matrix handed the whole cache to speaker 0. Speaker IDs
        // are arbitrary slot assignments, so that is a bias toward whoever landed in slot 0.
        int frames = 300;
        var scores = new float[frames, Spk];   // all zero: every entry ties

        var selected = SortformerStreamer.SelectCacheFrames(Flat(scores), Keep, frames, frames);

        var perSpeaker = Enumerable.Range(0, Spk)
            .Select(s => selected.Count(x => x.sIdx == s))
            .ToArray();
        Assert.All(perSpeaker, c => Assert.Equal(Keep / Spk, c));

        // Frame-major: the earliest frames win, all speakers of frame t before frame t+1.
        Assert.True(selected.Max(x => x.tIdx) < Keep / Spk);
    }

    [Fact]
    public void PartialTie_TakesTheEarliestFramesAmongEquals()
    {
        // One speaker's frames all tie above everything else, so exactly which of them is
        // kept is decided by the tie rule alone.
        int frames = 300;
        var scores = new float[frames, Spk];
        for (int t = 0; t < frames; t++)
            for (int s = 0; s < Spk; s++)
                scores[t, s] = s == 1 ? 10f : 0f;

        var selected = SortformerStreamer.SelectCacheFrames(Flat(scores), Keep, frames, frames);

        var spk1 = selected.Where(x => x.sIdx == 1).Select(x => x.tIdx).OrderBy(t => t).ToArray();
        Assert.Equal(Enumerable.Range(0, spk1.Length).ToArray(), spk1);
    }

    [Fact]
    public void NegativeInfinityPicksAreDisabledAndSortLast()
    {
        // Fewer surviving frames than the cache holds -- sparse audio, and the early
        // stream. NeMo substitutes max_index for those picks, which both marks them
        // disabled (mean silence embedding, zero preds) and pushes them to the end.
        int frames = 100;
        var scores = new float[frames, Spk];
        for (int t = 0; t < frames; t++)
            for (int s = 0; s < Spk; s++)
                scores[t, s] = t < 10 ? 1f : float.NegativeInfinity;

        var selected = SortformerStreamer.SelectCacheFrames(Flat(scores), Keep, frames, frames);

        Assert.Equal(10 * Spk, selected.Count(x => !x.disabled));
        int lastLive = Array.FindLastIndex(selected, x => !x.disabled);
        int firstDead = Array.FindIndex(selected, x => x.disabled);
        Assert.True(lastLive < firstDead, "every -inf pick must sort after every live pick");
    }

    [Fact]
    public void SilencePadIsDisabledButKeepsItsPlace()
    {
        // The +inf silence pad is always picked, and is disabled so the mean silence
        // embedding is substituted -- but unlike a -inf pick it keeps its natural position
        // at the tail of its speaker's block rather than being pushed to the very end.
        const int real = 100, sil = 3, ext = real + sil;
        var scores = new float[ext, Spk];
        for (int t = 0; t < ext; t++)
            for (int s = 0; s < Spk; s++)
                scores[t, s] = t >= real ? float.PositiveInfinity : 1f;

        var selected = SortformerStreamer.SelectCacheFrames(Flat(scores), Keep, ext, real);

        Assert.Equal(sil * Spk, selected.Count(x => x.disabled));
        Assert.All(selected.Where(x => x.disabled), x => Assert.True(x.tIdx >= real));
        // Not all at the end: speaker 0's pad rows precede speaker 1's live rows.
        int lastPadOfSpk0 = Array.FindLastIndex(selected, x => x.disabled && x.sIdx == 0);
        int firstLiveOfSpk1 = Array.FindIndex(selected, x => !x.disabled && x.sIdx == 1);
        Assert.True(lastPadOfSpk0 < firstLiveOfSpk1);
    }
}
