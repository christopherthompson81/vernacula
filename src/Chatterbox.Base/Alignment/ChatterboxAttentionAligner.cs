using Chatterbox.Base.Tokenization;
using Vernacula.Base.Alignment;

namespace Chatterbox.Base.Alignment;

/// <summary>
/// Word-timing aligner that consumes an <see cref="AcousticLmAlignment"/>
/// matrix captured during Chatterbox's own LM rollout, instead of running
/// a separate forced-alignment model. Replaces
/// <see cref="NemoNfaAligner"/> in the Avalonia reader path.
///
/// Why it works better than NFA: the alignment is read directly from
/// the LM's cross-attention (Resemble AI's pre-identified alignment
/// heads — see <see cref="ChatterboxConstants.AlignmentLayerIndices"/>).
/// So the alignment grounds in what the LM *actually attended to* when
/// generating each speech token, not in what a separate ASR's reduced
/// vocabulary thinks it heard. Digits, multi-decimal section
/// references, fancy quotes, and proper nouns — all the cases that
/// broke NFA's SP-tokenizer vocab — align cleanly here because the LM
/// itself owns the text→audio mapping.
///
/// Doesn't implement <see cref="IForcedAligner"/> because the input
/// contract is different (alignment matrix + tokenizer instead of
/// audio + text). The output contract <see cref="WordTiming"/> is the
/// same so callers swap with no downstream changes.
/// </summary>
public static class ChatterboxAttentionAligner
{
    /// <summary>
    /// Convert a per-step attention matrix into per-input-word timings.
    ///
    /// <paramref name="alignment"/> is what <see cref="AcousticLM.Generate"/>
    /// returns when called with <c>captureAlignment: true</c>. Its
    /// NumTextCols must equal <c>WrapForLm(sourceText).Length</c>; this
    /// is checked.
    ///
    /// <paramref name="totalAudioSamples"/> + <paramref name="sampleRate"/>
    /// give the wall-clock duration of the synthesized audio; combined
    /// with NumSpeechRows they yield seconds-per-LM-step (the audio
    /// rate is fixed by Chatterbox's vocoder so this ratio is roughly
    /// constant ~40 ms/step).
    ///
    /// Algorithm:
    ///   1. Tokenize sourceText with per-token source spans
    ///      (<see cref="EnTokenizer.EncodeWithSpans"/>).
    ///   2. Split sourceText into whitespace-separated words with their
    ///      source-char ranges.
    ///   3. For each LM speech row (skipping row 0 — the prefill's
    ///      last-query slot, which predicts the START_SPEECH sentinel
    ///      and tends to attend to the conditioning tail rather than
    ///      content), argmax over text cols → wrapped-token index →
    ///      bpe-token index → source-char midpoint → containing word.
    ///   4. For each input word, aggregate the rows that argmaxed to
    ///      it: span = (min_row, max_row + 1) × secondsPerStep. Words
    ///      that never won an argmax get zero-duration entries at the
    ///      end of their would-be span (linear interpolation between
    ///      neighbours) so the consumer always has a 1:1 word list.
    /// </summary>
    public static IReadOnlyList<WordTiming> Align(
        AcousticLmAlignment alignment,
        string sourceText,
        EnTokenizer tokenizer,
        int totalAudioSamples,
        int sampleRate)
    {
        // Sanity: the LM input was WrapForLm(sourceText), which prepends
        // [EXAGGERATION, START] (2 tokens) and appends [STOP, START_SPEECH,
        // START_SPEECH] (3 tokens). So wrappedLen = bpeLen + 5.
        var bpeWithSpans = tokenizer.EncodeWithSpans(sourceText);
        int bpeLen = bpeWithSpans.Count;
        int expectedWrappedLen = bpeLen + 5;
        if (alignment.NumTextCols != expectedWrappedLen)
            throw new ArgumentException(
                $"Alignment NumTextCols={alignment.NumTextCols} but expected {expectedWrappedLen} "
                + $"(WrapForLm({bpeLen} bpe + 5 wrappers)). Was the alignment captured for this exact text?");

        int speechRows = alignment.NumSpeechRows;
        if (speechRows == 0) return Array.Empty<WordTiming>();

        double secondsPerStep = (double)totalAudioSamples / sampleRate / speechRows;

        // BPE column range in the wrapped sequence: [2, 2 + bpeLen).
        const int wrapperPrefix = 2;  // EXAGGERATION + START
        int bpeColStart = wrapperPrefix;
        int bpeColEnd = wrapperPrefix + bpeLen;

        // Split sourceText into words. Word = run of non-whitespace.
        var words = SplitIntoWords(sourceText);
        if (words.Count == 0) return Array.Empty<WordTiming>();

        // ── Build per-word attention scores per row ────────────────────
        // For each BPE token, find its containing input word (or -1 if
        // it's a SPACE / falls in a gap). Then per row, sum the
        // alignment weights of all BPE cols belonging to each word.
        // wordAttn[r, w] = total attention from speech row r to input word w.
        int[] bpeToWord = new int[bpeLen];
        for (int i = 0; i < bpeLen; i++)
        {
            int srcChar = (bpeWithSpans[i].SourceStart + bpeWithSpans[i].SourceEnd) / 2;
            bpeToWord[i] = FindWordContaining(words, srcChar);
        }

        int W = words.Count;
        int R = speechRows;
        // Flat row-major access for cache friendliness.
        var wordAttn = new float[R * W];
        for (int r = 1; r < R; r++)  // skip sentinel row 0
        {
            var row = alignment.Row(r);
            for (int c = bpeColStart; c < bpeColEnd; c++)
            {
                int w = bpeToWord[c - bpeColStart];
                if (w < 0) continue;
                wordAttn[r * W + w] += row[c];
            }
        }

        // ── Viterbi-style monotonic forced alignment ──────────────────
        // Find the monotonic non-decreasing row→word assignment that
        // maximizes total wordAttn. Solves the argmax-noise problem
        // (per-row argmax can spike forward to anticipate the next
        // word, putting that word's StartSeconds too early → "highlight
        // ahead of audio, audio catches up" symptom) by enforcing
        // global monotonicity rather than trusting any single row.
        //
        // State: dp[r, w] = max total score reaching row r at word w.
        // Transition: dp[r, w] = max(dp[r-1, w], dp[r-1, w-1]) + wordAttn[r, w]
        //   "stay on word w"     "advance to w from w-1"
        // Boundary: dp[1, 0] = wordAttn[1, 0]; dp[1, w>0] = -inf
        //   (row 1 is the first audible speech token; it must start at word 0)
        //
        // Every word ends up with at least one row in the assignment
        // (you can't skip words in a monotonic +0/+1 walk), so the
        // separate interpolation phase is no longer needed.
        const float NEG = float.MinValue / 2f;
        var dp = new float[R * W];
        var advance = new bool[R * W];  // true = arrived via "advance from w-1"
        for (int i = 0; i < dp.Length; i++) dp[i] = NEG;
        dp[1 * W + 0] = wordAttn[1 * W + 0];
        for (int r = 2; r < R; r++)
        {
            int rowBase = r * W;
            int prevBase = (r - 1) * W;
            for (int w = 0; w < W; w++)
            {
                float stayScore = dp[prevBase + w];
                float advanceScore = w > 0 ? dp[prevBase + w - 1] : NEG;
                if (stayScore >= advanceScore)
                {
                    dp[rowBase + w] = stayScore + wordAttn[rowBase + w];
                    advance[rowBase + w] = false;
                }
                else
                {
                    dp[rowBase + w] = advanceScore + wordAttn[rowBase + w];
                    advance[rowBase + w] = true;
                }
            }
        }

        // Backtrack from the best end-state. Pin the end to the LAST
        // word — typical chunks end exactly there, and forcing it
        // prevents the DP from "leaving" the last few words unassigned
        // when the model briefly drifts back to earlier text mid-rollout.
        int currR = R - 1;
        int currW = W - 1;
        var assignment = new int[R];
        assignment[0] = -1;  // sentinel
        while (currR >= 1)
        {
            assignment[currR] = currW;
            if (currR > 1 && advance[currR * W + currW]) currW--;
            currR--;
        }

        // ── Per-word row ranges → WordTiming entries ──────────────────
        var firstRow = new int[W];
        var lastRow = new int[W];
        for (int w = 0; w < W; w++) { firstRow[w] = -1; lastRow[w] = -1; }
        for (int r = 1; r < R; r++)
        {
            int w = assignment[r];
            if (firstRow[w] < 0) firstRow[w] = r;
            lastRow[w] = r;
        }

        var result = new List<WordTiming>(W);
        for (int w = 0; w < W; w++)
        {
            // Every word has at least one row because the DP must
            // monotonically walk through all of them. Empty case
            // shouldn't happen but guard anyway.
            double s, e;
            if (firstRow[w] >= 0)
            {
                s = firstRow[w] * secondsPerStep;
                e = (lastRow[w] + 1) * secondsPerStep;
            }
            else
            {
                // Theoretical: a word the DP couldn't visit. Pin to
                // chunk end as a zero-duration entry so the consumer
                // still has a 1:1 list.
                s = e = (double)totalAudioSamples / sampleRate;
            }
            result.Add(new WordTiming(words[w].Text, s, e));
        }
        return result;
    }

    private sealed record SourceWord(string Text, int CharStart, int CharEnd);

    private static List<SourceWord> SplitIntoWords(string text)
    {
        var result = new List<SourceWord>();
        int i = 0;
        int n = text.Length;
        while (i < n)
        {
            // Skip leading whitespace.
            while (i < n && char.IsWhiteSpace(text[i])) i++;
            if (i >= n) break;
            int wordStart = i;
            while (i < n && !char.IsWhiteSpace(text[i])) i++;
            result.Add(new SourceWord(text.Substring(wordStart, i - wordStart), wordStart, i));
        }
        return result;
    }

    private static int FindWordContaining(List<SourceWord> words, int srcChar)
    {
        // Words list is short (typically <100 per chunk); linear is fine.
        // If srcChar falls in whitespace between words, attribute it to
        // the next word (the BPE [SPACE] token's char position is the
        // gap itself; assigning to the trailing word is intuitive for
        // word-highlight UX).
        for (int w = 0; w < words.Count; w++)
        {
            if (srcChar < words[w].CharStart) return w;     // in pre-word whitespace → next word
            if (srcChar < words[w].CharEnd) return w;       // inside this word
        }
        return -1;  // past the end (shouldn't happen if spans are correct)
    }
}
