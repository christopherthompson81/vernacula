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

        // For each speech row, compute the word index it lands on (or -1
        // if argmax falls in a wrapper column / on a SPACE token / on
        // BPE that doesn't fall in any word's char range).
        var rowWord = new int[speechRows];
        for (int r = 0; r < speechRows; r++)
        {
            if (r == 0) { rowWord[r] = -1; continue; }  // sentinel predictor; skip
            int argCol = ArgmaxRow(alignment.Row(r));
            int bpeIdx = argCol - bpeColStart;
            if (bpeIdx < 0 || bpeIdx >= bpeLen) { rowWord[r] = -1; continue; }
            var span = bpeWithSpans[bpeIdx];
            // Midpoint of the BPE token's source span — robust to BPE
            // pieces that straddle word boundaries (rare in this vocab).
            int srcChar = (span.SourceStart + span.SourceEnd) / 2;
            rowWord[r] = FindWordContaining(words, srcChar);
        }

        // Aggregate per input word: min/max row that argmaxed to it.
        // Words with no row entries get zero-duration timings inserted
        // between their neighbours' spans (so the consumer always has
        // one entry per input word, even for short/skipped words).
        var perWord = new (int firstRow, int lastRow)[words.Count];
        for (int w = 0; w < perWord.Length; w++) perWord[w] = (-1, -1);
        for (int r = 0; r < speechRows; r++)
        {
            int w = rowWord[r];
            if (w < 0) continue;
            if (perWord[w].firstRow < 0) perWord[w] = (r, r);
            else perWord[w] = (perWord[w].firstRow, r);
        }

        // Emit one WordTiming per input word. For words that argmax'd
        // somewhere: timing = (firstRow, lastRow + 1). For un-seen
        // words: interpolate between the nearest seen neighbours,
        // preserving input-order monotonicity.
        var result = new List<WordTiming>(words.Count);
        for (int w = 0; w < words.Count; w++)
        {
            double startSec, endSec;
            if (perWord[w].firstRow >= 0)
            {
                startSec = perWord[w].firstRow * secondsPerStep;
                endSec = (perWord[w].lastRow + 1) * secondsPerStep;
            }
            else
            {
                // Find previous seen word's end and next seen word's
                // start; place this word in the gap. If no prior word
                // is seen, pin to the next's start; if no next, pin to
                // prior's end. As a final fallback, mid-audio.
                double prevEnd = -1, nextStart = -1;
                for (int u = w - 1; u >= 0; u--)
                    if (perWord[u].lastRow >= 0) { prevEnd = (perWord[u].lastRow + 1) * secondsPerStep; break; }
                for (int u = w + 1; u < words.Count; u++)
                    if (perWord[u].firstRow >= 0) { nextStart = perWord[u].firstRow * secondsPerStep; break; }
                if (prevEnd >= 0 && nextStart >= 0) { startSec = prevEnd; endSec = nextStart; }
                else if (prevEnd >= 0) { startSec = prevEnd; endSec = prevEnd; }
                else if (nextStart >= 0) { startSec = nextStart; endSec = nextStart; }
                else { startSec = 0; endSec = 0; }
            }
            result.Add(new WordTiming(words[w].Text, startSec, endSec));
        }
        return result;
    }

    private static int ArgmaxRow(ReadOnlySpan<float> row)
    {
        int best = 0;
        float bv = row[0];
        for (int c = 1; c < row.Length; c++)
            if (row[c] > bv) { bv = row[c]; best = c; }
        return best;
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
