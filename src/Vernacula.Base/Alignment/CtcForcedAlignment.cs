namespace Vernacula.Base.Alignment;

/// <summary>
/// One token's frame span in a CTC Viterbi alignment.
/// </summary>
/// <param name="TokenId">The token's id in the model's vocabulary.</param>
/// <param name="TokenPosition">Index in the original (un-blanked) target
/// sequence — useful for matching back to per-token metadata like word
/// boundaries.</param>
/// <param name="StartFrame">First frame (inclusive) the Viterbi backtrack
/// assigned to this token.</param>
/// <param name="EndFrame">Last frame (inclusive) the backtrack assigned
/// to this token.</param>
public sealed record TokenAlignment(int TokenId, int TokenPosition, int StartFrame, int EndFrame);

/// <summary>
/// CTC forced-alignment Viterbi DP — the algorithmic core of NFA, MMS
/// forced aligner, torchaudio.functional.forced_align, etc. All those
/// systems run the same algorithm over CTC log-probs; only the
/// acoustic model differs.
///
/// Given:
///   <see cref="LogProbs"/>: [T, V+1] log-softmax over vocab + blank
///   target token sequence (no blanks): the reference text tokenized
///   blank_id: which column is the CTC blank (conventionally V)
///
/// Returns: one <see cref="TokenAlignment"/> per target token, with
/// the frames assigned to it by the optimal-path backtrack. Per-token
/// frames are contiguous and non-overlapping; tokens may be assigned
/// zero frames (degenerate; happens when the optimal path emits the
/// token at a single frame between blanks).
///
/// Algorithm summary (the "modified Viterbi" for CTC):
/// - Build a blanked target of length 2N+1: [blank, t0, blank, t1, …, t_{N-1}, blank]
/// - DP: log_alpha[t, s] = log_probs[t, blankedTarget[s]] + max of:
///   - log_alpha[t-1, s]          (stay)
///   - log_alpha[t-1, s-1]        (advance)
///   - log_alpha[t-1, s-2]        (skip blank — only when current is
///                                 not blank and the prev-token ≠ this
///                                 token; otherwise CTC merging rules
///                                 forbid the skip)
/// - Final: max(log_alpha[T-1, 2N-1], log_alpha[T-1, 2N])
/// - Backtrack to recover per-frame state, then aggregate frames per
///   original-target-token position.
/// </summary>
public static class CtcForcedAlignment
{
    /// <summary>
    /// Run Viterbi forced alignment. <paramref name="logProbs"/> is a
    /// row-major [T, V+1] log-softmax matrix; <paramref name="numFrames"/>
    /// = T; <paramref name="vocabPlusBlank"/> = V+1.
    /// </summary>
    /// <param name="targetTokens">Target token ids (no blanks). Must be
    /// non-empty and all ids must be in [0, vocabPlusBlank) and != blankId.</param>
    /// <param name="blankId">CTC blank token id (conventionally V, i.e.
    /// vocabPlusBlank-1 — but pass it explicitly for clarity).</param>
    /// <exception cref="ArgumentException">When the inputs are
    /// internally inconsistent (empty target, target token equals blank,
    /// T &lt; required minimum length to fit the target).</exception>
    public static IReadOnlyList<TokenAlignment> Align(
        float[] logProbs, int numFrames, int vocabPlusBlank,
        IReadOnlyList<int> targetTokens, int blankId)
    {
        int n = targetTokens.Count;
        if (n == 0) throw new ArgumentException("targetTokens must be non-empty.", nameof(targetTokens));
        if (blankId < 0 || blankId >= vocabPlusBlank)
            throw new ArgumentException($"blankId {blankId} out of range [0,{vocabPlusBlank}).", nameof(blankId));
        for (int i = 0; i < n; i++)
        {
            int t = targetTokens[i];
            if (t < 0 || t >= vocabPlusBlank)
                throw new ArgumentException($"targetTokens[{i}] = {t} out of range.", nameof(targetTokens));
            if (t == blankId)
                throw new ArgumentException($"targetTokens[{i}] is the blank id — target must not contain blanks.", nameof(targetTokens));
        }
        // Minimum frames needed: each token needs ≥1 frame; same-token
        // pairs additionally need a blank between them (CTC merging rule).
        int minFrames = n;
        for (int i = 1; i < n; i++)
            if (targetTokens[i] == targetTokens[i - 1]) minFrames++;
        if (numFrames < minFrames)
            throw new ArgumentException(
                $"numFrames={numFrames} is too small to fit a {n}-token target requiring ≥{minFrames} frames "
                + "(same-token repeats need separating blanks).",
                nameof(numFrames));

        // Build the blanked target: 2N+1 states alternating blank/token.
        int s_n = 2 * n + 1;
        var blanked = new int[s_n];
        for (int s = 0; s < s_n; s++)
            blanked[s] = (s % 2 == 0) ? blankId : targetTokens[s / 2];

        const float NegInf = float.NegativeInfinity;
        // log_alpha[t, s] flattened as alpha[t * s_n + s].
        var alpha = new float[numFrames * s_n];
        for (int i = 0; i < alpha.Length; i++) alpha[i] = NegInf;
        // backptr[t, s] in {0=stay, 1=advance, 2=skip}; -1 means unreachable.
        // Stored as sbyte to keep the trellis small for long alignments.
        var backptr = new sbyte[numFrames * s_n];

        // Init (t=0): can be in state 0 (blank) or state 1 (first token).
        alpha[0 * s_n + 0] = logProbs[0 * vocabPlusBlank + blankId];
        backptr[0 * s_n + 0] = 0;
        alpha[0 * s_n + 1] = logProbs[0 * vocabPlusBlank + blanked[1]];
        backptr[0 * s_n + 1] = 0;

        // DP.
        for (int t = 1; t < numFrames; t++)
        {
            int alphaRow = t * s_n;
            int prevRow = (t - 1) * s_n;
            int logRow = t * vocabPlusBlank;
            for (int s = 0; s < s_n; s++)
            {
                float stay = alpha[prevRow + s];
                float advance = (s >= 1) ? alpha[prevRow + s - 1] : NegInf;
                bool canSkip = s >= 2
                    && blanked[s] != blankId
                    && blanked[s] != blanked[s - 2];
                float skip = canSkip ? alpha[prevRow + s - 2] : NegInf;

                float best = stay;
                sbyte choice = 0;
                if (advance > best) { best = advance; choice = 1; }
                if (skip > best) { best = skip; choice = 2; }
                if (float.IsNegativeInfinity(best))
                {
                    backptr[alphaRow + s] = -1;
                    continue;
                }
                alpha[alphaRow + s] = best + logProbs[logRow + blanked[s]];
                backptr[alphaRow + s] = choice;
            }
        }

        // Pick the better of the two valid final states (final token or
        // trailing blank).
        int sLastTok = 2 * n - 1;
        int sLastBlank = 2 * n;
        int lastRow = (numFrames - 1) * s_n;
        int finalState = alpha[lastRow + sLastTok] >= alpha[lastRow + sLastBlank] ? sLastTok : sLastBlank;
        if (float.IsNegativeInfinity(alpha[lastRow + finalState]))
            throw new InvalidOperationException(
                "Viterbi DP found no valid path. This usually means logProbs are degenerate "
                + "(all -inf) or numFrames is too small for the target.");

        // Backtrack: record state per frame.
        var statePerFrame = new int[numFrames];
        int state = finalState;
        statePerFrame[numFrames - 1] = state;
        for (int t = numFrames - 1; t > 0; t--)
        {
            sbyte choice = backptr[t * s_n + state];
            state -= choice;  // 0=stay, 1=advance, 2=skip
            statePerFrame[t - 1] = state;
        }

        // Aggregate frames per token-position. A token at original index k
        // owns the frames where statePerFrame == 2k+1.
        var startFrame = new int[n];
        var endFrame = new int[n];
        for (int k = 0; k < n; k++) { startFrame[k] = -1; endFrame[k] = -1; }
        for (int t = 0; t < numFrames; t++)
        {
            int s = statePerFrame[t];
            if (s % 2 == 1)  // odd state = token
            {
                int k = s / 2;
                if (startFrame[k] < 0) startFrame[k] = t;
                endFrame[k] = t;
            }
        }

        var result = new TokenAlignment[n];
        for (int k = 0; k < n; k++)
        {
            // A token with no frames assigned can happen if the Viterbi
            // path skipped through it as a blank-flanked advance; we
            // surface the degenerate case (start=end=-1 wouldn't be
            // meaningful) by collapsing to the frame just before the
            // next token took over, but in practice CTC Viterbi over
            // a feasible target always assigns ≥1 frame per token.
            int sf = startFrame[k] >= 0 ? startFrame[k] : (k > 0 ? endFrame[k - 1] : 0);
            int ef = endFrame[k] >= 0 ? endFrame[k] : sf;
            result[k] = new TokenAlignment(targetTokens[k], k, sf, ef);
        }
        return result;
    }
}
