using System.Text;
using System.Text.RegularExpressions;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.Tts.Base;

/// <summary>
/// Cuts text into pieces that each stay inside Kokoro's 512-token context window, measuring
/// with the G2P frontend rather than with characters — the budget is in PHONEMES, and how many
/// phonemes a paragraph becomes is not a function of how long it is written.
///
/// <para>
/// Lifted out of <see cref="KokoroTts"/> so it can run without an ONNX session. The ONNX path
/// needs the split because the graph's position embedding fails above the window; the audio.cpp
/// path needs it because the engine takes a caller's phonemes as a LIST of chunks, each at most
/// 510 symbols, and only the caller's own G2P knows where the cuts belong. One chunker keeps
/// those two answers identical, which is the whole reason the same text says the same thing on
/// either engine.
/// </para>
/// </summary>
public sealed class KokoroChunker
{
    private readonly KokoroPhonemizer _g2p;

    public KokoroChunker(KokoroPhonemizer g2p) => _g2p = g2p;

    // Kokoro's BERT context_length is 512 tokens (incl. 2 pad); a chunk over that fails
    // the graph's position-embedding Expand. Pack to a conservative budget, then verify
    // each piece against the hard limit and resplit if the packing approximation slipped.
    private const int PackBudgetTokens = 460;   // target inner tokens when packing
    private const int HardInnerLimit = 508;     // never exceed 510 (= 512 - 2 pad); margin of 2

    private static readonly Regex SentenceSplitRe = new(@"(?<=[.!?])\s+", RegexOptions.Compiled);
    // Clause boundaries for over-long sentences — split after , ; : — (keeping the mark with
    // the preceding clause so the seam lands on a Kokoro pause token at a natural place).
    private static readonly Regex ClauseSplitRe = new(@"(?<=[,;:—])\s+", RegexOptions.Compiled);

    /// <summary>
    /// Split <paramref name="text"/> into synthesis chunks that each stay within Kokoro's
    /// 512-token context window. Paragraph/char chunking first (<see cref="ParagraphChunker"/>),
    /// then any chunk still over the token budget is sub-split on sentence — and, if a single
    /// sentence is still too long, word — boundaries. All splits are at whitespace, so the
    /// concatenated word sequence is unchanged (alignment stays 1:1 with the source text).
    /// </summary>
    public IReadOnlyList<string> ChunkForSynthesis(string text, bool british = false)
    {
        var pieces = new List<string>();
        foreach (var chunk in ParagraphChunker.Chunk(text))
            SplitToTokenBudget(chunk, british, pieces);
        return pieces;
    }

    /// <summary>Inner phoneme-token count (excludes the 2 pad tokens) for <paramref name="text"/>.</summary>
    public int CountTokens(string text, bool british = false) => _g2p.CountTokens(text, british);

    private void SplitToTokenBudget(string chunk, bool british, List<string> output)
    {
        if (CountTokens(chunk, british) <= HardInnerLimit) { output.Add(chunk); return; }
        // Sentence-level packing; an over-budget sentence descends to clause level.
        PackToBudget(SentenceSplitRe.Split(chunk), british, output, SplitClausesToBudget);
    }

    // Over-budget sentence → split on clause boundaries (commas etc.) so the seam falls at a
    // natural pause; a single over-budget clause descends to word level.
    private void SplitClausesToBudget(string sentence, bool british, List<string> output)
        => PackToBudget(ClauseSplitRe.Split(sentence), british, output, SplitWordsToBudget);

    // Last resort — pack individual words (no further fallback; a lone giant word, which
    // shouldn't occur, is accepted by EmitVerified).
    private void SplitWordsToBudget(string clause, bool british, List<string> output)
        => PackToBudget(clause.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries),
            british, output, overBudget: null);

    /// <summary>
    /// Greedily pack <paramref name="segments"/> into pieces of ≤ <see cref="PackBudgetTokens"/>
    /// inner tokens. The cost of joining a segment includes the inter-segment space token (+1) —
    /// the chunk's real token count has one space per gap, which a naive sum of isolated counts
    /// misses. A segment that alone exceeds the budget is handed to <paramref name="overBudget"/>
    /// (the next finer split); every emitted piece passes through <see cref="EmitVerified"/>.
    /// </summary>
    private void PackToBudget(IEnumerable<string> segments, bool british, List<string> output,
        Action<string, bool, List<string>>? overBudget)
    {
        var buf = new StringBuilder();
        var bufTokens = 0;
        void Flush() { if (buf.Length > 0) { EmitVerified(buf.ToString(), british, output); buf.Clear(); bufTokens = 0; } }

        foreach (var raw in segments)
        {
            var s = raw.Trim();
            if (s.Length == 0) continue;
            var st = CountTokens(s, british);
            if (st > PackBudgetTokens && overBudget is not null) { Flush(); overBudget(s, british, output); continue; }
            var cost = st + (buf.Length > 0 ? 1 : 0);
            if (bufTokens > 0 && bufTokens + cost > PackBudgetTokens) { Flush(); cost = st; }
            if (buf.Length > 0) buf.Append(' ');
            buf.Append(s);
            bufTokens += cost;
        }
        Flush();
    }

    // Safety net: emit a piece, but if the packing approximation under-counted and the
    // piece's real token count exceeds the hard limit, halve it on word boundaries.
    private void EmitVerified(string piece, bool british, List<string> output)
    {
        if (CountTokens(piece, british) <= HardInnerLimit) { output.Add(piece); return; }
        var words = piece.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length <= 1) { output.Add(piece); return; }  // can't split further
        var mid = words.Length / 2;
        EmitVerified(string.Join(' ', words[..mid]), british, output);
        EmitVerified(string.Join(' ', words[mid..]), british, output);
    }
}
