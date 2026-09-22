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

    // ⚠ CJK TERMINATORS TOO. A Japanese or Chinese paragraph ends its sentences with 。！？ and no
    // following space, so the ASCII-only pattern matched nothing and an over-budget paragraph fell
    // straight past sentence and clause splitting to the word level -- which, in a script without
    // spaces, is one "word" the length of the paragraph.
    private static readonly Regex SentenceSplitRe =
        new(@"(?<=[.!?])\s+|(?<=[\u3002\uFF01\uFF1F])", RegexOptions.Compiled);
    // Clause boundaries for over-long sentences — split after , ; : — (keeping the mark with
    // the preceding clause so the seam lands on a Kokoro pause token at a natural place).
    private static readonly Regex ClauseSplitRe =
        new(@"(?<=[,;:—])\s+|(?<=[\u3001\uFF0C\uFF1B])", RegexOptions.Compiled);

    /// <summary>
    /// Split <paramref name="text"/> into synthesis chunks that each stay within Kokoro's
    /// 512-token context window. Paragraph/char chunking first (<see cref="ParagraphChunker"/>),
    /// then any chunk still over the token budget is sub-split on sentence — and, if a single
    /// sentence is still too long, word — boundaries. All splits are at whitespace, so the
    /// concatenated word sequence is unchanged (alignment stays 1:1 with the source text).
    /// </summary>
    public IReadOnlyList<string> ChunkForSynthesis(string text, bool british = false)
        => ChunkForSynthesis(text, british ? "en-GB" : "en");

    /// <inheritdoc cref="ChunkForSynthesis(string, bool)"/>
    public IReadOnlyList<string> ChunkForSynthesis(string text, string lang)
    {
        var pieces = new List<string>();
        foreach (var chunk in ParagraphChunker.Chunk(text))
            SplitToTokenBudget(chunk, lang, pieces);
        return pieces;
    }

    /// <summary>Inner phoneme-token count (excludes the 2 pad tokens) for <paramref name="text"/>.</summary>
    public int CountTokens(string text, bool british = false) => _g2p.CountTokens(text, british);

    /// <inheritdoc cref="CountTokens(string, bool)"/>
    public int CountTokens(string text, string lang) => _g2p.CountTokens(text, lang);

    private void SplitToTokenBudget(string chunk, string lang, List<string> output)
    {
        if (CountTokens(chunk, lang) <= HardInnerLimit) { output.Add(chunk); return; }
        // Sentence-level packing; an over-budget sentence descends to clause level.
        PackToBudget(SentenceSplitRe.Split(chunk), lang, output, SplitClausesToBudget);
    }

    // Over-budget sentence → split on clause boundaries (commas etc.) so the seam falls at a
    // natural pause; a single over-budget clause descends to word level.
    private void SplitClausesToBudget(string sentence, string lang, List<string> output)
        => PackToBudget(ClauseSplitRe.Split(sentence), lang, output, SplitWordsToBudget);

    // Last resort — pack individual words (no further fallback; a lone giant word, which
    // shouldn't occur, is accepted by EmitVerified).
    private void SplitWordsToBudget(string clause, string lang, List<string> output)
    {
        var words = clause.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        // ⚠ A SCRIPT WITHOUT SPACES HAS ONE "WORD" HERE, so packing them changes nothing and the
        // over-budget piece is emitted whole. Falling to characters is the only split left, and it
        // is a legitimate one: the entries are merged back into one buffer, so a cut mid-phrase
        // costs prosody at the seam rather than correctness.
        if (words.Length <= 1 && clause.Length > 1)
        {
            PackToBudget(clause.Select(c => c.ToString()), lang, output, overBudget: null);
            return;
        }
        PackToBudget(words, lang, output, overBudget: null);
    }

    /// <summary>
    /// Greedily pack <paramref name="segments"/> into pieces of ≤ <see cref="PackBudgetTokens"/>
    /// inner tokens. The cost of joining a segment includes the inter-segment space token (+1) —
    /// the chunk's real token count has one space per gap, which a naive sum of isolated counts
    /// misses. A segment that alone exceeds the budget is handed to <paramref name="overBudget"/>
    /// (the next finer split); every emitted piece passes through <see cref="EmitVerified"/>.
    /// </summary>
    private void PackToBudget(IEnumerable<string> segments, string lang, List<string> output,
        Action<string, string, List<string>>? overBudget)
    {
        var buf = new StringBuilder();
        var bufTokens = 0;
        void Flush() { if (buf.Length > 0) { EmitVerified(buf.ToString(), lang, output); buf.Clear(); bufTokens = 0; } }

        foreach (var raw in segments)
        {
            var s = raw.Trim();
            if (s.Length == 0) continue;
            var st = CountTokens(s, lang);
            if (st > PackBudgetTokens && overBudget is not null) { Flush(); overBudget(s, lang, output); continue; }
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
    private void EmitVerified(string piece, string lang, List<string> output)
    {
        if (CountTokens(piece, lang) <= HardInnerLimit) { output.Add(piece); return; }
        var words = piece.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length <= 1)
        {
            // No whitespace to halve on: halve the characters instead, which is the only cut a
            // script without spaces leaves. A single character that is still over budget is
            // accepted, as a lone giant word always was.
            if (piece.Length <= 1) { output.Add(piece); return; }
            var half = piece.Length / 2;
            EmitVerified(piece[..half], lang, output);
            EmitVerified(piece[half..], lang, output);
            return;
        }
        var mid = words.Length / 2;
        EmitVerified(string.Join(' ', words[..mid]), lang, output);
        EmitVerified(string.Join(' ', words[mid..]), lang, output);
    }
}
