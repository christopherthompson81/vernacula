namespace Vernacula.Base.Alignment;

/// <summary>
/// Minimal greedy longest-prefix-match sentencepiece-style tokenizer
/// over <c>vocab.txt</c>. Stopgap for forced-alignment input encoding
/// while we wait for a working unigram-SP loader in
/// <c>Microsoft.ML.Tokenizers</c> — version 2.0.0's
/// <c>SentencePieceTokenizer.Create</c> throws <c>IndexOutOfRangeException</c>
/// on the NeMo FastConformer's SP model (the loader's
/// <c>SentencePieceUnigramModel..ctor</c> assumes BOS/EOS pieces the
/// NeMo proto doesn't define).
///
/// Behavior:
/// - Lowercases NOTHING (matches the <c>_pc</c> "punctuation + capitalization"
///   models the export targets).
/// - Splits the input on Unicode whitespace; prepends the SP word-boundary
///   marker <c>▁</c> (U+2581) to the first piece of each word.
/// - Greedy longest-prefix-match against vocab.txt entries. Falls back to
///   the unknown token id when no prefix matches (skips the char so the
///   tokenization makes forward progress; a more faithful byte-fallback
///   would split into UTF-8 bytes and look up <c>&lt;0xXX&gt;</c> tokens).
///
/// Quality envelope: for clean in-domain English (the
/// <c>stt_en_fastconformer_hybrid_large_pc</c> domain), greedy LPM
/// matches the unigram-optimal tokenization most of the time; the worst
/// case is occasional sub-optimal piece breaks (e.g. splitting "evening"
/// as ▁ev/ening instead of ▁eve/ning) that change a few CTC-frame
/// boundaries. Acceptable for word-level alignment; not appropriate
/// for downstream uses needing exact-tokenization parity with the
/// model's training pipeline.
/// </summary>
public sealed class SimpleSpTokenizer
{
    // U+2581 LOWER ONE EIGHTH BLOCK = sentencepiece word boundary
    private const char SpWordBoundary = '▁';

    private readonly Dictionary<string, int> _pieceToId;
    private readonly string[] _idToPiece;
    private readonly int _unknownId;

    /// <summary>Number of vocabulary entries (excludes the CTC blank,
    /// which lives at index <see cref="VocabSize"/> in the model's output
    /// but isn't part of this tokenizer's surface).</summary>
    public int VocabSize => _idToPiece.Length;

    /// <summary>Build from vocab.txt. Expected format: one line per
    /// piece, <c>{piece}{ws}{id}</c>. The trailing blank entry (id ==
    /// vocab_size) is recognized and excluded from the tokenizer's
    /// surface; the caller uses it as the CTC blank id directly.</summary>
    public SimpleSpTokenizer(string vocabPath, string blankToken = "<blk>", string unknownToken = "<unk>")
    {
        var lines = File.ReadAllLines(vocabPath)
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToList();
        // Build id → piece array; the file is already sorted by id so we
        // just track the max id we see and pad. Skip the blank entry.
        var pairs = new List<(int Id, string Piece)>(lines.Count);
        foreach (var line in lines)
        {
            int sp = line.LastIndexOf(' ');
            if (sp < 0) continue;
            string piece = line.Substring(0, sp);
            if (!int.TryParse(line.AsSpan(sp + 1), out int id)) continue;
            if (piece == blankToken) continue;
            pairs.Add((id, piece));
        }
        int maxId = pairs.Max(p => p.Id);
        _idToPiece = new string[maxId + 1];
        foreach (var (id, piece) in pairs) _idToPiece[id] = piece;
        _pieceToId = new Dictionary<string, int>(pairs.Count, StringComparer.Ordinal);
        foreach (var (id, piece) in pairs) _pieceToId[piece] = id;

        _unknownId = _pieceToId.TryGetValue(unknownToken, out var unkId) ? unkId : 0;
    }

    /// <summary>Encode <paramref name="text"/> to (piece_id, piece_text) pairs.
    /// Whitespace becomes the SP <c>▁</c> word-boundary marker on the next
    /// piece. Punctuation is treated as part of the current word for the
    /// purposes of greedy matching (the SP vocab contains pieces with and
    /// without leading punctuation; LPM lets the longer matches win).</summary>
    public List<(int Id, string Piece)> Encode(string text)
    {
        var result = new List<(int Id, string Piece)>();
        if (string.IsNullOrEmpty(text)) return result;

        // Split on whitespace runs into "words" (any non-whitespace span).
        int i = 0;
        int n = text.Length;
        while (i < n)
        {
            // Skip leading whitespace.
            while (i < n && char.IsWhiteSpace(text[i])) i++;
            if (i >= n) break;
            // Find the end of this whitespace-delimited span.
            int wordStart = i;
            while (i < n && !char.IsWhiteSpace(text[i])) i++;
            int wordEnd = i;
            // Prepend ▁ to mark the word start, then greedy LPM through it.
            EncodeWordWithBoundary(text.AsSpan(wordStart, wordEnd - wordStart), result);
        }
        return result;
    }

    private void EncodeWordWithBoundary(ReadOnlySpan<char> word, List<(int Id, string Piece)> result)
    {
        // Strategy: build a working buffer that starts with ▁ and contains
        // the word; greedy-match longest prefix against vocab; emit; repeat
        // from the consumed offset.
        string buf = SpWordBoundary + word.ToString();
        int pos = 0;
        while (pos < buf.Length)
        {
            int matchLen = LongestPrefixMatch(buf, pos);
            if (matchLen == 0)
            {
                // No vocab piece matches even the single char at pos. Emit
                // the unknown token and advance one char. (A faithful
                // byte-fallback would emit one <0xXX> piece per UTF-8 byte;
                // we trade fidelity for simplicity here since
                // English-domain inputs hit this branch ~never.)
                result.Add((_unknownId, _idToPiece[_unknownId] ?? "<unk>"));
                pos++;
                continue;
            }
            string piece = buf.Substring(pos, matchLen);
            result.Add((_pieceToId[piece], piece));
            pos += matchLen;
        }
    }

    private int LongestPrefixMatch(string text, int start)
    {
        // Scan from longest possible match down to 1 char. The SP vocab
        // has a maximum piece length (typically <= 16 chars for English),
        // so this O(maxLen × dict-lookup) per piece is fine.
        int maxTry = Math.Min(32, text.Length - start);
        for (int len = maxTry; len >= 1; len--)
        {
            if (_pieceToId.ContainsKey(text.Substring(start, len)))
                return len;
        }
        return 0;
    }
}
