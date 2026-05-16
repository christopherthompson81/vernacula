// C# port of chatterbox.models.tokenizers.EnTokenizer (upstream wraps the
// HuggingFace `tokenizers` Rust crate). The model is a small BPE tokenizer
// (704 vocab, 265 merges, Whitespace pre-tokenizer, no normalizer), loaded
// from a tokenizer.json file. The upstream `encode` does just one extra
// step before BPE: replace literal ' ' with the `[SPACE]` added-token text,
// so spaces become a distinct token id rather than affecting word
// boundaries.
//
// Algorithm:
//   1. text.Replace(" ", "[SPACE]")
//   2. Split on added-token spans (longest-match), emitting their token IDs
//      directly. The text between special spans goes through BPE.
//   3. BPE encode each non-special chunk:
//      a. Decompose into single-char tokens.
//      b. Repeatedly merge adjacent pairs using the priority order from
//         tokenizer.json's merges list (lower rank = higher priority).
//      c. Map final tokens to vocab IDs.
//
// Pre-tokenizer skip: tokenizer.json specifies `pre_tokenizer.type =
// "Whitespace"` (HF's regex split on `\w+|[^\w\s]+`). This implementation
// intentionally skips that step. It's load-bearing safe ONLY because none
// of chatterbox's 265 BPE merges span a word/punct character-class
// boundary — checked at construction time via AssertNoBoundarySpanningMerges.
// If a future tokenizer.json adds such a merge, that assertion fails loudly
// and forces re-evaluation; we don't silently corrupt the output.
//
// Verified equivalent to the upstream EnTokenizer on the canonical Ezreal
// sentence. Only the encode path is implemented; decode is unused by the
// smoke test orchestration.

using System.Text.Json;
using System.Text.RegularExpressions;

namespace Vernacula.ChatterboxSmoke;

internal sealed class EnTokenizer
{
    // LM-vocab IDs that wrap the BPE-encoded text in the full LM input.
    // These live in the LM's vocabulary (>6000), outside the text-tokenizer
    // vocab — hardcoded because they're not in tokenizer.json.
    public const long ExaggerationToken = 6563;   // _common.py::EXAGGERATION_TOKEN
    public const long StartSpeechToken = 6561;    // _common.py::START_SPEECH_TOKEN

    // Text-vocab special tokens — derived from vocab in the ctor so they
    // tolerate upstream renumbering (originally 255 / 0 in chatterbox).
    public long StartTextToken { get; }    // [START]
    public long StopTextToken { get; }     // [STOP]

    private readonly Dictionary<string, long> _vocab;
    private readonly Dictionary<(string, string), int> _mergeRank;  // pair → rank (lower = higher priority)
    private readonly List<(string text, long id)> _addedTokensByLongestFirst;

    public EnTokenizer(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root = doc.RootElement;
        var model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE")
            throw new InvalidOperationException("EnTokenizer only supports BPE tokenizer.json");

        // vocab: { "token_string": id }
        _vocab = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var prop in model.GetProperty("vocab").EnumerateObject())
            _vocab[prop.Name] = prop.Value.GetInt64();

        // merges: ["a b", "c d", ...] in priority order; rank = position in list
        var mergesArr = model.GetProperty("merges");
        _mergeRank = new Dictionary<(string, string), int>(mergesArr.GetArrayLength());
        int rank = 0;
        foreach (var m in mergesArr.EnumerateArray())
        {
            var parts = m.GetString()!.Split(' ', 2);
            _mergeRank[(parts[0], parts[1])] = rank++;
        }

        // added_tokens: special tokens recognized as-is in the input, e.g.
        // [STOP]=0, [UNK]=1, [SPACE]=2, [START]=255, [UH]=604.
        // Order by length DESC so longest-match split works (avoids splitting
        // "[SPACE]" mid-token if a shorter prefix were ambiguous).
        var added = new List<(string text, long id)>();
        if (root.TryGetProperty("added_tokens", out var addedArr))
        {
            foreach (var t in addedArr.EnumerateArray())
            {
                added.Add((t.GetProperty("content").GetString()!, t.GetProperty("id").GetInt64()));
            }
        }
        _addedTokensByLongestFirst = added.OrderByDescending(t => t.text.Length).ToList();

        // Derive text-vocab special token IDs from the loaded vocab so we
        // automatically pick up any upstream renumbering.
        if (!_vocab.TryGetValue("[START]", out var startId))
            throw new InvalidOperationException("tokenizer.json vocab is missing [START]");
        if (!_vocab.TryGetValue("[STOP]", out var stopId))
            throw new InvalidOperationException("tokenizer.json vocab is missing [STOP]");
        StartTextToken = startId;
        StopTextToken = stopId;

        AssertNoBoundarySpanningMerges();
    }

    /// <summary>
    /// Pre-tokenizer-skip safety check. HF's Whitespace pre-tokenizer splits
    /// input on the regex `\w+|[^\w\s]+`, so a merge like `i ,` (word + punct)
    /// could only fire in HF if input had no whitespace between them — but the
    /// pre-tokenizer prevents that. We skip the pre-tokenizer entirely (Encode
    /// only splits on added-token spans), so we'd silently apply such a merge
    /// when HF wouldn't. Chatterbox's 265 merges happen to never span a word/
    /// punct boundary, so the skip is currently safe. Fail loudly if that
    /// invariant ever breaks (new model rev, different tokenizer.json).
    /// </summary>
    private void AssertNoBoundarySpanningMerges()
    {
        var specials = _addedTokensByLongestFirst.Select(t => t.text).ToHashSet();
        var wordOnly = new Regex(@"^\w+$");
        var punctOnly = new Regex(@"^[^\w\s]+$");
        foreach (var ((a, b), _) in _mergeRank)
        {
            if (specials.Contains(a) || specials.Contains(b)) continue;
            bool aWord = wordOnly.IsMatch(a), aPunct = punctOnly.IsMatch(a);
            bool bWord = wordOnly.IsMatch(b), bPunct = punctOnly.IsMatch(b);
            if ((aWord && bPunct) || (aPunct && bWord))
                throw new InvalidOperationException(
                    $"BPE merge {{\"{a}\" \"{b}\"}} spans a word/punct boundary. "
                    + "Skipping HF's Whitespace pre-tokenizer would diverge from upstream "
                    + "here; re-add the pre-tokenizer before using this tokenizer.json.");
        }
    }

    /// <summary>
    /// BPE-encode `text` into vocab IDs. Matches upstream EnTokenizer.encode:
    /// replaces literal spaces with [SPACE] first, then BPE.
    /// </summary>
    public long[] Encode(string text)
    {
        text = text.Replace(" ", "[SPACE]");
        var ids = new List<long>(text.Length);
        int pos = 0;
        while (pos < text.Length)
        {
            // Try to match an added (special) token at pos.
            long? specialId = null;
            int specialLen = 0;
            foreach (var (special, id) in _addedTokensByLongestFirst)
            {
                if (pos + special.Length <= text.Length
                    && string.CompareOrdinal(text, pos, special, 0, special.Length) == 0)
                {
                    specialId = id;
                    specialLen = special.Length;
                    break;
                }
            }
            if (specialId is not null)
            {
                ids.Add(specialId.Value);
                pos += specialLen;
                continue;
            }
            // Otherwise consume a run of ordinary characters up to the next
            // special-token boundary, then BPE-encode that run.
            int runStart = pos;
            while (pos < text.Length)
            {
                bool atSpecial = false;
                foreach (var (special, _) in _addedTokensByLongestFirst)
                {
                    if (pos + special.Length <= text.Length
                        && string.CompareOrdinal(text, pos, special, 0, special.Length) == 0)
                    {
                        atSpecial = true;
                        break;
                    }
                }
                if (atSpecial) break;
                pos++;
            }
            if (pos > runStart)
                BpeEncodeRun(text.AsSpan(runStart, pos - runStart), ids);
        }
        return ids.ToArray();
    }

    /// <summary>
    /// Standard BPE: start from single-character tokens, repeatedly merge the
    /// adjacent pair with the lowest rank in `_mergeRank` until no in-vocab
    /// pair remains, then map tokens to IDs.
    /// </summary>
    private void BpeEncodeRun(ReadOnlySpan<char> run, List<long> outIds)
    {
        // Initial token decomposition: one token per character (as a string).
        var tokens = new List<string>(run.Length);
        foreach (var ch in run) tokens.Add(ch.ToString());

        while (tokens.Count >= 2)
        {
            int bestRank = int.MaxValue;
            int bestI = -1;
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                if (_mergeRank.TryGetValue((tokens[i], tokens[i + 1]), out int rank)
                    && rank < bestRank)
                {
                    bestRank = rank;
                    bestI = i;
                }
            }
            if (bestI < 0) break;
            // Merge all occurrences of the best pair in one pass.
            var merged = new List<string>(tokens.Count - 1);
            int j = 0;
            while (j < tokens.Count)
            {
                if (j + 1 < tokens.Count
                    && _mergeRank.TryGetValue((tokens[j], tokens[j + 1]), out int r)
                    && r == bestRank)
                {
                    merged.Add(tokens[j] + tokens[j + 1]);
                    j += 2;
                }
                else
                {
                    merged.Add(tokens[j]);
                    j++;
                }
            }
            tokens = merged;
        }

        foreach (var t in tokens)
        {
            if (!_vocab.TryGetValue(t, out var id))
            {
                // TODO([UNK] fallback): upstream HF tokenizers emit [UNK]
                // (id 1) for chars not in the BPE base alphabet — typically
                // anything beyond ASCII letters, digits, basic punctuation
                // (emoji, smart quotes, non-Latin scripts). Throwing here is
                // safer for the smoke test (loud failure beats silent wrong
                // output) but needs to become a soft [UNK] before we use this
                // tokenizer for arbitrary user input.
                throw new InvalidOperationException(
                    $"BPE produced token \"{t}\" not in vocab — likely an unknown character. "
                    + "Upstream EnTokenizer would emit [UNK] here; not implemented.");
            }
            outIds.Add(id);
        }
    }

    /// <summary>
    /// Build the full LM input sequence in the layout chatterbox's
    /// listen_test uses:
    ///   [EXAGGERATION, START, ...BPE(text)..., STOP, START_SPEECH, START_SPEECH]
    /// </summary>
    public long[] WrapForLm(string text)
    {
        var bpe = Encode(text);
        var wrapped = new long[bpe.Length + 5];
        wrapped[0] = ExaggerationToken;
        wrapped[1] = StartTextToken;
        Array.Copy(bpe, 0, wrapped, 2, bpe.Length);
        wrapped[^3] = StopTextToken;
        wrapped[^2] = StartSpeechToken;
        wrapped[^1] = StartSpeechToken;
        return wrapped;
    }
}
