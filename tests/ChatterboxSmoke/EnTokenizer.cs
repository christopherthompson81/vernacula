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
// Verified equivalent to the upstream EnTokenizer on the canonical Ezreal
// sentence (see EnTokenizerSelfTest). Only the encode path is implemented;
// decode is unused by the smoke test orchestration.

using System.Text.Json;

namespace Vernacula.ChatterboxSmoke;

internal sealed class EnTokenizer
{
    // Special tokens that wrap the BPE-encoded text in the full LM input.
    // Match scripts/chatterbox_export/_common.py + listen_test.py's INPUT_IDS layout.
    public const long StartTextToken = 255;       // [START] in vocab
    public const long StopTextToken = 0;          // [STOP] in vocab
    public const long ExaggerationToken = 6563;   // _common.py::EXAGGERATION_TOKEN
    public const long StartSpeechToken = 6561;    // _common.py::START_SPEECH_TOKEN

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
                throw new InvalidOperationException(
                    $"BPE produced token \"{t}\" not in vocab — likely an unknown character. "
                    + "Upstream EnTokenizer would emit [UNK] here; not implemented.");
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
