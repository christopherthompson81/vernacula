// Byte-level BPE encoder for the Qwen3 tokenizer (text -> token IDs), as used by
// OmniVoice. The repo's other Qwen/GPT-2-family tokenizers (Qwen3Asr, WhisperTurbo,
// GraniteSpeech, VibeVoiceAsr) are all decode-only or bake fixed token sequences
// offline; OmniVoice synthesises arbitrary user text, so it needs the real encoder.
//
// This composes three existing patterns:
//   - the GPT-2 bytes->unicode table (cf. WhisperTurbo's decode table, used here in
//     the encode direction),
//   - the merge-rank BPE inner loop (cf. EnTokenizer.FlushBpeRun), and
//   - tokenizer.json loading (cf. Qwen3Asr).
// New vs EnTokenizer: the GPT-2/Qwen split-regex pre-tokenizer + byte-encoding each
// piece into the 256-symbol byte alphabet (so every symbol is in vocab — no UNK),
// and NFC normalization.
//
// HuggingFace pipeline reproduced: NFC normalizer, ByteLevel pre-tokenizer with the
// Qwen Split regex, BPE model. add_special_tokens is a no-op (Qwen adds no auto BOS/EOS).
// Verified against scripts/omnivoice_export/dump_tokenizer_fixtures.py output by
// Vernacula.Tests/Qwen3TokenizerParityTests.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Vernacula.Tts.Base.Tokenization;

public sealed partial class Qwen3Tokenizer
{
    // Qwen / GPT-2 pre-tokenizer split pattern (from tokenizer.json
    // pre_tokenizer.Split.pattern). .NET regex is Unicode-aware for \p{L}, \p{N}, \s.
    [GeneratedRegex(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")]
    private static partial Regex SplitRegex();

    private readonly Dictionary<string, int> _vocab;                 // byte-char token string -> id
    private readonly Dictionary<(string, string), int> _mergeRank;   // pair -> rank (lower = higher priority)
    private readonly List<(string text, int id)> _addedLongestFirst; // added tokens, longest content first
    private readonly Dictionary<string, int> _addedByContent;        // content -> id
    private readonly char[] _byteToChar = new char[256];             // GPT-2 byte -> unicode char

    public Qwen3Tokenizer(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root = doc.RootElement;
        var model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE")
            throw new InvalidOperationException("Qwen3Tokenizer expects a BPE tokenizer.json");

        _vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var prop in model.GetProperty("vocab").EnumerateObject())
            _vocab[prop.Name] = prop.Value.GetInt32();

        // merges: array of ["a","b"] pairs in priority order (rank = index).
        var mergesArr = model.GetProperty("merges");
        _mergeRank = new Dictionary<(string, string), int>(mergesArr.GetArrayLength());
        int rank = 0;
        foreach (var m in mergesArr.EnumerateArray())
        {
            string a, b;
            if (m.ValueKind == JsonValueKind.Array)
            {
                a = m[0].GetString()!;
                b = m[1].GetString()!;
            }
            else
            {
                var parts = m.GetString()!.Split(' ', 2);
                a = parts[0];
                b = parts[1];
            }
            _mergeRank[(a, b)] = rank++;
        }

        // added_tokens: matched as whole units in the raw text (both special and
        // non-special; normalized:false means they match pre-normalization).
        var added = new List<(string text, int id)>();
        _addedByContent = new Dictionary<string, int>(StringComparer.Ordinal);
        if (root.TryGetProperty("added_tokens", out var addedArr))
        {
            foreach (var t in addedArr.EnumerateArray())
            {
                var content = t.GetProperty("content").GetString()!;
                int id = t.GetProperty("id").GetInt32();
                added.Add((content, id));
                _addedByContent[content] = id;
            }
        }
        _addedLongestFirst = added.OrderByDescending(t => t.text.Length).ToList();

        BuildByteToChar();
    }

    /// <summary>Resolve an added/special token id by its literal content, e.g. "&lt;|denoise|&gt;".</summary>
    public int TokenId(string content) =>
        _addedByContent.TryGetValue(content, out int id)
            ? id
            : throw new KeyNotFoundException($"Added token not found: {content}");

    /// <summary>
    /// Encode text to token IDs. Splits out added/special tokens (longest-match) on the
    /// raw text, then NFC-normalizes and byte-level-BPEs the gaps. add_special_tokens is
    /// accepted for call-site symmetry but is a no-op: Qwen3 adds no automatic BOS/EOS.
    /// </summary>
    public int[] Encode(string text, bool addSpecialTokens = false)
    {
        var ids = new List<int>(text.Length);
        EncodeInto(text, ids);
        return ids.ToArray();
    }

    private void EncodeInto(string text, List<int> ids)
    {
        int pos = 0;
        while (pos < text.Length)
        {
            (int len, int id) hit = (0, 0);
            foreach (var (special, id) in _addedLongestFirst)
            {
                if (special.Length > 0 && pos + special.Length <= text.Length
                    && string.CompareOrdinal(text, pos, special, 0, special.Length) == 0)
                {
                    hit = (special.Length, id);
                    break;
                }
            }
            if (hit.len > 0)
            {
                ids.Add(hit.id);
                pos += hit.len;
                continue;
            }
            // Consume up to the next added-token boundary, then BPE the gap.
            int runStart = pos;
            while (pos < text.Length && !MatchesAddedTokenAt(text, pos))
                pos++;
            EncodeOrdinary(text.Substring(runStart, pos - runStart), ids);
        }
    }

    private bool MatchesAddedTokenAt(string text, int pos)
    {
        foreach (var (special, _) in _addedLongestFirst)
            if (special.Length > 0 && pos + special.Length <= text.Length
                && string.CompareOrdinal(text, pos, special, 0, special.Length) == 0)
                return true;
        return false;
    }

    /// <summary>NFC-normalize a special-token-free chunk, split with the pre-tokenizer regex,
    /// and byte-level-BPE each piece.</summary>
    private void EncodeOrdinary(string chunk, List<int> ids)
    {
        if (chunk.Length == 0) return;
        string norm = chunk.Normalize(NormalizationForm.FormC);
        foreach (Match m in SplitRegex().Matches(norm))
            BpeBytes(m.Value, ids);
    }

    /// <summary>UTF-8 encode a pre-token piece, map each byte to its GPT-2 unicode char,
    /// then run merge-rank BPE over the byte-char symbols and emit vocab IDs.</summary>
    private void BpeBytes(string piece, List<int> ids)
    {
        if (piece.Length == 0) return;
        byte[] utf8 = Encoding.UTF8.GetBytes(piece);
        var symbols = new List<string>(utf8.Length);
        foreach (byte b in utf8)
            symbols.Add(_byteToChar[b].ToString());

        // Standard GPT-2 BPE: repeatedly merge all occurrences of the lowest-rank pair.
        while (symbols.Count >= 2)
        {
            int bestRank = int.MaxValue, bestI = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
                if (_mergeRank.TryGetValue((symbols[i], symbols[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r;
                    bestI = i;
                }
            if (bestI < 0) break;
            var merged = new List<string>(symbols.Count - 1);
            int j = 0;
            while (j < symbols.Count)
            {
                if (j + 1 < symbols.Count
                    && _mergeRank.TryGetValue((symbols[j], symbols[j + 1]), out int r) && r == bestRank)
                {
                    merged.Add(symbols[j] + symbols[j + 1]);
                    j += 2;
                }
                else
                {
                    merged.Add(symbols[j]);
                    j++;
                }
            }
            symbols = merged;
        }

        foreach (var s in symbols)
        {
            if (!_vocab.TryGetValue(s, out int id))
                throw new InvalidOperationException(
                    $"Byte-level BPE produced a symbol not in vocab: {s.Length} chars. "
                    + "This should be impossible (all 256 byte chars are in vocab).");
            ids.Add(id);
        }
    }

    /// <summary>
    /// Mirror OmniVoice's _tokenize_with_nonverbal_tags: split the text on the supported
    /// nonverbal tags and tokenize each segment and each tag independently (standalone),
    /// so tag token IDs are stable regardless of surrounding script. Identical IDs to
    /// Encode for tag-free text.
    /// </summary>
    public int[] EncodeWithNonverbalTags(string text)
    {
        var ids = new List<int>(text.Length);
        int last = 0;
        foreach (Match m in NonverbalRegex().Matches(text))
        {
            if (m.Index > last)
                EncodeInto(text.Substring(last, m.Index - last), ids);
            EncodeInto(m.Value, ids);
            last = m.Index + m.Length;
        }
        if (last < text.Length)
            EncodeInto(text.Substring(last), ids);
        return ids.ToArray();
    }

    [GeneratedRegex(@"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]")]
    private static partial Regex NonverbalRegex();

    // GPT-2 / OpenAI bytes_to_unicode: a reversible map from each of 256 byte values to a
    // printable Unicode char, so byte sequences become BPE-able strings.
    private void BuildByteToChar()
    {
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }
        for (int i = 0; i < bs.Count; i++)
            _byteToChar[bs[i]] = (char)cs[i];
    }
}
