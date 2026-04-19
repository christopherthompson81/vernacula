using System.IO.Compression;
using System.Text;

namespace Vernacula.Base;

/// <summary>
/// Pure-C# scorer for ARPA-format n-gram language models built over a
/// Parakeet-compatible subword vocabulary. Integer token IDs are used as
/// "words" in the ARPA file so the decoder can look up the LM without any
/// string<->ID round-trip. Supports Katz backoff with per-context weights,
/// matching KenLM's default behavior.
///
/// Not intended as a full KenLM replacement — we only need the subset the
/// shallow-fusion beam decoder calls on the hot path (single-token score
/// given a variable-length history).
/// </summary>
public sealed class KenLmScorer
{
    /// <summary>Packs up to four 14-bit token IDs into a single ulong for fast hashing.</summary>
    private const int TokenBits = 14;                   // supports vocab up to 16384 (covers Parakeet's 8193)
    private const ulong TokenMask = (1UL << TokenBits) - 1;
    private const int MaxOrder = 4;                     // higher orders would overflow ulong (5×14 = 70 bits)

    private readonly Dictionary<ulong, (float logProb, float backoff)>[] _ngrams;
    private readonly float _unkLogProb;
    private readonly int _order;

    public int Order => _order;
    public float LogProbNatural(float log10) => log10 * 2.302585093f;

    private KenLmScorer(
        Dictionary<ulong, (float, float)>[] ngrams,
        float unkLogProb,
        int order)
    {
        _ngrams     = ngrams;
        _unkLogProb = unkLogProb;
        _order      = order;
    }

    /// <summary>
    /// Loads an ARPA file (optionally gzip-compressed — detected by .gz suffix
    /// or a gzip magic header). The token column is expected to contain
    /// integer subword IDs matching the Parakeet vocabulary.
    /// </summary>
    public static KenLmScorer LoadArpa(string path)
    {
        using var fs     = File.OpenRead(path);
        using var reader = OpenMaybeGzip(fs, path);
        return LoadArpaCore(reader);
    }

    private static StreamReader OpenMaybeGzip(Stream raw, string path)
    {
        // Detect by magic or .gz extension
        Stream inner = raw;
        if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
        {
            inner = new GZipStream(raw, CompressionMode.Decompress);
        }
        else
        {
            int b0 = raw.ReadByte();
            int b1 = raw.ReadByte();
            raw.Position = 0;
            if (b0 == 0x1f && b1 == 0x8b)
                inner = new GZipStream(raw, CompressionMode.Decompress);
        }
        return new StreamReader(inner, Encoding.UTF8);
    }

    private static KenLmScorer LoadArpaCore(StreamReader reader)
    {
        // Phase 1: scan \data\ block for ngram counts to pre-size dictionaries
        string? line;
        int order = 0;
        var counts = new List<int>(); // counts[i] = number of (i+1)-grams
        while ((line = reader.ReadLine()) != null)
        {
            line = line.Trim();
            if (line == "\\data\\") break;
        }
        if (line is null)
            throw new FormatException("ARPA file missing \\data\\ header.");

        while ((line = reader.ReadLine()) != null)
        {
            line = line.Trim();
            if (line.Length == 0) break;
            if (!line.StartsWith("ngram ", StringComparison.Ordinal)) continue;
            // "ngram N=COUNT"
            int eq    = line.IndexOf('=');
            int nOrd  = int.Parse(line.AsSpan(6, eq - 6));
            int count = int.Parse(line.AsSpan(eq + 1));
            while (counts.Count < nOrd) counts.Add(0);
            counts[nOrd - 1] = count;
            if (nOrd > order) order = nOrd;
        }

        if (order < 1 || order > MaxOrder)
            throw new FormatException($"Unsupported LM order {order} (max {MaxOrder}).");

        var ngrams = new Dictionary<ulong, (float, float)>[order];
        for (int i = 0; i < order; i++)
            ngrams[i] = new Dictionary<ulong, (float, float)>(Math.Max(counts[i], 16));

        // Phase 2: read each "\N-grams:" section
        float unkLogProb = -10f; // fallback if LM has no explicit <unk>
        int currentOrder = 0;
        while ((line = reader.ReadLine()) != null)
        {
            if (line.Length == 0) continue;
            if (line == "\\end\\") break;

            if (line.Length > 2 && line[0] == '\\' && line.EndsWith("-grams:", StringComparison.Ordinal))
            {
                currentOrder = int.Parse(line.AsSpan(1, line.IndexOf('-') - 1));
                continue;
            }
            if (currentOrder == 0) continue;

            ParseAndStoreNgram(line, currentOrder, ngrams, ref unkLogProb);
        }

        return new KenLmScorer(ngrams, unkLogProb, order);
    }

    private static void ParseAndStoreNgram(
        string line,
        int currentOrder,
        Dictionary<ulong, (float, float)>[] ngrams,
        ref float unkLogProb)
    {
        // Format: "logprob\tt1 t2 ... tN[\tbackoff]"  — whitespace may be tab OR space
        var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        int needed = 1 + currentOrder; // logprob + N tokens
        if (parts.Length < needed) return;

        float logProb = float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture);
        float backoff = parts.Length >= needed + 1
            ? float.Parse(parts[needed], System.Globalization.CultureInfo.InvariantCulture)
            : 0f;

        ulong key = 0;
        bool ok = true;
        for (int k = 0; k < currentOrder; k++)
        {
            string tok = parts[1 + k];
            if (tok == "<unk>")
            {
                if (currentOrder == 1) unkLogProb = logProb;
                ok = false;
                break;
            }
            if (tok == "<s>" || tok == "</s>")
            {
                // We skip sentence boundaries: decoder-level fusion operates
                // on per-segment token streams, not sentence-delimited text,
                // so <s>/</s> lookups would just miss.
                ok = false;
                break;
            }
            if (!int.TryParse(tok, out int id) || id < 0 || (ulong)id > TokenMask)
            {
                ok = false;
                break;
            }
            key |= (ulong)id << (k * TokenBits);
        }

        if (ok) ngrams[currentOrder - 1][key] = (logProb, backoff);
    }

    /// <summary>
    /// Log-probability (natural log, not log10) of <paramref name="nextToken"/>
    /// given <paramref name="context"/>. Applies Katz backoff when the full
    /// n-gram is missing.
    /// </summary>
    public float LogProb(IReadOnlyList<int> context, int nextToken)
    {
        int ctxUsed = Math.Min(context.Count, _order - 1);

        // Try longest-first: (ctxUsed+1)-gram down to unigram
        for (int n = ctxUsed; n >= 0; n--)
        {
            ulong key = 0;
            int start = context.Count - n;
            bool inRange = true;
            for (int k = 0; k < n; k++)
            {
                int id = context[start + k];
                if (id < 0 || (ulong)id > TokenMask) { inRange = false; break; }
                key |= (ulong)id << (k * TokenBits);
            }
            if (!inRange) continue;
            if (nextToken < 0 || (ulong)nextToken > TokenMask) continue;
            key |= (ulong)nextToken << (n * TokenBits);

            if (_ngrams[n].TryGetValue(key, out var entry))
            {
                float s = entry.logProb;
                // Add backoff weights for the (shorter) contexts we skipped past.
                // For each skipped order m in (n+1 .. ctxUsed), accumulate
                // backoff of the m-length context that was too long to match.
                for (int skip = n + 1; skip <= ctxUsed; skip++)
                {
                    ulong ctxKey = 0;
                    int cs = context.Count - skip;
                    bool ok = true;
                    for (int k = 0; k < skip; k++)
                    {
                        int id = context[cs + k];
                        if (id < 0 || (ulong)id > TokenMask) { ok = false; break; }
                        ctxKey |= (ulong)id << (k * TokenBits);
                    }
                    if (ok && _ngrams[skip - 1].TryGetValue(ctxKey, out var be))
                        s += be.backoff;
                }
                return Log10ToNat(s);
            }
        }

        return Log10ToNat(_unkLogProb);
    }

    private static float Log10ToNat(float log10) => log10 * 2.302585093f;
}
