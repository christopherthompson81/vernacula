using System.Text;
using System.Text.RegularExpressions;
using Vernacula.Tts.Base.Markdown;
using Vernacula.Base.Models;

namespace Vernacula.Tts.Base;

/// <summary>One aligned word: its source text and [start, end) seconds in the audio.</summary>
public sealed record KokoroWord(string Text, double StartSec, double EndSec);

/// <summary>Result of <see cref="KokoroTts.SpeakAligned"/>: audio plus per-word timings.</summary>
public sealed record KokoroSpeech(float[] Audio, IReadOnlyList<KokoroWord> Words);

/// <summary>
/// End-to-end Kokoro-82M text-to-speech: text → phonemes → audio. Composes the G2P frontend
/// (<see cref="KokoroPhonemizer"/>: vernacula-phonemizer IPA rendered into Kokoro's alphabet by
/// <see cref="KokoroFormat"/>) and the ONNX inference path (<see cref="Kokoro"/>).
///
/// Not thread-safe (wraps <see cref="Kokoro"/> / ORT). One instance per caller.
/// </summary>
public sealed class KokoroTts : IDisposable
{
    private readonly Kokoro _kokoro;
    private readonly KokoroPhonemizer _g2p;

    /// <param name="onnxDir">Directory holding kokoro_batched.onnx (or the older kokoro.onnx)
    /// and voices/.</param>
    /// <param name="phonemizerDataDir">The vernacula-phonemizer <c>data/</c> root, or null to
    /// resolve it (VERNACULA_DATA_DIR, then the submodule — see <see cref="PhonemizerData"/>).</param>
    public KokoroTts(string onnxDir, string? phonemizerDataDir, ExecutionProvider ep,
                     SessionLoadObserver? onLoad = null)
    {
        _g2p = new KokoroPhonemizer(phonemizerDataDir);   // before the model: the cheaper failure first
        _kokoro = new Kokoro(onnxDir, ep, onLoad);
    }

    /// <summary>Output sample rate (24 kHz).</summary>
    public int SampleRate => Kokoro.SampleRate;

    /// <summary>True when the loaded graph renders a whole batch in one ONNX call
    /// (<c>kokoro_batched.onnx</c>). Callers use this to decide whether grouping work into
    /// <see cref="SpeakAlignedBatch"/> is worth the regrouping it costs them.</summary>
    public bool SupportsBatching => _kokoro.SupportsBatching;

    /// <summary>
    /// Synthesize 24 kHz mono float32 audio from <paramref name="text"/> using the
    /// given <paramref name="voice"/> (e.g. "af_heart"). Set <paramref name="british"/>
    /// for en-gb pronunciation + voices (bf_*/bm_*).
    /// </summary>
    public float[] Speak(string text, string voice, float speed = 1.0f, bool british = false)
        => _kokoro.Synthesize(ToPhonemes(text, british), voice, speed);

    /// <summary>
    /// Synthesize <paramref name="text"/> and return the audio plus per-word timings for
    /// karaoke-style highlighting. Each phoneme group (run of tokens between space/pad tokens)
    /// is mapped to its source word via the phonemizer's source-word map, so spell-out
    /// expansions (e.g. "$3.14" → "three dollars and fourteen cents", five groups) all
    /// collapse onto the one written word with the union of their durations. Per-group
    /// predicted durations give exact start/end times.
    /// </summary>
    public KokoroSpeech SpeakAligned(string text, string voice, float speed = 1.0f, bool british = false)
    {
        var (phonemes, groupSourceWords) = _g2p.Phonemize(text, british);
        return Align(text, groupSourceWords, _kokoro.SynthesizeWithDurations(phonemes, voice, speed));
    }

    /// <summary>
    /// <see cref="SpeakAligned"/> for several texts in one ONNX call. Word timings come from
    /// <c>pred_dur</c>, which is bit-identical to the solo render regardless of batch composition,
    /// so alignment is unaffected by which texts share a batch.
    ///
    /// <para>Only faster when the loaded graph is <c>kokoro_batched.onnx</c>
    /// (<see cref="Kokoro.SupportsBatching"/>); otherwise it loops and matches
    /// <see cref="SpeakAligned"/> exactly. Padding is wasted compute, so pass texts of similar
    /// length together where the caller is free to choose the grouping.</para>
    /// </summary>
    public IReadOnlyList<KokoroSpeech> SpeakAlignedBatch(
        IReadOnlyList<string> texts, string voice, float speed = 1.0f, bool british = false)
    {
        ArgumentNullException.ThrowIfNull(texts);
        if (texts.Count == 0) return [];
        var ph = new KokoroPhonemization[texts.Count];
        var phonemes = new string[texts.Count];
        for (var i = 0; i < texts.Count; i++)
        {
            ph[i] = _g2p.Phonemize(texts[i], british);
            phonemes[i] = ph[i].Phonemes;
        }
        var results = new KokoroSpeech[texts.Count];
        foreach (var group in BucketByLength(phonemes))
        {
            var outs = _kokoro.SynthesizeBatch([.. group.Select(i => phonemes[i])], voice, speed);
            for (var g = 0; g < group.Count; g++)
            {
                var i = group[g];
                results[i] = Align(texts[i], ph[i].GroupSourceWords, outs[g]);
            }
        }
        return results;
    }

    // A batch is padded to its longest item, so mixing a heading with a long paragraph spends
    // most of the GPU on padding — measured at 37% fill on an ordinary document, which wipes the
    // batching win out entirely (0.98x, i.e. slower than sequential). Sorting by phoneme length
    // and cutting a new batch when the spread gets too wide keeps each batch close to square.
    //
    // ⚠ This is purely a THROUGHPUT concern. Fidelity does not depend on how items are grouped:
    // the padding error is a step function (two frames of padding cost as much as two hundred)
    // and it is masked out either way. So grouping is free to optimise for fill.
    private const int MaxBatchItems = 16;      // throughput saturates around 8-16 (Run 33)
    private const double MaxLengthSpread = 1.5; // start a new batch past this longest/shortest ratio

    /// <summary>Indices of <paramref name="phonemes"/> grouped into batches of similar length.
    /// Groups are returned in no particular order; each carries the original indices.</summary>
    private static List<List<int>> BucketByLength(IReadOnlyList<string> phonemes)
    {
        var order = Enumerable.Range(0, phonemes.Count)
                              .Where(i => phonemes[i].Length > 0)
                              .OrderBy(i => phonemes[i].Length)
                              .ToArray();
        var groups = new List<List<int>>();
        var current = new List<int>();
        var shortest = 0;
        foreach (var i in order)
        {
            var len = phonemes[i].Length;
            if (current.Count > 0 && (current.Count >= MaxBatchItems || len > shortest * MaxLengthSpread))
            {
                groups.Add(current);
                current = [];
            }
            if (current.Count == 0) shortest = len;
            current.Add(i);
        }
        if (current.Count > 0) groups.Add(current);

        // Empty strings never reach the model; hand them back so every index gets a result.
        var empties = Enumerable.Range(0, phonemes.Count).Where(i => phonemes[i].Length == 0).ToList();
        if (empties.Count > 0) groups.Add(empties);
        return groups;
    }

    /// <summary>Map one synthesis result onto per-word timings. Shared by the single and
    /// batched paths so they cannot drift apart.</summary>
    private static KokoroSpeech Align(string text, IReadOnlyList<int>? groupSourceWords, KokoroOutput o)
    {
        if (o.Audio.Length == 0)
            return new KokoroSpeech([], []);

        long durSum = 0;
        foreach (var d in o.PredDur) durSum += d;
        double secPerDur = durSum > 0 ? o.Audio.Length / (double)durSum / Kokoro.SampleRate : 0;

        // Cumulative seconds at the start of each token (cum[k] = Σ dur[0..k-1]).
        var cum = new double[o.PredDur.Length + 1];
        for (var k = 0; k < o.PredDur.Length; k++)
            cum[k + 1] = cum[k] + o.PredDur[k] * secPerDur;

        // Phoneme groups: maximal token spans between space/pad tokens. Group g aligns to
        // groupSourceWords[g] (1:1 — Render/punctuation re-injection preserve group order/count).
        var runs = new List<(double Start, double End)>();
        var i = 0;
        while (i < o.InputIds.Length)
        {
            var id = o.InputIds[i];
            if (id == KokoroVocab.Space || id == KokoroVocab.Pad) { i++; continue; }
            var first = i;
            while (i < o.InputIds.Length && o.InputIds[i] != KokoroVocab.Space && o.InputIds[i] != KokoroVocab.Pad)
                i++;
            runs.Add((cum[first], cum[i]));
        }

        var sourceWords = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var words = new List<KokoroWord>(sourceWords.Length);

        if (groupSourceWords is not null && runs.Count == groupSourceWords.Count && sourceWords.Length > 0)
        {
            // Collect each source word's group span. A word's groups are contiguous and in
            // time order, so first start / last end gives its [start, end].
            var hasRun = new bool[sourceWords.Length];
            var starts = new double[sourceWords.Length];
            var ends = new double[sourceWords.Length];
            for (var g = 0; g < runs.Count; g++)
            {
                var src = groupSourceWords[g];
                if (src < 0 || src >= sourceWords.Length) continue;
                if (!hasRun[src]) { starts[src] = runs[g].Start; hasRun[src] = true; }
                ends[src] = runs[g].End;
            }
            // Emit one word per source word — including unpronounceable words that produced
            // no groups (zero-length marker at the running cursor), so the display shows every
            // word and the index stays 1:1 with the source-text whitespace split.
            var cursor = 0.0;
            for (var w = 0; w < sourceWords.Length; w++)
            {
                if (hasRun[w]) { words.Add(new KokoroWord(sourceWords[w], starts[w], ends[w])); cursor = ends[w]; }
                else words.Add(new KokoroWord(sourceWords[w], cursor, cursor));
            }
        }
        else
        {
            // Defensive fallback: even split (should not happen with the source map intact).
            var total = o.Audio.Length / (double)Kokoro.SampleRate;
            for (var w = 0; w < sourceWords.Length; w++)
                words.Add(new KokoroWord(sourceWords[w],
                    total * w / sourceWords.Length, total * (w + 1) / sourceWords.Length));
        }
        return new KokoroSpeech(o.Audio, words);
    }

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

    /// <summary>
    /// Text → Kokoro-alphabet phoneme string, without running the vocoder. Useful
    /// for inspection, caching, or feeding <see cref="Kokoro.Synthesize"/> directly.
    /// </summary>
    public string ToPhonemes(string text, bool british = false) => _g2p.ToPhonemes(text, british);

    public void Dispose() => _kokoro.Dispose();
}
