using System.Text;
using System.Text.RegularExpressions;
using Chatterbox.Base.Markdown;
using Vernacula.Base.Models;
using Vernacula.Phonemizer;
using Vernacula.Phonemizer.Data;
using Vernacula.Phonemizer.Types;

namespace Chatterbox.Base;

/// <summary>One aligned word: its source text and [start, end) seconds in the audio.</summary>
public sealed record KokoroWord(string Text, double StartSec, double EndSec);

/// <summary>Result of <see cref="KokoroTts.SpeakAligned"/>: audio plus per-word timings.</summary>
public sealed record KokoroSpeech(float[] Audio, IReadOnlyList<KokoroWord> Words);

/// <summary>
/// End-to-end Kokoro-82M text-to-speech: text → phonemes → audio. Composes the
/// pure-C# espeak-ng port (<see cref="Phonemize"/>), the Kokoro render format
/// (<see cref="KokoroFormat"/>), and the ONNX inference path (<see cref="Kokoro"/>).
///
/// G2P uses the misaki-style frontend for English: phonemize to IPA, then render
/// to Kokoro's alphabet. The phonemizer needs its language data directory (the
/// <c>data/</c> tree shipped with the espeak-ng-portable submodule) — pass its
/// path as <c>phonemizerDataDir</c>; en/en-gb subfolders are loaded from there.
///
/// Not thread-safe (wraps <see cref="Kokoro"/> / ORT). One instance per caller.
/// </summary>
public sealed class KokoroTts : IDisposable
{
    private readonly Kokoro _kokoro;
    private readonly string _dataDir;
    private readonly Language _enUs;
    private Language? _enGb;

    /// <param name="onnxDir">Directory holding kokoro.onnx and voices/.</param>
    /// <param name="phonemizerDataDir">The espeak-ng-portable <c>data/</c> directory
    /// (contains <c>en/</c>, <c>en-gb/</c>, …).</param>
    public KokoroTts(string onnxDir, string phonemizerDataDir, ExecutionProvider ep,
                     SessionLoadObserver? onLoad = null)
    {
        _kokoro = new Kokoro(onnxDir, ep, onLoad);
        _dataDir = phonemizerDataDir;
        _enUs = LanguageLoader.Load("en", Path.Combine(_dataDir, "en"));
    }

    /// <summary>Output sample rate (24 kHz).</summary>
    public int SampleRate => Kokoro.SampleRate;

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
        var (phonemes, groupSourceWords) = ToPhonemesWithSourceMap(text, british);
        var o = _kokoro.SynthesizeWithDurations(phonemes, voice, speed);
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

        if (runs.Count == groupSourceWords.Count && sourceWords.Length > 0)
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
    public int CountTokens(string text, bool british = false)
        => System.Math.Max(0, KokoroVocab.Encode(ToPhonemes(text, british)).Length - 2);

    private void SplitToTokenBudget(string chunk, bool british, List<string> output)
    {
        if (CountTokens(chunk, british) <= HardInnerLimit) { output.Add(chunk); return; }
        // Pack whole sentences up to the budget; a single over-budget sentence goes to
        // word-level packing. Account for the inter-piece space token (+1) so the running
        // count tracks the real combined count, not the sum of isolated counts.
        var buf = new StringBuilder();
        var bufTokens = 0;
        void Flush() { if (buf.Length > 0) { EmitVerified(buf.ToString(), british, output); buf.Clear(); bufTokens = 0; } }

        foreach (var raw in SentenceSplitRe.Split(chunk))
        {
            var s = raw.Trim();
            if (s.Length == 0) continue;
            var st = CountTokens(s, british);
            if (st > PackBudgetTokens) { Flush(); SplitWordsToBudget(s, british, output); continue; }
            var cost = st + (buf.Length > 0 ? 1 : 0);
            if (bufTokens > 0 && bufTokens + cost > PackBudgetTokens) { Flush(); cost = st; }
            if (buf.Length > 0) buf.Append(' ');
            buf.Append(s);
            bufTokens += cost;
        }
        Flush();
    }

    private void SplitWordsToBudget(string sentence, bool british, List<string> output)
    {
        var buf = new StringBuilder();
        var bufTokens = 0;
        foreach (var w in sentence.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            // +1 for the space token that joins this word to the buffer (the chunk's real
            // token count includes one space per inter-word gap — the bug this fixes).
            var cost = CountTokens(w, british) + (buf.Length > 0 ? 1 : 0);
            if (bufTokens > 0 && bufTokens + cost > PackBudgetTokens)
            {
                EmitVerified(buf.ToString(), british, output);
                buf.Clear();
                bufTokens = 0;
                cost = CountTokens(w, british);
            }
            if (buf.Length > 0) buf.Append(' ');
            buf.Append(w);
            bufTokens += cost;
        }
        if (buf.Length > 0) EmitVerified(buf.ToString(), british, output);
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
    public string ToPhonemes(string text, bool british = false)
        => ToPhonemesWithSourceMap(text, british).Phonemes;

    // Phonemize + render + punctuation re-injection, also returning one source-word index
    // per output phoneme group. Render (per-char) and ReinjectPunctuation (clause join)
    // preserve group order and count, so the map stays 1:1 with the final token runs.
    private (string Phonemes, IReadOnlyList<int> GroupSourceWords) ToPhonemesWithSourceMap(string text, bool british)
    {
        var lang = british ? (_enGb ??= LanguageLoader.Load("en-gb", Path.Combine(_dataDir, "en-gb"))) : _enUs;
        var (ipa, groupSourceWords) = Phonemize.RunWithSourceWords(text, lang);
        var kok = KokoroFormat.Render(ipa, british);
        return (ReinjectPunctuation(kok, text), groupSourceWords);
    }

    // The phonemizer collapses every clause/sentence punctuation mark to a single
    // '\n' and drops the final one — but Kokoro's vocab carries punctuation tokens
    // (',' '.' ';' …) that drive its prosodic pauses. Re-inject them by correlating
    // the source text's clause punctuation (in order) with the '\n' breaks: the i-th
    // break and the trailing position get the i-th source mark. Number-internal dots
    // (e.g. "3.14", normalized to words upstream) produce no break and are excluded
    // so the alignment holds. Defaults to a comma if a mark is somehow missing.
    private static readonly Regex ClausePunctRe = new(@"[,;:!?…—]|(?<![0-9])\.(?![0-9])", RegexOptions.Compiled);

    private static string ReinjectPunctuation(string kokoro, string sourceText)
    {
        var clauses = kokoro.Split('\n');
        if (clauses.Length == 1 && !ClausePunctRe.IsMatch(sourceText))
            return kokoro;

        var marks = ClausePunctRe.Matches(sourceText);
        var sb = new StringBuilder(kokoro.Length + clauses.Length);
        for (var i = 0; i < clauses.Length; i++)
        {
            sb.Append(clauses[i]);
            var mark = i < marks.Count ? marks[i].Value : (i < clauses.Length - 1 ? "," : "");
            if (mark.Length == 1 && KokoroVocab.Contains(mark[0]))
                sb.Append(mark);
            if (i < clauses.Length - 1)
                sb.Append(' ');
        }
        return sb.ToString();
    }

    public void Dispose() => _kokoro.Dispose();
}
