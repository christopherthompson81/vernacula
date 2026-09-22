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
    private readonly KokoroChunker _chunker;

    /// <param name="onnxDir">Directory holding kokoro_batched.onnx (or the older kokoro.onnx)
    /// and voices/.</param>
    /// <param name="phonemizerDataDir">The vernacula-phonemizer <c>data/</c> root, or null to
    /// resolve it (VERNACULA_DATA_DIR, then the submodule — see <see cref="PhonemizerData"/>).</param>
    public KokoroTts(string onnxDir, string? phonemizerDataDir, ExecutionProvider ep,
                     SessionLoadObserver? onLoad = null)
    {
        _g2p = new KokoroPhonemizer(phonemizerDataDir);   // before the model: the cheaper failure first
        _chunker = new KokoroChunker(_g2p);
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
        var spoken = _g2p.Phonemize(text, british);
        return Align(text, spoken, _kokoro.SynthesizeWithDurations(spoken.Phonemes, voice, speed));
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
                results[i] = Align(texts[i], ph[i], outs[g]);
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
    private static KokoroSpeech Align(string text, KokoroPhonemization spoken, KokoroOutput o)
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
        var runs = new List<KokoroAlignment.GroupSpan>();
        var i = 0;
        while (i < o.InputIds.Length)
        {
            var id = o.InputIds[i];
            if (id == KokoroVocab.Space || id == KokoroVocab.Pad) { i++; continue; }
            var first = i;
            while (i < o.InputIds.Length && o.InputIds[i] != KokoroVocab.Space && o.InputIds[i] != KokoroVocab.Pad)
                i++;
            runs.Add(new KokoroAlignment.GroupSpan(cum[first], cum[i]));
        }

        // The join is shared with the audio.cpp backend, which gets its groups from the engine
        // instead of cutting pred_dur itself — see KokoroAlignment. Only the sentence above this
        // one differs between the two engines.
        var words = KokoroAlignment.WordsFromGroups(
            text, spoken.Words, spoken.GroupSourceWords, runs, o.Audio.Length / (double)Kokoro.SampleRate);
        return new KokoroSpeech(o.Audio, words);
    }

    /// <inheritdoc cref="KokoroChunker.ChunkForSynthesis"/>
    public IReadOnlyList<string> ChunkForSynthesis(string text, bool british = false)
        => _chunker.ChunkForSynthesis(text, british);

    /// <summary>Inner phoneme-token count (excludes the 2 pad tokens) for <paramref name="text"/>.</summary>
    public int CountTokens(string text, bool british = false) => _g2p.CountTokens(text, british);

    /// <summary>
    /// Text → Kokoro-alphabet phoneme string, without running the vocoder. Useful
    /// for inspection, caching, or feeding <see cref="Kokoro.Synthesize"/> directly.
    /// </summary>
    public string ToPhonemes(string text, bool british = false) => _g2p.ToPhonemes(text, british);

    public void Dispose() => _kokoro.Dispose();
}
