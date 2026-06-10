using System.Text;
using System.Text.RegularExpressions;
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
    /// Synthesize <paramref name="text"/> and return the audio plus per-word timings
    /// for karaoke-style highlighting. Word boundaries are recovered from the phoneme
    /// token stream (runs of tokens between space/pad tokens) and timed from the model's
    /// per-token predicted durations; words are labelled by the whitespace-split source
    /// text. On a count mismatch (e.g. number expansion like "$3.14") timings fall back
    /// to an even split across the utterance.
    /// </summary>
    public KokoroSpeech SpeakAligned(string text, string voice, float speed = 1.0f, bool british = false)
    {
        var phonemes = ToPhonemes(text, british);
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

        // Word runs: maximal spans of tokens that are neither space nor pad.
        var runs = new List<(double Start, double End)>();
        var i = 0;
        while (i < o.InputIds.Length)
        {
            var id = o.InputIds[i];
            if (id == KokoroVocab.Space || id == KokoroVocab.Pad) { i++; continue; }
            var first = i;
            while (i < o.InputIds.Length && o.InputIds[i] != KokoroVocab.Space && o.InputIds[i] != KokoroVocab.Pad)
                i++;
            runs.Add((cum[first], cum[i]));   // i is one past the run's last token
        }

        var sourceWords = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var words = new List<KokoroWord>(sourceWords.Length);
        if (runs.Count == sourceWords.Length)
        {
            for (var w = 0; w < sourceWords.Length; w++)
                words.Add(new KokoroWord(sourceWords[w], runs[w].Start, runs[w].End));
        }
        else
        {
            // Fallback: even split of total audio duration across the source words.
            var total = o.Audio.Length / (double)Kokoro.SampleRate;
            for (var w = 0; w < sourceWords.Length; w++)
                words.Add(new KokoroWord(sourceWords[w],
                    total * w / sourceWords.Length, total * (w + 1) / sourceWords.Length));
        }
        return new KokoroSpeech(o.Audio, words);
    }

    /// <summary>
    /// Text → Kokoro-alphabet phoneme string, without running the vocoder. Useful
    /// for inspection, caching, or feeding <see cref="Kokoro.Synthesize"/> directly.
    /// </summary>
    public string ToPhonemes(string text, bool british = false)
    {
        var lang = british ? (_enGb ??= LanguageLoader.Load("en-gb", Path.Combine(_dataDir, "en-gb"))) : _enUs;
        var ipa = Phonemize.Run(text, lang);
        var kok = KokoroFormat.Render(ipa, british);
        return ReinjectPunctuation(kok, text);
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
