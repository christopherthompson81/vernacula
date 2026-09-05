using Vernacula.Base.Models;
using Vernacula.Phonemizer;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.Tts.Base;

/// <summary>One spoken word: its source text and [start, end) seconds in the audio.</summary>
public sealed record OmniVoiceIpaWord(string Text, double StartSec, double EndSec);

/// <summary>Result of <see cref="OmniVoiceIpaTts.SpeakAligned"/>: audio plus per-word timings.</summary>
public sealed record OmniVoiceIpaSpeech(float[] Audio, IReadOnlyList<OmniVoiceIpaWord> Words);

/// <summary>
/// End-to-end IPA-native text-to-speech: text → canonical IPA (vernacula-phonemizer, any of its
/// languages) → the OmniVoice IPA fine-tune → audio. The engine behind <c>vernacula-tts</c>, as a
/// class the reader can drive chunk by chunk: the CLI's setup (tokenizer, the versioned diff, the
/// phonemizer data tree), its conditioning (a stored voice's codes and IPA transcript, the IPA-side
/// duration estimate) and its finishing chain, with a chunker and a word map on top.
///
/// ⚠ WORD TIMINGS HERE ARE ESTIMATED, NOT MEASURED. Chatterbox reads them off its cross-attention
/// and Kokoro off its predicted durations; this model exposes neither, so each chunk's audio is
/// divided among its words in proportion to their IPA script-weight (the same weight the duration
/// estimator uses). Good enough to keep a highlight moving with the speech; not good enough to
/// seek to a word. A forced aligner (Vernacula.Base.Alignment) would be the measured answer.
///
/// Not thread-safe (wraps ORT sessions). One instance per caller.
/// </summary>
public sealed class OmniVoiceIpaTts : IDisposable
{
    public const int SampleRate = OmniVoiceTts.SampleRate;

    /// <summary>Audio tokens per chunk above which a paragraph is split on sentences. 25 tokens
    /// ≈ 1 s; 750 ≈ 30 s. The CLI warns at 1500 that output degrades in one shot; half that keeps
    /// each chunk well inside what the fine-tune saw (FLEURS read sentences, median 12 s).</summary>
    public const int MaxChunkTokens = 750;

    private readonly OmniVoiceTts _tts;

    /// <summary>The reference voice. Null renders in auto mode — which the fine-tune handles badly
    /// under ~5 s of text (noise rather than speech, deterministically), so the reader always
    /// sets one.</summary>
    public StoredVoice? Voice { get; set; }

    /// <param name="onnxDir">Holds the base transformer, the Higgs codec graphs and (by default)
    /// the versioned IPA diff.</param>
    /// <param name="tokenizerJson">Qwen3 tokenizer.json; null resolves it via
    /// <see cref="LocateTokenizerJson"/>.</param>
    /// <param name="phonemizerDataDir">vernacula-phonemizer data/ root; null resolves it via
    /// <see cref="PhonemizerData.Resolve"/>.</param>
    /// <param name="diffPath">The IPA fine-tune diff; null means <see cref="IpaFineTune.DefaultDiffFile"/>
    /// beside the graphs. There is no orthographic fallback here: without the diff the model reads
    /// IPA as spelling and produces confident nonsense, so a missing diff throws.</param>
    public OmniVoiceIpaTts(string onnxDir, string? tokenizerJson, string? phonemizerDataDir,
                           ExecutionProvider ep, SessionLoadObserver? onLoad = null, string? diffPath = null)
    {
        if (PhonemizerData.Resolve(phonemizerDataDir) is null)
            throw new DirectoryNotFoundException(PhonemizerData.NotFoundMessage());
        Registry.EnsureLanguages();

        tokenizerJson ??= LocateTokenizerJson(onnxDir)
            ?? throw new FileNotFoundException("Qwen3 tokenizer.json not found: looked beside the graphs in "
                + $"{onnxDir}, in OMNIVOICE_MODEL_DIR, and in the k2-fsa/OmniVoice HuggingFace cache.");
        diffPath ??= Path.Combine(onnxDir, IpaFineTune.DefaultDiffFile);
        if (!File.Exists(diffPath))
            throw new FileNotFoundException($"IPA fine-tune diff not found: {diffPath}");

        _tts = new OmniVoiceTts(onnxDir, tokenizerJson, ep, onLoad, transformerFile: null, diffPath);
    }

    /// <summary>tokenizer.json beside the graphs, else in OMNIVOICE_MODEL_DIR, else in the
    /// k2-fsa/OmniVoice snapshot of the HuggingFace hub cache. Null when none exists.</summary>
    public static string? LocateTokenizerJson(string onnxDir)
    {
        var beside = Path.Combine(onnxDir, "tokenizer.json");
        if (File.Exists(beside)) return beside;
        var modelDir = Environment.GetEnvironmentVariable("OMNIVOICE_MODEL_DIR");
        if (!string.IsNullOrEmpty(modelDir) && File.Exists(Path.Combine(modelDir, "tokenizer.json")))
            return Path.Combine(modelDir, "tokenizer.json");
        var snapshots = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub", "models--k2-fsa--OmniVoice", "snapshots");
        if (Directory.Exists(snapshots))
            foreach (var snap in Directory.EnumerateDirectories(snapshots))
                if (File.Exists(Path.Combine(snap, "tokenizer.json"))) return Path.Combine(snap, "tokenizer.json");
        return null;
    }

    /// <summary>Text → canonical IPA in <paramref name="lang"/> (a vernacula-phonemizer code).
    /// The best-output entry: languages with a neural model use it. Blocking, like
    /// <see cref="KokoroPhonemizer"/> and for the same reason — every caller is off the UI thread.</summary>
    public static string Phonemize(string text, string lang)
        => global::Vernacula.Phonemizer.Phonemizer.PhonemizeAsync(text, lang).GetAwaiter().GetResult().Trim();

    private string? CondRefIpa => Voice is null ? null : OmniVoiceTextPrep.AddPunctuation(Voice.RefIpa);
    private int? RefTokens => Voice?.Codes.GetLength(1);

    /// <summary>Estimated audio tokens for <paramref name="text"/> under the current voice.</summary>
    public int EstimateTokens(string text, string lang)
        => OmniVoiceDuration.EstimateTargetTokens(Phonemize(text, lang), CondRefIpa, RefTokens);

    /// <summary>
    /// Split <paramref name="text"/> into synthesis chunks: paragraph/char chunking first
    /// (<see cref="ParagraphChunker"/>), then any chunk estimated over <see cref="MaxChunkTokens"/>
    /// is packed from its sentences. All splits are at whitespace, so the concatenated word
    /// sequence is unchanged and word timings stay 1:1 with the source text.
    /// </summary>
    public IReadOnlyList<string> ChunkForSynthesis(string text, string lang)
    {
        var pieces = new List<string>();
        foreach (var chunk in ParagraphChunker.Chunk(text))
        {
            if (EstimateTokens(chunk, lang) <= MaxChunkTokens) { pieces.Add(chunk); continue; }
            var buf = new List<string>();
            var bufTokens = 0;
            foreach (var raw in SentenceSplitRe.Split(chunk))
            {
                var s = raw.Trim();
                if (s.Length == 0) continue;
                var st = EstimateTokens(s, lang);
                if (buf.Count > 0 && bufTokens + st > MaxChunkTokens)
                {
                    pieces.Add(string.Join(' ', buf)); buf.Clear(); bufTokens = 0;
                }
                buf.Add(s); bufTokens += st;
            }
            if (buf.Count > 0) pieces.Add(string.Join(' ', buf));
        }
        return pieces;
    }

    private static readonly System.Text.RegularExpressions.Regex SentenceSplitRe =
        new(@"(?<=[.!?…])\s+", System.Text.RegularExpressions.RegexOptions.Compiled);

    /// <summary>Synthesize one chunk in the current voice and return audio plus estimated
    /// per-word timings (see the class remarks).</summary>
    /// <param name="leveler">Shared across the chunks of one document so their loudness stays
    /// consistent; null finishes this chunk as a single utterance.</param>
    public OmniVoiceIpaSpeech SpeakAligned(string text, string lang, int numStep = 32,
        OmniVoiceAudioPost.StoredVoiceLeveler? leveler = null)
    {
        var trace = global::Vernacula.Phonemizer.Phonemizer.PhonemizeTrace(text, lang);
        var ipa = Phonemize(text, lang);
        if (string.IsNullOrWhiteSpace(ipa)) return new OmniVoiceIpaSpeech([], []);

        // Duration is estimated on the IPA, not the orthography — both sides of the ratio must be
        // the same representation, and IPA-on-both is the pacing the fine-tune was accepted with.
        int target = OmniVoiceDuration.EstimateTargetTokens(ipa, CondRefIpa, RefTokens);
        // lang: null is the IPA fine-tune's conditioning — the language never reaches the model.
        var tokens = _tts.GenerateTokens(ipa, target, CondRefIpa, Voice?.Codes, lang: null, instruct: null,
            new OmniVoiceTts.GenConfig(NumStep: numStep));
        // A stored voice was never boosted at encode time, so Python's un-boost would just make a
        // quiet source quieter; an encoded reference WAS boosted and needs it undone.
        var decoded = _tts.DecodeTokens(tokens);
        var audio = Voice is null
            ? OmniVoiceAudioPost.Finish(decoded, SampleRate, refRms: null)
            : OmniVoiceAudioPost.FinishStoredVoice(decoded, SampleRate, leveler);

        return new OmniVoiceIpaSpeech(audio, OmniVoiceIpaAlignment.Proportional(text, trace, audio.Length / (double)SampleRate));
    }

    public void Dispose() => _tts.Dispose();
}

/// <summary>
/// The estimated word map: each whitespace-delimited source word gets a share of the chunk's
/// duration proportional to the IPA script-weight of what it became. Separate from the engine so
/// it can be tested without a model.
/// </summary>
public static class OmniVoiceIpaAlignment
{
    public static IReadOnlyList<OmniVoiceIpaWord> Proportional(string text, PhonemeTrace trace, double totalSec)
    {
        var sourceWords = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var weights = new double[sourceWords.Length];

        // Character offset → index of the whitespace-delimited word containing it.
        var wordAt = new int[text.Length];
        var w = -1; var inWord = false;
        for (var i = 0; i < text.Length; i++)
        {
            if (char.IsWhiteSpace(text[i])) { inWord = false; wordAt[i] = w + 1; continue; }
            if (!inWord) { w++; inWord = true; }
            wordAt[i] = w;
        }

        var mapped = trace.Traced;
        if (mapped)
        {
            foreach (var tok in trace.Tokens)
            {
                if (tok.InputSpan is not { } input || tok.IpaSpan is not { } span
                    || input.Start < 0 || input.Start >= text.Length) { mapped = false; break; }
                var word = wordAt[input.Start];
                if (word >= sourceWords.Length) { mapped = false; break; }
                weights[word] += OmniVoiceDuration.TotalWeight(trace.Ipa[span.Start..span.End]);
            }
        }
        if (!mapped)
            for (var i = 0; i < weights.Length; i++) weights[i] = sourceWords[i].Length;

        double sum = 0;
        foreach (var x in weights) sum += x;
        var words = new List<OmniVoiceIpaWord>(sourceWords.Length);
        double cursor = 0;
        for (var i = 0; i < sourceWords.Length; i++)
        {
            var dur = sum > 0 ? totalSec * weights[i] / sum : totalSec / Math.Max(1, sourceWords.Length);
            words.Add(new OmniVoiceIpaWord(sourceWords[i], cursor, cursor + dur));
            cursor += dur;
        }
        return words;
    }
}
