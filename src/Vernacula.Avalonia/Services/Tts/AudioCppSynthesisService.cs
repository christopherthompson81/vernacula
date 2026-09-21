#if AUDIOCPP_BACKEND
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Vernacula.AudioCpp;
using Vernacula.Phonemizer;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Alignment;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Kokoro-82M through audio.cpp's C ABI, beside the ONNX Kokoro rather than instead of it.
/// Same segmented-synthesis contract as every other engine — one paragraph per segment, streamed
/// as it is rendered, per-segment WAVs, one sidecar at the end.
///
/// <para>
/// ⚠ ENGLISH IS SPOKEN FROM OUR OWN PHONEMES, NOT THE ENGINE'S. audio.cpp's Kokoro phonemizes
/// with eSpeak-ng; upstream audio.cpp#577 added a <c>phonemes</c> request option that takes a
/// caller's stream instead, and for an English voice this passes vernacula-phonemizer's reading
/// rendered through <see cref="KokoroFormat"/> — the same frontend the ONNX Kokoro uses. The
/// point is that the two engines now say a word the SAME WAY: the dictionary the user can see,
/// correct and re-hear is the one that decides, whichever backend renders it. The five
/// non-English voices keep the engine's own G2P, and so does any run where the phonemizer's data
/// tree is absent; there is no setting, because "which pronunciation dictionary is in force" is
/// not a thing a reader should have to choose per document.
/// </para>
///
/// <para>
/// ⚠ THE WORD TIMINGS ARE MEASURED NOW, AND THEY USED TO BE ESTIMATES. The <c>kokoro_tts</c>
/// family reported <c>SupportsTimestamps=false</c> and returned no words, so a paragraph's words
/// had to be spread across its buffer by some proxy for length — first spelling, then, once we
/// supplied the phonemes, phoneme count. Neither was a measurement. The information was always
/// there: Kokoro predicts a per-token frame count BEFORE the decoder runs and the decoder
/// upsamples by exactly those counts, which is the same <c>pred_dur</c> the ONNX path reads. The
/// engine now reports it per phoneme group, and since the phonemizer already says which source
/// word each group came from, the join is exact — the same join, in the same code
/// (<see cref="KokoroAlignment"/>), as the ONNX engine's.
/// </para>
///
/// <para>
/// The proportional tiers stay as fallbacks, for a non-English voice, a missing phonemizer tree,
/// or an engine built before the family declared the capability. The sidecar's aligner name says
/// which was in force: <c>audiocpp_duration</c> against <c>audiocpp_proportional</c>.
/// </para>
///
/// <para>
/// Compiled only when the AudioCpp-Bindings submodule is present; see the project file. The
/// model loads lazily on the first call and is reused.
/// </para>
/// </summary>
public sealed class AudioCppSynthesisService : ITtsBackend
{
    private readonly string   _modelPath;
    private readonly string[] _backends;
    private readonly int      _threads;
    private readonly string?  _dataDir;

    private AudioCppTts?      _tts;
    private KokoroPhonemizer? _g2p;       // null when the phonemizer's data tree is not installed
    private KokoroChunker?    _chunker;
    private readonly object _gate = new();

    public AudioCppSynthesisService(string modelPath, string[] backends, int threads,
                                    string? phonemizerDataDir = null)
    {
        _modelPath = modelPath;
        _backends  = backends;
        _threads   = threads;
        _dataDir   = phonemizerDataDir;
    }

    public int SampleRate => AudioCppTts.SampleRate;

    public async Task<SynthesisResult> SynthesizeStreamingAsync(
        TtsRequest request,
        Action<ChunkProducedEvent>? onChunkProduced = null,
        Action<ProgressEvent>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var voice = request.Voice;
        var speed = request.Speed;
        // The phonemizer covers every language this package speaks, but KokoroFormat is an
        // English render target: the other five would need their own mapping onto Kokoro's
        // alphabet, and a symbol we got wrong is a refused paragraph rather than an accent.
        var lang = AudioCppKokoroVoices.PhonemizerLanguage(voice);
        bool british = lang == "en-GB";
        bool ours    = lang is "en" or "en-GB";

        onProgress?.Invoke(new ProgressEvent("loading models"));
        await Task.Run(EnsureLoaded, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        return await Task.Run(() =>
        {
            var tts = _tts!;
            // Set by the first segment that actually joins the engine's timings to its words.
            var measuredAny = false;

            (float[] Audio, IReadOnlyList<AlignedWord> Words) SynthesizeSegment(
                Vernacula.Tts.Base.Markdown.TextSegment seg, Action<string> warn)
            {
                cancellationToken.ThrowIfCancellationRequested();
                // No chunker on the text path: the family splits on its own text_chunk_size and
                // joins, so a paragraph of any length comes back as one buffer (investigation
                // Run 4). On the phoneme path the split is ours to make, because only this G2P
                // knows where its own stream may be cut.
                var supplied = ours && _g2p is not null && _chunker is not null
                    ? Supply(_g2p, _chunker, seg.Text, british, warn)
                    : null;
                var spoken = tts.SpeakAligned(seg.Text, supplied?.Chunks, voice, speed);
                var seconds = spoken.Audio.Length / (double)SampleRate;
                var aligned = Align(seg.Text, supplied, spoken, seconds, out var measured);
                measuredAny |= measured;
                return (spoken.Audio, aligned);
            }

            // No batch synthesizer: the ABI takes one request at a time, so there is nothing to
            // batch and offering a delegate that loops would only disable the reuse decorator.
            var result = SegmentedSynthesis.Run(request, SampleRate, "audiocpp_proportional",
                SynthesizeSegment, onChunkProduced, onProgress, cancellationToken);

            // ⚠ NAMED FROM WHAT HAPPENED, NOT FROM WHAT WAS INTENDED, and the earlier version
            // could not be. It decided the name before any paragraph rendered, from a capability
            // flag — and audio.cpp#626's review removed that flag for this family, so it would now
            // read "proportional" on every job the engine measured perfectly well. Nothing can
            // answer the question before a render: the option is opt-in, the family declares no
            // capability for it, and a published package's contract predates the option, so the
            // only way to know is to ask and see what came back.
            //
            // Which is all this needs. The sidecar is written at the end, so the name can simply
            // describe the run: measured if any paragraph used the engine's own timings, estimated
            // otherwise. That also retires the two ways the old name could lie.
            if (measuredAny) result.Alignment.Aligner = "audiocpp_duration";
            return result;
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// One segment's phoneme stream: the chunks to send, and — when the phonemizer accounted for
    /// every group — how many phoneme symbols each source word became.
    /// </summary>
    /// <param name="GroupSourceWords">For each phoneme group the engine will render, in order, the
    /// index of the segment's source word it came from. This is what turns the engine's measured
    /// group timings into word timings; <see cref="PhonemesPerWord"/> is only the fallback for an
    /// engine that reports no timings.</param>
    internal sealed record SuppliedPhonemes(
        IReadOnlyList<string> Chunks, IReadOnlyList<int>? GroupSourceWords, double[]? PhonemesPerWord);

    /// <summary>
    /// Phonemize <paramref name="text"/> into chunks the engine will accept, or null to let the
    /// engine pronounce the text itself.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every failure here falls back rather than throwing. A paragraph the engine can say in its
    /// own accent is better than a paragraph that does not render, and the difference between
    /// the two readings is a pronunciation, not a document.
    /// </para>
    /// </remarks>
    internal static SuppliedPhonemes? Supply(KokoroPhonemizer g2p, KokoroChunker chunker,
                                            string text, bool british, Action<string> warn)
    {
        var sourceWords = SplitWords(text);
        if (sourceWords.Length == 0) return null;

        var chunks = new List<string>();
        var weights = new double[sourceWords.Length];
        var map = new List<int>();
        var weightsUsable = true;
        var wordOffset = 0;
        var dropped = new List<char>();

        foreach (var chunk in chunker.ChunkForSynthesis(text, british))
        {
            var (rendered, groupSourceWords) = g2p.Phonemize(chunk, british);

            // Filtered, not sent raw. The engine refuses a symbol its vocabulary has no id for
            // — deliberately, because a caller with its own G2P can correct one — and the ONNX
            // Kokoro drops it. Dropping keeps the two engines saying the same thing; refusing
            // would make one stray diacritic anywhere in a document fatal on one of them only.
            var stream = KokoroVocab.KeepKnown(rendered, out var gone);
            if (gone.Count > 0) dropped.AddRange(gone);

            // An entry must not be empty: the engine treats "set but blank" as a caller error
            // rather than a cue to read the text. A chunk that phonemizes to nothing is real
            // (a paragraph of bare punctuation), so the whole segment goes back to the engine.
            if (string.IsNullOrWhiteSpace(stream))
            {
                warn("A part of this paragraph produced no phonemes, so audio.cpp's own "
                     + "pronunciation was used for the whole paragraph.");
                return null;
            }
            chunks.Add(stream);

            // Weights come from the RENDERED stream, before filtering: a dropped diacritic
            // changes a group's length by one and must not be able to change its group COUNT,
            // which is what the map is indexed by.
            var groups = SplitWords(rendered);
            var chunkWords = SplitWords(chunk).Length;
            if (!weightsUsable) { wordOffset += chunkWords; continue; }

            if (groupSourceWords is null || groupSourceWords.Count != groups.Length)
            {
                map.Clear();
                // The phonemizer could not account for every group, so no group→word map is
                // trustworthy; the stream is still good, only the timings lose their weighting.
                weightsUsable = false;
            }
            else
            {
                for (var g = 0; g < groups.Length; g++)
                {
                    var word = wordOffset + groupSourceWords[g];
                    if (word < 0 || word >= weights.Length) { weightsUsable = false; map.Clear(); break; }
                    weights[word] += groups[g].Length;
                    map.Add(word);
                }
            }
            wordOffset += chunkWords;
        }

        if (chunks.Count == 0) return null;

        if (dropped.Count > 0)
            warn("Dropped " + dropped.Count + " phoneme symbol(s) Kokoro has no token for ("
                 + string.Join(", ", dropped.Distinct().Select(c => $"U+{(int)c:X4}")) + ").");

        // The chunker splits at whitespace, so the chunks' words are the segment's words in
        // order. If that ever stops holding, the map's indices point at the wrong words — which
        // is a silently wrong highlight, so it is checked rather than assumed.
        if (wordOffset != sourceWords.Length) weightsUsable = false;

        return new SuppliedPhonemes(chunks, weightsUsable ? map : null, weightsUsable ? weights : null);
    }

    /// <summary>
    /// Word timings for one segment, taking the best source available.
    ///
    /// <para>
    /// ⚠ THREE TIERS, AND ONLY THE FIRST IS A MEASUREMENT. When the engine reports per-group
    /// timings (it predicts the durations the decoder then upsamples by) and the phonemizer said
    /// which word each group came from, the join is exact and the reader's highlight follows the
    /// voice the way it does on the ONNX engine. Failing that, the phoneme COUNTS still weight a
    /// proportional spread. Failing that, word length does. Each tier is a worse answer than the
    /// one above and a better one than nothing, and the sidecar records which was used.
    /// </para>
    /// </summary>
    internal static IReadOnlyList<AlignedWord> Align(
        string text, SuppliedPhonemes? supplied, AudioCppSpeech spoken, double seconds,
        out bool measured)
    {
        measured = false;
        // The counts must agree: the engine cuts groups at ITS view of the stream, the map was
        // built from ours, and a silent disagreement would put every word after it on the wrong
        // audio. Falling back is visibly worse; being one word out is not visible at all.
        if (supplied?.GroupSourceWords is { } map && spoken.Groups.Count == map.Count && map.Count > 0)
        {
            var spans = new KokoroAlignment.GroupSpan[spoken.Groups.Count];
            for (var i = 0; i < spans.Length; i++)
                spans[i] = new KokoroAlignment.GroupSpan(spoken.Groups[i].StartSeconds, spoken.Groups[i].EndSeconds);
            measured = true;
            return [.. KokoroAlignment.WordsFromGroups(text, map, spans, seconds)
                          .Select(w => new AlignedWord { Text = w.Text, StartSeconds = w.StartSec, EndSeconds = w.EndSec })];
        }
        return supplied?.PhonemesPerWord is { } weights
            ? SpreadByWeight(text, weights, seconds)
            : EstimateWords(text, seconds);
    }

    /// <summary>
    /// Spreads <paramref name="totalSeconds"/> across the words of <paramref name="text"/> in
    /// proportion to <paramref name="phonemesPerWord"/>. A word that became no phonemes at all —
    /// a bare "—", an emoji — gets a zero-length marker where the voice has reached, so the
    /// reader still shows every word and the indices stay 1:1 with the source split, exactly as
    /// the ONNX path does with measured durations.
    /// </summary>
    internal static IReadOnlyList<AlignedWord> SpreadByWeight(
        string text, IReadOnlyList<double> phonemesPerWord, double totalSeconds)
    {
        var words = SplitWords(text);
        if (words.Length == 0 || words.Length != phonemesPerWord.Count) return [];

        double total = 0;
        foreach (var w in phonemesPerWord) total += w;
        if (total <= 0) return EstimateWords(text, totalSeconds);

        var aligned = new List<AlignedWord>(words.Length);
        var cursor = 0.0;
        for (var i = 0; i < words.Length; i++)
        {
            var span = totalSeconds * phonemesPerWord[i] / total;
            aligned.Add(new AlignedWord { Text = words[i], StartSeconds = cursor, EndSeconds = cursor + span });
            cursor += span;
        }
        return aligned;
    }

    /// <summary>
    /// Spreads the segment's words across its duration, weighted by word length.
    ///
    /// <para>
    /// The reading used when nothing better is available: a non-English voice, or a run with no
    /// phonemizer data installed. Reuses OmniVoice's estimator rather than growing a second one.
    /// That class is deliberately separate from its engine so it can run without a model, and
    /// its untraced path — which is what an empty <see cref="PhonemeTrace"/> selects — is
    /// precisely "weight each word by its length", which is all that can be known when the
    /// phonemes are the engine's own: audio.cpp phonemizes with eSpeak-ng and hands back no
    /// trace, no phonemes and no timings.
    /// </para>
    /// </summary>
    internal static IReadOnlyList<AlignedWord> EstimateWords(string text, double totalSeconds) =>
        [.. OmniVoiceIpaAlignment.Proportional(text, new PhonemeTrace(), totalSeconds)
                .Select(w => new AlignedWord { Text = w.Text, StartSeconds = w.StartSec, EndSeconds = w.EndSec })];

    /// <summary>The whitespace split every index here is against — the one the ONNX Kokoro
    /// aligns on, so a word means the same thing on both engines.</summary>
    private static string[] SplitWords(string text) =>
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    private void EnsureLoaded()
    {
        if (_tts is not null) return;
        lock (_gate)
        {
            // The phonemizer first: it is the cheaper failure, and a missing data tree is not a
            // failure at all here — the engine has a G2P of its own, so an install without the
            // phonemizer's data renders in eSpeak's accent rather than not rendering.
            if (_g2p is null)
            {
                try
                {
                    _g2p = new KokoroPhonemizer(_dataDir);
                    _chunker = new KokoroChunker(_g2p);
                }
                catch (DirectoryNotFoundException)
                {
                    Console.WriteLine("[audio.cpp] no vernacula-phonemizer data tree; "
                                      + "English will be pronounced by the engine's own eSpeak-ng.");
                }
            }
            _tts ??= new AudioCppTts(_modelPath, _backends, _threads);
        }
    }

    public void Dispose() => _tts?.Dispose();
}
#endif
