using AudioCpp;

namespace Vernacula.AudioCpp;

/// <summary>
/// Kokoro's preset voices as audio.cpp's <c>kokoro_tts</c> family carries them, and the
/// language each one demands.
/// </summary>
/// <remarks>
/// <para>
/// ⚠ THE VOICE DECIDES THE LANGUAGE, and a mismatch is a hard failure rather than a fallback:
/// asking for <c>bm_george</c> with <c>en-us</c> fails the run with "voice bm_george requires
/// lang_code=b but request resolved to a". So nothing here lets the two be chosen apart — the
/// language is a function of the voice's first letter, which is exactly what Kokoro's naming
/// convention encodes.
/// </para>
/// <para>
/// ⚠ THIS TABLE IS MEASURED, NOT COPIED, AND IT NOW LISTS ALL 54. It used to list 41: the
/// thirteen Japanese and Chinese voices were excluded because the installed package refused
/// them, each failure naming the package rather than the voice — "Kokoro UniDic resources are
/// not bundled in this GGUF" for the five Japanese, "Kokoro vocab is missing phoneme symbol: H"
/// for the eight Chinese.
/// </para>
/// <para>
/// ⚠ BOTH REFUSALS WERE ABOUT THE ENGINE'S OWN G2P, WHICH THIS APP NO LONGER USES. Every voice
/// embedding was in the package all along — 54 of 54 sidecars — and only the grapheme-to-phoneme
/// resources were missing. Supplying the phonemes means no built-in G2P runs, so the Japanese
/// refusal simply does not arise: measured, jf_alpha and jm_kumo render 2.38 s and 3.08 s of
/// audio where the text path still refuses outright. The Chinese one turned out to be stale
/// besides — that engine defect was fixed upstream, and those voices now work on either path.
/// See docs/investigations/audiocpp_multilingual_investigation.md.
/// </para>
/// </remarks>
public static class AudioCppKokoroVoices
{
    /// <summary>The engine's own language code per voice-name prefix, as the request wants it.</summary>
    private static readonly Dictionary<char, string> EngineLanguages = new()
    {
        ['a'] = "en-us", ['b'] = "en-gb", ['e'] = "es", ['f'] = "fr-fr",
        ['h'] = "hi",    ['i'] = "it",    ['p'] = "pt-br",
        ['j'] = "ja",    ['z'] = "zh",
    };

    /// <summary>
    /// The vernacula-phonemizer code for the same prefix, so the reader annotates a rendered
    /// job in the language it was actually spoken in. Separate from
    /// <see cref="EngineLanguages"/> because they are two different vocabularies that merely
    /// happen to be keyed alike — the phonemizer has no "en-us", and its Brazilian Portuguese
    /// is "pt-BR".
    /// </summary>
    private static readonly Dictionary<char, string> PhonemizerLanguages = new()
    {
        ['a'] = "en", ['b'] = "en-GB", ['e'] = "es", ['f'] = "fr",
        ['h'] = "hi", ['i'] = "it",    ['p'] = "pt-BR",
        // ⚠ `cmn`, NOT `zh`. The engine's language code and the phonemizer's are two different
        // vocabularies that merely overlap, which is why these tables are separate at all.
        ['j'] = "ja", ['z'] = "cmn",
    };

    /// <summary>Every voice this package renders, grouped by language in the order it lists them.</summary>
    public static readonly string[] All =
    [
        // American English (a → en-us)
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
        "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
        "am_puck", "am_santa",
        // British English (b → en-gb)
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        // Spanish (e), French (f), Hindi (h), Italian (i), Brazilian Portuguese (p)
        "ef_dora", "em_alex", "em_santa",
        "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "pf_dora", "pm_alex", "pm_santa",
        // Japanese (j → ja), Mandarin (z → zh)
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ];

    /// <summary>Whether this build would offer <paramref name="voice"/> at all.</summary>
    public static bool IsKnown(string? voice) => voice is not null && Array.IndexOf(All, voice) >= 0;

    /// <summary>
    /// The engine's language for <paramref name="voice"/>, defaulting to American English for a
    /// voice this table does not know — a package with more voices than the one measured must
    /// still be usable, and the engine rejects a genuinely wrong pairing itself.
    /// </summary>
    public static string EngineLanguage(string? voice) => Lookup(EngineLanguages, voice, "en-us");

    /// <summary>The phonemizer language the reader annotates such a job in.</summary>
    public static string PhonemizerLanguage(string? voice) => Lookup(PhonemizerLanguages, voice, "en");

    private static string Lookup(Dictionary<char, string> table, string? voice, string fallback) =>
        !string.IsNullOrEmpty(voice) && table.TryGetValue(voice[0], out var code) ? code : fallback;
}

/// <summary>
/// One phoneme group the engine rendered: the group's phonemes, and where in the returned buffer
/// it landed. A group is a run between Kokoro's own space tokens — one spoken word — and the
/// timings are the model's predicted per-token durations, not an estimate over the buffer.
/// </summary>
public sealed record AudioCppPhonemeGroup(string Phonemes, double StartSeconds, double EndSeconds);

/// <summary>Audio plus the per-group timings the engine reported, empty when it reported none.</summary>
public sealed record AudioCppSpeech(float[] Audio, IReadOnlyList<AudioCppPhonemeGroup> Groups);

/// <summary>
/// Synthesises through audio.cpp's C ABI, the TTS counterpart to <see cref="AudioCppAsr"/>.
/// </summary>
/// <remarks>
/// <para>
/// Deliberately thin, for the same reason that class is: the segmentation, the streaming, the
/// per-segment files and the alignment sidecar are all Vernacula's own and unchanged, and only
/// "this text, this voice → these samples" crosses the boundary. What is being tested is the
/// ABI, not a second synthesis pipeline.
/// </para>
/// <para>
/// ⚠ THE ABI REPORTS NO WORD TIMINGS FOR SYNTHESIS. <c>SupportsTimestamps</c> is false on this
/// family and <c>result.Words</c> comes back empty, which is the reverse of the ASR path, where
/// measured word boundaries were the reason to prefer it. The caller has to estimate them; this
/// class returns audio and says nothing about words rather than inventing any here.
/// </para>
/// <para>
/// One session is created and reused. Creating one per paragraph would reload 190 MB of weights
/// each time, and keeping the loaded model behind the session is the whole reason to embed
/// rather than shell out.
/// </para>
/// </remarks>
public sealed class AudioCppTts : IDisposable
{
    /// <summary>
    /// What this family produces, verified against the installed package: 24 kHz mono — the
    /// same rate as the ONNX Kokoro, so nothing downstream needs a second case. The caller
    /// needs it before any audio exists, which is why it is a constant; <see cref="Speak"/>
    /// checks the buffer that comes back actually is at this rate rather than trusting it.
    /// </summary>
    public const int SampleRate = 24_000;

    /// <summary>The family name to load the package as.</summary>
    private const string Family = "kokoro_tts";

    private readonly AudioCppRegistry _registry;
    private readonly AudioCppModel    _model;
    private readonly AudioCppSession  _session;
    private readonly object           _gate = new();

    /// <summary>The backend the session actually opened on.</summary>
    public string Backend { get; } = "";

    /// <summary>
    /// Whether this engine has accepted a request for timings so far. Starts true and goes false
    /// only once one has been refused as an unknown option.
    ///
    /// <para>
    /// ⚠ NOT A PRE-RENDER ANSWER, and it used to pretend to be one. It was read off
    /// <c>SupportsTimestamps</c>, which the engine no longer sets for this family at all, so it
    /// would now say "no" on an engine that reports timings perfectly well. Nothing can answer
    /// this before a render — see <see cref="ReturnTimestamps"/> — so callers that need to
    /// describe what a job actually got should look at what came back instead.
    /// </para>
    /// </summary>
    public bool ReportsTimings => !_timingsRefused;

    /// <param name="modelPath">The package's .gguf, as <see cref="ResolveKokoro"/> finds it.</param>
    /// <param name="backends">
    /// Backends to try, in order, taking the first that opens — the same "auto is a list"
    /// treatment the ASR path uses, and for the same reason: whether the engine has CUDA
    /// registered is a property of how it was BUILT, which no caller can see.
    /// </param>
    public AudioCppTts(string modelPath, IReadOnlyList<string> backends, int threads = 1)
    {
        ArgumentOutOfRangeException.ThrowIfZero(backends.Count);
        _registry = AudioCppRegistry.Create();
        try
        {
            _model = _registry.Load(modelPath, new ModelConfig(Family));

            AudioCppException? last = null;
            AudioCppSession? session = null;
            for (int i = 0; i < backends.Count; i++)
            {
                try
                {
                    session = _model.CreateSession("tts", "offline",
                                                   new BackendConfig(backends[i], 0, threads));
                    Backend = backends[i];
                    break;
                }
                catch (AudioCppException failure) when (i < backends.Count - 1)
                {
                    last = failure;
                }
            }
            _session = session
                ?? throw (Exception?)last
                ?? new InvalidOperationException("no backend opened and none reported why");
        }
        catch
        {
            // Load and CreateSession both throw; without this a failure leaks the registry
            // and, on the second, the model as well.
            _model?.Dispose();
            _registry.Dispose();
            throw;
        }
    }

    /// <summary>
    /// One pass of synthesis. <paramref name="voice"/> is a preset id and decides the language
    /// (see <see cref="AudioCppKokoroVoices"/>); <paramref name="speed"/> is a rate multiplier.
    /// </summary>
    /// <remarks>
    /// Text of any length may be passed: the family chunks internally on its own
    /// <c>text_chunk_size</c> and joins, so unlike the ONNX Kokoro there is no context window
    /// for the caller to cut around.
    /// </remarks>
    public float[] Speak(string text, string voice, float speed)
        => Speak(text, null, voice, speed);

    /// <summary>
    /// One pass of synthesis from a phoneme stream the CALLER produced, bypassing the engine's
    /// built-in eSpeak-ng G2P (upstream audio.cpp#577). <paramref name="phonemes"/> is one
    /// Kokoro-alphabet string per chunk, rendered in order and merged into one buffer; passing
    /// null or an empty list falls back to the engine's own pronunciation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠ THE LIST IS THE INTERFACE, not a convenience. On the text path the family chunks on its
    /// own <c>text_chunk_size</c>; on this one it cannot, because where a phoneme stream may be
    /// cut is known only to the G2P that produced it. So the caller cuts, one entry per chunk,
    /// each at most 510 symbols and none of them empty — the engine refuses an empty entry
    /// rather than quietly speaking the text instead.
    /// </para>
    /// <para>
    /// ⚠ AND THE ENGINE VALIDATES WHAT IT IS GIVEN, where it does not validate its own G2P's
    /// output: a symbol outside Kokoro's vocabulary fails the run naming the entry. That is the
    /// right way round — a caller can correct a stream it generated — but it means the caller
    /// must filter before sending, not after being refused.
    /// </para>
    /// <para>
    /// <paramref name="text"/> is still required and its language must still match the voice:
    /// it is what the engine reports and caches on, not what it speaks.
    /// </para>
    /// </remarks>
    public float[] Speak(string text, IReadOnlyList<string>? phonemes, string voice, float speed)
        => SpeakAligned(text, phonemes, voice, speed).Audio;

    /// <summary>
    /// <see cref="Speak(string, IReadOnlyList{string}, string, float)"/>, also returning where each
    /// phoneme group landed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠ MEASURED, NOT ESTIMATED, AND THAT IS NEW. This family used to report
    /// <c>SupportsTimestamps=false</c> and hand back no words, so a caller had to spread a
    /// paragraph's words across its buffer by some proxy for length. Kokoro's architecture predicts
    /// a per-token frame count BEFORE the decoder runs and the decoder upsamples by exactly those
    /// counts, so the information was always there — it simply was not reported. It is now
    /// (upstream audio.cpp, `word_timestamps` on the kokoro_tts family).
    /// </para>
    /// <para>
    /// ⚠ THE LABEL IS PHONEMIC, because nothing in the engine maps tokens back to written words:
    /// the built-in G2P keeps no span, and on the supplied-phoneme path there is no text to map
    /// to. A caller whose own G2P produced the stream knows which of its words became which group
    /// and can join the two; that is what <c>KokoroAlignment</c> does.
    /// </para>
    /// <para>
    /// An older engine reports nothing here rather than failing, so <see cref="AudioCppSpeech.Groups"/>
    /// comes back empty and the caller falls back to whatever it did before.
    /// </para>
    /// </remarks>
    public AudioCppSpeech SpeakAligned(string text, IReadOnlyList<string>? phonemes, string voice, float speed)
    {
        ArgumentException.ThrowIfNullOrEmpty(voice);

        // The session is shared state across paragraphs and the ABI makes no thread-safety
        // promise. The synthesis loop is sequential today, so this costs nothing; it is here so
        // that a caller that parallelises paragraphs later gets a slow answer, not a wrong one.
        lock (_gate)
        {
            AudioBuffer audio;
            IReadOnlyList<AudioCppPhonemeGroup> groups;
            try
            {
                (audio, groups) = Render(text, phonemes, voice, speed);
            }
            catch (AudioCppException failure) when (failure.Message.Contains(UnknownSymbol, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    phonemes is { Count: > 0 }
                        ? DescribeRefusedStream(phonemes, failure)
                        : DescribeUnsayable(text, voice, speed, failure),
                    failure);
            }

            // A rate other than 24 kHz would play back at the wrong pitch and the only symptom
            // would be that the voice sounds wrong, so say what happened instead.
            if (audio.SampleRate != SampleRate)
                throw new InvalidOperationException(
                    $"audio.cpp {Family} returned {audio.SampleRate} Hz audio; this backend and "
                    + $"everything downstream of it assume {SampleRate} Hz.");
            if (audio.Channels != 1)
                throw new InvalidOperationException(
                    $"audio.cpp {Family} returned {audio.Channels} channels; mono is assumed.");

            return new AudioCppSpeech(audio.Samples, groups);
        }
    }

    private AudioBuffer Render(string text, string voice, float speed)
        => Render(text, null, voice, speed).Audio;

    private (AudioBuffer Audio, IReadOnlyList<AudioCppPhonemeGroup> Groups) Render(
        string text, IReadOnlyList<string>? phonemes, string voice, float speed)
    {
        try
        {
            return RenderOnce(text, phonemes, voice, speed);
        }
        catch (AudioCppException failure) when (!_timingsRefused && IsUnknownOption(failure, ReturnTimestamps))
        {
            // An engine older than #626. Remember it, so this costs one refused request per
            // session rather than one per paragraph, and render again without asking.
            Console.WriteLine($"[audio.cpp] this engine does not accept {ReturnTimestamps}; "
                              + "word timings will be estimated rather than measured.");
            _timingsRefused = true;
            return RenderOnce(text, phonemes, voice, speed);
        }
    }

    /// <summary>Whether <paramref name="failure"/> is the engine rejecting <paramref name="option"/>
    /// as one it does not know — as opposed to refusing its VALUE, which is our bug, not its age.</summary>
    private static bool IsUnknownOption(AudioCppException failure, string option) =>
        failure.Message.Contains("unknown", StringComparison.OrdinalIgnoreCase)
        && failure.Message.Contains(option, StringComparison.Ordinal);

    private (AudioBuffer Audio, IReadOnlyList<AudioCppPhonemeGroup> Groups) RenderOnce(
        string text, IReadOnlyList<string>? phonemes, string voice, float speed)
    {
        using var request = new AudioCppRequest();
        request.SetText(text, AudioCppKokoroVoices.EngineLanguage(voice));
        request.SetVoiceId(voice);
        request.SetSpeakingRate(speed);
        // Never an empty list: "set but holding nothing" is a caller error to the engine, and
        // rightly so, but here it just means this caller had no phonemes to offer for this
        // paragraph and wants the built-in G2P.
        if (phonemes is { Count: > 0 }) request.SetOptionArray(SuppliedPhonemes, phonemes);
        if (!_timingsRefused) request.SetOption(ReturnTimestamps, "true");

        using var result = _session.Run(request);
        var audio = result.Audio
            ?? throw new InvalidOperationException(
                $"audio.cpp {Family} returned no audio for {text.Length} characters.");
        // Samples, not seconds, across the ABI — and the rate is the buffer's own, not the
        // constant, because a buffer at the wrong rate is checked for by the caller and a timing
        // divided by the wrong rate would silently agree with it.
        var rate = audio.SampleRate > 0 ? audio.SampleRate : SampleRate;
        var groups = new AudioCppPhonemeGroup[result.Words.Count];
        for (var i = 0; i < groups.Length; i++)
        {
            var w = result.Words[i];
            groups[i] = new AudioCppPhonemeGroup(w.Word, w.StartSample / (double)rate, w.EndSample / (double)rate);
        }
        return (audio, groups);
    }

    /// <summary>The engine's own words for "this phoneme has no token id".</summary>
    private const string UnknownSymbol = "Kokoro vocab is missing phoneme symbol";

    /// <summary>The request option carrying a caller's phoneme stream (audio.cpp#577).</summary>
    private const string SuppliedPhonemes = "phonemes";

    /// <summary>
    /// The request option asking for per-group timings (audio.cpp#626), which are OPT-IN.
    /// </summary>
    /// <remarks>
    /// ⚠ AND THERE IS NO WAY TO ASK WHETHER THE ENGINE HAS IT. The family declares no capability
    /// for this — review deliberately removed the one the change originally added, because a
    /// phoneme-group alignment is not the written-word timeline <c>word_timestamps</c> means
    /// elsewhere — and the option does not appear in a published package's declared options
    /// either, since every shipped contract predates it. Measured against the installed package:
    /// <c>SupportsTimestamps=False</c>, request options <c>language, seed, phonemes,
    /// text_chunk_size</c>, and the engine serves <c>return_timestamps</c> regardless. So the only
    /// way to find out is to ask and see what happens, which is what <see cref="_timingsRefused"/>
    /// records.
    /// </remarks>
    private const string ReturnTimestamps = "return_timestamps";

    /// <summary>
    /// Set once an engine has rejected <see cref="ReturnTimestamps"/> as an unknown option, i.e.
    /// one built before audio.cpp#626. AUDIOCPP_NATIVE_DIR is read at BUILD time and points this
    /// at whatever engine someone has, so that is a real configuration and not a formality — and
    /// an unknown request option is a HARD refusal, not a degraded result, so it has to be caught
    /// and retried rather than allowed to fail the paragraph.
    /// </summary>
    private bool _timingsRefused;

    /// <summary>
    /// A refusal of OUR OWN stream is a different bug from a refusal of the engine's, and must
    /// not borrow the other message: nothing here was pronounced by eSpeak-ng, so there is no
    /// word to name and no newer engine to recommend.
    /// </summary>
    /// <remarks>
    /// The caller filters its stream against Kokoro's vocabulary before sending, so reaching
    /// this means the two vocabularies disagree — the package's embedded table has fewer symbols
    /// than the one the caller filtered against. The engine names the offending entry and symbol
    /// itself; the useful addition is the entry's CONTENT, which is what identifies the rule
    /// that emitted it.
    /// </remarks>
    private static string DescribeRefusedStream(IReadOnlyList<string> phonemes, AudioCppException failure)
    {
        // "Kokoro supplied phoneme entry 7: Kokoro vocab is missing phoneme symbol: R"
        var entry = Entry(failure.Message) is { } index && index < phonemes.Count
            ? $"Entry {index} of {phonemes.Count} was: {phonemes[index]}"
            : $"The entry was not named in the message; {phonemes.Count} were sent.";

        return "audio.cpp's Kokoro refused a phoneme in the stream this app supplied. " + entry
             + "\n\nThe stream is filtered against Kokoro's vocabulary before it is sent, so a "
             + "refusal means this package's embedded vocabulary is missing a symbol that "
             + "filtering kept — the two tables disagree. Use the ONNX Kokoro engine for this "
             + "document, and report the symbol below.\n\n"
             + $"Engine's own message: {failure.Message}";
    }

    /// <summary>The entry index out of "…entry N: …", or null when the message has no such shape.</summary>
    private static int? Entry(string message)
    {
        const string marker = "entry ";
        var at = message.IndexOf(marker, StringComparison.Ordinal);
        if (at < 0) return null;
        var digits = at + marker.Length;
        var end = digits;
        while (end < message.Length && char.IsAsciiDigit(message[end])) end++;
        return end > digits && int.TryParse(message[digits..end], out var index) ? index : null;
    }

    /// <summary>
    /// Turns the engine's report of an unusable phoneme into the question a reader actually has:
    /// WHICH WORD did it refuse?
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠ UNREACHABLE ON THE PINNED ENGINE, AND KEPT ANYWAY. This was a defect in the engine, not
    /// in the text: audio.cpp's Kokoro phonemizes with eSpeak-ng and then THREW on any symbol its
    /// vocab had no id for — where the reference implementation (misaki/KModel,
    /// <c>filter(None, map(vocab.get, phonemes))</c>) drops it and carries on. eSpeak glottalises
    /// /t/ before a syllabic nasal, so "button" becomes <c>bˈæʔn̩</c> and the engine refused the
    /// syllabic mark its own G2P had just produced — which made the whole <c>-tten</c>/<c>-tton</c>
    /// family fatal, one occurrence anywhere in a document being enough.
    /// </para>
    /// <para>
    /// Fixed upstream in audio.cpp #564 (drop instead of throw) and #565 (the English arm of the
    /// misaki port, which turns the mark into the <c>ᵊ</c> Kokoro was trained on rather than merely
    /// dropping it). The bindings pin an engine that has both, so this path does not run there.
    /// </para>
    /// <para>
    /// It stays because the pin is not a guarantee: <c>AUDIOCPP_NATIVE_DIR</c> is read at BUILD
    /// time and points this at whatever engine someone has, which may predate those fixes. See
    /// docs/investigations/audiocpp_tts_backend_investigation.md Runs 6-7 and 11.
    /// </para>
    /// <para>
    /// So the words are found by asking, one at a time. That is N more engine calls, which is
    /// affordable ONLY because this runs on a path that has already failed: the alternative is a
    /// message naming a combining codepoint, which tells the reader nothing about their document.
    /// </para>
    /// </remarks>
    private string DescribeUnsayable(string text, string voice, float speed, AudioCppException failure)
    {
        var offenders = new List<string>();
        // Distinct, because one bad word usually appears more than once, and ordered as written so
        // the reader can find the first one in their document.
        foreach (var word in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Distinct(StringComparer.Ordinal))
        {
            if (offenders.Count >= 5) break;      // enough to see the pattern; the rest add noise
            try { Render(word, voice, speed); }
            catch (AudioCppException probe) when (probe.Message.Contains(UnknownSymbol, StringComparison.Ordinal))
            {
                offenders.Add(word);
            }
            catch (AudioCppException) { /* a word that fails for some OTHER reason is not the one */ }
        }

        string named = offenders.Count > 0
            ? $"It cannot say: {string.Join(", ", offenders.Select(w => $"\"{w}\""))}."
            : "The word could not be narrowed down — the paragraph fails as a whole but no single "
              + "word does.";

        return $"audio.cpp's Kokoro refused a phoneme its own pronunciation produced. {named}\n\n"
             + "This is a limitation of that engine: it rejects any phoneme missing from Kokoro's "
             + "vocabulary instead of dropping it the way the reference implementation does, and "
             + "its eSpeak-ng pronunciation of words like \"button\" and \"written\" produces one. "
             + "It was fixed in audio.cpp 31d00b5c, so the engine this was built against is older "
             + "than that — rebuild with a current one. Until then, use the ONNX Kokoro engine "
             + "for this document.\n\n"
             + $"Engine's own message: {failure.Message}";
    }

    /// <summary>
    /// The Kokoro package under <paramref name="modelsRoot"/>, or null. Mirrors
    /// <see cref="AudioCppAsr.ResolveParakeet"/>: the ABI loads a MODEL, not a models root, so
    /// the file has to be found rather than the directory handed over.
    /// </summary>
    public static string? ResolveKokoro(string modelsRoot)
    {
        if (!Directory.Exists(modelsRoot)) return null;

        var found = Directory
            .EnumerateFiles(modelsRoot, "kokoro-82m*.gguf", SearchOption.AllDirectories)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToList();

        // With two quantisations installed the ordinal sort decides, which means f16 beats q8_0
        // for no better reason than the alphabet. Say so rather than let someone wonder why the
        // model they installed second is the one being used.
        if (found.Count > 1)
            Console.WriteLine($"[audio.cpp] {found.Count} kokoro packages under {modelsRoot}; "
                              + $"using {Path.GetFileName(found[0])}");

        return found.FirstOrDefault();
    }

    public void Dispose()
    {
        _session.Dispose();
        _model.Dispose();
        _registry.Dispose();
    }
}
