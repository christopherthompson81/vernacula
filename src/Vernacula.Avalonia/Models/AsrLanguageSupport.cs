using System.Collections.Frozen;

namespace Vernacula.App.Models;

/// <summary>
/// Static table of ISO 639-1 language codes supported by each ASR backend,
/// plus helpers for LID-driven backend selection and compatibility checks.
///
/// <para>
/// The per-backend sets are derived from:
/// <list type="bullet">
/// <item><see cref="AsrBackend.Parakeet"/>: NVIDIA Parakeet TDT 0.6B v3 —
/// 25 European languages (per the HF model card).</item>
/// <item><see cref="AsrBackend.Cohere"/>: Cohere Transcribe — 14 languages
/// (mirrors <c>SettingsViewModel.CohereLanguages</c>).</item>
/// <item><see cref="AsrBackend.Qwen3Asr"/>: Qwen3-ASR 1.7B — 29 languages
/// (mirrors <c>SettingsViewModel.Qwen3AsrLanguages</c>).</item>
/// <item><see cref="AsrBackend.VibeVoice"/>: VibeVoice-ASR — 12 languages
/// explicitly evaluated in the model card. The card claims 50+ supported
/// via code-switching, but only the evaluated set is enumerated, so we
/// list those (any broader use is opt-in via "auto").</item>
/// </list>
/// </para>
///
/// <para>
/// Codes are lowercase ISO 639-1. Entries in the force-language pickers
/// that are empty-string (meaning "auto-detect") are not listed here;
/// this table is about *positive* language support.
/// </para>
/// </summary>
public static class AsrLanguageSupport
{
    // NVIDIA Parakeet TDT 0.6B v3 — 25 European languages
    // (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
    private static readonly FrozenSet<string> ParakeetLangs = new HashSet<string>
    {
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    }.ToFrozenSet();

    private static readonly FrozenSet<string> CohereLangs = new HashSet<string>
    {
        "ar", "de", "el", "en", "es", "fr", "it", "ja",
        "ko", "nl", "pl", "pt", "vi", "zh",
    }.ToFrozenSet();

    // Official Qwen3-ASR 1.7B supported languages (https://github.com/QwenLM/Qwen3-ASR).
    // "tl" (Tagalog) covers the officially listed "Filipino". Cantonese (yue) omitted
    // pending token-ID support in Qwen3Asr.IsoToLanguageName.
    private static readonly FrozenSet<string> Qwen3AsrLangs = new HashSet<string>
    {
        "zh", "en", "ar", "de", "fr", "es", "pt", "id", "it", "ko",
        "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da",
        "fi", "pl", "cs", "tl", "fa", "el", "hu", "mk", "ro",
    }.ToFrozenSet();

    // VibeVoice-ASR — 12 languages explicitly evaluated in the model card
    // (https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md).
    // The card claims 50+ via code-switching but doesn't enumerate them.
    private static readonly FrozenSet<string> VibeVoiceLangs = new HashSet<string>
    {
        "en", "fr", "de", "it", "ja", "ko", "pt", "ru", "es", "th", "vi", "zh",
    }.ToFrozenSet();

    // OpenAI Whisper large-v3-turbo — 99 languages from the model's
    // generation_config.json lang_to_id dict. Covers the widest surface of
    // any backend Vernacula ships: all of Parakeet / Cohere / Qwen3 plus many
    // that no other backend supports (Arabic, Japanese, Korean, Turkish, a
    // long tail of low-resource languages, and full Chinese incl. Cantonese).
    // Codes as they appear in the Whisper vocab, with one mapping applied:
    // Whisper's "<|jw|>" is the pre-1989 ISO code for Javanese; we store "jv"
    // here to stay consistent with NormalizeIso, and WhisperTurbo maps back
    // at token-lookup time. "haw" / "yue" are ISO 639-3 codes without 639-1
    // equivalents — stored as-is, matching Whisper's convention.
    private static readonly FrozenSet<string> WhisperTurboLangs = new HashSet<string>
    {
        "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
        "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
        "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
        "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
        "jv", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
        "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
        "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
        "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
        "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
        "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh",
    }.ToFrozenSet();

    // AI4Bharat IndicConformer 600M — the 22 official Indian languages, mixed
    // ISO 639-1 / 639-3 convention because five have no 639-1 assignment:
    // brx (Bodo), doi (Dogri), kok (Konkani), mai (Maithili), mni (Manipuri),
    // sat (Santali). The remaining seventeen use their 639-1 forms. The
    // model's language_spans.json keys use the same convention, so matching
    // is straightforward — we do not normalize between 639-1 / 639-3 here.
    private static readonly FrozenSet<string> IndicConformerLangs = new HashSet<string>
    {
        "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok",
        "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa",
        "sa", "sat", "sd", "ta", "te", "ur",
    }.ToFrozenSet();

    /// <summary>
    /// Map deprecated ISO 639-1 codes to their modern equivalents and
    /// lowercase the result. Used at every lookup boundary so the rest of
    /// the codebase only ever sees current codes.
    ///
    /// <para>VoxLingua107's <c>lang_map.json</c> emits the pre-1989 ISO
    /// 639-1 codes for two languages:
    /// <list type="bullet">
    /// <item><c>iw</c> → <c>he</c> (Hebrew, retired 1989)</item>
    /// <item><c>jw</c> → <c>jv</c> (Javanese, retired 1989)</item>
    /// </list>
    /// No current ASR backend supports either language, so this is forward
    /// compatibility — when one is added (under the modern code, which
    /// every model card uses today) the LID-driven fallback will Just Work
    /// without revisiting this mapping.
    /// </para>
    ///
    /// <para>
    /// Genuinely-ambiguous cases (different ISO codes for mutually
    /// intelligible languages) are deliberately NOT aliased here — see
    /// the "Cross-language fallback policy" note above this method.
    /// </para>
    /// </summary>
    public static string NormalizeIso(string iso)
    {
        if (string.IsNullOrWhiteSpace(iso)) return iso ?? string.Empty;
        return iso.ToLowerInvariant() switch
        {
            "iw" => "he",
            "jw" => "jv",
            var c => c,
        };
    }

    /// <summary>
    /// Returns the ISO 639-1 language codes the given ASR backend can
    /// transcribe. Result is lowercase.
    /// </summary>
    public static FrozenSet<string> Get(AsrBackend backend) => backend switch
    {
        AsrBackend.Parakeet       => ParakeetLangs,
        AsrBackend.Cohere         => CohereLangs,
        AsrBackend.Qwen3Asr       => Qwen3AsrLangs,
        AsrBackend.VibeVoice      => VibeVoiceLangs,
        AsrBackend.IndicConformer => IndicConformerLangs,
        AsrBackend.WhisperTurbo   => WhisperTurboLangs,
        _                         => FrozenSet<string>.Empty,
    };

    // ── Cross-language fallback policy ───────────────────────────────────────
    //
    // VoxLingua107 distinguishes some closely related languages that no
    // installed ASR backend supports under the same code. Examples:
    //
    //   sr (Serbian)   — no backend; Croatian (hr) is mutually intelligible.
    //   bs (Bosnian)   — no backend; Croatian (hr) is mutually intelligible.
    //   no/nn (Norwegian) — no backend; Swedish/Danish are nearby cousins.
    //
    // We deliberately do NOT auto-alias these in NormalizeIso. Reasoning:
    //
    //   1. Script mismatch matters. Serbian commonly uses Cyrillic; Parakeet's
    //      Croatian support is Latin-script only. Silent fallback would
    //      produce nonsense or transliterated mush on Cyrillic input.
    //   2. Auto-routing erases user agency. A Serbian speaker may prefer to
    //      hear "no direct support" over a confidently-wrong Croatian-flavoured
    //      transcript with no warning.
    //   3. Preferred UX is an *explicit affordance* in the AsrMismatch popup
    //      — "Serbian isn't directly supported; try Croatian (Latin script)?"
    //      — surfaced as a separate user choice, not as `Supports("sr")=true`.
    //
    // When that affordance is built, it should consult a separate
    // "near-language fallback" map and clearly label the substitution in the
    // UI and in the DB metadata. Until then, leaving these languages as
    // "unsupported" is honest and recoverable.
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// True when <paramref name="backend"/> can transcribe the given ISO
    /// 639-1 code. Case-insensitive; deprecated codes are normalized via
    /// <see cref="NormalizeIso"/>.
    /// </summary>
    public static bool Supports(AsrBackend backend, string iso) =>
        !string.IsNullOrWhiteSpace(iso) && Get(backend).Contains(NormalizeIso(iso));

    /// <summary>
    /// Returns the list of ASR backends that support <paramref name="iso"/>,
    /// ordered by preference for generic use:
    /// Qwen3-ASR (widest coverage) → Cohere → Parakeet → VibeVoice.
    /// Used when LID detects a language the user's current backend can't
    /// handle and we need to suggest an alternative. Deprecated codes are
    /// normalized via <see cref="NormalizeIso"/>.
    /// </summary>
    public static IReadOnlyList<AsrBackend> BackendsSupporting(string iso)
    {
        if (string.IsNullOrWhiteSpace(iso)) return Array.Empty<AsrBackend>();
        string code = NormalizeIso(iso);
        // IndicConformer leads the preference order but its 22-language set
        // is disjoint from Parakeet/Cohere/VibeVoice and barely overlaps
        // Qwen3-ASR ("hi" is the only overlap), so promoting it to first
        // only effects Indic languages — other codes still pick the
        // generalist backends in the usual order.
        // IndicConformer first (specialist; small disjoint set).
        // Whisper second — its 99-language set is the widest by far and
        // covers ~everything, so it's the best generalist when the detected
        // language isn't in the more tightly-scoped sets.
        // Qwen3-ASR / Cohere / Parakeet / VibeVoice round out in preference
        // order for the languages they do cover.
        var order = new[] {
            AsrBackend.IndicConformer,
            AsrBackend.WhisperTurbo,
            AsrBackend.Qwen3Asr,
            AsrBackend.Cohere,
            AsrBackend.Parakeet,
            AsrBackend.VibeVoice,
        };
        return order.Where(b => Get(b).Contains(code)).ToArray();
    }

    /// <summary>
    /// Preferred ASR backend for a detected language, or null if no
    /// backend supports it. Preference order matches
    /// <see cref="BackendsSupporting"/>.
    /// </summary>
    public static AsrBackend? PickBestBackend(string iso)
    {
        var options = BackendsSupporting(iso);
        return options.Count > 0 ? options[0] : null;
    }

    /// <summary>Human-readable display name for a backend.</summary>
    public static string DisplayName(AsrBackend backend) => backend switch
    {
        AsrBackend.Parakeet       => "Parakeet",
        AsrBackend.Cohere         => "Cohere Transcribe",
        AsrBackend.Qwen3Asr       => "Qwen3-ASR",
        AsrBackend.VibeVoice      => "VibeVoice-ASR",
        AsrBackend.IndicConformer => "IndicConformer",
        AsrBackend.WhisperTurbo   => "Whisper Turbo",
        _                         => backend.ToString(),
    };

    /// <summary>
    /// Maps the pipeline's ASR model-name string to an <see cref="AsrBackend"/>
    /// enum value. Returns null for unknown model names (safe fallback).
    /// </summary>
    public static AsrBackend? BackendOf(string modelName) => modelName switch
    {
        "nvidia/parakeet-tdt-0.6b-v3"                  => AsrBackend.Parakeet,
        "CohereLabs/cohere-transcribe-03-2026"         => AsrBackend.Cohere,
        "Qwen/Qwen3-ASR-1.7B"                          => AsrBackend.Qwen3Asr,
        "vibevoice/vibevoice-asr"                      => AsrBackend.VibeVoice,
        "ai4bharat/indic-conformer-600m-multilingual"  => AsrBackend.IndicConformer,
        "openai/whisper-large-v3-turbo"                => AsrBackend.WhisperTurbo,
        _                                              => (AsrBackend?)null,
    };

    /// <summary>Inverse of <see cref="BackendOf"/>: enum → pipeline model-name string.</summary>
    public static string ModelName(AsrBackend backend) => backend switch
    {
        AsrBackend.Parakeet       => "nvidia/parakeet-tdt-0.6b-v3",
        AsrBackend.Cohere         => "CohereLabs/cohere-transcribe-03-2026",
        AsrBackend.Qwen3Asr       => "Qwen/Qwen3-ASR-1.7B",
        AsrBackend.VibeVoice      => "vibevoice/vibevoice-asr",
        AsrBackend.IndicConformer => "ai4bharat/indic-conformer-600m-multilingual",
        AsrBackend.WhisperTurbo   => "openai/whisper-large-v3-turbo",
        _ => throw new ArgumentOutOfRangeException(nameof(backend)),
    };

    // English display names for every ISO 639-1 code that appears in any
    // backend's supported set. Centralised here so the force-language picker
    // (AsrMismatchDialog) can label languages for backends like Parakeet
    // and VibeVoice that don't otherwise carry per-code display names.
    private static readonly FrozenDictionary<string, string> IsoDisplayNamesMap =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["ar"] = "Arabic",      ["bg"] = "Bulgarian",  ["cs"] = "Czech",
            ["da"] = "Danish",      ["de"] = "German",     ["el"] = "Greek",
            ["en"] = "English",     ["es"] = "Spanish",    ["et"] = "Estonian",
            ["fa"] = "Persian",     ["fi"] = "Finnish",    ["fr"] = "French",
            ["hi"] = "Hindi",       ["hr"] = "Croatian",   ["hu"] = "Hungarian",
            ["id"] = "Indonesian",  ["it"] = "Italian",    ["ja"] = "Japanese",
            ["ko"] = "Korean",      ["lt"] = "Lithuanian", ["lv"] = "Latvian",
            ["mk"] = "Macedonian",  ["ms"] = "Malay",      ["mt"] = "Maltese",
            ["nl"] = "Dutch",       ["pl"] = "Polish",     ["pt"] = "Portuguese",
            ["ro"] = "Romanian",    ["ru"] = "Russian",    ["sk"] = "Slovak",
            ["sl"] = "Slovenian",   ["sv"] = "Swedish",    ["th"] = "Thai",
            ["tl"] = "Filipino",    ["tr"] = "Turkish",    ["uk"] = "Ukrainian",
            ["vi"] = "Vietnamese",  ["zh"] = "Chinese",
            // IndicConformer — 22 official Indian languages. Five use 639-3
            // because they have no 639-1 code; see IndicConformerLangs above.
            ["as"] = "Assamese",    ["bn"] = "Bengali",    ["brx"] = "Bodo",
            ["doi"] = "Dogri",      ["gu"] = "Gujarati",   ["kn"] = "Kannada",
            ["kok"] = "Konkani",    ["ks"] = "Kashmiri",   ["mai"] = "Maithili",
            ["ml"] = "Malayalam",   ["mni"] = "Manipuri",  ["mr"] = "Marathi",
            ["ne"] = "Nepali",      ["or"] = "Odia",       ["pa"] = "Punjabi",
            ["sa"] = "Sanskrit",    ["sat"] = "Santali",   ["sd"] = "Sindhi",
            ["ta"] = "Tamil",       ["te"] = "Telugu",     ["ur"] = "Urdu",
            // Whisper-only additions (languages no other backend supports).
            ["af"] = "Afrikaans",   ["am"] = "Amharic",    ["az"] = "Azerbaijani",
            ["ba"] = "Bashkir",     ["be"] = "Belarusian", ["bo"] = "Tibetan",
            ["br"] = "Breton",      ["bs"] = "Bosnian",    ["ca"] = "Catalan",
            ["cy"] = "Welsh",       ["eu"] = "Basque",     ["fo"] = "Faroese",
            ["gl"] = "Galician",    ["ha"] = "Hausa",      ["haw"] = "Hawaiian",
            ["he"] = "Hebrew",      ["ht"] = "Haitian Creole", ["hy"] = "Armenian",
            ["is"] = "Icelandic",   ["jv"] = "Javanese",   ["ka"] = "Georgian",
            ["kk"] = "Kazakh",      ["km"] = "Khmer",      ["la"] = "Latin",
            ["lb"] = "Luxembourgish", ["ln"] = "Lingala",  ["lo"] = "Lao",
            ["mg"] = "Malagasy",    ["mi"] = "Maori",      ["mn"] = "Mongolian",
            ["my"] = "Burmese",     ["nn"] = "Norwegian Nynorsk",
            ["no"] = "Norwegian",   ["oc"] = "Occitan",    ["ps"] = "Pashto",
            ["si"] = "Sinhala",     ["sn"] = "Shona",      ["so"] = "Somali",
            ["sq"] = "Albanian",    ["sr"] = "Serbian",    ["su"] = "Sundanese",
            ["sw"] = "Swahili",     ["tg"] = "Tajik",      ["tk"] = "Turkmen",
            ["tt"] = "Tatar",       ["uz"] = "Uzbek",      ["yi"] = "Yiddish",
            ["yo"] = "Yoruba",      ["yue"] = "Cantonese",
        }.ToFrozenDictionary(StringComparer.OrdinalIgnoreCase);

    // Right-to-left scripts. Keep tight — only codes whose *normal* writing
    // direction is RTL. Urdu, Kashmiri and Sindhi all write in Perso-Arabic
    // RTL in this model's training data. Arabic / Hebrew / Persian aren't
    // in IndicConformer's set but are included so other backends (Cohere,
    // Qwen3) benefit from the same helper.
    private static readonly FrozenSet<string> RtlLangs = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "ar", "fa", "he", "iw", "ur", "ks", "sd",
    }.ToFrozenSet(StringComparer.OrdinalIgnoreCase);

    /// <summary>True when the language's normal writing direction is
    /// right-to-left. Used by the editor to flip FlowDirection on the
    /// per-token ASR runs and the Edit box when rendering one of these
    /// languages. Deprecated codes normalized via <see cref="NormalizeIso"/>.</summary>
    public static bool IsRtl(string? iso) =>
        !string.IsNullOrWhiteSpace(iso) && RtlLangs.Contains(NormalizeIso(iso));

    /// <summary>
    /// English display name for an ISO 639-1 code, or the code itself if
    /// no name is registered. Case-insensitive; deprecated codes are
    /// normalized via <see cref="NormalizeIso"/>.
    /// </summary>
    public static string LanguageDisplayName(string iso)
    {
        if (string.IsNullOrWhiteSpace(iso)) return iso ?? string.Empty;
        string code = NormalizeIso(iso);
        return IsoDisplayNamesMap.TryGetValue(code, out string? name) ? name : code;
    }

    /// <summary>
    /// Returns the user-pickable language list for a backend, sorted by
    /// display name. Used by the force-language affordance in the LID
    /// mismatch popup; does not include an "auto-detect" entry.
    /// </summary>
    public static IReadOnlyList<AsrLanguageOption> LanguageOptions(AsrBackend backend) =>
        Get(backend)
            .Select(c => new AsrLanguageOption(c, LanguageDisplayName(c)))
            .OrderBy(o => o.DisplayName, StringComparer.Ordinal)
            .ToArray();
}
