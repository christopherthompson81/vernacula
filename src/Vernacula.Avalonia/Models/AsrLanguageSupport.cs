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

    /// <summary>
    /// Returns the ISO 639-1 language codes the given ASR backend can
    /// transcribe. Result is lowercase.
    /// </summary>
    public static FrozenSet<string> Get(AsrBackend backend) => backend switch
    {
        AsrBackend.Parakeet  => ParakeetLangs,
        AsrBackend.Cohere    => CohereLangs,
        AsrBackend.Qwen3Asr  => Qwen3AsrLangs,
        AsrBackend.VibeVoice => VibeVoiceLangs,
        _                    => FrozenSet<string>.Empty,
    };

    /// <summary>
    /// True when <paramref name="backend"/> can transcribe the given ISO
    /// 639-1 code. Case-insensitive.
    /// </summary>
    public static bool Supports(AsrBackend backend, string iso) =>
        !string.IsNullOrWhiteSpace(iso) && Get(backend).Contains(iso.ToLowerInvariant());

    /// <summary>
    /// Returns the list of ASR backends that support <paramref name="iso"/>,
    /// ordered by preference for generic use:
    /// Qwen3-ASR (widest coverage) → Cohere → Parakeet → VibeVoice.
    /// Used when LID detects a language the user's current backend can't
    /// handle and we need to suggest an alternative.
    /// </summary>
    public static IReadOnlyList<AsrBackend> BackendsSupporting(string iso)
    {
        if (string.IsNullOrWhiteSpace(iso)) return Array.Empty<AsrBackend>();
        string code = iso.ToLowerInvariant();
        var order = new[] { AsrBackend.Qwen3Asr, AsrBackend.Cohere, AsrBackend.Parakeet, AsrBackend.VibeVoice };
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
        AsrBackend.Parakeet  => "Parakeet",
        AsrBackend.Cohere    => "Cohere Transcribe",
        AsrBackend.Qwen3Asr  => "Qwen3-ASR",
        AsrBackend.VibeVoice => "VibeVoice-ASR",
        _                    => backend.ToString(),
    };

    /// <summary>
    /// Maps the pipeline's ASR model-name string to an <see cref="AsrBackend"/>
    /// enum value. Returns null for unknown model names (safe fallback).
    /// </summary>
    public static AsrBackend? BackendOf(string modelName) => modelName switch
    {
        "nvidia/parakeet-tdt-0.6b-v3"          => AsrBackend.Parakeet,
        "CohereLabs/cohere-transcribe-03-2026" => AsrBackend.Cohere,
        "Qwen/Qwen3-ASR-1.7B"                  => AsrBackend.Qwen3Asr,
        "vibevoice/vibevoice-asr"              => AsrBackend.VibeVoice,
        _                                      => (AsrBackend?)null,
    };

    /// <summary>Inverse of <see cref="BackendOf"/>: enum → pipeline model-name string.</summary>
    public static string ModelName(AsrBackend backend) => backend switch
    {
        AsrBackend.Parakeet  => "nvidia/parakeet-tdt-0.6b-v3",
        AsrBackend.Cohere    => "CohereLabs/cohere-transcribe-03-2026",
        AsrBackend.Qwen3Asr  => "Qwen/Qwen3-ASR-1.7B",
        AsrBackend.VibeVoice => "vibevoice/vibevoice-asr",
        _ => throw new ArgumentOutOfRangeException(nameof(backend)),
    };
}
