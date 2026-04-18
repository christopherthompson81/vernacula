using System.Collections.Frozen;

namespace Vernacula.App.Models;

/// <summary>
/// Static table of ISO 639-1 language codes supported by each ASR backend,
/// plus helpers for LID-driven backend selection and compatibility checks.
///
/// <para>
/// The per-backend sets are derived from:
/// <list type="bullet">
/// <item><see cref="AsrBackend.Parakeet"/>: NVIDIA Parakeet TDT 0.6B v3 — English only.</item>
/// <item><see cref="AsrBackend.Cohere"/>: Cohere Transcribe — 14 languages
/// (mirrors <c>SettingsViewModel.CohereLanguages</c>).</item>
/// <item><see cref="AsrBackend.Qwen3Asr"/>: Qwen3-ASR 1.7B — 57 languages
/// (mirrors <c>SettingsViewModel.Qwen3AsrLanguages</c>).</item>
/// <item><see cref="AsrBackend.VibeVoice"/>: VibeVoice-ASR — English only
/// in the current export.</item>
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
    private static readonly FrozenSet<string> ParakeetLangs =
        new HashSet<string> { "en" }.ToFrozenSet();

    private static readonly FrozenSet<string> CohereLangs = new HashSet<string>
    {
        "ar", "de", "el", "en", "es", "fr", "it", "ja",
        "ko", "nl", "pl", "pt", "vi", "zh",
    }.ToFrozenSet();

    private static readonly FrozenSet<string> Qwen3AsrLangs = new HashSet<string>
    {
        "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr",
        "cs", "da", "nl", "en", "et", "fi", "fr", "gl", "de", "el",
        "he", "hi", "hu", "is", "id", "it", "ja", "kn", "kk", "ko",
        "lv", "lt", "mk", "ms", "mr", "ne", "no", "fa", "pl", "pt",
        "ro", "ru", "sr", "sk", "sl", "es", "sw", "sv", "tl", "ta",
        "th", "tr", "uk", "ur", "vi", "cy",
    }.ToFrozenSet();

    private static readonly FrozenSet<string> VibeVoiceLangs =
        new HashSet<string> { "en" }.ToFrozenSet();

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
}
