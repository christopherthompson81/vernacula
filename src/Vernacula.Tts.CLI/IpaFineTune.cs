namespace Vernacula.Tts.CLI;

/// <summary>
/// Facts about the shipped OmniVoice IPA fine-tune that the CLI needs at runtime.
///
/// These describe a FROZEN artifact (the v6 LoRA checkpoint), not a live configuration, which is
/// why they are literals here rather than read from the training tree — the training tree lives on
/// the machine that produced the checkpoint and is not part of a user install.
/// </summary>
internal static class IpaFineTune
{
    /// <summary>Default diff filename inside --onnx-dir. Bump alongside the checkpoint; the
    /// version is in the name precisely so a stale diff cannot masquerade as current
    /// (see commit "diff: version the IPA fine-tune patch").</summary>
    public const string DefaultDiffFile = "ipa_diff_v6.onnx";

    /// <summary>
    /// The 28 languages v6 actually trained on, as vernacula-phonemizer codes.
    ///
    /// Source of truth: the `language_id` values in /mnt/data/omnivoice_ipa/train/data_config.json
    /// (the v6 build; v5's is kept beside it as data_config_v5.json.bak), mapped out of FLEURS
    /// config codes into phonemizer codes — en_us→en, cmn_hans_cn→cmn, es_419→es, ar_eg→ar,
    /// pt_br→pt, and so on for the rest.
    ///
    /// ⚠ This is a COVERAGE SET, not a support list. It was chosen as a greedy cover over IPA
    /// primitives (English the generalist Latin base, Zulu the clicks and breathy voice, Hausa the
    /// ejectives, Fula the prenasals) — see build_webdataset.py. The whole premise of an
    /// IPA-conditioned model is that a language outside it still renders, from phones the model
    /// already holds; that is what the off-corpus notice says rather than refusing.
    /// </summary>
    public static readonly IReadOnlySet<string> TrainedLanguages = new HashSet<string>(StringComparer.Ordinal)
    {
        "en", "cmn", "hi", "es", "ar", "fr", "pt", "ru", "de", "ja",
        "tr", "vi", "ta", "ko", "ha", "th", "am", "om", "sd", "ff",
        "kk", "zu", "cs", "sv", "xh", "ca", "ga", "cy",
    };
}
