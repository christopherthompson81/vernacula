using Vernacula.App.Models;
using Vernacula.Base;
using Vernacula.Tts.Base;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// One text-to-speech model set: a directory the Settings → TTS tab shows a status row for,
/// that an engine lists in <see cref="TtsEngine.RequiredSets"/>, and that the job runner and
/// the New TTS Job dialog check before anything loads.
/// <para>
/// Every per-set fact is a <c>required</c> member, and there is no enum beside it: adding a set
/// is one entry in <see cref="TtsModelSets"/>, and leaving any piece out is a compile error
/// rather than a switch's default arm quietly reporting the set as ready with nothing to
/// download (#133). The only things a new set needs outside this file are the AppSettings
/// field its override lives in and the Config constant naming its default subfolder — and both
/// are referenced from here, so they cannot be forgotten either.
/// </para>
/// </summary>
internal sealed class TtsModelSet
{
    /// <summary>The row's title, and the name the job runner uses when the set is incomplete.</summary>
    public required string Name { get; init; }
    public required string Description { get; init; }

    /// <summary>Default location: this subfolder of the models directory.</summary>
    public required string SubDir { get; init; }
    /// <summary>The user's pick from Settings, "" for the default; where it is stored.</summary>
    public required Func<AppSettings, string>   GetOverride { get; init; }
    public required Action<AppSettings, string> SetOverride { get; init; }
    /// <summary>
    /// Where the set is found when there is no pick and before the models-dir default — an
    /// environment variable, a checkout beside a source build. Null result falls through.
    /// </summary>
    public Func<string?>? ResolveDefault { get; init; }

    /// <summary>Hub URL prefix, or "" while the set is not hosted (download button hidden).</summary>
    public required string RepoBase { get; init; }
    /// <summary>The repo's manifest.json, or "" where it has none (update check skipped).</summary>
    public required string ManifestUrl { get; init; }
    /// <summary>The set's files, (local path under the set's directory, path in the repo).</summary>
    public required ModelAsset[] Assets { get; init; }
    /// <summary>Fetched with the set but judged for presence another way (a tokenizer the engine can also find elsewhere).</summary>
    public ModelAsset[] ExtraDownloads { get; init; } = [];

    /// <summary>
    /// What the set still needs, relative to its directory, given the directory and settings.
    /// Empty means the backend can load. This is per set because presence is not always "each
    /// asset exists": an alternative export layout, any-one-voice-pack, a tokenizer found
    /// through a cache, a sentinel file for a data tree.
    /// </summary>
    public required Func<string, SettingsService, IEnumerable<string>> Missing { get; init; }

    public bool CanDownload => RepoBase.Length > 0;
    public bool HasManifest => ManifestUrl.Length > 0;

    /// <summary>Everything a download of this set fetches.</summary>
    public IEnumerable<ModelAsset> Downloadable => Assets.Concat(ExtraDownloads);

    /// <summary>
    /// The set's directory: the user's pick as given (even while still empty — it may be the
    /// folder they are about to fill or have chosen as the download target), else
    /// <see cref="ResolveDefault"/>, else the subfolder of the models directory.
    /// </summary>
    public string Dir(SettingsService settings)
    {
        string pick = GetOverride(settings.Current);
        if (!string.IsNullOrWhiteSpace(pick)) return pick;
        return ResolveDefault?.Invoke() ?? Path.Combine(settings.GetModelsDir(), SubDir);
    }

    public IReadOnlyList<string> MissingFiles(SettingsService settings) => Missing(Dir(settings), settings).ToList();

    public override string ToString() => Name;

    /// <summary>The usual presence rule: each listed asset exists under <paramref name="dir"/>.</summary>
    public static IEnumerable<string> MissingAssets(string dir, IEnumerable<ModelAsset> assets) =>
        assets.Where(a => !File.Exists(Path.Combine(dir, a.LocalRelativePath))).Select(a => a.LocalRelativePath);
}

/// <summary>
/// The table of TTS model sets, in the order the Settings → TTS tab lists them. The three
/// engine exports are our own (scripts/chatterbox_export, kokoro_export, omnivoice_export) and
/// ship as Hub repos with the same (local path, remote path) asset tables as the ASR bundles in
/// <see cref="ModelManagerService"/>; the OmniVoice voice library rides in the OmniVoice repo
/// under voices/. The phonemizer data tree is not hosted: its RepoBase is empty, so the
/// download button stays hidden and the status line names the folder to fill by hand — the
/// "not hosted yet" convention Qwen3-ASR uses.
/// </summary>
internal static class TtsModelSets
{
    private const string ChatterboxRepoBase =
        "https://huggingface.co/christopherthompson81/chatterbox-tts-onnx/resolve/main";
    private const string KokoroRepoBase =
        "https://huggingface.co/christopherthompson81/kokoro-82m-onnx/resolve/main";
    private const string OmniVoiceRepoBase =
        "https://huggingface.co/christopherthompson81/omnivoice-ipa-onnx/resolve/main";

    // Kokoro v1.0's 28 English voices — what scripts/kokoro_export/export_voices.py emits by
    // default and what the repo holds. A voice missing from here would not be fetched.
    internal static readonly string[] KokoroVoices =
        [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
            "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
            "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        ];

    // The four Chatterbox stages. Every graph keeps its weights in an external-data sidecar
    // spelled `<graph>.onnx_data` (underscore — that is what the graphs reference). The
    // vocoder is listed in its merged-Loop layout, which is what the C# Vocoder loads first;
    // the split graphs (flow_encoder / cfm_estimator / mel2wav) are in the repo too but are
    // only a fallback, so they are neither required nor fetched. tokenizer.json may also come
    // from the HF cache (ChatterboxPipeline.LocateCachedTokenizerJson) and is checked apart.
    private static readonly ModelAsset[] ChatterboxFiles =
        [
            new("speech_encoder.onnx",                 "speech_encoder.onnx"),
            new("speech_encoder.onnx_data",            "speech_encoder.onnx_data"),
            new("embed_tokens.onnx",                   "embed_tokens.onnx"),
            new("embed_tokens.onnx_data",              "embed_tokens.onnx_data"),
            new("language_model.onnx",                 "language_model.onnx"),
            new("language_model.onnx_data",            "language_model.onnx_data"),
            new("conditional_decoder_loop.onnx",       "conditional_decoder_loop.onnx"),
            new("conditional_decoder_loop.onnx_data",  "conditional_decoder_loop.onnx_data"),
        ];

    private static readonly ModelAsset[] KokoroFiles =
        [
            new("kokoro.onnx", "kokoro.onnx"),
            .. KokoroVoices.Select(v => new ModelAsset(Path.Combine("voices", $"{v}.bin"), $"voices/{v}.bin")),
        ];

    // The base transformer + codec + the IPA fine-tune diff + tokenizer. The repo names the
    // diff `ipa_diff.onnx` (its card says which extraction it is); locally it keeps the
    // versioned name IpaFineTune.DefaultDiffFile so a stale diff cannot pass as current —
    // bump both when the fine-tune moves. The int4 browser build in the repo is not fetched.
    // OmniVoice's manifest.json covers the files fetched here and keys the diff by its Hub name.
    private static readonly ModelAsset[] OmniVoiceFiles =
        [
            new("omnivoice_transformer.onnx",       "omnivoice_transformer.onnx"),
            new("omnivoice_transformer.onnx.data",  "omnivoice_transformer.onnx.data"),
            new("higgs_encoder.onnx",               "higgs_encoder.onnx"),
            new("higgs_decoder.onnx",               "higgs_decoder.onnx"),
            new(IpaFineTune.DefaultDiffFile,        "ipa_diff.onnx"),
        ];

    // The library sits at the root of its own directory locally (what StoredVoice.IsLibrary
    // checks) and under voices/ in the repo, beside the graphs it belongs with.
    private static readonly ModelAsset[] OmniVoiceVoiceLibFiles =
        [
            new("voices.jsonc",     "voices/voices.jsonc"),
            new("voice-codes.json", "voices/voice-codes.json"),
        ];

    private static readonly ModelAsset[] TokenizerJson = [new("tokenizer.json", "tokenizer.json")];

    public static readonly TtsModelSet Kokoro = new()
    {
        Name        = "Kokoro-82M",
        Description = "kokoro.onnx + voices/*.bin from scripts/kokoro_export. English voices; fast, light.",
        SubDir      = Config.KokoroSubDir,
        GetOverride = a => a.KokoroModelDir,
        SetOverride = (a, v) => a.KokoroModelDir = v,
        RepoBase    = KokoroRepoBase,
        ManifestUrl = KokoroRepoBase + "/manifest.json",
        Assets      = KokoroFiles,
        Missing     = (dir, _) =>
        {
            var missing = new List<string>();
            if (!File.Exists(Path.Combine(dir, "kokoro.onnx"))) missing.Add("kokoro.onnx");
            // Any voice pack will do to run; the download fills in the full set.
            string voices = Path.Combine(dir, "voices");
            if (!Directory.Exists(voices) || !Directory.EnumerateFiles(voices, "*.bin").Any())
                missing.Add("voices/*.bin");
            return missing;
        },
    };

    public static readonly TtsModelSet OmniVoice = new()
    {
        Name        = "OmniVoice-IPA",
        Description = "The OmniVoice base transformer, Higgs codec graphs and the IPA fine-tune diff (scripts/omnivoice_export). Any language the phonemizer covers.",
        SubDir      = Config.OmniVoiceSubDir,
        GetOverride = a => a.OmniVoiceOnnxDir,
        SetOverride = (a, v) => a.OmniVoiceOnnxDir = v,
        ResolveDefault = () => Environment.GetEnvironmentVariable("OMNIVOICE_ONNX_DIR"),
        RepoBase    = OmniVoiceRepoBase,
        ManifestUrl = OmniVoiceRepoBase + "/manifest.json",
        Assets      = OmniVoiceFiles,
        ExtraDownloads = TokenizerJson,
        Missing     = (dir, s) =>
        {
            var missing = TtsModelSet.MissingAssets(dir, OmniVoiceFiles).ToList();
            if (!File.Exists(s.Current.OmniVoiceTokenizerJson)
                && (!Directory.Exists(dir) || OmniVoiceIpaTts.LocateTokenizerJson(dir) is null))
                missing.Add("tokenizer.json");
            return missing;
        },
    };

    public static readonly TtsModelSet OmniVoiceVoices = new()
    {
        Name        = "OmniVoice voice library",
        Description = "voices.jsonc + voice-codes.json — 530 stored reference voices OmniVoice reads in, one or more per language (shared with the web demo).",
        SubDir      = Config.OmniVoiceVoiceLibSubDir,
        GetOverride = a => a.OmniVoiceVoiceLib,
        SetOverride = (a, v) => a.OmniVoiceVoiceLib = v,
        // The web demo's library beside a source build, when there is one.
        ResolveDefault = StoredVoice.ResolveDefaultLibrary,
        RepoBase    = OmniVoiceRepoBase,
        ManifestUrl = OmniVoiceRepoBase + "/manifest.json",
        Assets      = OmniVoiceVoiceLibFiles,
        Missing     = (dir, _) => TtsModelSet.MissingAssets(dir, OmniVoiceVoiceLibFiles),
    };

    public static readonly TtsModelSet Chatterbox = new()
    {
        Name        = "Chatterbox",
        Description = "The Chatterbox ONNX bundle (scripts/chatterbox_export) + tokenizer.json. English; clones a reference clip.",
        SubDir      = Config.ChatterboxSubDir,
        GetOverride = a => a.ChatterboxBundleDir,
        SetOverride = (a, v) => a.ChatterboxBundleDir = v,
        RepoBase    = ChatterboxRepoBase,
        ManifestUrl = ChatterboxRepoBase + "/manifest.json",
        Assets      = ChatterboxFiles,
        ExtraDownloads = TokenizerJson,
        Missing     = (dir, _) =>
        {
            // A monolithic conditional_decoder.onnx (the --no-split --no-merge export) stands
            // in for the merged-Loop pair.
            bool monolithic = File.Exists(Path.Combine(dir, "conditional_decoder.onnx"));
            var missing = TtsModelSet.MissingAssets(dir, ChatterboxFiles
                .Where(a => !(monolithic && a.LocalRelativePath.StartsWith("conditional_decoder_loop", StringComparison.Ordinal))))
                .ToList();
            if (!File.Exists(Path.Combine(dir, "tokenizer.json"))
                && ChatterboxPipeline.LocateCachedTokenizerJson() is null)
                missing.Add("tokenizer.json");
            return missing;
        },
    };

    public static readonly TtsModelSet PhonemizerData = new()
    {
        Name        = "Phonemizer data",
        Description = "The vernacula-phonemizer data/ tree (text → IPA) that Kokoro and OmniVoice need. Found automatically beside a source checkout.",
        SubDir      = Config.PhonemizerDataSubDir,
        GetOverride = a => a.PhonemizerDataDir,
        SetOverride = (a, v) => a.PhonemizerDataDir = v,
        // VERNACULA_DATA_DIR, then the submodule beside a source build.
        ResolveDefault = () => Vernacula.Tts.Base.PhonemizerData.Resolve(null),
        RepoBase    = "",
        ManifestUrl = "",
        Assets      = [],
        Missing     = (dir, _) =>
            Vernacula.Tts.Base.PhonemizerData.IsDataRoot(dir) ? [] : ["core/phonology.jsonc (the data/ tree)"],
    };

    /// <summary>Every set, in Settings row order.</summary>
    public static IReadOnlyList<TtsModelSet> All { get; } = [Kokoro, OmniVoice, OmniVoiceVoices, Chatterbox, PhonemizerData];
}
