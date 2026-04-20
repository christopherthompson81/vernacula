using Vernacula.Base.Models;

namespace Vernacula.App.Models;

public enum AppTheme    { Dark, Light }
public enum PlaybackMode { Single, AutoAdvance, Continuous }
// New enum values MUST be appended. Persisted settings store the integer
// value, so inserting mid-enum silently re-interprets existing user state.
public enum AsrBackend { Parakeet, Cohere, Qwen3Asr, VibeVoice, IndicConformer }

public class AppSettings
{
    public AppTheme           Theme               { get; set; } = AppTheme.Dark;
    public SegmentationMode   Segmentation        { get; set; } = SegmentationMode.SileroVad;
    public AsrBackend         AsrBackend          { get; set; } = AsrBackend.Parakeet;
    // Parakeet TDT beam search. 1 = greedy (default, fastest). 4–8 enables
    // beam search — ~3–5× slower per segment but improves accuracy on hard
    // or ambiguous audio and is a prerequisite for shallow LM fusion.
    public int                ParakeetBeamWidth   { get; set; } = 1;

    // Which KenLM the Parakeet decoder should shallow-fuse.
    // Stable key from KenLmCatalog — "none" (default), a built-in key like
    // "en-general", or "custom" (paired with ParakeetLmPath below).
    public string             ParakeetLmSelection     { get; set; } = "none";
    // Free-form path used only when ParakeetLmSelection == "custom".
    // When a built-in option is selected, the app resolves the path via
    // ModelManagerService (downloading on demand) instead.
    public string             ParakeetLmPath          { get; set; } = "";
    // Shallow-fusion weight. Typical 0.1–0.5.
    public float              ParakeetLmWeight        { get; set; } = 0.3f;
    // Per-emitted-token reward that offsets the LM's shortening bias.
    // Typical 0.0–1.0.
    public float              ParakeetLmLengthPenalty { get; set; } = 0.6f;
    public string             CohereLanguage      { get; set; } = "";
    public string             Qwen3AsrLanguage    { get; set; } = "";
    // IndicConformer is strictly per-language at inference — the model has
    // 22 CTC heads and picking one is mandatory, so this is not optional
    // like Cohere/Qwen3 "auto". Default to Hindi (largest / most common).
    public string             IndicConformerLanguage { get; set; } = "hi";
    public DenoiserMode       Denoiser            { get; set; } = DenoiserMode.None;
    public PlaybackMode       EditorPlaybackMode  { get; set; } = PlaybackMode.Continuous;
    public string             ModelsDir           { get; set; } = "";
    public string             DiariZenModelsDir   { get; set; } = "";
    public bool               DiariZenNoticeAccepted { get; set; } = false;

    // Language identification (VoxLingua107)
    public bool               LidEnabled          { get; set; } = false;
    // Per-segment LID: when both this and LidEnabled are on, classify
    // every segment ≥ 2 s and persist the result alongside ASR output.
    // Shorter segments inherit the file-level language detected in Phase 3b.
    public bool               LidPerSegment       { get; set; } = false;
    public string             VoxLinguaModelsDir  { get; set; } = "";

    public List<string>       AcceptedGatedModels { get; set; } = [];
    public string             Language            { get; set; } = "";

    // Column widths — 0 means "use default"
    public double HomeColTitleWidth      { get; set; } = 0;
    public double HomeColAudioWidth      { get; set; } = 0;
    public double ResultsColSpeakerWidth { get; set; } = 0;
    public double ResultsColStartWidth   { get; set; } = 0;
    public double ResultsColEndWidth     { get; set; } = 0;

    // Window state — null Left/Top means "let Windows decide"
    public double? WindowLeft      { get; set; } = null;
    public double? WindowTop       { get; set; } = null;
    public double  WindowWidth     { get; set; } = 920;
    public double  WindowHeight    { get; set; } = 840;
    public bool    WindowMaximized { get; set; } = false;
}
