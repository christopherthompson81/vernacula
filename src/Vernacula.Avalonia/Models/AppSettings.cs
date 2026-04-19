using Vernacula.Base.Models;

namespace Vernacula.App.Models;

public enum AppTheme    { Dark, Light }
public enum PlaybackMode { Single, AutoAdvance, Continuous }
public enum AsrBackend { Parakeet, Cohere, Qwen3Asr, VibeVoice }

public class AppSettings
{
    public AppTheme           Theme               { get; set; } = AppTheme.Dark;
    public SegmentationMode   Segmentation        { get; set; } = SegmentationMode.SileroVad;
    public AsrBackend         AsrBackend          { get; set; } = AsrBackend.Parakeet;
    // Parakeet TDT beam search. 1 = greedy (default, fastest). 4–8 enables
    // beam search — ~3–5× slower per segment but improves accuracy on hard
    // or ambiguous audio and is a prerequisite for shallow LM fusion.
    public int                ParakeetBeamWidth   { get; set; } = 1;

    // Optional ARPA (.arpa or .arpa.gz) subword language model used for
    // shallow fusion during Parakeet beam search. Empty string = fusion off.
    // When set, auto-bumps beam width to at least 4 since greedy can't
    // benefit from fusion.
    public string             ParakeetLmPath          { get; set; } = "";
    // Shallow-fusion weight. Typical 0.1–0.5.
    public float              ParakeetLmWeight        { get; set; } = 0.3f;
    // Per-emitted-token reward that offsets the LM's shortening bias.
    // Typical 0.0–1.0.
    public float              ParakeetLmLengthPenalty { get; set; } = 0.6f;
    public string             CohereLanguage      { get; set; } = "";
    public string             Qwen3AsrLanguage    { get; set; } = "";
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
