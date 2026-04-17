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
    public string             CohereLanguage      { get; set; } = "";
    public DenoiserMode       Denoiser            { get; set; } = DenoiserMode.None;
    public PlaybackMode       EditorPlaybackMode  { get; set; } = PlaybackMode.Continuous;
    public string             ModelsDir           { get; set; } = "";
    public string             DiariZenModelsDir   { get; set; } = "";
    public bool               DiariZenNoticeAccepted { get; set; } = false;
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
