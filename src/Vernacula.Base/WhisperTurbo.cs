namespace Vernacula.Base;

/// <summary>
/// Whisper large-v3-turbo backend — constants and file layout.
///
/// Phase-1 scaffolding: declares the ONNX filenames and tokenizer/config
/// assets that <c>ModelManagerService</c> downloads from the
/// <a href="https://huggingface.co/onnx-community/whisper-large-v3-turbo">
/// onnx-community pre-exported repo</a>. The actual inference class will be
/// added in Phase 2 (see <c>docs/whisper_turbo_investigation.md</c>).
///
/// File layout mirrors Cohere / Qwen3-ASR: separate prefill and step decoders
/// to keep per-session activation memory small and leave VRAM headroom for
/// larger decode batches.
/// </summary>
public static class WhisperTurbo
{
    // ── ONNX graphs (fp16 variants — single file, no external-data sidecar) ──
    //
    // fp16 was chosen for v1: Whisper's quality loss at fp16 is negligible,
    // the files fit comfortably under the 2 GiB protobuf limit (no .onnx_data
    // sidecar), download is reasonable (~2 GB total), and inference works on
    // both GPU (native fp16) and CPU (slower but functional). int8 / q4
    // variants can be added later as user-selectable precision.
    public const string EncoderFile         = "encoder_model_fp16.onnx";
    public const string DecoderInitFile     = "decoder_model_fp16.onnx";
    public const string DecoderStepFile     = "decoder_with_past_model_fp16.onnx";

    // ── Tokenizer and config ─────────────────────────────────────────────────
    public const string TokenizerFile       = "tokenizer.json";
    public const string ConfigFile          = "config.json";
    public const string GenerationConfigFile = "generation_config.json";
    public const string PreprocessorConfigFile = "preprocessor_config.json";

    // ── Canonical HF model name (for TranscriptionDb metadata and
    //     AsrLanguageSupport.BackendOf / ModelName lookups) ──────────────────
    public const string ModelName = "openai/whisper-large-v3-turbo";
}
