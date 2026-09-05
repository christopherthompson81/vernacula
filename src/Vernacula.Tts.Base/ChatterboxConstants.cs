namespace Vernacula.Tts.Base;

/// <summary>
/// Model-wide constants for the Chatterbox TTS pipeline. Mirror
/// <c>scripts/chatterbox_export/_common.py</c> and the upstream
/// <c>chatterbox</c> values it pulls from. If those change, this list
/// silently drifts — a future pass should read the dynamic ones from
/// <c>export-report.json</c>.
/// </summary>
public static class ChatterboxConstants
{
    public const int StartSpeechToken = 6561;   // _common.py::START_SPEECH_TOKEN
    public const int StopSpeechToken = 6562;    // _common.py::STOP_SPEECH_TOKEN
    public const int ExaggerationToken = 6563;  // _common.py::EXAGGERATION_TOKEN

    public const int LlmLayers = 30;            // _common.py::LLM_NUM_LAYERS
    public const int LlmKvHeads = 16;           // _common.py::LLM_NUM_KV_HEADS
    public const int LlmHeadDim = 64;           // _common.py::LLM_HEAD_DIM
    public const int LlmHidden = 1024;          // _common.py::LLM_HIDDEN_SIZE

    public const int S3GenSr = 24_000;          // _common.py::S3GEN_SR
    public const int DummyAudioSamples = 312_936;  // _common.py::DUMMY_AUDIO_SAMPLES (13.04 s @ 24 kHz)

    public const int MelBins = 80;              // chatterbox.s3gen.flow.output_size
    public const int PromptLen = 500;           // speaker_features.shape[1], fixed by speech_encoder export
    public const int CfmSteps = 10;             // flow.decoder n_timesteps
    public const float CfgRate = 0.7f;          // chatterbox.s3gen.flow.decoder.inference_cfg_rate

    public const float DefaultExaggeration = 0.5f;     // listen_test.py default
    public const float DefaultRepetitionPenalty = 1.2f;  // listen_test.py LM-loop rep-penalty divisor
    public const int DefaultMaxLmSteps = 256;          // listen_test.py LM-loop max_new_tokens

    // Cross-attention alignment: Resemble AI's AlignmentStreamAnalyzer
    // (chatterbox/models/t3/inference/alignment_stream_analyzer.py)
    // identifies these three (layer, head) pairs as carrying clean
    // text→speech alignment signal. The LM ONNX export emits the
    // attention tensors for these layers as additional outputs named
    // `attentions.{layer}`; the C# AcousticLM picks the head index
    // listed here, mean-averages across the three, and exposes the
    // result as a (speech_step, text_token) alignment matrix.
    //
    // Index pairs must stay in sync with LLAMA_ALIGNED_LAYERS in
    // scripts/chatterbox_export/export_chatterbox_to_onnx.py — order
    // is significant (Layers[i] pairs with Heads[i]).
    public static readonly int[] AlignmentLayerIndices = [9, 12, 13];
    public static readonly int[] AlignmentHeadIndices = [2, 15, 11];
}
