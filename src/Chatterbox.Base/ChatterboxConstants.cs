namespace Chatterbox.Base;

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
}
