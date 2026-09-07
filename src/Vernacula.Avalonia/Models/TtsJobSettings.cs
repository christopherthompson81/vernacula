namespace Vernacula.App.Models;

/// <summary>
/// The per-job text-to-speech choices, snapshotted into the jobs table when a TTS job is
/// created so a later requeue renders exactly what the user asked for, whatever the Settings
/// defaults have become since. Model/data locations are NOT here — those are app settings and
/// resolve at run time.
/// </summary>
/// <param name="Backend">TtsBackendKind name: "Chatterbox", "Kokoro" or "OmniVoice".</param>
/// <param name="Language">vernacula-phonemizer language code (OmniVoice); "" for backends without a choice.</param>
/// <param name="Voice">Chatterbox: reference WAV path. Kokoro: voice name. OmniVoice: library voice id.</param>
/// <param name="Speed">Kokoro speech-rate multiplier.</param>
/// <param name="NumStep">OmniVoice diffusion steps.</param>
public sealed record TtsJobSettings(
    string Backend,
    string Language,
    string Voice,
    float  Speed   = 1.0f,
    int    NumStep = 32);
