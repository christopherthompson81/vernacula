---
title: "Audio Input Quality"
description: "How to get the best transcription results: recording practices, common problems, and picking the right ASR backend for your audio."
topic_id: first_steps_audio_input_quality
---

# Audio Input Quality

Transcription quality depends far more on the audio you feed in than on any post-processing Vernacula can apply. This page covers what actually matters and what to do when audio is less than ideal.

## What makes ASR work well

Modern speech-recognition models are trained on millions of hours of varied audio and are quite robust, but they share a few consistent preferences:

- **Clear, full-bandwidth voice.** The model was trained on mic-quality speech. Anything up to about 8 kHz of useful content is what it expects.
- **Moderate, consistent levels.** Peaks around -6 to -3 dBFS, with voice RMS around -20 dBFS, is the sweet spot. Very quiet recordings force the model to work harder; heavily clipped audio is unrecoverable.
- **Low stationary background noise.** HVAC hum, fan noise, and mains hum are rarely fatal but can shift accuracy by a few percent.
- **One speaker clearly dominant at a time.** Heavy overlap, crosstalk, or loud background voices (TV, restaurant chatter) are the hardest cases for any ASR.

## Recording guidance

If you are recording audio you will later transcribe, these are high-leverage choices — far more useful than anything done in post-processing.

| Factor | Goal | Why |
|---|---|---|
| **Microphone** | Any reasonably modern mic — headset, USB condenser, lavalier, or a decent smartphone — works well. Avoid laptop built-in mics for multi-speaker content. | Built-in laptop mics pick up keyboard noise and fan hiss at close range. |
| **Distance** | 15–30 cm for a headset or directional mic; closer to the speaker than to noise sources. | Doubling the distance to the speaker quadruples the noise relative to voice. |
| **Recording level** | Aim for voice peaks at around -6 dBFS. Avoid any clipping. | Clipped audio cannot be recovered; too-quiet audio amplifies the noise floor. |
| **Room** | A carpeted or soft-furnished room beats a bathroom, stairwell, or glass-walled office. | Echo and reverb confuse the model more than background noise does. |
| **Format** | 16-bit WAV at 16 kHz or higher. 48 kHz is fine. Lossless codecs (FLAC, WAV) preferred over lossy (MP3, OGG) when you have the choice. | Lossy compression at low bitrates introduces codec artefacts the model was not trained on. |

## Common problems and what to do

**Mains hum (50/60 Hz buzz).** Vernacula automatically applies a notch filter at both frequencies during transcription, so the hum should not hurt accuracy. If you still hear it on playback of the source file, it is there in the recording — reposition the mic away from power cables and fluorescent lights.

**Rumble, handling noise, HVAC low-end.** Vernacula applies a high-pass filter at 75 Hz by default. This removes low-frequency energy that is below the range of spoken fundamentals and which otherwise dominates the model's input. If you still have audible rumble, a better mic stand or shock mount solves it at the source.

**Telephone / codec-quality audio.** Recordings from phone calls, VoIP, or old sources are band-limited (typically 300–3400 Hz) and sometimes codec-compressed. ASR handles this well in most cases — do **not** run denoising on such audio, as most denoisers were trained on full-bandwidth speech and introduce artefacts that hurt accuracy more than the band-limiting does.

**Echo or reverb.** Hard to fix after the fact. Recording closer to the mic and using a less reverberant space is the only practical solution.

**Loud background voices.** Not something ASR can separate. Consider whether a diarization-capable setup would help, or re-record if possible.

**Clipping.** Clipped regions are lost. Re-record with a lower input gain.

## Picking the right ASR backend

Vernacula ships several ASR engines, each with different strengths. Switch backends in **Settings → ASR**.

| Backend | Best for | Notes |
|---|---|---|
| **Parakeet** | Clean to moderately-noisy English. Fast. | The default. Good general choice. Strong on native-English wide-band audio; less robust to strong accents or severe noise. |
| **Qwen3-ASR** | Multilingual, heavier accents, noisy audio. Slower than Parakeet. | More robust to adverse conditions; trades latency for quality. |
| **VibeVoice-ASR** | Long-form with built-in diarization. Requires CUDA. | Whole-recording ASR with automatic speaker attribution, useful when segmentation matters. |
| **Cohere Transcribe** | Cloud-quality transcription. Requires model download. | Strong general accuracy; forced-language option available. |
| **IndicConformer** | 22 Indian languages. | Per-language model; mandatory language selection. |

If Parakeet is struggling on a file, the most effective next step is usually switching to a more robust backend, not preprocessing. We tested learned denoisers extensively and found they consistently hurt Parakeet's accuracy — modern ASR absorbs noise robustness during training and penalises preprocessing artefacts.

## What Vernacula does automatically

Every file gets the following before it reaches the ASR model:

- Downmix to mono
- Resample to 16 kHz (most ASR models' training rate)
- High-pass at 75 Hz (removes rumble / HVAC low-end)
- Narrow notches at 50 Hz and 60 Hz (removes mains hum)

These are deterministic DSP filters, not learned models. They target only the specific low-frequency interferers that the ASR was not trained to handle, and they do not otherwise alter the speech signal. There is no setting — they are always on because they are always safe.

Vernacula intentionally does **not** ship a learned denoiser. Our own evaluation showed such denoisers either help negligibly or actively hurt transcription quality on modern ASR backends.
