---
title: "Settings"
description: "Overview of all options in the Settings window."
topic_id: first_steps_settings_window
---

# Settings

The **Settings** window gives you control over hardware configuration, model management, segmentation mode, editor behaviour, appearance, and language. Open it from the menu bar: `Settings…`.

## Hardware & Performance

This section shows the status of your NVIDIA GPU and CUDA software stack, and reports the dynamic batch ceiling used during GPU transcription.

| Item | Description |
|---|---|
| GPU name and VRAM | Detected NVIDIA GPU and available video memory. |
| CUDA Toolkit | Whether the CUDA runtime libraries were found via `CUDA_PATH`. |
| cuDNN | Whether the cuDNN runtime DLLs are available. |
| CUDA Acceleration | Whether ONNX Runtime successfully loaded the CUDA execution provider. |

Click `Re-check` to re-run hardware detection without restarting the application — useful after installing CUDA or cuDNN.

Direct download links for the CUDA Toolkit and cuDNN are shown when those components are not detected.

The **batch ceiling** message reports how many seconds of audio are processed in each GPU run. This value is derived from free VRAM after models are loaded and adjusts automatically.

For full CUDA setup instructions, see [Installing CUDA and cuDNN](cuda_installation.md).

## Models

This section manages the AI model files required for transcription.

- **Download Missing Models** — downloads any model files not yet present on disk. A progress bar and status line track each file as it downloads.
- **Check for Updates** — checks whether newer model weights are available. An update banner also appears on the home screen automatically when updated weights are detected.

## ASR Backend

Controls which speech-recognition model transcribes the audio. Each backend covers a different set of languages and has different tradeoffs.

| Backend | Languages | Notes |
|---|---|---|
| **Parakeet** | 25 European | Default. Multilingual shared-vocab model; auto-detects language during decode. Optional KenLM shallow fusion for domain biasing. |
| **Cohere Transcribe** | 14 | Whisper-family decoder path. Supports optional per-file forced language. |
| **Qwen3-ASR 1.7B** | 29 | Batched decoder with the widest language coverage. Optional forced language; otherwise auto-detects. |
| **VibeVoice-ASR** | 12 | Combined diarization + ASR in a single model pass. Requires a CUDA-capable GPU. |
| **IndicConformer 600M** | 22 Indic | AI4Bharat's multilingual Indic model. Covers the 22 official Indian languages across multiple scripts (Devanagari, Bengali, Tamil, Arabic, Ol Chiki, etc.). Requires a language to be picked at inference — see below. |

### IndicConformer language selection

Unlike Parakeet's shared-vocab auto-routing, IndicConformer's 22 languages have disjoint per-language CTC heads. The decoder needs to be told which head to use, so the Settings page exposes a **Language** picker just below the backend radio group. Five languages use ISO 639-3 codes because they have no 639-1 assignment: `brx` (Bodo), `doi` (Dogri), `kok` (Konkani), `mai` (Maithili), `mni` (Manipuri), `sat` (Santali). The remaining 17 use their 639-1 codes.

When **Language Identification** is enabled (see below), IndicConformer uses the detected language per segment and falls back to the manual picker for segments whose detected language isn't one of the 14 VoxLingua107 covers. The 8 languages LID cannot detect — `brx`, `doi`, `kok`, `ks`, `mai`, `mni`, `or`, `sat` — always require the manual picker for files whose primary language is one of them.

## Segmentation Mode

Controls how the audio is divided into segments before speech recognition.

| Mode | Description |
|---|---|
| **Speaker Diarization** | Uses the Sortformer model to identify individual speakers and label each segment. Best for interviews, meetings, and multi-speaker recordings. |
| **Voice Activity Detection** | Uses Silero VAD to detect speech regions only — no speaker labels. Faster than diarization and well-suited to single-speaker audio. |

## Transcript Editor

**Default Playback Mode** — sets the playback mode used when you open the transcript editor. You can also change it directly in the editor at any time. See [Editing Transcripts](../operations/editing_transcripts.md) for a description of each mode.

## Appearance

Select **Dark** or **Light** theme. The change applies immediately. See [Picking a Theme](theme.md).

## Language

Select the display language for the application interface. The change applies immediately. See [Picking a Language](language.md).
