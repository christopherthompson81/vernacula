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
