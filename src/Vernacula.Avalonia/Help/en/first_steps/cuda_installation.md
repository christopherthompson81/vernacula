---
title: "Installing CUDA and cuDNN for GPU Acceleration"
description: "How to set up NVIDIA CUDA and cuDNN so Parakeet Transcription can use your GPU."
topic_id: first_steps_cuda_installation
---

# Installing CUDA and cuDNN for GPU Acceleration

Parakeet Transcription can use an NVIDIA GPU to accelerate transcription significantly. GPU acceleration requires the NVIDIA CUDA Toolkit and cuDNN runtime libraries to be installed on your system.

## Requirements

- An NVIDIA GPU that supports CUDA (GeForce GTX 10-series or later is recommended).
- Windows 10 or 11 (64-bit).
- The model files must already be downloaded. See [Downloading Models](downloading_models.md).

## Installation Steps

### 1. Install the CUDA Toolkit

Download and run the CUDA Toolkit installer from the NVIDIA developer website. During installation, accept the default paths. The installer sets the `CUDA_PATH` environment variable automatically — Parakeet uses this variable to locate the CUDA libraries.

### 2. Install cuDNN

Download the cuDNN ZIP archive for your installed CUDA version from the NVIDIA developer website. Extract the archive and copy the contents of its `bin`, `include`, and `lib` folders into the corresponding folders inside your CUDA Toolkit installation directory (the path shown by `CUDA_PATH`).

Alternatively, install cuDNN using the NVIDIA cuDNN installer if one is available for your CUDA version.

### 3. Restart the Application

Close and reopen Parakeet Transcription after installation. The application checks for CUDA on startup.

## GPU Status in Settings

Open `Settings…` from the menu bar and look at the **Hardware & Performance** section. Each component shows a checkmark (✓) when detected:

| Item | What it means |
|---|---|
| GPU name and VRAM | Your NVIDIA GPU was found |
| CUDA Toolkit ✓ | CUDA libraries located via `CUDA_PATH` |
| cuDNN ✓ | cuDNN runtime DLLs found |
| CUDA Acceleration ✓ | ONNX Runtime loaded the CUDA execution provider |

If any item is missing after installation, click `Re-check` to re-run hardware detection without restarting the application.

The Settings window also provides direct download links for the CUDA Toolkit and cuDNN if they are not yet installed.

### Troubleshooting

If `CUDA Acceleration` does not show a checkmark, verify that:

- The `CUDA_PATH` environment variable is set (check `System > Advanced system settings > Environment Variables`).
- The cuDNN DLLs are in a directory on your system `PATH` or inside the CUDA `bin` folder.
- Your GPU driver is up to date.

### Batch Sizing

When CUDA is active, the **Hardware & Performance** section also shows the current dynamic batch ceiling — the maximum seconds of audio processed in one GPU run. This value is calculated from free VRAM after models are loaded and adjusts automatically if your available memory changes.

## Running Without a GPU

If CUDA is not available, Parakeet falls back to CPU processing automatically. Transcription still works but will be slower, especially for long audio files.
