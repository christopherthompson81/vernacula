---
title: "Picking Model Weight Precision"
description: "How to choose between INT8 and FP32 model precision and what the trade-offs are."
topic_id: first_steps_model_precision
---

# Picking Model Weight Precision

Model precision controls the numerical format used by the AI model weights. It affects download size, memory usage, and accuracy.

## Precision Options

### INT8 (smaller download)

- Smaller model files — faster to download and less disk space required.
- Slightly lower accuracy on some audio.
- Recommended if you have limited disk space or a slower internet connection.

### FP32 (more accurate)

- Larger model files.
- Higher accuracy, especially on difficult audio with accents or background noise.
- Recommended when accuracy is the priority and you have sufficient disk space.
- Required for CUDA GPU acceleration — the GPU path always uses FP32 regardless of this setting.

## How to Change Precision

Open `Settings…` from the menu bar, then go to the **Models** section and select either `INT8 (smaller download)` or `FP32 (more accurate)`.

## After Changing Precision

Changing precision requires a different set of model files. If the new precision's models have not been downloaded yet, click `Download Missing Models` in Settings. Previously downloaded files for the other precision are kept on disk and do not need to be re-downloaded if you switch back.
