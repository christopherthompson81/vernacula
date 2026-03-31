---
title: "Downloading Models"
description: "How to download the AI model files required for transcription."
topic_id: first_steps_downloading_models
---

# Downloading Models

Parakeet Transcription requires AI model files to run. These are not bundled with the application and must be downloaded before your first transcription.

## Model Status (Home Screen)

A slim status line at the top of the home screen shows whether your models are ready. When files are missing it also shows an `Open Settings` button that takes you directly to model management.

| Status | Meaning |
|---|---|
| `All N model file(s) present ✓` | All required files are downloaded and ready. |
| `N model file(s) missing: …` | One or more files are absent; open Settings to download. |

When models are ready the `New Transcription` and `Bulk Add Jobs` buttons become active.

## How to Download Models

1. On the home screen, click `Open Settings` (or go to `Settings… > Models`).
2. In the **Models** section, click `Download Missing Models`.
3. A progress bar and a status line appear showing the current file, its position in the queue, and the download size — for example: `[1/3] encoder-model.onnx — 42 MB`.
4. Wait for the status to read `Download complete.`

## Cancelling a Download

To stop a download in progress, click `Cancel`. The status line will show `Download cancelled.` Partially downloaded files are preserved so the download resumes from where it left off next time you click `Download Missing Models`.

## Download Errors

If a download fails, the status line shows `Download failed: <reason>`. Check your internet connection and click `Download Missing Models` again to retry. The application resumes from the last successfully completed file.

## Changing Precision

The model files that need to be downloaded depend on the selected **Model Precision**. To change it, go to `Settings… > Models > Model Precision`. If you switch precision after downloading, the new set of files must be downloaded separately. See [Picking Model Weight Precision](model_precision.md).
