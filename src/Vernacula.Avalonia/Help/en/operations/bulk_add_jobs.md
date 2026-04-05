---
title: "Enqueueing Multiple Audio Files"
description: "How to add several audio files to the job queue at once."
topic_id: operations_bulk_add_jobs
---

# Enqueueing Multiple Audio Files

Use **Bulk Add Jobs** to queue multiple audio or video files for transcription in one step. The application processes them one at a time in the order they were added.

## Prerequisites

- All model files must be downloaded. The **Model Status** card must show `All N model file(s) present ✓`. See [Downloading Models](../first_steps/downloading_models.md).

## How to Bulk Add Jobs

1. On the home screen, click `Bulk Add Jobs`.
2. A file picker opens. Select one or more audio or video files — hold `Ctrl` or `Shift` to select multiple files.
3. Click **Open**. Each selected file is added to the **Transcription History** table as a separate job.

> **Video files with multiple audio streams:** If a video file contains more than one audio stream (for example, multiple languages or a director's commentary track), the application creates one job per stream automatically.

## Job Names

Each job is named automatically from its audio file name. You can rename a job at any time by clicking its name in the **Title** column of the Transcription History table, editing the text, and pressing `Enter` or clicking away.

## Queue Behaviour

- If no job is currently running, the first file starts immediately and the rest are shown as `queued`.
- If a job is already running, all newly added files are shown as `queued` and will start automatically in sequence.
- To monitor the active job, click `Monitor` in its **Actions** column. See [Monitoring Jobs](monitoring_jobs.md).
- To pause or remove a queued job before it starts, use the `Pause` or `Remove` buttons in its **Actions** column. See [Pausing, Resuming, or Removing Jobs](pausing_resuming_removing.md).
