---
title: "New Transcription Workflow"
description: "Step-by-step guide to transcribing an audio file."
topic_id: operations_new_transcription
---

# New Transcription Workflow

Use this workflow to transcribe a single audio file.

## Prerequisites

- All model files must be downloaded. The **Model Status** card must show `All N model file(s) present ✓`. See [Downloading Models](../first_steps/downloading_models.md).

## Supported Formats

### Audio

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.aiff`, `.aif`, `.webm`

### Video

`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.ts`, `.mts`, `.m2ts`, `.3gp`

Video files are decoded via FFmpeg. If a video file contains **multiple audio streams** (e.g. multiple languages or commentary tracks), one transcription job is created automatically for each stream.

## Steps

### 1. Open the New Transcription form

Click `New Transcription` on the home screen, or go to `File > New Transcription`.

### 2. Select a media file

Click `Browse…` next to the **Audio File** field. A file picker opens filtered to supported audio and video formats. Select your file and click **Open**. The file path appears in the field.

### 3. Name the job

The **Job Name** field is pre-filled from the file name. Edit it if you want a different label — this name appears in the Transcription History on the home screen.

### 4. Start transcription

Click `Start Transcription`. The application switches to the **Progress** view.

To go back without starting, click `← Back`.

## What Happens Next

The job runs through two phases shown in the progress bar:

1. **Audio Analysis** — speaker diarization: identifying who is speaking and when.
2. **Speech Recognition** — converting speech to text segment by segment.

Transcribed segments appear in the live table as they are produced. When processing is complete the application moves to the **Results** view automatically.

If you add the job while another is already running, the new job will show `queued` status and start when the current job finishes. See [Monitoring Jobs](monitoring_jobs.md).
