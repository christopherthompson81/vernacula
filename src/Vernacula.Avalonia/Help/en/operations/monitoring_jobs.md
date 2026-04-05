---
title: "Monitoring Jobs"
description: "How to watch a running or queued job's progress."
topic_id: operations_monitoring_jobs
---

# Monitoring Jobs

The **Progress** view gives you a live view of a running transcription job.

## Opening the Progress View

- When you start a new transcription the application goes to the Progress view automatically.
- For a job already running or queued, find it in the **Transcription History** table and click `Monitor` in its **Actions** column.

## Reading the Progress View

| Element | Description |
|---|---|
| Progress bar | Overall completion percentage. Indeterminate (animated) while the job is starting or resuming. |
| Percentage label | Numeric percentage shown to the right of the bar. |
| Status message | Current activity — for example `Audio Analysis` or `Speech Recognition`. Shows `Waiting in queue…` if the job has not started yet. |
| Segments table | Live feed of transcribed segments with **Speaker**, **Start**, **End**, and **Content** columns. Scrolls automatically as new segments arrive. |

## Progress Phases

The phases shown depend on the **Segmentation Mode** selected in Settings.

**Speaker Diarization mode** (default):

1. **Audio Analysis** — Sortformer diarization runs over the whole file to identify speaker boundaries. The bar may stay near 0% until this phase completes.
2. **Speech Recognition** — each speaker segment is transcribed. The percentage climbs steadily during this phase.

**Voice Activity Detection mode**:

1. **Detecting speech segments** — Silero VAD scans the file to find regions of speech. This phase is fast.
2. **Speech Recognition** — each detected speech region is transcribed.

In both modes the live segment table fills in as transcription proceeds.

## Navigating Away

Click `← Back to Home` to return to the home screen without interrupting the job. The job continues running in the background and its status updates in the **Transcription History** table. Click `Monitor` again at any time to return to the Progress view.
