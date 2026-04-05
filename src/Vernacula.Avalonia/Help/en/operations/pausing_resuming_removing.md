---
title: "Pausing, Resuming, or Removing Jobs"
description: "How to pause a running job, resume a stopped one, or delete a job from history."
topic_id: operations_pausing_resuming_removing
---

# Pausing, Resuming, or Removing Jobs

## Pausing a Job

You can pause a running or queued job from two places:

- **Progress view** — click `Pause` in the bottom-right corner while watching the active job.
- **Transcription History table** — click `Pause` in the **Actions** column of any row whose status is `running` or `queued`.

After you click `Pause`, the status line shows `Pausing…` while the application finishes the current processing unit. The job status then changes to `cancelled` in the history table.

> Pausing saves all segments transcribed so far. You can resume the job later without losing that work.

## Resuming a Job

To resume a paused or failed job:

1. On the home screen, find the job in the **Transcription History** table. Its status will be `cancelled` or `failed`.
2. Click `Resume` in the **Actions** column.
3. The application returns to the **Progress** view and continues from where processing stopped.

The status line shows `Resuming…` briefly while the job re-initialises.

## Removing a Job

To permanently delete a job and its transcript from the history:

1. In the **Transcription History** table, click `Remove` in the **Actions** column of the job you want to delete.

The job is removed from the list and its data is deleted from the local database. This action cannot be undone. Exported files saved to disk are not affected.
