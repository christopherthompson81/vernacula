---
title: "Editing Speaker Names"
description: "How to replace generic speaker IDs with real names in a transcript."
topic_id: operations_editing_speaker_names
---

# Editing Speaker Names

The transcription engine automatically labels each speaker with a generic ID (for example, `speaker_0`, `speaker_1`). You can replace these with real names that will appear throughout the transcript and in any exported files.

## How to Edit Speaker Names

1. Open a completed job. See [Loading Completed Jobs](loading_completed_jobs.md).
2. In the **Results** view, click `Edit Speaker Names`.
3. The **Edit Speaker Names** dialog opens with two columns:
   - **Speaker ID** — the original label assigned by the model (read-only).
   - **Display Name** — the name shown in the transcript (editable).
4. Click a cell in the **Display Name** column and type the speaker's name.
5. Press `Tab` or click another row to move to the next speaker.
6. Click `Save` to apply the changes, or `Cancel` to discard them.

## Where Names Appear

Updated display names replace the generic IDs in:

- The segments table in the Results view.
- All exported files (Excel, CSV, SRT, Markdown, Word, JSON, SQLite).

## Editing Names Again

You can reopen the Edit Speaker Names dialog at any time while the job is loaded in the Results view. Changes are saved to the local database and persist across sessions.
