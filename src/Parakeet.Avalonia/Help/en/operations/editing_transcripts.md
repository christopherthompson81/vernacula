---
title: "Editing Transcripts"
description: "How to review, correct, and verify transcribed segments in the transcript editor."
topic_id: operations_editing_transcripts
---

# Editing Transcripts

The **Transcript Editor** lets you review ASR output, correct text, rename speakers inline, adjust segment timing, and mark segments as verified — all while listening to the original audio.

## Opening the Editor

1. Load a completed job (see [Loading Completed Jobs](loading_completed_jobs.md)).
2. In the **Results** view, click `Edit Transcript`.

The editor opens as a separate window and can remain open alongside the main application.

## Layout

Each segment is shown as a card with two side-by-side panels:

- **Left panel** — the original ASR output with per-word confidence colouring. Words the model was less certain about appear in red; high-confidence words appear in the normal text colour.
- **Right panel** — an editable text box. Make corrections here; the diff against the original is highlighted as you type.

The speaker label and time range appear above each card. Click a card to focus it and reveal its action icons. Hover over any icon to see a tooltip describing its function.

## Icon Legend

### Playback Bar

| Icon | Action |
|------|--------|
| ▶ | Play |
| ⏸ | Pause |
| ⏮ | Jump to previous segment |
| ⏭ | Jump to next segment |

### Segment Card Actions

| Icon | Action |
|------|--------|
| <mdl2 ch="E77B"/> | Reassign segment to a different speaker |
| <mdl2 ch="E916"/> | Adjust segment start and end times |
| <mdl2 ch="EA39"/> | Suppress or unsuppress the segment |
| <mdl2 ch="E72B"/> | Merge with the previous segment |
| <mdl2 ch="E72A"/> | Merge with the next segment |
| <mdl2 ch="E8C6"/> | Split the segment |
| <mdl2 ch="E72C"/> | Redo ASR on this segment |

## Audio Playback

A playback bar runs across the top of the editor window:

| Control | Action |
|---------|--------|
| Play / Pause icon | Start or pause playback |
| Seek bar | Drag to jump to any position in the audio |
| Speed slider | Adjust playback speed (0.5× – 2×) |
| Prev / Next icons | Jump to the previous or next segment |
| Playback mode dropdown | Select one of three playback modes (see below) |
| Volume slider | Adjust playback volume |

While playing, the word currently being spoken is highlighted in the left panel. When paused after a seek, the highlight updates to the word at the seek position.

### Playback Modes

| Mode | Behaviour |
|------|-----------|
| `Single` | Play the current segment once, then stop. |
| `Auto-advance` | Play the current segment; when it ends, mark it as verified and advance to the next. |
| `Continuous` | Play all segments in sequence without marking any as verified. |

Select the active mode from the dropdown in the playback bar.

## Editing a Segment

1. Click a card to focus it.
2. Edit the text in the right panel. Changes are saved automatically when you move focus to another card.

## Renaming a Speaker

Click the speaker label inside the focused card and type a new name. Press `Enter` or click away to save. The new name is applied to that card only; to rename a speaker globally, use [Edit Speaker Names](editing_speaker_names.md) from the Results view.

## Verifying a Segment

Click the `Verified` checkbox on a focused card to mark it as reviewed. Verified status is saved to the database and is visible in the editor on future loads.

## Suppressing a Segment

Click `Suppress` on a focused card to hide the segment from exports (useful for noise, music, or other non-speech sections). Click `Unsuppress` to restore it.

## Adjusting Segment Times

Click `Adjust Times` on a focused card to open the time-adjustment dialog. Use the scroll wheel over the **Start** or **End** field to nudge the value in 0.1-second increments, or type a value directly. Click `Save` to apply.

## Merging Segments

- Click `⟵ Merge` to merge the focused segment with the segment immediately before it.
- Click `Merge ⟶` to merge the focused segment with the segment immediately after it.

The combined text and time range of both cards are joined. This is useful when a single spoken utterance was split across two segments.

## Splitting a Segment

Click `Split…` on a focused card to open the split dialog. Position the split point within the text and confirm. Two new segments are created covering the original time range. This is useful when two distinct utterances were merged into one segment.

## Redo ASR

Click `Redo ASR` on a focused card to re-run speech recognition on that segment's audio. The model processes only the audio slice for that segment and produces a fresh, single-source transcription.

Use this when:

- A segment came from a merge and cannot be split (merged segments span multiple ASR sources; Redo ASR collapses them into one, after which `Split…` becomes available).
- The original transcription is poor and you want a clean second pass without editing manually.

**Note:** Any text you have already typed in the right panel is discarded and replaced with the new ASR output. The operation requires the audio file to be loaded; the button is disabled if audio is unavailable.