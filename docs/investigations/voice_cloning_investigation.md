# Voice cloning with the OmniVoice IPA fine-tune — investigation log

How well `vernacula-tts --voice` transfers a speaker, and what it takes to get a usable
reference out of found audio rather than a studio clip.

## Run 1 — 2026-09-09 11:10 — cloning a regional accent from a 60 s interview

**Question.** Given an ordinary two-speaker recording — an accent-comparison interview,
one speaker in Irish English and one in Newfoundland English, ~60 s, room noise — can the
pipeline pick out one speaker, encode him, and re-speak new text in his voice? And is the
result the same speaker by any measure other than "it sounds right to me"?

Source is a local test recording of two identifiable people. It is not copied into the repo,
and the clips it produced stay on the machine.

### Diarization decided the whole job, and two of three backends got it wrong

To pick a reference you first need to know which stretches are one speaker.

| backend | result |
|---|---|
| DiariZen | **1 segment for the whole recording** — both speakers merged |
| Sortformer | 8 segments, but the long monologue split across two speaker labels, and the label that owns "…where I'm from" flips mid-monologue |
| VibeVoice-ASR Streaming (1.5B) | 14 segments, consistent: one speaker holds every long stretch, the other only backchannels ("Yeah", "Right", "Really?") |

The VibeVoice segmentation is the only one whose speaker assignment survives a content check
— the same label owns every first-person statement about living in the place under discussion.
Worth remembering that the model with *built-in* attribution beat both dedicated diarizers on
a short, noisy, heavily-accented two-hander; the accent is the plausible reason, since the
diarizers' embeddings are trained on rather more standard speech.

### Cutting the reference

The interviewer's backchannels are the constraint: they land in the middle of the long turns,
so "longest turn" is not the same as "longest clean stretch". Taking the boundaries from
`silencedetect` rather than from ASR timings matters — the ASR segment started at 23.00 s and
speech actually starts at 22.99, so cutting on the ASR boundary clips the first phoneme.

Chosen: 22.90 → 31.60 (8.7 s), which sits inside a 0.89 s silence at the start and a 0.44 s
silence at the end, one complete sentence, no backchannel.

**ASR is not good enough to be the reference transcript here.** Parakeet on the clip gave
"once you get to the vehicles and whip the spay" and "the Evelyn Peninsula" for what are
place names; on the whole file, with more context, it got most of them right. The transcript
was hand-corrected before use — a wrong `--ref-text` means the IPA the model is conditioned on
does not describe the audio it is conditioned on.

### Reference language: GenAm, not the British delta

`--ref-lang` picks the phonemizer, and the candidates for this speaker were `en` (GenAm) and
`en-GB` (SSBE). GenAm, because the variety in the reference is **rhotic** and SSBE is not; a
non-rhotic IPA would claim /r/-less vowels the audio plainly has. The vowel mismatches that
remain (LOT/CLOTH) are smaller than losing the rhotics.

### Result

8.70 s of reference → 217 codec codes. Synthesis at 32 steps on the 3090: ~1.8-2.0× real time.

Round-tripping the output through ASR returns the input sentence essentially verbatim, so the
output is speech rather than the noise this fine-tune emits when it is out of distribution.

### Is it the same speaker? Measured, not asserted

Cosine similarity between WeSpeaker embeddings (the model DiariZen already ships), all clips
16 kHz mono:

| pair | cos |
|---|---|
| reference vs two other genuine segments of the same speaker | 0.849, 0.884 |
| reference vs clone, same sentence | 0.833 |
| reference vs clone, unseen sentence | 0.850 |
| reference vs the other speaker in the recording | **0.090** |
| clone vs the other speaker | 0.10 – 0.17 |

The clone is as close to the reference as the speaker's own other clips are, and the
different-speaker floor is an order of magnitude away. Note the same-speaker baseline is
itself only 0.85-0.88 — clip-to-clip variation within one speaker is the ceiling here, and
the clone is at it.

**Negative result: diarization is the wrong instrument for this question.** The first attempt
at an objective check was to concatenate [reference + clone] and ask a diarizer whether it
heard one speaker or two. It said one — but concatenating [other speaker + clone] *also* came
back as one speaker, while [reference + other speaker] correctly came back as two. A 3.4 s
clip is too short for the pipeline's clustering to commit, so the test cannot distinguish
"same voice" from "not enough evidence". The embedding cosine answers the same question
directly and does not have that failure mode.

### Open

- The reference is 8.7 s. The fine-tune's corpus median is 12 s and nothing here establishes
  how short a reference can get before timbre transfer degrades — a sweep over reference
  length against this cosine would settle it, and would be worth having as guidance.
- Nothing persists an encoded voice. The codes exist only inside the run; the stored-voice
  library (`web-demo/public/models`) is read-only from the CLI's side, so re-cloning re-encodes
  and re-loads the 654 MB encoder every time. A "save this as a voice" path is missing.
