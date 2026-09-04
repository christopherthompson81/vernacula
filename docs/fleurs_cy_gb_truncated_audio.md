# FLEURS `cy_gb`: 585 train-split audio files are truncated (17.1% of the split)

**Status:** ready to file upstream against the FLEURS dataset (`google/fleurs`).
**Companion file:** `fleurs_cy_gb_truncated_audio.txt` — the 585 affected filenames, one per line.

## Summary

585 of the 3,427 utterances in the Welsh (`cy_gb`) **train** split ship audio that is a small
fraction of the length its transcript requires. The transcript and the audio are individually fine;
only the **pair** is broken, which is why no text-side check finds it.

    median duration, affected     1.44 s
    median duration, unaffected  14.16 s
    affected                    585 / 3,427  (17.1%)

A representative case: `10089284382541497316.wav` is 0.72 s of audio against a transcript needing
roughly 100 phones.

## It is not a download artifact

Checked the member sizes inside the source tar directly rather than trusting our extracted copies —
the truncated files really are small there:

    median member size, affected      99,898 bytes
    median member size, unaffected   954,298 bytes

(The ~2:1 discrepancy against the decoded duration is just stereo, which our loader averages.)
Re-fetching does not help; this is what the dataset ships.

## What the audio contains

Of the 585, **333 decode to nothing at all** under a multilingual phone recognizer
(`facebook/wav2vec2-xlsr-53-espeak-cv-ft`) and 252 return short fragments. Where the fragments are
intelligible they look like **English**, in files whose transcript is Welsh — e.g.
`ð ə s eɪ m ɪ k s p ɪ ɹ ɪ ə n s ə` ("the same experience"), `h aʊ s`, `ɡ eɪ v ð ə … m eɪ ʃ ə n`.

That last point is **suggestive but weak on its own** — a one-second noisy fragment biases this
recognizer toward English whatever is actually in it. The **duration is the unambiguous part**, and
the English-sounding content is offered only as a hint at the cause (possibly a segmentation or
alignment step that cut against the wrong source audio).

## Why it matters for downstream users

These are not merely short clips, they are **catastrophic training pairs**: a full sentence of
target text against ~1.5 s of audio teaches a model to compress a sentence into a tenth of its
duration. Any speech task that consumes (text, audio) pairs from this split is affected, and nothing
in the transcript signals the problem.

For our own use we exclude them, which costs real coverage: Welsh was in our corpus as the sole
source of one phonetic primitive (U+0325, the voiceless ring), and the exclusion removes 15.7% of
its occurrences. It survives — 1,557 of 1,846 remain across 1,110 utterances, and no phone
disappears entirely — but only because we checked.

## How this was found

A phone-recognizer pass over the whole corpus, comparing recognized phones against our own
phonemization per utterance, then flagging utterances whose **phones-per-second implied by the
transcript** was an outlier relative to that language's own median (3×MAD, not a fixed threshold).

## The detector is not simply aggressive

The same detector, unchanged, has since been run over **102 FLEURS languages / 270,106 utterances** —
essentially the whole dataset. Outside `cy_gb` and one milder case it flags almost nothing:

    cy_gb                        585 / 3,427    = 17.1%
    oc_fr                         94 / 3,379    =  2.8%   (a different, milder defect — see below)
    every other language          64 / 263,300  =  0.024%

The full tail after those two, in order: sd_in 12, ff_sn 9, ny_mw 6, fa_ir 5, ckb_iq 4, ceb_ph 4, so_so 4,
then a dozen languages with 2 or fewer. No language other than Welsh and Occitan reaches even 0.5% of its
own split.

**`oc_fr` is not a second instance of this defect**, and is called out here so it is not mistaken for one.
Its 94 files are short for their transcripts, but they contain the *beginning of the correct sentence* at a
normal speaking rate — scored against the best-matching prefix of the expected phone string they reach
0.634 against an Occitan baseline of 0.639, whereas the Welsh files reach only 0.231 against a baseline of
0.696. They also end in silence (last 100 ms at 3% of overall RMS) rather than being cut mid-signal, where
half the Welsh files stop at full amplitude. And the Occitan population is a *continuous* left tail with no
gap, while Welsh is cleanly bimodal — 583 files below 0.3 of their implied duration, then a gap, then the
healthy split. Occitan looks like audio that stops early; Welsh looks like the wrong audio entirely.

Against the 100 languages that are not Welsh or Occitan the gap is nearly three orders of magnitude, and
it is the reason this report is about `cy_gb` specifically rather than about a threshold. A detector tuned
too tightly would light up everywhere; this one is near-silent across 100 languages and then finds a sixth
of the Welsh split.

(One unrelated defect shows up at comparable scale and is **not** the same problem: `es_419` ships 490
files that are full-length and **empty** — normal duration, no signal. Their phones-per-second is
unremarkable, so this detector cannot see them; they were found by measuring the waveform directly.
Mentioned only so the 490 is not mistaken for a second instance of the truncation.)

## Reproduction

```python
import soundfile as sf, statistics
durs = [len(sf.read(p)[0]) / sr for p, sr in cy_gb_train_files]
print(statistics.median(durs))   # affected files cluster near 1.4 s
```

Or directly against the shipped tar, without decoding:

```bash
tar tvf cy_gb.tar.gz | awk '$3 < 200000 {n++} END {print n, "small members"}'
```
