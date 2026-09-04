# The ASR-alignment tooling moved to vernacula-phonemizer

`asr_align_corpus.py`, `asr_align_report.py`, `asr_align_label.py`, `scan_silent_audio.py`,
`consonant_skeleton.py`, `confusion_pairs.py`, `judge_alignment.py` and `judge_cascade.py` now live in
**`vernacula-phonemizer/tools/corpus/asr-align/`**, alongside `tools/referee-eval` and the FLEURS audio
fetcher that was already there.

⚠ **They were moved, not copied.** Two divergent copies is exactly what this move was meant to end: the
two FLEURS downloaders had drifted until each held half of one fix — the phonemizer's verified against
the REMOTE SIZE, the one here had a stall watchdog after an eleven-hour silent hang. Both halves are now
in `vernacula-phonemizer/tools/corpus/fetch-fleurs-audio.py`.

**Why there:** everything that tooling finds is a phonemizer fix, and a fix should be able to land in the
same commit as its evidence. It also sat on two definitions of "the segmental backbone" — one in
`referee-eval/config.ts`, one in `asr_align_report.py` — which can now be seen side by side.

## The audio downloader is superseded

`fetch_fleurs_all.py` is kept only for its TRANSCRIPT fetching. For AUDIO use
`vernacula-phonemizer/tools/corpus/fetch-fleurs-audio.py`, which is strictly better and now carries both
halves of the fix: it batches one metadata call instead of a HEAD per language, verifies each file against
the REMOTE SIZE (so a short or interrupted download is re-fetched rather than counted as complete), and
has the stall watchdog. ⚠ The presence check here was `os.listdir` of `data/`, which counts a language as
cached the moment its FOLDER exists — `ast_es` and `nso_za` were both skipped indefinitely on the strength
of an empty directory.

## What is still here, and why

The training-corpus pipeline, which answers *which pairs do we train on* rather than *is our IPA right*:

    ingest_fleurs.py  corpus_filter.py  exclude_defective.py  sampling_budget.py
    build_webdataset.py  publish_hf.py  publish_hf_dataset.py  patch_manifest_ipa.py

`exclude_defective.py` reads the alignment DB's `status` column directly and has no code dependency on
the moved tools, so this side still runs unchanged. The DB itself stays at
`work/asr_align/align.sqlite`; both repos point at it via `ASR_ALIGN_ROOT` (default unchanged).
