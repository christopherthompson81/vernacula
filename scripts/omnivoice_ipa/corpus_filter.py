"""The one place the training pipeline decides which utterances are USABLE.

⚠ WHY THIS EXISTS AS A SHARED MODULE RATHER THAN A FLAG ON ONE SCRIPT. The exclusion has to be applied
BEFORE `sampling_budget.py`, not just before `build_webdataset.py`. That script sets each language's
oversampling weight so its scarcest owned primitive reaches a minimum exposure per epoch — computing
that over pairs which are then discarded targets the wrong number, and the error is silent. The order
is:

    exclude  ->  patch manifests  ->  sampling weights  ->  webdataset

Both `sampling_budget.py` and `build_webdataset.py` load their manifests through `load_manifest()`
here, so neither can forget.

⚠ WHAT GETS EXCLUDED, AND WHAT DELIBERATELY DOES NOT. Only `defective_audio` — the FLEURS-side data
defect the wav2vec2 sweep found (Run 36): 611 utterances whose audio is truncated to a fraction of
its transcript, 585 of them Welsh (17.1% of cy_gb). Those are catastrophic TRAINING PAIRS — a full
sentence of IPA against ~1.5s of audio teaches the model to compress a sentence into a tenth of its
time. Not ours to fix; the action is to drop the pair and report upstream.

Everything else in the `status` column stays IN:
  · `investigate` (1,782) is a QC QUEUE, not a verdict — most are recognizer noise on a fine pair.
  · `recognizer_short` (737) is a fact about the RECOGNIZER, not the audio.
  · `verified` (74,446) is the bulk of the corpus.
A status column is a work log. Only one of its values is a statement about the data being unusable.

The exclusion list is MATERIALIZED to `work/exclusions.tsv` by `exclude_defective.py` so the training
pipeline does not depend on the alignment DB being present, and so the set that fed any given run is
auditable after the fact.
"""
from __future__ import annotations

import json
import os

ROOT = "/mnt/data/omnivoice_ipa"
TOKENS = f"{ROOT}/corpus/tokens"
EXCLUSIONS = f"{ROOT}/work/exclusions.tsv"

# ⚠ The ONLY status that means "this pair cannot be trained on". See the module note.
EXCLUDE_STATUSES = ("defective_audio",)


def load_exclusions(path: str = EXCLUSIONS) -> dict[str, set[str]]:
    """{lang: {utterance id}} — empty (and silent) if the file has not been generated.

    ⚠ Returns empty rather than raising: the pipeline must still run on a corpus that never had an
    audio gate. But every caller PRINTS what it dropped, so a missing file cannot pass unnoticed.
    """
    out: dict[str, set[str]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            out.setdefault(parts[0], set()).add(parts[1])
    return out


def load_manifest(lang: str, exclusions: dict[str, set[str]] | None = None,
                  tokens_dir: str = TOKENS) -> tuple[list[dict], int]:
    """(rows, n_dropped) for one language, with defective pairs already removed.

    The manifest `id` is the wav stem, which is what `exclude_defective.py` writes.
    """
    ex = (exclusions if exclusions is not None else load_exclusions()).get(lang, set())
    rows, dropped = [], 0
    with open(f"{tokens_dir}/manifest_{lang}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in ex:
                dropped += 1
                continue
            rows.append(d)
    return rows, dropped
