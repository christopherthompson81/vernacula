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

Everything else in the `status` column stays IN. Counts are as of 2026-08-22, over all 102 languages
(the corpus grew from 28; the old counts in this note were from that smaller set):
  · `verified` (258,045) is the bulk of the corpus.
  · `investigate` (7,440) is a QC QUEUE, not a verdict. ⚠ 79.3% of it is `sibling=exonerated` — a
    same-text recording scores fine, so our IPA is demonstrably not the cause.
  · `recognizer_short` (797) is MOSTLY a fact about the RECOGNIZER, not the audio, and the shape of the
    distribution is what says so — an average would have hidden a mixture. Characters of text per second
    of audio (a cut file has too much text for its length):

        status              min   p25   med   p75    max   >20cps      n
        defective_audio     3.8  11.5  21.0  64.0  341.6     744    1248
        recognizer_short    2.0   6.3   7.6   8.4   33.6      40     797
        verified            1.4   6.5   9.1  10.9   30.2     239  258045

    `recognizer_short`'s median sits BELOW `verified` — longer audio for the same text, i.e. slow
    reading the recognizer gave up on — and only 40 rows (5%) look cut at all. ⚠ THE 5% IS REAL THOUGH:
    that is ~55x the rate in `verified`, so the status is a mixture of "the recognizer bailed" and a
    small "the audio really is short" tail, and it is kept IN because the tail is 40 rows, not because
    the tail is empty. ⚠ An investigation-doc note once listed the whole status as excludable; it is
    not, and this table is why.
  · ⚠ `defective_audio` IS NOT ONLY TRUNCATION. 60% is cut or blank (the >20 cps mass, out to 341),
    but the other 40% has ORDINARY DURATION and wrong CONTENT — audio uncorrelated with the text or
    the language, including a reader asking for a retake in English inside the Welsh set. No duration
    or transcript check can see that; it took the phone recognizer, which is the reason this status
    exists at all rather than being derivable from the tsv.
  · `instrument_blind` (455) — the recognizers cannot adjudicate the language (<50% of our phones come
    back unchanged). ⚠ A statement about the INSTRUMENT, never about the audio or the IPA. Excluding on
    it would throw away nine languages' hardest rows for no reason.
  · `convention` (470), `artefact` (77), `examined_clean` (50) — human verdicts that the divergence is
    notation, the recognizer's error, or nothing. All mean the pair is FINE.
  · `defect` (1,339) — our phonemization WAS wrong. ⚠ NOT a permanent exclusion: it is a staleness
    flag. 1,333 are ckb_iq's free conjunction, fixed in the engine, and the rows are good the moment
    the IPA is re-derived. Excluding them permanently would discard a language over a landed fix.

⚠ `reader_divergence` IS SPLIT, AND THE SPLIT IS THE WHOLE POINT OF `read_text`. The reader did not say
what the transcript says, which makes the PAIR bad — unless someone wrote down what they DID say. 144
of 185 now carry a hand `read_text` (with `{en:…}`/`{pt:…}` code-switch spans where the reader switched
language) and their `ipa` is re-derived from it, so those are CORRECTED pairs and must stay in. The 41
without one are the genuinely unusable remainder.

A status column is a work log. Only two of its values are statements about the data being unusable, and
one of those is conditional on whether the row was repaired.

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

# ⚠ The only status that is UNCONDITIONALLY untrainable. See the module note.
EXCLUDE_STATUSES = ("defective_audio",)

# ⚠ Untrainable ONLY WHERE UNREPAIRED. A `reader_divergence` row with a hand `read_text` has had what
# the reader actually said written down and its IPA re-derived from that, so it is a corrected pair;
# without one, the transcript and the audio disagree and the pair teaches a wrong alignment.
EXCLUDE_UNLESS_HAND_READ_TEXT = ("reader_divergence",)


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
            # ⚠ THE MANIFEST'S OWN `status` IS THE FILTER, with exclusions.tsv as the fallback for
            #   manifests written before the field existed. Exclusion used to happen at ENCODE time,
            #   which fused a revisable judgement to a GPU artifact: cy_gb and es_419 were encoded
            #   2026-07-01 and carried 970 `defective_audio` rows for two months while the other 96
            #   languages were pruned, with nothing in any log to say so. Codes are a function of the
            #   audio; what to train on is a decision. Label at build, decide here.
            # ⚠ AN EMPTY `status` MEANS NO VERDICT, NOT "CLEAN" — the align pass does not cover every
            #   row, and reading absence as a verdict once dropped 60% of Assamese as "deliberate".
            st = d.get("status")
            if st and (st in EXCLUDE_STATUSES or (
                    st in EXCLUDE_UNLESS_HAND_READ_TEXT and d.get("ipa_src") != "hand")):
                dropped += 1
                continue
            if d["id"] in ex:
                dropped += 1
                continue
            rows.append(d)
    return rows, dropped
