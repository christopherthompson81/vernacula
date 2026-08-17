#!/usr/bin/env python3
"""
Read EVERY alignment row with a local model, rather than only the rows a distance heuristic flagged.

Why this is worth the compute rather than a refinement of the scorer:

  · A DISTANCE CANNOT SEE A CLOSE-BUT-WRONG ROW. `2008 400` read as *two million eight thousand four
    hundred* scores well — it is fluent, correctly-stressed English of about the right length. It was
    caught only because a human read it. Every defect of that shape is invisible to 3xMAD by construction.
  · THE HEURISTIC ITSELF WAS WRONG THREE TIMES in this run — modifier letters, tone digits, and recognizer
    failures each produced a whole language of false or missing flags. A second, independent reader is
    worth having precisely because the first one is a metric I wrote.
  · THE BULK IS THE POINT. 74k rows currently say "verified" on the strength of a median. That claim is
    what the training set rests on, and nothing has actually looked at it.

⚠ THE MODEL TRIAGES, IT DOES NOT DECIDE. Its verdicts land in `judge_verdict` / `judge_note`, beside — not
on top of — the `status` column, so a hand verdict and a heuristic verdict and a model verdict can
disagree visibly. Disagreement between the three is itself the most interesting thing this produces.

⚠ AND IT IS NOT ASKED WHETHER THE IPA IS "CORRECT". It cannot know that. It is asked the one question the
three columns can actually settle: do the transcript, our IPA, and the phones the recognizer heard tell
the same story, and if not, which one is the odd one out.

Rows already labelled `defective_audio` or `recognizer_short` are skipped — their audio side is known
broken, so there is nothing for a reader to adjudicate.

Usage:
  python3 judge_alignment.py --probe 200        # timed sample, prints an ETA and stops
  python3 judge_alignment.py --batch 10
  python3 judge_alignment.py --langs en_us fr_fr
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/mnt/data/Programming/vernacula/scripts/omnivoice_ipa")
from asr_align_report import fold as _fold

ROOT = "/mnt/data/omnivoice_ipa"
DB = f"{ROOT}/work/asr_align/align.sqlite"
ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

SKIP = ("defective_audio", "recognizer_short")

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["agree", "ipa_suspect", "reader_diverged", "notation_only", "unclear"],
                    },
                    "note": {"type": "string"},
                },
                "required": ["id", "verdict", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

SYSTEM = """For each item, answer TWO questions in order. They are separate; do not let one answer the other.

T = the sentence the reader was given.  I = our phonemization of T.  H = phones a recognizer
(wav2vec2-xlsr-53, espeak-flavoured) got from the audio, run together, no separators.

QUESTION 1 — IS I A CORRECT READING OF T?  Ignore H completely. Read T, work out what it should sound
like, compare with I. Check every one of these, because this is where we know defects live:

  · numbers        each number read separately? two adjacent numbers must NOT merge into one bigger one
  · years / dates  read the way THIS language reads them. A DAY-OF-MONTH before or after a month name is
                   an ORDINAL in German, French, Welsh, Turkish, Catalan, Portuguese and Tamil (German
                   `24 september` must be *vierundzwanzigsTEN*, not the cardinal). English is the odd one
                   out in using a cardinal there.
  · abbreviations  expanded to the word (Dr., St., etc., bzw., f.Kr.) rather than spelled as a cluster
  · acronyms       letter runs said as letter NAMES, not read as a word (u.s. must not be the word "us").
                   A short letter run that is not a word of the language is an acronym even when it looks
                   pronounceable: es/pt `irm` (MRI) is "i-ere-eme", not the word *ˈiɾm*; cs `rs`, sv `bnp`,
                   fr `pdg` likewise. If I renders such a run as a single syllable, that is ipa_suspect.
  · units          km/kg/mm expanded to the unit word
  · symbols        % ° $ £ spoken
  · anything in T with no counterpart in I at all, or vice versa

If any of those is wrong -> ipa_suspect, and name it. A defect here is usually FLUENT: `2008 400` read as
"two million eight thousand four hundred" is well-formed speech and still wrong. Do not be reassured by I
looking plausible on its own.

QUESTION 2 — DOES H SHOW THE SAME UTTERANCE AS I?  H is a different, lossy notation. These are measured
correspondences on rows that DO match, so every one of them is agreement, not difference:

  rhotics all collapse   ʁ ɹ ɾ r are one thing; German ɐ̯ also returns as ɾ
  spirantisation         d->ð  s->θ  b->β   (it over-produces ð θ β)
  vowel quality is FREE  ɔ~o  e~ɛ  a~ɑ  ə~ɪ~ɐ  i~ɪ  — never judge on vowel identity
  ɫ->l  t͡ʃ->tʃ  d͡ʒ->dʒ  eᶦ->eɪ  oᶷ->oʊ ; it inserts ɪ ʊ freely and drops t d s ə

H NEVER contains stress, length, tie bars, or modifier letters (ʰ ˠ ʲ ʷ) — absence proves nothing. Tone is
a trailing DIGIT (siɛ5). H runs 5-10% shorter than I on every utterance, EVENLY, not at the end. On
unintelligible audio it emits plausible English-ish phones rather than going quiet.

  reader_diverged  a word you can NAME is in one and not the other. Never from length, never from failing
                   to align the tail.
  notation_only    differences are only the conventions above AND question 1 was clean.
  unclear          H too garbled to judge.
  agree            question 1 clean and H shows the same utterance.

⚠ DO NOT IMPORT ENGLISH CONVENTIONS into question 1. Most languages read a year as a plain cardinal —
Amharic 1981 as "thousand nine hundred eighty one" is CORRECT; so are Chinese, Tamil, Hindi and Arabic
year readings. English's year-pair reading is the exception. Judge against THIS language.

Verdict priority: ipa_suspect > reader_diverged > unclear > notation_only > agree.
Note under 12 words, only when not agree. One entry per item, using its id."""


def ask(batch: list[dict], retries: int = 3) -> list[dict]:
    # ⚠ PROMPT-BOUND, so the payload is compressed rather than the batch enlarged. Measured at batch 20:
    # enlarging the batch moved 0.76 -> 0.73 s/row (nothing), because the cost is prompt processing at
    # ~401 tok/row. Where that goes: TEXT 137, IPA 124, HEARD 118 tokens per row.
    #
    # The recognizer emits phones SPACE-SEPARATED (`t eː ʃ a w a`), and those spaces are pure overhead —
    # each costs about a token. Removing them takes HEARD from 118 to 59 tok/row, the single biggest
    # saving available, and it does not cost information the model was using: our own IPA has no phone
    # spaces either, so both sides are read as strings in the same shape.
    #
    # `lang` is emitted once per batch, not per row: the query orders by language, so a batch is almost
    # always one language.
    # ⚠ NO TRUNCATION, and this is a correctness requirement rather than a nicety. The first version
    # clipped T and I at 200 characters and H at 150 — ASYMMETRIC, because H had already had its spaces
    # removed. On the longest utterances the model therefore saw a whole sentence against a clipped H and
    # concluded, reasonably, that "H cuts off mid-sentence, the reader omitted the final clause". Every one
    # of the 17 `reader_diverged` verdicts in the first probe was that artefact: their heard/ours phone
    # ratio is 0.92, indistinguishable from the 0.94 of the rows called `agree`. Nothing was truncated in
    # the data; I had truncated it in the prompt. The caps bit on 30% of IPA rows and 14% of H rows.
    lines = []
    for b in batch:
        heard = b["phones"].replace(" ", "")
        lines.append(f"[{b['id']}]T:{b['text']}\nI:{b['ipa']}\nH:{heard}")
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"lang={batch[0]['lang']}\n" + "\n".join(lines)},
        ],
        "temperature": 0,
        "max_tokens": 60 * len(batch) + 150,
        # ⚠ Qwen3 is a reasoning model: left on, it spends the whole budget in reasoning_content and
        # returns empty content. This cost a wasted run in the initialism sweep.
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "verdicts", "schema": SCHEMA}},
    }
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                content = json.load(resp)["choices"][0]["message"]["content"]
            return json.loads(content).get("verdicts", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            print(f"    retry {attempt+1}/{retries}: {type(e).__name__}: {e}", file=sys.stderr)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--probe", type=int, default=0, help="judge N rows, print an ETA, stop")
    # ⚠ CONCURRENCY IS THE LEVER, not batch size. The work is prompt-bound: at one slot, going from batch
    # 10 to 30 moved 0.76 -> 0.73 s/row, i.e. nothing. Overlapping requests across server slots is what
    # actually fills the GPU. Keep `--workers` <= the server's `-np`.
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    db = sqlite3.connect(a.db)
    cols = {r[1] for r in db.execute("PRAGMA table_info(utt)")}
    if "judge_verdict" not in cols:
        db.execute("ALTER TABLE utt ADD COLUMN judge_verdict TEXT")
        db.execute("ALTER TABLE utt ADD COLUMN judge_note TEXT")
        db.execute("CREATE INDEX IF NOT EXISTS utt_judge ON utt(judge_verdict)")
        db.commit()

    where = ["ipa IS NOT NULL", "phones IS NOT NULL", "phones != ''", "judge_verdict IS NULL",
             f"COALESCE(status,'') NOT IN {SKIP}"]
    args: list = []
    if a.langs:
        where.append(f"lang IN ({','.join('?' * len(a.langs))})")
        args += a.langs
    q = f"SELECT rowid,lang,text,ipa,phones FROM utt WHERE {' AND '.join(where)} ORDER BY lang,rowid"
    rows = [dict(id=r[0], lang=r[1], text=r[2] or "", ipa=r[3] or "", phones=r[4] or "")
            for r in db.execute(q, args)]
    total_left = len(rows)
    if a.probe:
        rows = rows[: a.probe]
    print(f"# {total_left} rows to judge; this run: {len(rows)}, batch {a.batch}", file=sys.stderr)

    batches = [rows[i : i + a.batch] for i in range(0, len(rows), a.batch)]
    t0, done, counts = time.time(), 0, {}
    # The DB write stays on this thread — sqlite3 connections are not shared across threads, and the
    # writes are trivial next to a 27B forward pass anyway.
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for batch, verdicts in zip(batches, pool.map(ask, batches)):
            got = {v["id"]: v for v in verdicts if isinstance(v, dict) and "id" in v}
            # ⚠ VALIDATE reader_diverged AGAINST DATA THE MODEL DOES NOT HAVE. It confabulates "H cuts off
            # mid-sentence, missing the final clause" on rows whose phone counts are EQUAL — measured at
            # ratios of 0.96-1.02 — while the recognizer does not degrade with length at all: heard/ours is
            # flat at 0.940-0.943 from 0-10 s through 20-60 s. So the claim is false by construction and is
            # downgraded here.
            #
            # ⚠ AND NOT BECAUSE IT CANNOT READ THE SCRIPT — I asserted that first and it is wrong. Asked
            # directly, the model quotes the last two words of an Amharic sentence correctly, transliterates
            # them to plausible IPA, and translates the whole thing. BPE reaches every codepoint; there is
            # no unreachable script. What it actually does is over-trust its own alignment of T against an
            # approximate phone string, with "cuts off mid-sentence" as the plausible-sounding default when
            # that alignment is hard. Which is why the guard is arithmetic rather than an instruction: the
            # instruction addresses a reason, and I had the reason wrong.
            for v in got.values():
                if v.get("verdict") == "reader_diverged":
                    b = next((x for x in batch if x["id"] == v["id"]), None)
                    if b is None:
                        continue
                    ni, nh = len(_fold(b["ipa"])), len(_fold(b["phones"]))
                    if ni >= 12 and nh / ni > 0.85:
                        v["verdict"] = "agree"
                        v["note"] = ""
            for b in batch:
                v = got.get(b["id"])
                verdict = (v or {}).get("verdict", "no_reply")
                note = (v or {}).get("note", "")[:160]
                counts[verdict] = counts.get(verdict, 0) + 1
                db.execute("UPDATE utt SET judge_verdict=?, judge_note=? WHERE rowid=?",
                           (verdict, note, b["id"]))
            db.commit()
            done += len(batch)
            if done % (a.batch * 10) < a.batch or done == len(rows):
                el = time.time() - t0
                print(f"  {done}/{len(rows)}  {el/done:.3f}s/row  {counts}", file=sys.stderr)

    el = time.time() - t0
    if done:
        rate = el / done
        print(f"\n{done} rows in {el:.0f}s = {rate:.2f}s/row ({done/el*60:.0f} rows/min)", file=sys.stderr)
        print(f"ETA for the remaining {total_left - done}: "
              f"{(total_left - done) * rate / 3600:.1f} h", file=sys.stderr)
    db.close()


if __name__ == "__main__":
    main()
