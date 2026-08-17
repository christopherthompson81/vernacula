#!/usr/bin/env python3
"""
Two-stage judge: a cheap screen over every row, then a REASONING pass over only what the screen doubted.

The single-stage judge forced an unwinnable trade. Tuned for recall it caught the known defects and
produced confident nonsense on languages the model reads poorly ("year 1981 misread as cardinal 'three
hundred twenty-nine'" when the IPA plainly reads *thousand nine hundred eighty one*); tuned for precision
it agreed with all four confirmed defects. One prompt cannot be both, because the two jobs are different
sizes: 99% of rows need a glance, and the remaining 1% need actual work.

  STAGE 1  every row, thinking OFF, binary output. "Is there anything to look at here?" Biased toward
           saying yes — a false `check` costs one stage-2 slot, a false `agree` is a defect that ships.
  STAGE 2  only the `check` rows, thinking ON, full checklist, one row per request. Assigns the real
           verdict and must justify it.

The economics are what make it worth doing: stage 1 is the whole corpus at ~0.45 s/row, stage 2 is a few
hundred rows at a few seconds each. Reasoning on 1% of the data costs a rounding error and is where all
the judgement is.

Usage:
  python3 judge_cascade.py --stage 1 --probe 300
  python3 judge_cascade.py --stage 2
  python3 judge_cascade.py --stage 1 --langs en_us
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

# ── STAGE 1 ────────────────────────────────────────────────────────────────────────────────────────
# Deliberately short: it is re-sent for every batch across the whole corpus, and it only has to make a
# binary call. No categories, no notes, no justification — those cost output tokens on 76k rows to
# produce text that stage 2 will overwrite anyway.
SCREEN_SYSTEM = """T = a sentence. I = our phonemization of T. H = phones a recognizer heard in the audio
(espeak-flavoured, no separators, lossy).

For each item, QUOTE the one token from T whose rendering in I is wrong — copy it exactly. If nothing is
wrong, return an EMPTY STRING "". Never return "?" or any other placeholder: an empty string is the
answer for "nothing wrong here", and it is the answer most items should get.

An empty string is the expected answer; most rows are fine. Quoting a token you cannot actually see
rendered wrongly is worse than returning "".

What qualifies as wrong:
  · two adjacent numbers rendered as one larger number
  · a year, date or ordinal rendered in a form this language does not use
  · an abbreviation left as a consonant cluster instead of its word (bzw, st, kl, srl)
  · a letter run rendered as one syllable instead of letter names (u.s. as "us", irm as one word)
  · a unit or symbol (km, %, °, $) in T with nothing corresponding in I
  · a whole WORD of T with no counterpart in I, or in I with none in T

NEVER a reason to quote anything: rhotics collapsing (ʁ ɹ ɾ r), d~ð, s~θ, b~β, ANY vowel quality
difference, ɫ~l, tie bars, missing stress/length/aspiration marks, tone as a digit, H running 5-10% short,
H being sloppy or approximate anywhere."""

SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "r": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "bad": {"type": "string"}},
                "required": ["id", "bad"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["r"],
    "additionalProperties": False,
}

# ── STAGE 2 ────────────────────────────────────────────────────────────────────────────────────────
ADJ_SYSTEM = """Adjudicate ONE utterance. Answer two questions in order; do not let one answer the other.

T = the sentence the reader was given.  I = our phonemization of T.  H = phones a recognizer
(wav2vec2-xlsr-53, espeak-flavoured) heard in the audio, run together, no separators.

Q1 — IS I A CORRECT READING OF T?  Ignore H. Work out what T should sound like, compare with I. Check:
  · numbers — each read separately; two adjacent numbers must NOT merge into a larger one
  · years/dates — as THIS language reads them. A day-of-month by a month name is an ORDINAL in German,
    French, Welsh, Turkish, Catalan, Portuguese, Tamil; English is the odd one out with a cardinal.
  · abbreviations — expanded to the word (Dr., St., etc., bzw., f.Kr.), not left as a cluster
  · acronyms — letter NAMES, not a word. `u.s.` is not "us"; es/pt `irm` is "i-ere-eme"; cs `rs`, sv
    `bnp`, fr `pdg` likewise. A short letter run that is not a word of the language is an acronym.
  · units (km kg mm) and symbols (% ° $ £) spoken as words
  · anything in T absent from I, or in I absent from T
A defect here is usually FLUENT — `2008 400` as "two million eight thousand four hundred" is well-formed
and still wrong. I looking plausible alone is not evidence.

Q2 — DOES H SHOW THE SAME UTTERANCE AS I?  H is lossy. All of this is AGREEMENT, measured on matching
rows: rhotics collapse (ʁ ɹ ɾ r; German ɐ̯ -> ɾ); d->ð, s->θ, b->β; vowel quality is free (ɔ~o, e~ɛ, a~ɑ,
ə~ɪ~ɐ, i~ɪ); ɫ->l, t͡ʃ->tʃ, d͡ʒ->dʒ, eᶦ->eɪ, oᶷ->oʊ; ɪ/ʊ inserted, t/d/s/ə dropped. H never has stress,
length, tie bars or modifier letters (ʰ ˠ ʲ ʷ) — absence proves nothing. Tone is a trailing digit. H runs
5-10% short EVENLY, not at the end. On unintelligible audio it emits plausible English-ish phones.

⚠ IF YOU CANNOT CONFIDENTLY READ THE NUMBERS OR WORDS OF THIS LANGUAGE, answer `unsure` rather than
guessing. A wrong confident verdict is worse than an abstention — you have produced claims like "1981
misread as three hundred twenty-nine" about IPA that plainly read otherwise.

Verdicts: ipa_suspect · reader_diverged · notation_only · unsure · agree.
Priority: ipa_suspect > reader_diverged > unsure > notation_only > agree.
In `note`, state the specific token and what it should have been, under 16 words."""

ADJ_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "enum": ["agree", "ipa_suspect", "reader_diverged", "notation_only", "unsure"]},
        "note": {"type": "string"},
    },
    "required": ["verdict", "note"],
    "additionalProperties": False,
}


def post(body: dict, retries: int = 3):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.load(resp)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            print(f"    retry {attempt+1}: {type(e).__name__}", file=sys.stderr)
    return None


def screen(batch: list[dict]) -> dict[int, str]:
    lines = [f"[{b['id']}]T:{b['text']}\nI:{b['ipa']}\nH:{b['phones'].replace(' ', '')}" for b in batch]
    out = post({
        "messages": [{"role": "system", "content": SCREEN_SYSTEM},
                     {"role": "user", "content": f"lang={batch[0]['lang']}\n" + "\n".join(lines)}],
        "temperature": 0, "max_tokens": 24 * len(batch) + 80,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": SCREEN_SCHEMA}},
    })
    if not out:
        return {}
    try:
        # A quoted token means check; an empty string means ok. The label is DERIVED from whether the
        # model could produce evidence, which it cannot bluff as cheaply as it can pick a safe enum value.
        # Keep the quoted token: it is the evidence, and it is what lets a programmatic filter check
        # whether the model actually pointed at something real.
        # ⚠ A PLACEHOLDER IS NOT EVIDENCE. Asked for "" when nothing is wrong, the model returns `?`
        # instead — 62% of all `check` verdicts were a bare question mark, which the non-empty test счёл
        # evidence. Anything with no letter or digit in it is an abstention, not a quote.
        out2 = {}
        for v in json.loads(out).get("r", []):
            if "id" not in v:
                continue
            tok = (v.get("bad") or "").strip()
            out2[v["id"]] = tok if any(c.isalnum() for c in tok) else ""
        return out2
    except json.JSONDecodeError:
        return {}


def adjudicate(row: dict) -> dict:
    out = post({
        "messages": [{"role": "system", "content": ADJ_SYSTEM},
                     {"role": "user", "content": f"lang={row['lang']}\nT:{row['text']}\n"
                                                 f"I:{row['ipa']}\nH:{row['phones'].replace(' ', '')}"}],
        "temperature": 0, "max_tokens": 2200,
        # ⚠ THINKING ON — the whole point of stage 2. It needs the budget: with it off this model produced
        # the confident-nonsense verdicts that motivated the cascade.
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": ADJ_SCHEMA}},
    })
    if not out:
        return {"verdict": "no_reply", "note": ""}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"verdict": "no_reply", "note": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--probe", type=int, default=0)
    a = ap.parse_args()

    db = sqlite3.connect(a.db)
    cols = {r[1] for r in db.execute("PRAGMA table_info(utt)")}
    for c in ("judge_screen", "judge_verdict", "judge_note"):
        if c not in cols:
            db.execute(f"ALTER TABLE utt ADD COLUMN {c} TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS utt_screen ON utt(judge_screen)")
    db.commit()

    base = ["ipa IS NOT NULL", "phones IS NOT NULL", "phones != ''", f"COALESCE(status,'') NOT IN {SKIP}"]
    if a.stage == 1:
        base.append("judge_screen IS NULL")
    else:
        base += ["judge_screen = 'check'", "judge_verdict IS NULL"]
    args: list = []
    if a.langs:
        base.append(f"lang IN ({','.join('?' * len(a.langs))})")
        args += a.langs
    rows = [dict(id=r[0], lang=r[1], text=r[2] or "", ipa=r[3] or "", phones=r[4] or "")
            for r in db.execute(
                f"SELECT rowid,lang,text,ipa,phones FROM utt WHERE {' AND '.join(base)} ORDER BY lang,rowid",
                args)]
    left = len(rows)
    if a.probe:
        rows = rows[: a.probe]
    print(f"# stage {a.stage}: {left} eligible, this run {len(rows)}", file=sys.stderr)

    t0, done, counts = time.time(), 0, {}
    if a.stage == 1:
        batches = [rows[i : i + a.batch] for i in range(0, len(rows), a.batch)]
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for batch, got in zip(batches, pool.map(screen, batches)):
                for b in batch:
                    tok = got.get(b["id"], "?")  # a missing answer is a `check`, never an `ok`
                    v = "ok" if tok == "" else "check"
                    counts[v] = counts.get(v, 0) + 1
                    db.execute("UPDATE utt SET judge_screen=?, judge_note=? WHERE rowid=?",
                               (v, tok[:120], b["id"]))
                db.commit()
                done += len(batch)
        el = time.time() - t0
        print(f"\nstage 1: {done} rows in {el:.0f}s = {el/max(done,1):.3f}s/row  {counts}", file=sys.stderr)
        if done:
            chk = counts.get("check", 0)
            print(f"  check rate {100*chk/done:.1f}%  -> ETA stage 1 over {left} rows: "
                  f"{left*el/done/3600:.1f} h; stage 2 on ~{int(left*chk/done)} rows", file=sys.stderr)
    else:
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for row, res in zip(rows, pool.map(adjudicate, rows)):
                v = res.get("verdict", "no_reply")
                counts[v] = counts.get(v, 0) + 1
                db.execute("UPDATE utt SET judge_verdict=?, judge_note=? WHERE rowid=?",
                           (v, (res.get("note") or "")[:200], row["id"]))
                db.commit()
                done += 1
        el = time.time() - t0
        print(f"\nstage 2: {done} rows in {el:.0f}s = {el/max(done,1):.1f}s/row  {counts}", file=sys.stderr)
    db.close()


if __name__ == "__main__":
    main()
