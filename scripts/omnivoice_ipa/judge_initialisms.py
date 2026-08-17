#!/usr/bin/env python3
"""
Classify the initialism-casing candidates with a LOCAL model, so the hand gate only reads what matters.

`scan_initialism_candidates.mts` produces ~2,164 candidate letter runs, of which the 46 with
cross-language spread >= 4 were reviewed by hand (see initialism_casing.mts). The remaining ~1,818 are
too many to read one by one and too risky to discard unread: a language-specific abbreviation is
legitimate and appears in exactly one corpus, so low spread is not proof of innocence.

Why a model can do this at all: the question is not phonetic, it is lexical — "is this token an
abbreviation read as letters, an ordinary word of this language, a unit, or an abbreviation that wants
a full word?" That is knowledge about the language, and the example sentence supplies the context that
settles most cases. The phonotactic predicate cannot: it flags Welsh `bwrdd` and Czech `smrt` as
unreadable because it is an ENGLISH test, and it misses `us`/`uk` because readability is not convention.

Output is CONSTRAINED to a JSON schema, so each verdict is a few tokens and a batch is one short
inference. Batches are graded, not serial: everything lands in a TSV for one bulk review pass.

⚠ THE MODEL DOES NOT DECIDE. It triages. Its LETTERS verdicts are a proposal that still gets read
before anything enters INITIALISM_UPPERCASE — this file only shrinks the reading, it does not replace it.

Usage:
  python3 judge_initialisms.py                       # all unreviewed candidates
  python3 judge_initialisms.py --bucket A            # no-vowel runs only (small, high yield)
  python3 judge_initialisms.py --batch 25 --limit 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = "/mnt/data/omnivoice_ipa"
GATE = f"{ROOT}/work/initialism_gate"
ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

# Already dispositioned by hand (initialism_casing.mts). Excluded from the sweep so the model spends no
# time re-deciding settled cases -- and so a disagreement cannot silently overwrite a human call.
REVIEWED = set("""utc gmt adt bce rspca usgs nsa npws nhc ptwc ndp pmo hjr afcfta plc pbs wned qvc vpn
pstn dslr gp xdr qc png hk cg kv tt mps km cm kg kph sq mbit zmapp jagr dzong angkor rossby bhog bhutha
bhajan lakkha rr rd isn didn wouldn wasn hadn doesn couldn mown jousts""".split())

VOWEL = re.compile(r"[aeiouy]")

# FLEURS code -> a language name the model will recognise. Only the 28 in the corpus.
LANG_NAME = {
    "am_et": "Amharic", "ar_eg": "Egyptian Arabic", "ca_es": "Catalan", "cmn_hans_cn": "Mandarin Chinese",
    "cs_cz": "Czech", "cy_gb": "Welsh", "de_de": "German", "en_us": "English", "es_419": "Latin American Spanish",
    "ff_sn": "Fula", "fr_fr": "French", "ga_ie": "Irish", "ha_ng": "Hausa", "hi_in": "Hindi",
    "ja_jp": "Japanese", "kk_kz": "Kazakh", "ko_kr": "Korean", "om_et": "Oromo", "pt_br": "Brazilian Portuguese",
    "ru_ru": "Russian", "sd_in": "Sindhi", "sv_se": "Swedish", "ta_in": "Tamil", "th_th": "Thai",
    "tr_tr": "Turkish", "vi_vn": "Vietnamese", "xh_za": "Xhosa", "zu_za": "Zulu",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "token": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["LETTERS", "WORD", "UNIT", "EXPAND", "UNSURE"]},
                    "expansion": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["token", "verdict", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

SYSTEM = """You classify short letter runs taken from speech-corpus transcripts that have been \
LOWERCASED, destroying the capitalisation that would mark an abbreviation. For each token decide how a \
native speaker READING THE SENTENCE ALOUD would say it.

Verdicts:
  LETTERS - an initialism/acronym said as a sequence of letter names (BBC -> "bee bee cee"). Also use \
this for alphanumeric codes and model designations said as letters (CG4684, KV62, Audi TT).
  WORD    - an ordinary word, name, or place of the language given. This includes words that merely \
look unpronounceable in English: Welsh bwrdd, Czech smrt, Irish bhfuil, Vietnamese khi are all WORDS.
  UNIT    - a unit of measurement said as its full unit name (km -> "kilometres", kg, mbit, sq mi).
  EXPAND  - an abbreviation said as a full word or phrase rather than letters (Rd -> "road", \
bzw. -> "beziehungsweise", St -> "saint", no. -> "number").
  UNSURE  - genuinely ambiguous from the evidence given.

Rules:
  - The example sentence is your main evidence. Use it.
  - An acronym pronounced as a WORD, not letters (NASA, UNESCO, AIDS), is WORD, not LETTERS.
  - Roman numerals (xv, xx, iii) are WORD - they are handled elsewhere, never letters.
  - Default to WORD when the token is plausibly native vocabulary of the stated language.
  - Be conservative: prefer UNSURE over a confident wrong LETTERS.
  - Keep "note" under 12 words. Give "expansion" only for UNIT and EXPAND."""


def load_unreviewed(bucket: str | None) -> list[dict]:
    out = []
    with open(f"{GATE}/candidates.tsv", encoding="utf8") as f:
        # ⚠ QUOTE_NONE. FLEURS transcripts contain `"`, and csv's DEFAULT quotechar made a quoted example
        # swallow every following line — 1,464 rows were silently read as 1,064, with no error anywhere.
        # A TSV has no quoting convention; asking the reader to honour one is the bug.
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            t = r["token"]
            if t in REVIEWED:
                continue
            b = "A" if not VOWEL.search(t) else "B"
            if bucket and b != bucket:
                continue
            r["bucket"] = b
            out.append(r)
    # No-vowel first (smaller and higher yield), then by corpus frequency: the most consequential first,
    # so a truncated run still covers what matters most.
    out.sort(key=lambda r: (r["bucket"], -int(r["count"])))
    return out


def ask(batch: list[dict], retries: int = 3) -> list[dict]:
    lines = []
    for r in batch:
        langs = ", ".join(LANG_NAME.get(l, l) for l in r["langs"].split(",")[:4])
        lines.append(
            f'token: {r["token"]}\n  language(s): {langs}\n  occurrences: {r["count"]}\n'
            f'  example: {r["example"][:130]}'
        )
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Classify every token below.\n\n" + "\n\n".join(lines)},
        ],
        "temperature": 0,
        "max_tokens": 90 * len(batch) + 200,
        # ⚠ THINKING OFF, and this is load-bearing rather than a preference. Qwen3 is a reasoning model:
        # left on, it spends the whole token budget in `reasoning_content`, returns `content: ""` with
        # finish_reason "length", and every row comes back NO_REPLY. Caught by testing one request — the
        # first sweep was launched without doing that and produced a file of nothing.
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
            print(f"    retry {attempt + 1}/{retries}: {type(e).__name__}: {e}", file=sys.stderr)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", choices=["A", "B"])
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=f"{GATE}/verdicts.tsv")
    a = ap.parse_args()

    cands = load_unreviewed(a.bucket)
    if a.limit:
        cands = cands[: a.limit]
    print(f"# {len(cands)} unreviewed candidates, batches of {a.batch}", file=sys.stderr)

    by_token = {r["token"]: r for r in cands}
    # Resume: never re-ask a token already on disk. Batches are cheap but not free, and a crash
    # mid-sweep should not restart from zero.
    done: dict[str, dict] = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf8") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                # NO_REPLY is not an answer, so it must not count as done — the model returns valid JSON
                # but occasionally omits a token from the array, and resume would otherwise cement that
                # omission forever. Dropping it here means a re-run re-asks exactly the missing ones.
                if r.get("verdict") != "NO_REPLY":
                    done[r["token"]] = r
        print(f"# resuming: {len(done)} already judged", file=sys.stderr)

    todo = [r for r in cands if r["token"] not in done]
    for i in range(0, len(todo), a.batch):
        batch = todo[i : i + a.batch]
        got = {v["token"]: v for v in ask(batch) if isinstance(v, dict) and "token" in v}
        for r in batch:
            v = got.get(r["token"])
            done[r["token"]] = {
                "token": r["token"],
                "bucket": r["bucket"],
                "count": r["count"],
                "n_langs": str(len(r["langs"].split(","))),
                "langs": r["langs"],
                "verdict": (v or {}).get("verdict", "NO_REPLY"),
                "expansion": (v or {}).get("expansion", ""),
                "note": (v or {}).get("note", "").replace("\t", " "),
                "as_lowercase": r["as_lowercase"],
                "as_uppercase": r["as_uppercase"],
                "example": r["example"][:110].replace("\t", " "),
            }
        # Flush every batch, so the sweep is resumable and reviewable while it runs.
        cols = ["token", "bucket", "count", "n_langs", "verdict", "expansion", "note",
                "as_lowercase", "as_uppercase", "langs", "example"]
        with open(a.out, "w", encoding="utf8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore",
                               quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writeheader()
            for t in sorted(done, key=lambda t: (-int(done[t]["count"]), t)):
                w.writerow(done[t])
        n = min(i + a.batch, len(todo))
        tally = {}
        for d in done.values():
            tally[d["verdict"]] = tally.get(d["verdict"], 0) + 1
        print(f"  {n}/{len(todo)}  {tally}", file=sys.stderr)

    print(f"\nwrote {a.out} ({len(done)} tokens)", file=sys.stderr)
    void = by_token  # keep the mapping referenced for future per-token lookups
    del void


if __name__ == "__main__":
    main()
