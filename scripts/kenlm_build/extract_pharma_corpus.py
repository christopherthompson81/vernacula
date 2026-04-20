#!/usr/bin/env python3
"""
Build the pharma-domain training corpus for Parakeet shallow-fusion KenLM.

Source: um-ids/dailymed-annotations (HF, derived from NLM DailyMed SPL XML).

Each row is one FDA-approved drug label with its cleaned INDICATIONS AND USAGE
section — i.e. drug names in natural prescribing prose:

    "Amlodipine besylate tablets are indicated for the treatment of hypertension.
     It may be used alone or in combination with other antihypertensive agents."

~35 k labels, ~8 M words. This is exactly the corpus we want for biasing
the decoder toward correct drug-name transcription: names appear in
context (dose, indication, mechanism) rather than as a bare gazetteer.

Output: one sentence per line, cased + punctuated, ready to be concatenated
over the en-general base corpus with upweighting before `build_kenlm_parakeet.py`.
"""
import argparse
import gzip
import re
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("install: pip install datasets", file=sys.stderr)
    sys.exit(1)


SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for para in re.split(r"\n\s*\n|\n(?=[A-Z][a-z])", text):
        para = para.strip().replace("\n", " ")
        if not para:
            continue
        for sent in SENTENCE_END.split(para):
            sent = re.sub(r"\s+", " ", sent).strip()
            # Drop leading section numbers / bullets like "1.1" or "1 INDICATIONS"
            sent = re.sub(r"^(\d+(\.\d+)*\s+)+", "", sent)
            if len(sent) >= 10:
                out.append(sent)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output",   required=True, type=Path)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = full dataset (default).")
    args = ap.parse_args()

    open_ctx = gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) \
               if args.output.suffix == ".gz" else args.output.open("w", encoding="utf-8")

    ds = load_dataset("um-ids/dailymed-annotations", split="train", streaming=True)

    rows = 0
    sents = 0
    words = 0
    with open_ctx as fout:
        for r in ds:
            rows += 1
            txt = r.get("indication_cleaned") or r.get("text") or ""
            if not txt:
                continue
            for s in split_sentences(txt):
                fout.write(s)
                fout.write("\n")
                sents += 1
                words += s.count(" ") + 1
            if args.max_rows and rows >= args.max_rows:
                break
            if rows % 5000 == 0:
                print(f"  {rows:,} rows -> {sents:,} sentences, {words:,} words",
                      file=sys.stderr)

    print(f"\n[total] {rows:,} rows -> {sents:,} sentences, {words:,} words "
          f"-> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
