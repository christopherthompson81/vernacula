#!/usr/bin/env python3
"""
Extract speech-register synthetic doctor-patient dialogue for the
en-medical corpus layer.

Source: CodCodingCode/cleaned-clinical-conversations — a 269 k-row
LLM-generated dialogue corpus covering a broad range of conditions,
specialties, and symptom presentations. Each row holds a conversation
grown turn-by-turn, so adjacent rows overlap heavily; we dedupe at
the turn level to emit each unique utterance exactly once.

Register properties that match what `en-medical` was missing:
  * Actual dialogue form ("DOCTOR: ..." / "PATIENT: ...")
  * Broad specialty coverage (oncology, cardio, GI, neuro, psych, etc.)
  * Natural speech patterns — disfluencies like "um", "kind of", "you know"
  * Lay vocabulary ("trouble breathing", "my chest hurts") alongside the
    occasional clinical term

We strip the DOCTOR:/PATIENT: prefixes before emitting — the LM doesn't
need speaker tags, just the utterance text.
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


SPEAKER_RE = re.compile(r"^\s*(DOCTOR|PATIENT):\s*", re.IGNORECASE)


def split_turns(raw: str) -> list[str]:
    # Literal '\n' to real newline; then split on newlines.
    text = raw.replace("\\n", "\n")
    turns = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = SPEAKER_RE.sub("", line)
        if line:
            turns.append(line)
    return turns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-turns", type=int, default=0,
                    help="0 = extract all deduped turns (default).")
    args = ap.parse_args()

    open_ctx = gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) \
               if args.output.suffix == ".gz" else args.output.open("w", encoding="utf-8")

    ds = load_dataset("CodCodingCode/cleaned-clinical-conversations",
                      split="train", streaming=True).select_columns(["input"])

    seen: set[str] = set()
    emitted_turns = 0
    total_words  = 0
    rows = 0

    with open_ctx as fout:
        for r in ds:
            rows += 1
            for turn in split_turns(r["input"] or ""):
                # Dedupe on the full turn text — LLM-sourced conversations
                # overlap hugely so this drops ~95% of rows.
                if turn in seen:
                    continue
                seen.add(turn)
                fout.write(turn)
                fout.write("\n")
                emitted_turns += 1
                total_words += turn.count(" ") + 1
            if rows % 10_000 == 0:
                print(f"  {rows:,} rows -> {emitted_turns:,} unique turns, "
                      f"{total_words:,} words", file=sys.stderr)
            if args.max_turns and emitted_turns >= args.max_turns:
                break

    print(f"\n[total] {rows:,} rows -> {emitted_turns:,} unique turns, "
          f"{total_words:,} words -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
