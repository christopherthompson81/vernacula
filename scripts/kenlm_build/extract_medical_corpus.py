#!/usr/bin/env python3
"""
Build the `en-medical` training corpus for Parakeet shallow-fusion KenLM.

Sources (streamed from HF; no audio to skip, these are text-only):
  1. galileo-ai/medical_transcription_40  (MTSamples mirror, ~3M words)
     Clinical dictation style — SOAP notes, H&P, op reports. The single
     best-matched register for medical-dictation ASR.
  2. lighteval/med_dialog [healthcaremagic]  (~37.5M words available)
     Patient ↔ doctor Q&A dialogue. Carries conversational medical
     register — questions phrased by patients, answers phrased by clinicians.
  3. MedRAG/pubmed  (full PubMed Central title/abstract index)
     Formal biomedical prose. Bounded below ~5M words here so formal-
     register priors don't dominate the more ASR-relevant dictation /
     dialogue slices.

All three have precedent for public redistribution of derivative LMs in
the medical-NLP literature. The resulting LM is licensed CC-BY-4.0 with
attribution to each upstream corpus.

Output: one sentence per line, cased + punctuated to match Parakeet's
output style.
"""
import argparse
import gzip
import re
import sys
import time
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("install: pip install datasets", file=sys.stderr)
    sys.exit(1)


# MTSamples items are paragraph-concatenated ("HISTORY:,The patient,...").
# We split on the comma-after-section-header pattern AND on regular periods
# to recover sentence boundaries.
MTS_SECTION_SPLIT = re.compile(r"(?<=[:,])\s+(?=[A-Z])")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # First break paragraph blocks on two-plus newlines, then on sentences.
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip().replace("\n", " ")
        if not para:
            continue
        for sent in SENTENCE_END.split(para):
            sent = sent.strip()
            if sent:
                chunks.append(sent)
    return chunks


def clean(text: str) -> str:
    # Collapse runs of whitespace; strip stray leading commas from MTSamples
    # "SECTION:," fragments.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[,;:\s]+", "", text)
    return text


def normalize_mtsamples(row: dict) -> list[str]:
    text = row.get("text") or row.get("transcription") or ""
    return [clean(s) for s in split_sentences(text) if s]


def normalize_meddialog_hc(row: dict) -> list[str]:
    out = []
    # lighteval/med_dialog "healthcaremagic": src = patient, tgt = doctor
    for key in ("src", "tgt"):
        for s in split_sentences(row.get(key, "") or ""):
            out.append(clean(s))
    return [s for s in out if s]


def normalize_pubmed(row: dict) -> list[str]:
    # Concatenate title + abstract, then split.
    title   = (row.get("title") or "").strip()
    content = (row.get("content") or row.get("abstract") or "").strip()
    joined  = f"{title}. {content}" if title and content else (title or content)
    return [clean(s) for s in split_sentences(joined) if s]


SOURCES = [
    {
        "name":      "mtsamples",
        "hf_id":     "galileo-ai/medical_transcription_40",
        "config":    None,
        "text_cols": ["text"],
        "normalize": normalize_mtsamples,
        # MTSamples is small — take all of it.
        "max_words": 10_000_000,
    },
    {
        "name":      "healthcaremagic",
        "hf_id":     "lighteval/med_dialog",
        "config":    "healthcaremagic",
        "text_cols": ["src", "tgt"],
        "normalize": normalize_meddialog_hc,
        "max_words": 15_000_000,
    },
    {
        "name":      "pubmed",
        "hf_id":     "MedRAG/pubmed",
        "config":    None,
        "text_cols": ["title", "content"],
        "normalize": normalize_pubmed,
        "max_words": 5_000_000,
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path, help=".txt or .txt.gz")
    ap.add_argument("--sources", default=",".join(s["name"] for s in SOURCES),
                    help="Comma-separated source names to include.")
    args = ap.parse_args()

    wanted = set(s.strip() for s in args.sources.split(",") if s.strip())

    if args.output.suffix == ".gz":
        fout = gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6)
    else:
        fout = args.output.open("w", encoding="utf-8")

    grand_lines = 0
    grand_words = 0

    try:
        for src in SOURCES:
            if src["name"] not in wanted:
                continue

            print(f"[{src['name']}] streaming {src['hf_id']} "
                  f"({src.get('config') or 'default'})…", file=sys.stderr)
            try:
                kwargs = {"split": "train", "streaming": True}
                if src.get("config"):
                    ds = load_dataset(src["hf_id"], src["config"], **kwargs)
                else:
                    ds = load_dataset(src["hf_id"], **kwargs)
                ds = ds.select_columns(src["text_cols"])
            except Exception as e:
                print(f"[{src['name']}] ERROR opening: {e}", file=sys.stderr)
                continue

            src_lines = 0
            src_words = 0
            t_start = time.time()
            for row in ds:
                try:
                    sentences = src["normalize"](row)
                except Exception:
                    continue
                for s in sentences:
                    if not s:
                        continue
                    fout.write(s)
                    fout.write("\n")
                    n = s.count(" ") + 1
                    src_words += n
                    src_lines += 1
                    grand_words += n
                    grand_lines += 1
                if src_words >= src["max_words"]:
                    break
                if src_lines % 50_000 == 0 and src_lines > 0:
                    elapsed = time.time() - t_start
                    rate    = src_words / max(elapsed, 0.1)
                    print(f"  [{src['name']}] {src_lines:,} lines / "
                          f"{src_words:,} words @ {rate/1000:.1f}k w/s "
                          f"(grand {grand_words:,})", file=sys.stderr)
            print(f"[{src['name']}] done: {src_lines:,} lines / {src_words:,} words",
                  file=sys.stderr)
    finally:
        fout.close()

    print(f"[total] {grand_lines:,} lines / {grand_words:,} words -> {args.output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
