#!/usr/bin/env python3
"""
fetch_whisper_test_audio.py

Pulls one FLEURS clip + ground-truth transcript per language into
~/Programming/test_audio/<locale>/, matching the layout the rest of the
test_audio library already uses (locale dir, <locale>_fleurs_01.wav,
<locale>_fleurs_01.md with a [speaker_0] block + metadata header).

Targets languages that aren't already covered by the existing test_audio
corpus and that exercise parts of Whisper's coverage Vernacula's other
backends don't reach:

  - Arabic (ar-EG)     — RTL script; no other backend supports it
  - Japanese (ja-JP)   — CJK; no other backend
  - Korean (ko-KR)     — CJK; no other backend
  - Mandarin (zh-CN)   — Qwen3 + Cohere also cover it
  - Swahili (sw-KE)    — Whisper-unique; widely spoken in East Africa
  - Hebrew (he-IL)     — Whisper-unique; RTL script

FLEURS clips are 16 kHz mono WAVs, ~10-15 s each, read speech with clean
ground-truth transcripts. Idempotent: skips a language if its
<locale>_fleurs_01.wav is already present.

Usage:
  python scripts/whisper_export/fetch_whisper_test_audio.py
  python scripts/whisper_export/fetch_whisper_test_audio.py --only-langs ar-EG,ja-JP
  python scripts/whisper_export/fetch_whisper_test_audio.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path


# BCP-47 locale : FLEURS config code.
# FLEURS uses pretty-verbose config names (hi_in, cmn_hans_cn, …). The mapping
# is stable; see https://huggingface.co/datasets/google/fleurs for the full list.
TARGET_LANGS: dict[str, str] = {
    "ar-EG": "ar_eg",
    "ja-JP": "ja_jp",
    "ko-KR": "ko_kr",
    "zh-CN": "cmn_hans_cn",
    "sw-KE": "sw_ke",
    "he-IL": "he_il",
}

MAX_SECONDS = 15.0
MIN_SECONDS = 5.0


@dataclass
class WrittenClip:
    locale: str
    source: str
    wav_path: Path
    md_path: Path
    duration_sec: float
    transcript_preview: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default=str(Path.home() / "Programming" / "test_audio"),
        help="Root of the test_audio library.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and replace existing clips.",
    )
    p.add_argument(
        "--only-langs",
        help="Comma-separated list of BCP-47 locales to restrict to, e.g. 'ar-EG,ja-JP'.",
    )
    return p.parse_args()


def format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_transcript(
    locale: str, source: str, duration: float, text: str, dst: Path,
) -> None:
    """Ground-truth transcript in Vernacula.CLI's [speaker_N] shape, with a
    metadata header so these files are distinguishable from leftover CLI
    output files."""
    header = (
        f"<!-- ground-truth transcript\n"
        f"source: {source}\n"
        f"locale: {locale}\n"
        f"duration_sec: {duration:.2f}\n"
        f"-->\n\n"
    )
    body = (
        f"## [speaker_0] [00:00:00 - {format_timestamp(duration)}]\n"
        f"\n"
        f"{text.strip()}\n"
    )
    dst.write_text(header + body, encoding="utf-8")


def _fetch_one(
    out_root: Path, locale: str, fleurs_cfg: str, overwrite: bool,
) -> "WrittenClip | None":
    """Direct HF Hub download — avoids the datasets local-cache bug some
    versions have. FLEURS layout is stable:
      data/<cfg>/audio/test.tar.gz
      data/<cfg>/test.tsv
    TSV columns: id, filename, raw_transcription, transcription, phonetic,
                 num_samples, gender
    Audio is already 16 kHz mono WAV — no resampling needed.
    """
    from huggingface_hub import hf_hub_download

    target_min_samples = int(MIN_SECONDS * 16000)
    target_max_samples = int(MAX_SECONDS * 16000)

    dst_dir = out_root / locale
    dst_dir.mkdir(parents=True, exist_ok=True)
    wav_path = dst_dir / f"{locale}_fleurs_01.wav"
    md_path = dst_dir / f"{locale}_fleurs_01.md"

    if wav_path.exists() and md_path.exists() and not overwrite:
        print(f"[{locale}] already present — skipping")
        return None

    print(f"[{locale}] fetching test.tsv …")
    tsv_path = hf_hub_download(
        repo_id="google/fleurs",
        repo_type="dataset",
        filename=f"data/{fleurs_cfg}/test.tsv",
    )

    chosen = None
    with open(tsv_path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            _id, filename, _raw, transcription, _phon, num_samples, *_ = row
            try:
                ns = int(num_samples)
            except ValueError:
                continue
            if (target_min_samples <= ns <= target_max_samples
                    and transcription.strip()):
                chosen = (filename, transcription, ns / 16000.0)
                break

    if chosen is None:
        print(f"[{locale}] no clip in [{MIN_SECONDS}, {MAX_SECONDS}]s — skipping")
        return None
    target_filename, text, dur = chosen

    print(f"[{locale}] fetching test.tar.gz (picking {target_filename}) …")
    tar_path = hf_hub_download(
        repo_id="google/fleurs",
        repo_type="dataset",
        filename=f"data/{fleurs_cfg}/audio/test.tar.gz",
    )

    with tarfile.open(tar_path, "r:gz") as tar:
        member = None
        for m in tar:
            if m.name.endswith("/" + target_filename) or m.name == target_filename:
                member = m
                break
        if member is None:
            raise RuntimeError(f"{target_filename} not in {tar_path}")
        src = tar.extractfile(member)
        if src is None:
            raise RuntimeError(f"Failed to open {target_filename} in tar")
        wav_path.write_bytes(src.read())

    write_transcript(locale, "fleurs", dur, text, md_path)
    preview = text[:60] + ("…" if len(text) > 60 else "")
    print(f"[{locale}] wrote {wav_path.name} ({dur:.1f}s) — {preview}")
    return WrittenClip(locale, "fleurs", wav_path, md_path, dur, preview)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    langs = dict(TARGET_LANGS)
    if args.only_langs:
        wanted = {s.strip() for s in args.only_langs.split(",") if s.strip()}
        langs = {k: v for k, v in langs.items() if k in wanted}
        if not langs:
            raise SystemExit(f"No matching locales in --only-langs={args.only_langs!r}")

    written: list[WrittenClip] = []
    for locale, cfg in langs.items():
        try:
            clip = _fetch_one(out_root, locale, cfg, args.overwrite)
            if clip is not None:
                written.append(clip)
        except Exception as exc:
            print(f"[{locale}] FAILED: {type(exc).__name__}: {exc}")

    if written:
        summary_path = out_root / "whisper_fetch_summary.json"
        # Append-merge rather than clobber so subsequent runs add to the log.
        existing = []
        if summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text())
            except Exception:
                existing = []
        if not isinstance(existing, list):
            existing = []
        new_entries = [{
            "locale": w.locale,
            "source": w.source,
            "duration_sec": w.duration_sec,
            "wav": str(w.wav_path),
            "md":  str(w.md_path),
            "transcript_preview": w.transcript_preview,
        } for w in written]
        # de-dupe by (locale, source) — new overwrites old.
        keyed = {(e["locale"], e["source"]): e for e in existing}
        for e in new_entries:
            keyed[(e["locale"], e["source"])] = e
        summary_path.write_text(
            json.dumps(sorted(keyed.values(), key=lambda e: e["locale"]), indent=2,
                       ensure_ascii=False))
        print(f"\nWrote {len(written)} clip(s); updated {summary_path}")
    else:
        print("\nNo new clips fetched.")


if __name__ == "__main__":
    main()
