#!/usr/bin/env python3
"""
fetch_indic_test_audio.py

Pulls one short clip + ground-truth transcript per Indic language into
~/Programming/test_audio/<locale>/, in the layout the rest of the
test_audio library already uses (locale dir, <locale>_<source>_NN.wav,
<locale>_<source>_NN.md with a single [speaker_0] block).

Two sources:
  - Fleurs (google/fleurs, CC-BY-4.0, not gated) for 14 languages:
    hi, bn, ta, te, mr, gu, kn, ml, pa, or, as, ne, sd, ur
  - IndicVoices-R (ai4bharat/indicvoices_r, gated) for 8 languages:
    sa, ks, brx, doi, kok, mai, mni, sat
    — model was trained on IndicVoices (sibling dataset), so these 8
    are smoke-test material, not a rigorous WER benchmark.

Resamples to 16 kHz mono. Idempotent: skips a language if a matching
<source>_01.wav is already present.

Usage:
  # Fleurs 14 (no HF gate):
  python scripts/indicconformer_export/fetch_indic_test_audio.py --source fleurs

  # IndicVoices-R 8 (needs HF gate approval):
  python scripts/indicconformer_export/fetch_indic_test_audio.py --source indicvoices_r

  # Everything in one go:
  python scripts/indicconformer_export/fetch_indic_test_audio.py --source all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# BCP-47 locale → (HF config, transcript field, source tag).
# Keep these aligned with the existing EU locale dir naming style.

FLEURS_LANGS: dict[str, str] = {
    # BCP-47 locale : Fleurs config code
    "hi-IN": "hi_in",
    "bn-IN": "bn_in",
    "ta-IN": "ta_in",
    "te-IN": "te_in",
    "mr-IN": "mr_in",
    "gu-IN": "gu_in",
    "kn-IN": "kn_in",
    "ml-IN": "ml_in",
    "pa-IN": "pa_in",
    "or-IN": "or_in",
    "as-IN": "as_in",
    "ne-NP": "ne_np",
    "sd-IN": "sd_in",  # Fleurs has Indian Sindhi (not Pakistani)
    "ur-PK": "ur_pk",  # Fleurs has Pakistani Urdu only
}

# IndicVoices-R config names are listed on the dataset card once accessed;
# Sanskrit, Kashmiri, Bodo, Dogri, Konkani, Maithili, Manipuri, Santali.
# The exact config keys will be filled in when we actually load the dataset
# and inspect available configs — the webcard doesn't publish them.
IV_R_LANGS: dict[str, str] = {
    # BCP-47 locale : IndicVoices-R directory name (TitleCase)
    "sa-IN": "Sanskrit",
    "ks-IN": "Kashmiri",
    "brx-IN": "Bodo",
    "doi-IN": "Dogri",
    "kok-IN": "Konkani",
    "mai-IN": "Maithili",
    "mni-IN": "Manipuri",
    "sat-IN": "Santali",
}

# Max clip duration to keep things small; Fleurs clips are typically ~12s.
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
        "--source",
        choices=("fleurs", "indicvoices_r", "all"),
        default="fleurs",
        help="Which source to pull from.",
    )
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
        help="Comma-separated list of BCP-47 locales to restrict to, e.g. 'hi-IN,bn-IN'.",
    )
    return p.parse_args()


def format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_transcript(
    locale: str, source: str, duration: float, text: str, dst: Path
) -> None:
    """Write a .md in Vernacula.CLI's [speaker_N] [hh:mm:ss - hh:mm:ss] shape,
    prefixed with a header that distinguishes these ground-truth transcripts
    from leftover CLI output files."""
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
    dst.write_text(header + body)


def fetch_fleurs(
    out_root: Path,
    langs: dict[str, str],
    overwrite: bool,
) -> list[WrittenClip]:
    """Direct HF Hub download — avoids the `datasets 2.14.4`/local-cache bug.

    Fleurs layout:  data/<cfg>/audio/test.tar.gz  +  data/<cfg>/test.tsv
    TSV columns:    id, filename, raw_transcription, transcription, phonetic, num_samples, gender
    Audio is already 16 kHz mono .wav — no resampling needed.
    """
    target_min_samples = int(MIN_SECONDS * 16000)
    target_max_samples = int(MAX_SECONDS * 16000)

    written: list[WrittenClip] = []
    for locale, fleurs_cfg in langs.items():
        try:
            clip = _fetch_one_fleurs(out_root, locale, fleurs_cfg,
                                    overwrite, target_min_samples, target_max_samples)
            if clip is not None:
                written.append(clip)
        except Exception as exc:
            print(f"[{locale}] FAILED: {type(exc).__name__}: {exc}")
    return written


def _fetch_one_fleurs(
    out_root: Path, locale: str, fleurs_cfg: str, overwrite: bool,
    target_min_samples: int, target_max_samples: int,
) -> "WrittenClip | None":
    import csv
    import tarfile
    from huggingface_hub import hf_hub_download

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
            ns = int(num_samples)
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
        # Fleurs tars put files at either <name>.wav or test/<name>.wav;
        # walk the tar and match the filename tail.
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


def fetch_indicvoices_r(
    out_root: Path,
    langs: dict[str, str],
    overwrite: bool,
) -> list[WrittenClip]:
    """Direct HF Hub download — sidesteps the datasets/local-cache bug.

    IV-R layout:  <LanguageName>/test-0000N-of-000MM.parquet
    Audio lives inline in each row as `{"bytes": <wav bytes>, "path": str}`.
    Transcripts: prefer `verbatim`, fall back to `text` / `normalized`.

    `langs` values must be the TitleCase language dir name (e.g., "Sanskrit").
    """
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = api.list_repo_files("ai4bharat/indicvoices_r", repo_type="dataset")

    written: list[WrittenClip] = []
    for locale, dirname in langs.items():
        try:
            shards = sorted(
                f for f in all_files
                if f.startswith(dirname + "/") and "/test-" in f
            )
            if not shards:
                print(f"[{locale}] no IV-R test shards found for '{dirname}' — skipping")
                continue
            clip = _fetch_one_iv_r(out_root, locale, shards[0], overwrite)
            if clip is not None:
                written.append(clip)
        except Exception as exc:
            print(f"[{locale}] FAILED: {type(exc).__name__}: {exc}")
    return written


def _fetch_one_iv_r(
    out_root: Path, locale: str, shard_filename: str, overwrite: bool,
) -> "WrittenClip | None":
    import io as _io
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf
    import librosa

    dst_dir = out_root / locale
    dst_dir.mkdir(parents=True, exist_ok=True)
    wav_path = dst_dir / f"{locale}_indicvoices_r_01.wav"
    md_path = dst_dir / f"{locale}_indicvoices_r_01.md"

    if wav_path.exists() and md_path.exists() and not overwrite:
        print(f"[{locale}] already present — skipping")
        return None

    print(f"[{locale}] fetching {shard_filename} …")
    shard_path = hf_hub_download(
        repo_id="ai4bharat/indicvoices_r",
        repo_type="dataset",
        filename=shard_filename,
    )

    table = pq.read_table(shard_path)
    n = table.num_rows
    print(f"[{locale}] shard has {n} rows; scanning for [{MIN_SECONDS}, {MAX_SECONDS}]s clip")

    chosen = None
    for row in table.to_pylist():
        dur = float(row.get("duration") or 0.0)
        text = (row.get("verbatim") or row.get("text") or row.get("normalized") or "").strip()
        if MIN_SECONDS <= dur <= MAX_SECONDS and text:
            chosen = (row, dur, text)
            break
    if chosen is None:
        print(f"[{locale}] no clip in range found in first test shard — skipping")
        return None
    row, dur, text = chosen

    wav_bytes = row["audio"]["bytes"]
    arr, sr = sf.read(_io.BytesIO(wav_bytes), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        sr = 16000
    sf.write(str(wav_path), arr, sr, subtype="PCM_16")

    write_transcript(locale, "indicvoices_r", dur, text, md_path)
    preview = text[:60] + ("…" if len(text) > 60 else "")
    print(f"[{locale}] wrote {wav_path.name} ({dur:.1f}s) — {preview}")
    return WrittenClip(locale, "indicvoices_r", wav_path, md_path, dur, preview)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    only = None
    if args.only_langs:
        only = {s.strip() for s in args.only_langs.split(",") if s.strip()}

    written: list[WrittenClip] = []
    if args.source in ("fleurs", "all"):
        langs = FLEURS_LANGS if only is None else {
            k: v for k, v in FLEURS_LANGS.items() if k in only
        }
        if langs:
            written += fetch_fleurs(out_root, langs, args.overwrite)

    if args.source in ("indicvoices_r", "all"):
        langs = IV_R_LANGS if only is None else {
            k: v for k, v in IV_R_LANGS.items() if k in only
        }
        if langs:
            written += fetch_indicvoices_r(out_root, langs, args.overwrite)

    print(f"\nDone. {len(written)} clips written.")
    summary_path = out_root / "indic_fetch_summary.json"
    summary_path.write_text(json.dumps([
        {
            "locale": c.locale,
            "source": c.source,
            "duration_sec": c.duration_sec,
            "wav": str(c.wav_path),
            "md": str(c.md_path),
            "transcript_preview": c.transcript_preview,
        }
        for c in written
    ], indent=2, ensure_ascii=False))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
