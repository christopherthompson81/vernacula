#!/usr/bin/env python3
"""
Stream transcripts from permissive HF ASR datasets by reading ONLY the text
column of their Parquet shards (skipping audio bytes entirely), in parallel
across shards. Produces one plain-text corpus file, ready for
`build_kenlm_parakeet.py` downstream.

Sources (hard-coded for simplicity — extend if needed):
  - MLCommons/peoples_speech   "clean" subset (CC-BY-4.0, ungated)
                               Lowercase conversational text, no punctuation.
  - speechcolab/gigaspeech     "xl" subset (Apache 2.0, but GATED on HF —
                               must be logged-in with access granted).
                               Uppercase with <COMMA>/<PERIOD>/<QUESTIONMARK>/
                               <EXCLAMATIONPOINT>; we restore real punctuation.

Register target: conversational English with backchannels — the exact
register where Parakeet's multilingual drift (e.g. "uh uh" → "ajá") bites.

Output sentences are cased + punctuated to roughly match Parakeet's output
style. Case is restored crudely (sentence-initial only).
"""
import argparse
import gzip
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import fsspec
    import requests
except ImportError as e:
    print(f"missing dep: {e}. install: pip install pyarrow fsspec requests", file=sys.stderr)
    sys.exit(1)


def _hf_token() -> str | None:
    """Resolve the user's stored Hugging Face token. None if not logged in."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


_HF_TOKEN = _hf_token()
_HF_HEADERS = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}


GIGASPEECH_PUNCT = {
    "<COMMA>":            ",",
    "<PERIOD>":           ".",
    "<QUESTIONMARK>":     "?",
    "<EXCLAMATIONPOINT>": "!",
}


def normalize_gigaspeech(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    tokens = text.split()
    cleaned = []
    for tok in tokens:
        mapped = GIGASPEECH_PUNCT.get(tok)
        if mapped is not None:
            if cleaned:
                cleaned[-1] = cleaned[-1] + mapped
            continue
        if tok.startswith("<") and tok.endswith(">"):
            continue
        cleaned.append(tok.lower())
    if not cleaned:
        return None
    s = " ".join(cleaned)
    return s[0].upper() + s[1:]


def normalize_peoples_speech(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    return text[0].upper() + text[1:]


def list_parquet_shards(repo: str, subdir: str, name_prefix: str = "train-") -> list[str]:
    """
    Enumerate parquet files under repo/subdir matching name_prefix (default
    "train-" so test/validation shards are skipped) via the HF tree API.
    """
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{subdir}"
    paths: list[str] = []
    cursor = None
    while True:
        full = url + (f"?cursor={cursor}" if cursor else "")
        r = requests.get(full, timeout=60, headers=_HF_HEADERS)
        r.raise_for_status()
        items = r.json()
        if not items:
            break
        for it in items:
            path = it.get("path", "")
            base = path.rsplit("/", 1)[-1]
            if base.endswith(".parquet") and base.startswith(name_prefix):
                paths.append(path)
        link = r.headers.get("Link", "")
        if 'rel="next"' in link:
            m = re.search(r'cursor=([^>;]+)', link)
            if m:
                cursor = m.group(1)
                continue
        break
    return sorted(paths)


def resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def read_text_column(url: str, text_col: str = "text") -> list[str]:
    """Download one parquet, read ONLY the text column, return the strings."""
    # fsspec's http backend takes kwargs for client session; pass auth header
    # when available so gated repos (e.g. speechcolab/gigaspeech) work.
    client_kwargs = {"headers": _HF_HEADERS} if _HF_HEADERS else {}
    fs = fsspec.filesystem("http", client_kwargs=client_kwargs)
    with fs.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read(columns=[text_col])
    return [s.as_py() for s in table.column(text_col)]


def open_output(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=6)
    return path.open("w", encoding="utf-8")


def process_source(repo: str, subdir: str, normalize, max_words: int,
                   total_words_so_far: list[int], fout, fout_lock,
                   workers: int = 4, progress_every: int = 100_000):
    """Download parquet shards in parallel, write normalized lines sequentially."""
    print(f"[{repo}] enumerating shards under {subdir}/...", file=sys.stderr)
    paths = list_parquet_shards(repo, subdir)
    print(f"[{repo}] found {len(paths)} parquet shards", file=sys.stderr)
    if not paths:
        return 0, 0

    lines = 0
    words = 0
    t_start = time.time()

    # ThreadPoolExecutor downloads + reads shards in parallel. We submit in batches
    # and process results in submission order so output is deterministic-ish.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for p in paths:
            if total_words_so_far[0] + words >= max_words:
                break
            futures.append(pool.submit(read_text_column, resolve_url(repo, p)))

        for i, fut in enumerate(futures):
            try:
                texts = fut.result()
            except Exception as e:
                print(f"[{repo}] shard {i} error: {e}", file=sys.stderr)
                continue

            with fout_lock:
                for text in texts:
                    s = normalize(text) if text else None
                    if not s:
                        continue
                    fout.write(s)
                    fout.write("\n")
                    n = s.count(" ") + 1
                    words += n
                    lines += 1

            done_words = total_words_so_far[0] + words
            if lines // progress_every > (lines - len(texts)) // progress_every:
                elapsed = time.time() - t_start
                rate = words / max(elapsed, 0.1)
                print(f"  [{repo}] shard {i+1}/{len(paths)}: {lines:,} lines, "
                      f"{words:,} words @ {rate/1000:.1f}k words/s "
                      f"(total: {done_words:,})", file=sys.stderr)

            if done_words >= max_words:
                # Cancel outstanding futures so we stop quickly
                for f in futures[i+1:]:
                    f.cancel()
                break

    return lines, words


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output",   required=True, type=Path)
    ap.add_argument("--max-words", type=int, default=200_000_000,
                    help="Stop after this many words (default 200M)")
    ap.add_argument("--sources",  default="peoples",
                    help="Comma-separated subset of: peoples, gigaspeech "
                         "(gigaspeech is gated — requires HF auth)")
    ap.add_argument("--workers",  type=int, default=6,
                    help="Parallel parquet downloads per source (default 6)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    fout_lock = threading.Lock()
    total_words = [0]
    total_lines = 0

    config = {
        "peoples":    ("MLCommons/peoples_speech", "clean",            normalize_peoples_speech),
        # Use the "s" subset by default (~250h, 8 parquets) — big enough for case/
        # punctuation priors, small enough to pull over the wire quickly when column-
        # pruned. Bump to parquet-data/m or /xl if you need more volume.
        "gigaspeech": ("speechcolab/gigaspeech",   "parquet-data/s",   normalize_gigaspeech),
    }

    with open_output(args.output) as fout:
        for src in sources:
            if total_words[0] >= args.max_words:
                break
            if src not in config:
                print(f"[skip] unknown source: {src}", file=sys.stderr)
                continue
            repo, subdir, normalize = config[src]
            lines, words = process_source(
                repo, subdir, normalize,
                max_words=args.max_words,
                total_words_so_far=total_words,
                fout=fout, fout_lock=fout_lock,
                workers=args.workers)
            total_words[0] += words
            total_lines += lines
            print(f"[{src}] done: {lines:,} lines / {words:,} words "
                  f"(total so far: {total_words[0]:,})", file=sys.stderr)

    print(f"[total] {total_lines:,} lines / {total_words[0]:,} words -> {args.output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
