#!/usr/bin/env python3
"""
Build a subword-level n-gram language model for Parakeet-TDT shallow fusion.

The resulting ARPA file has Parakeet subword *IDs* (integers) as the "words",
matching the format KenLmScorer.cs (src/Vernacula.Base/KenLmScorer.cs) expects.
Users pick one via `--lm` on the CLI or via the Language picker in Settings.

Inputs:
  --corpus <path>   Plain-text training corpus (one sentence per line).
  --tokenizer <p>   HuggingFace tokenizer.json for nvidia/parakeet-tdt-0.6b-v3.
                    (grab with: huggingface-cli download nvidia/parakeet-tdt-0.6b-v3 tokenizer.json)
  --order <N>       N-gram order (default: 4).
  --prune <spec>    lmplz prune spec, e.g. "0 0 1 1" (default drops 3- and 4-grams seen once).
  --output <path>   Output ARPA path. Gzip if it ends in .gz.

Requires:
  pip install tokenizers
  KenLM tools (lmplz, build_binary) on PATH.

Build KenLM tools from source:
  git clone https://github.com/kpu/kenlm
  cd kenlm && mkdir build && cd build && cmake .. && make -j$(nproc)
  export PATH="$PWD/bin:$PATH"
"""
import argparse
import gzip
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus",    required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--order",     type=int, default=4)
    ap.add_argument("--prune",     default="0 0 1 1",
                    help="lmplz prune spec matching --order, e.g. '0 0 1 1'")
    ap.add_argument("--output",    required=True, type=Path)
    args = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("error: `pip install tokenizers` first.", file=sys.stderr)
        return 1

    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tok.get_vocab_size()
    print(f"loaded tokenizer, vocab size = {vocab_size}", file=sys.stderr)
    if vocab_size > 16384:
        print(f"error: vocab size {vocab_size} exceeds the 14-bit packing limit "
              f"(16384) in KenLmScorer.", file=sys.stderr)
        return 1

    tokenized_corpus = args.corpus.with_suffix(args.corpus.suffix + ".tok")
    print(f"tokenizing {args.corpus} -> {tokenized_corpus}", file=sys.stderr)
    lines = 0
    subwords = 0
    with open_maybe_gz(args.corpus, "r") as f_in, tokenized_corpus.open("w") as f_out:
        batch = []
        for raw in f_in:
            raw = raw.strip()
            if not raw:
                continue
            batch.append(raw)
            if len(batch) >= 1024:
                write_tokenized(tok, batch, f_out)
                lines += len(batch)
                for enc in tok.encode_batch(batch):
                    subwords += len(enc.ids)
                batch.clear()
        if batch:
            write_tokenized(tok, batch, f_out)
            lines += len(batch)
            for enc in tok.encode_batch(batch):
                subwords += len(enc.ids)

    print(f"tokenized {lines:,} sentences, {subwords:,} subwords "
          f"({subwords / max(lines, 1):.1f}/sentence)", file=sys.stderr)

    arpa_tmp = args.output.with_suffix(".arpa.tmp")
    lmplz = run_lmplz(tokenized_corpus, arpa_tmp, args.order, args.prune, vocab_size)
    if lmplz != 0:
        return lmplz

    if str(args.output).endswith(".gz"):
        print(f"gzipping -> {args.output}", file=sys.stderr)
        with arpa_tmp.open("rb") as f_in, gzip.open(args.output, "wb", compresslevel=6) as f_out:
            while chunk := f_in.read(1 << 20):
                f_out.write(chunk)
        arpa_tmp.unlink()
    else:
        arpa_tmp.rename(args.output)

    size_mb = args.output.stat().st_size / (1 << 20)
    print(f"wrote {args.output} ({size_mb:.1f} MB)", file=sys.stderr)
    return 0


def open_maybe_gz(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def write_tokenized(tok, batch, f_out):
    for enc in tok.encode_batch(batch):
        ids = enc.ids
        # Skip empty encodings (lines with no normal tokens, e.g. all-special).
        if not ids:
            continue
        f_out.write(" ".join(str(i) for i in ids))
        f_out.write("\n")


def run_lmplz(tokenized: Path, out_arpa: Path, order: int, prune: str, vocab_size: int) -> int:
    cmd = [
        "lmplz",
        "-o", str(order),
        "--prune", *prune.split(),
        "--discount_fallback",
        "--vocab_estimate", str(vocab_size),
        "--text", str(tokenized),
        "--arpa", str(out_arpa),
    ]
    print("running:", " ".join(cmd), file=sys.stderr)
    try:
        p = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("error: `lmplz` not found on PATH. See header for build instructions.", file=sys.stderr)
        return 1
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
