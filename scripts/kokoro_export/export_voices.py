#!/usr/bin/env python3
"""Export Kokoro voice packs to flat binary for the C# runtime.

Each voice is a [510, 1, 256] float32 tensor indexed by phoneme-string length:
the C# Kokoro path selects `ref_s = pack[len(phonemes) - 1]`. We dump each voice
to `<out>/voices/<name>.bin` as 510*256 little-endian float32 (the middle singleton
axis dropped), so C# can mmap/read and index without a tensor library.

By default exports all English voices (a*/b*); pass --all for every language.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

N_INDEX = 510   # voice-pack rows (= max phoneme-string length)
STYLE_DIM = 256


def list_voices(repo_id: str):
    files = HfApi().list_repo_files(repo_id)
    return sorted(f.split("/")[-1][:-3] for f in files
                  if f.startswith("voices/") and f.endswith(".pt"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Export Kokoro voices to flat binary")
    p.add_argument("--out", type=Path, default=Path("external/kokoro_onnx"))
    p.add_argument("--repo-id", default="hexgrad/Kokoro-82M")
    p.add_argument("--all", action="store_true", help="all languages (default: English a*/b* only)")
    args = p.parse_args(argv)

    voices = list_voices(args.repo_id)
    if not args.all:
        voices = [v for v in voices if v[0] in ("a", "b")]

    out_dir = args.out / "voices"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[voices] exporting {len(voices)} voices → {out_dir}")

    for name in voices:
        pt = hf_hub_download(repo_id=args.repo_id, filename=f"voices/{name}.pt")
        pack = torch.load(pt, weights_only=True)  # [510, 1, 256] float32
        assert pack.shape == (N_INDEX, 1, STYLE_DIM), f"{name}: unexpected shape {tuple(pack.shape)}"
        flat = pack.squeeze(1).contiguous().float().numpy().ravel()  # [510*256]
        dst = out_dir / f"{name}.bin"
        with open(dst, "wb") as f:
            f.write(struct.pack(f"<{flat.size}f", *flat.tolist()))
        print(f"[voices]   {name:12s} → {dst.name} ({dst.stat().st_size} B)")

    print(f"[voices] done. C# reads <name>.bin as {N_INDEX}×{STYLE_DIM} f32, ref_s = row[len(phonemes)-1].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
