#!/usr/bin/env python3
"""Dump the Phase-1 capture (reference.npz) to a C#-readable form for the Stage-B
graph-parity test: one raw little-endian .bin per array + a manifest.json with shapes
and dtypes. Written under capture/csharp_fixtures/ (gitignored — regenerate locally).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# array name -> on-disk dtype (bool is written as 1 byte per element)
ARRAYS = {
    "enc_input_values": "<f4",
    "enc_audio_codes": "<i8",
    "tf_input_ids": "<i8",
    "tf_audio_mask": "|u1",
    "tf_attention_mask": "|u1",
    "tf_logits": "<f4",
    "dec_audio_codes": "<i8",
    "dec_audio_values": "<f4",
    "final_audio": "<f4",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", default=str(Path(__file__).resolve().parent / "capture/reference.npz"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "capture/csharp_fixtures"))
    args = p.parse_args()

    cap = np.load(args.capture)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"arrays": {}}
    for name, dt in ARRAYS.items():
        if name not in cap:
            continue
        arr = cap[name]
        np.ascontiguousarray(arr.astype(np.dtype(dt))).tofile(out / f"{name}.bin")
        manifest["arrays"][name] = {"shape": list(arr.shape), "dtype": dt}

    # Generation params from reference_meta.txt, with the language resolved to its id (as
    # _prepare_inference_inputs sees it) so the C# loop test can rebuild the exact cond.
    meta_path = Path(args.capture).parent / "reference_meta.txt"
    if meta_path.exists():
        meta = dict(
            line.split("=", 1)
            for line in meta_path.read_text().splitlines() if "=" in line
        )
        from omnivoice.models.omnivoice import _resolve_language
        manifest["params"] = {
            "text": meta.get("text", ""),
            "ref_text": meta.get("ref_text", ""),
            "language": meta.get("language", "None"),
            "lang_resolved": _resolve_language(meta.get("language")) or "",
            "num_step": int(meta.get("num_step", "16")),
            "guidance_scale": float(meta.get("guidance_scale", "2.0")),
        }

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(manifest['arrays'])} arrays -> {out}")
    for k, v in manifest["arrays"].items():
        print(f"  {k:20s} {v['dtype']} {v['shape']}")


if __name__ == "__main__":
    main()
