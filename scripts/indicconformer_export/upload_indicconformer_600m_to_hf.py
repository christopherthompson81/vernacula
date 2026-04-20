#!/usr/bin/env python3
"""
upload_indicconformer_600m_to_hf.py

Publishes the repackaged AI4Bharat IndicConformer 600M ONNX package to
`christopherthompson81/indicconformer-600m-onnx` on the HF Hub so
Vernacula's ModelManagerService can download it at runtime.

What this uploads (from --package, default ~/models/indicconformer_600m_onnx/):
  - encoder-model.onnx + encoder-model.onnx.data (consolidated external data)
  - ctc_decoder-model.onnx
  - nemo128.onnx (DFT-conv1d 80-mel preprocessor, verified byte-identical
    to the upstream 600M preprocessor.ts config — see Phase 2 discovery)
  - vocab.txt, language_spans.json, config.json

Also generates and uploads:
  - manifest.json — {"files": {name: {"md5": …}, …}} shape; matches the
    other ModelManagerService manifests (ParseManifestHashes reads this).
  - README.md — model card with AI4Bharat attribution, MIT license,
    summary of the repackaging transformations so downstream users can
    reproduce what we did.

Usage:
  python scripts/indicconformer_export/upload_indicconformer_600m_to_hf.py \\
    [--package ~/models/indicconformer_600m_onnx] \\
    [--repo-id christopherthompson81/indicconformer-600m-onnx] \\
    [--private]   # default is public
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES_TO_UPLOAD = [
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "ctc_decoder-model.onnx",
    "nemo128.onnx",
    "vocab.txt",
    "language_spans.json",
    "config.json",
]

DEFAULT_REPO_ID = "christopherthompson81/indicconformer-600m-onnx"

MODEL_CARD = """---
license: mit
tags:
  - asr
  - speech-recognition
  - onnx
  - indic
language:
  - as
  - bn
  - brx
  - doi
  - gu
  - hi
  - kn
  - kok
  - ks
  - mai
  - ml
  - mni
  - mr
  - ne
  - or
  - pa
  - sa
  - sat
  - sd
  - ta
  - te
  - ur
base_model: ai4bharat/indic-conformer-600m-multilingual
library_name: onnxruntime
---

# IndicConformer 600M — ONNX (repackaged for Vernacula)

This repo republishes AI4Bharat's 22-language
[`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
as a single self-contained ONNX shipping package, in the shape the
[Vernacula](https://github.com/christopherthompson81/vernacula) desktop ASR
app expects its on-disk model directories to be. The CTC head only — the
RNNT components from the source repo are not shipped here.

All numerical behavior is identical to the upstream encoder+CTC graph; only
the on-disk packaging differs.

## Contents

| File | Purpose |
|---|---|
| `encoder-model.onnx` (+ `.data` sidecar) | Conformer encoder, `[features, features_lens] -> [encoded, encoded_lens]` |
| `ctc_decoder-model.onnx` | Single `Conv1d` → 5633-dim logits (22 × 256 language tokens + 1 shared CTC blank at id 5632) |
| `nemo128.onnx` | DFT-conv1d 80-mel preprocessor, `[waveforms, waveforms_lens] -> [features, features_lens]` |
| `vocab.txt` | Flat 5632-line vocab, id = line index; shared CTC blank is implicit at id 5632 |
| `language_spans.json` | 22 × `{start, length}` — which slice of `vocab.txt` each language's 256 tokens occupy |
| `config.json` | Preprocessor frontend params + CTC blank id |
| `manifest.json` | Per-file MD5 hashes (used by Vernacula's download verifier) |

## Transformations applied vs upstream

1. **Encoder ONNX**: consolidated the ~360 per-tensor external-data blob
   files (HF's xet layout in the upstream repo) into a single
   `encoder-model.onnx.data` sidecar so the file set is manageable. Also
   resolved external data from Constant-node attributes as well as graph
   initializers.
2. **Renamed ONNX IO tensors** so one C# backend loads either this 600M
   package or a NeMo-fork 120M export without branching:
   - Encoder: `audio_signal → features`, `length → features_lens`,
     `outputs → encoded`, `encoded_lengths → encoded_lens`.
   - CTC decoder: `encoder_output → encoded`, `logprobs → logits`.
3. **Vocab flatten**: upstream `vocab.json` is a 22-key dict with 257
   entries each (`[<unk>, t1..t256]`). Flattened to a single 5632-line
   `vocab.txt` keeping `<unk>` at local index 0 and the 255 real tokens
   at 1..255 per language. The 257th upstream slot is unused padding
   mirroring the RNNT head layout; it would never be decoded by the
   256-dim CTC softmax.
4. **Masks → spans**: upstream `language_masks.json` is 22 per-language
   length-5633 boolean arrays. Verified they resolve to contiguous
   256-token ranges, then compressed to `22 × {start, length}` entries.
5. **Preprocessor**: upstream ships TorchScript (`preprocessor.ts`). We
   replace it with a custom DFT-conv1d ONNX graph (no `STFT` op — ONNX
   Runtime's STFT diverges from PyTorch's on current toolchains). The
   frontend config is byte-identical to upstream: sample_rate 16 kHz,
   80 mel, n_fft 512, hop 160, win 400, hann, preemph 0.97, log+add
   guard, per-feature normalize, power spectrogram.
6. Parity verified end-to-end against upstream on a Hindi Fleurs clip —
   decodes to readable Devanagari with realistic WER; the full pipeline
   (nemo128 → encoder → ctc_decoder) is numerically equivalent to
   running AI4Bharat's reference model_onnx.py against the original
   assets/*.onnx files.

## Citation

Original model by AI4Bharat. Please cite their work when using this
repackaged copy; see their [model
card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
for details.

## License

MIT, same as upstream.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--package",
        default=str(Path.home() / "models" / "indicconformer_600m_onnx"),
        help="Directory holding the repackaged ONNX package.",
    )
    p.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help="Destination HF repo id (created if missing).",
    )
    p.add_argument(
        "--private", action="store_true",
        help="Create the repo as private (default is public).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Generate manifest.json + README.md locally; skip create/upload.",
    )
    return p.parse_args()


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    pkg = Path(args.package).expanduser()
    if not pkg.is_dir():
        raise SystemExit(f"--package dir not found: {pkg}")

    for f in FILES_TO_UPLOAD:
        if not (pkg / f).exists():
            raise SystemExit(f"missing required file: {pkg / f}")

    # Generate manifest.json with per-file MD5s. Written into the package
    # itself so it's uploaded alongside the other files.
    manifest = {
        "files": {f: {"md5": md5(pkg / f)} for f in FILES_TO_UPLOAD}
    }
    manifest_path = pkg / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")

    # Write the model card next to the package — it gets uploaded too.
    readme_path = pkg / "README.md"
    readme_path.write_text(MODEL_CARD)
    print(f"wrote {readme_path}")

    if args.dry_run:
        print("\n--dry-run: stopping before HF create/upload.")
        return

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    whoami = api.whoami()
    print(f"\nauthenticated as: {whoami.get('name', '?')}")

    print(f"creating repo {args.repo_id} (private={args.private}) …")
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # upload_folder with allow_patterns so we don't accidentally push
    # any stray files a user left in the package dir.
    allow = FILES_TO_UPLOAD + ["manifest.json", "README.md"]
    print(f"uploading {len(allow)} files from {pkg} → {args.repo_id} …")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(pkg),
        allow_patterns=allow,
        commit_message="Initial upload: repackaged IndicConformer 600M ONNX",
    )
    print(f"\nDone. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
