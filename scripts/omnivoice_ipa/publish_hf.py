"""Publish the OmniVoice base-ONNX + IPA fine-tune diff to a (private) HF model repo.

Hosts OUR ONNX conversion of k2-fsa/OmniVoice (the base isn't published in ONNX) plus the 31 MB
IPA fine-tune diff that folds onto the base transformer at load (no 2.45 GB merged graph). Run
under an authenticated HF session (`hf auth login`).

Usage: publish_hf.py [--repo-id NAME] [--public]
"""
import argparse
import os
from huggingface_hub import HfApi, whoami

ONNX_BASE = "/mnt/data/omnivoice_ipa/onnx_base"
ONNX = "/mnt/data/omnivoice_ipa/onnx"

# (local path, path-in-repo)
FILES = [
    (f"{ONNX_BASE}/omnivoice_transformer.onnx", "omnivoice_transformer.onnx"),
    (f"{ONNX_BASE}/omnivoice_transformer.onnx.data", "omnivoice_transformer.onnx.data"),
    (f"{ONNX}/higgs_encoder.onnx", "higgs_encoder.onnx"),
    (f"{ONNX}/higgs_decoder.onnx", "higgs_decoder.onnx"),
    (f"{ONNX}/ipa_diff.onnx", "ipa_diff.onnx"),
]

CARD = """---
license: apache-2.0
base_model: k2-fsa/OmniVoice
tags:
  - text-to-speech
  - onnx
  - ipa
  - phonemes
library_name: onnx
---

# OmniVoice ONNX + IPA fine-tune diff

An **ONNX conversion** of [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) (a
non-autoregressive diffusion-LM TTS), plus a small **IPA fine-tune diff** that teaches the model to
accept IPA phoneme strings (from [vernacula-phonemizer](https://github.com/christopherthompson81/vernacula-phonemizer))
as text input — so the phonemizer, not the model, owns the linguistic G2P: dictionary + neural G2P,
stress, pitch accent, and text normalization (numbers, %, currency, units spoken in-language) all
happen before the model sees a token.

## Files

| File | Size | What |
|---|---|---|
| `omnivoice_transformer.onnx` (+`.onnx.data`) | 2.45 GB | **base** transformer (embeds + Qwen3-0.6B + audio heads), fp32 |
| `higgs_encoder.onnx` | 654 MB | Higgs codec encoder (24 kHz audio → codes) |
| `higgs_decoder.onnx` | 86 MB | Higgs codec decoder (codes → 24 kHz audio) |
| `ipa_diff.onnx` | 31 MB | the IPA fine-tune, as a reconstruction diff over the base transformer |

The transformer is the base (un-fine-tuned) graph; the encoder/decoder are the codec, unchanged by
the fine-tune. The IPA fine-tune is distributed as a **31 MB diff** rather than a second 2.45 GB
merged graph.

## Applying the diff (load-time fold)

The diff holds the LoRA low-rank factors (q/k/v/o/gate/up/down × 28 layers) and the sparse changed
`embed_tokens` rows, as ONNX initializers. To reconstruct the fine-tuned transformer, fold it onto
the base: for each Linear `W += ((B@A)·scale)ᵀ`, and overwrite the changed embed rows. This is
exact (100% argmax parity vs a full PyTorch merge) and takes ~2.5–5 s at load — no merged file
needed. Reference implementations: `apply_diff.py` (Python) and `OmniVoiceDiff` (C#, via
`SessionOptions.AddInitializer`) in the source repo.

## Attribution & license

- Base model: [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) — Apache-2.0. This
  repo redistributes an ONNX conversion of it under the same license.
- The fine-tune was trained on codec tokens derived from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0),
  transcribed to IPA with [vernacula-phonemizer](https://github.com/christopherthompson81/vernacula-phonemizer)
  (see the [token corpus dataset](https://huggingface.co/datasets/christopherthompson81/omnivoice-ipa-corpus)).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    user = whoami()["name"]
    repo_id = a.repo_id or f"{user}/omnivoice-ipa-onnx"

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=not a.public, exist_ok=True)
    print(f"repo: https://huggingface.co/{repo_id}  (private={not a.public})")

    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")
    print("uploaded README.md")
    for local, name in FILES:
        mb = os.path.getsize(local) / 1e6
        print(f"uploading {name} ({mb:.0f} MB) ...", flush=True)
        api.upload_file(path_or_fileobj=local, path_in_repo=name,
                        repo_id=repo_id, repo_type="model")
    print(f"\ndone -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
