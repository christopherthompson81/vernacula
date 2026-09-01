"""Publish the OmniVoice base-ONNX + IPA fine-tune diff to a HF model repo.

Hosts OUR ONNX conversion of k2-fsa/OmniVoice (the base isn't published in ONNX) plus the 31 MB
IPA fine-tune diff that reconstructs the fine-tuned transformer at load (no 2.45 GB merged graph).
Run under an authenticated HF session (`hf auth login`).

⚠ THE DIFF VERSION IS AN ARGUMENT, NOT A CONSTANT. This was pinned to the v1-era `ipa_diff.onnx`
  and would have published it after v6 trained — the same stale-pin bug already fixed in
  extract_diff.py and apply_diff.py ("diff: version the IPA fine-tune patch"). --version selects
  WHICH LOCAL extraction is published; the repo path stays `ipa_diff.onnx`, and the card records
  which version that is plus its sha256.

⚠ THE BASE IS TAKEN FROM onnx_base/, NOT onnx/. Those two directories hold DIFFERENT 2.45 GB
  .onnx.data files under byte-identical graph protos. Only onnx_base/ matches the upstream
  checkpoint (verified against model.safetensors: max|d| = 0.0; the onnx/ copy differs in all
  151,676 embedding rows). The diff carries absolute replacement embed rows, so applying it to the
  other one yields a plausible-looking model that is quietly wrong, and nothing in the diff can
  detect that. BASE_SHA256 below is published in the card so downstream can check.

Usage: publish_hf.py [--repo-id NAME] [--version v6] [--public]
"""
import argparse
import hashlib
import os
from huggingface_hub import HfApi, whoami

ONNX_BASE = "/mnt/data/omnivoice_ipa/onnx_base"
ONNX = "/mnt/data/omnivoice_ipa/onnx"

# sha256 of onnx_base/omnivoice_transformer.onnx.data — the base that matches k2-fsa/OmniVoice.
BASE_SHA256 = "2ea0980e184bbf8457048fbb3ed2a01f8f8c3816a8ee9fbff3ce0886c1aeeb4a"


def files(version):
    """(local path, path-in-repo) for one diff version."""
    return [
        (f"{ONNX_BASE}/omnivoice_transformer.onnx", "omnivoice_transformer.onnx"),
        (f"{ONNX_BASE}/omnivoice_transformer.onnx.data", "omnivoice_transformer.onnx.data"),
        (f"{ONNX}/higgs_encoder.onnx", "higgs_encoder.onnx"),
        (f"{ONNX}/higgs_decoder.onnx", "higgs_decoder.onnx"),
        # ⚠ Versioned LOCALLY, single canonical name IN THE REPO. The local filename carries the
        # version so a stale extraction cannot masquerade as current on the build side; the repo is
        # a distribution point with one current answer, and each superseded diff was strictly worse.
        # Traceability lives in the card (which version, and the diff's sha256), not the filename.
        (f"{ONNX}/ipa_diff_{version}.onnx", "ipa_diff.onnx"),
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
| `ipa_diff.onnx` | 31 MB | the IPA fine-tune, as a reconstruction diff over the base transformer (currently the **{VERSION}** extraction, sha256 `{DIFF_SHA256}`) |

The transformer is the base (un-fine-tuned) graph; the encoder/decoder are the codec, unchanged by
the fine-tune. The IPA fine-tune is distributed as a **31 MB diff** rather than a second 2.45 GB
merged graph.

### ⚠ Verify the base before applying the diff

The diff holds a LoRA plus **absolute replacement** embedding rows, so it is only meaningful
against the exact base it was extracted from. Applied to any other weights it produces a
plausible-looking model that is quietly wrong — the perturbation is the same order as the fine-tune
itself — and **nothing in the diff can detect this**: the per-row delta distributions for a right
and a wrong base overlap. Check the fingerprint:

```
sha256sum omnivoice_transformer.onnx.data
# {BASE_SHA256}
```

## Applying the diff

The diff carries, per Linear (q/k/v/o/gate/up/down x 28 layers = 196 modules), the LoRA factors
`A (r=16, in)` and `B (out, r)` with `lora_scale = alpha/r = 2.0` in the model metadata, plus the
changed `embed_tokens` rows as `embed_rows` (fp16) and `embed_idx` (int32).

**Recommended — rewrite the graph, keeping the LoRA factored.** Leave the base weights alone and
add nodes:

```
MatMul(x, W) -> Y      becomes
MatMul(x, W) -> Y_base ;  MatMul(x, A^T) -> t ;  MatMul(t, scale*B^T) -> d ;  Add(Y_base, d) -> Y
```

with a sparse additive correction for the embedding (a compact `(changed+1, hidden)` delta table
with a zero row 0 and a `(vocab,)` int32 map, as two Gathers and an Add). Insert each chain at the
original node's position, not at the end of the graph, or the `Add` lands after its own consumers.
Because no weight is replaced, the runtime loads the base from the untouched `.onnx.data` and this
works on **every execution provider**. Measured against a full PyTorch merge: argmax agreement
100.000%, max|dlogit| 1.1e-4 (CPU) / 6.1e-5 (CUDA).

**Alternative — fold into the weights** (`W += ((B@A)*scale)^T`, overwrite the changed embed rows).
Also exact, but note that ⚠ **if you feed the folded weights back through
`SessionOptions.AddInitializer`, ONNX Runtime will silently ignore them on any non-CPU provider** —
a user-supplied initializer must be a CPU tensor, so when the session is planned on CUDA every one
is rejected ("Cannot use user supplied initializer ... planned memory location device is
different") and the base graph is served instead. The output is then bit-identical to using no diff
at all, which for IPA input is noise. If you fold, write the merged model out and load it from
disk.

Reference implementations: `apply_diff.py` (Python, fold) and `OmniVoiceDiff` (C#, graph rewrite)
in [vernacula](https://github.com/christopherthompson81/vernacula).

## Using it

Text -> IPA -> audio, with the phonemizer owning all G2P. The model is conditioned
**language-agnostically** (`<|lang_start|>None<|lang_end|>`); the IPA stream carries everything.

⚠ **Short input needs a reference voice.** The fine-tune corpus is FLEURS read prose — median 12 s,
only 0.21% under 3 s — so under ~5 s the model is out of distribution and can emit noise rather
than degrade gracefully. Supplying a reference clip for voice cloning fixes it, as does lengthening
the text. Generation is greedy (temperature 0) and therefore deterministic: re-running reproduces a
failure byte-for-byte.

## Attribution & license

- Base model: [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) — Apache-2.0. This
  repo redistributes an ONNX conversion of it under the same license.
- The fine-tune was trained on codec tokens derived from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0),
  transcribed to IPA with [vernacula-phonemizer](https://github.com/christopherthompson81/vernacula-phonemizer)
  (see the [token corpus dataset](https://huggingface.co/datasets/christopherthompson81/omnivoice-ipa-corpus)).
"""


def sha256_of(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--version", default="v6", help="ipa_diff_<version>.onnx")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    user = whoami()["name"]
    repo_id = a.repo_id or f"{user}/omnivoice-ipa-onnx"

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=not a.public, exist_ok=True)
    print(f"repo: https://huggingface.co/{repo_id}  (private={not a.public})")

    diff_local = dict((n, l) for l, n in files(a.version))["ipa_diff.onnx"]
    card = (CARD.replace("{VERSION}", a.version)
                .replace("{BASE_SHA256}", BASE_SHA256)
                .replace("{DIFF_SHA256}", sha256_of(diff_local)))
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")
    print("uploaded README.md")
    # Skip files already on the repo with identical content. The base + codec are 3.2 GB and
    # change ~never, while the diff changes every fine-tune; re-hashing and re-pushing the lot to
    # ship a 31 MB file is the slow way to a no-op.
    try:
        remote = {s.rfilename: (s.lfs.sha256 if s.lfs else None)
                  for s in api.repo_info(repo_id, repo_type="model", files_metadata=True).siblings}
    except Exception:
        remote = {}
    for local, name in files(a.version):
        mb = os.path.getsize(local) / 1e6
        if remote.get(name) and remote[name] == sha256_of(local):
            print(f"unchanged, skipping {name} ({mb:.0f} MB)")
            continue
        print(f"uploading {name} ({mb:.0f} MB) ...", flush=True)
        api.upload_file(path_or_fileobj=local, path_in_repo=name,
                        repo_id=repo_id, repo_type="model")
    print(f"\ndone -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
