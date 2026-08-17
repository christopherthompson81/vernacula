"""Publish the IPA token corpus to a (private) HF dataset repo.

The corpus that trained the OmniVoice IPA fine-tune: FLEURS utterances re-transcribed to IPA (via a
portable espeak-ng-compatible phonemizer) and encoded to Higgs codec tokens. We publish the CODES
and the IPA/metadata, NOT the source audio — the audio is FLEURS' (regenerate it from there if you
need waveforms). 28 languages, ~77k utterances, ~267 h.

Per language, two files:
  codes_<lang>.npz       dict {utt_id: int16 (8, T)} — 8-codebook Higgs codec tokens (0..1023)
  manifest_<lang>.jsonl  one row/utt: {id, sentence_id, lang, ipa, gender, dur_s, n_frames}
`id` joins the two (npz key == manifest "id"); `sentence_id` is the FLEURS per-sentence id.

Run under an authenticated HF session (`hf auth login`).
Usage: publish_hf_dataset.py [--repo-id NAME] [--public]
"""
import argparse
from huggingface_hub import HfApi, whoami

TOKENS_DIR = "/mnt/data/omnivoice_ipa/corpus/tokens"

CARD = """---
license: cc-by-4.0
language:
  - am
  - ar
  - ca
  - zh
  - cs
  - cy
  - de
  - en
  - es
  - ff
  - fr
  - ga
  - ha
  - hi
  - ja
  - kk
  - ko
  - om
  - pt
  - ru
  - sd
  - sv
  - ta
  - th
  - tr
  - vi
  - xh
  - zu
tags:
  - text-to-speech
  - ipa
  - phonemes
  - codec-tokens
  - fleurs
task_categories:
  - text-to-speech
pretty_name: OmniVoice IPA token corpus
---

# OmniVoice IPA token corpus

The training corpus for the [OmniVoice IPA fine-tune](https://huggingface.co/christopherthompson81/omnivoice-ipa-onnx):
[FLEURS](https://huggingface.co/datasets/google/fleurs) utterances re-transcribed to **IPA** with
[vernacula-phonemizer](https://github.com/christopherthompson81/vernacula-phonemizer) and encoded to
**Higgs codec tokens**. The fine-tune teaches OmniVoice to accept IPA phoneme strings as text input,
so the phonemizer (not the model) owns G2P — including stress, pitch accent, and text normalization
(numbers, %, currency, and units are already spoken words in each language's own vocabulary).

**28 languages · ~77k utterances · ~267 h.** Codes + IPA/metadata only — **not** the source audio
(that is FLEURS'; regenerate waveforms from there via the Higgs decoder if you need them).

## Layout

Per language `<lang>` (e.g. `en_us`, `zu_za`):

| File | What |
|---|---|
| `codes_<lang>.npz` | `dict {utt_id: int16 (8, T)}` — 8-codebook Higgs codec tokens, values 0..1023, T frames (~25 Hz) |
| `manifest_<lang>.jsonl` | one row per utterance |

Manifest row:

```json
{"id": "10004088536354799741", "sentence_id": "903", "lang": "en_us",
 "ipa": "ə tɔːɹnˈeᶦd̬oᶷ ɪz ə spˈɪnɪŋ kʰˈɑːləm ...", "gender": "FEMALE",
 "dur_s": 6.8, "n_frames": 170}
```

`id` is the join key (npz key == manifest `id`). `sentence_id` is the FLEURS per-sentence id (shared
across speakers of the same sentence). `n_frames` is T (matches the code array's second dim).

### IPA notes

The IPA follows a **one-symbol-one-sound** discipline so the same glyph isn't overloaded across
languages: English offglides use superscript nuclei (`eᶦ`, `oᶷ`) distinct from syllabic vowels, and the
American-English intervocalic flap is `t̬` (voiced-t) rather than the tap `ɾ` used elsewhere. This is
what lets a single model render, e.g., Zulu clicks and English vowels from IPA alone.

Transcription is narrow where the language is: aspiration is marked (`kʰ tʰ pʰ`), dental stops are
`t̪ d̪`, geminates use `ː`, Japanese carries pitch-accent downsteps (`ꜜ`) and mora conventions
(`ɯᵝ`, `e̞ o̞`), and Sindhi has its full implosive series (`ɓ ɗ ʄ ɠ`). Non-lexical text (numbers,
percent, currency, units, dates) is normalized to each language's own spoken words BEFORE
phonemization, so the IPA contains no unspoken symbols.

## Loading

```python
import numpy as np, json
codes = np.load("codes_en_us.npz", allow_pickle=True)      # {id: (8, T) int16}
meta  = {json.loads(l)["id"]: json.loads(l) for l in open("manifest_en_us.jsonl")}
utt = "10004088536354799741"
c, m = codes[utt], meta[utt]      # c: (8, 170) int16 ; m["ipa"]: the IPA string
```

## Attribution & license

Derived from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0); this corpus is
released under the same license. Codec tokens are produced by the Higgs codec from
[k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (Apache-2.0).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    user = whoami()["name"]
    repo_id = a.repo_id or f"{user}/omnivoice-ipa-corpus"

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
    print(f"repo: https://huggingface.co/datasets/{repo_id}  (private={not a.public})")

    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    print("uploaded README.md")

    # upload the codes + manifests as-is (LFS handles the .npz automatically)
    api.upload_folder(folder_path=TOKENS_DIR, path_in_repo="data", repo_id=repo_id,
                      repo_type="dataset", allow_patterns=["codes_*.npz", "manifest_*.jsonl"])
    print(f"\ndone -> https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
