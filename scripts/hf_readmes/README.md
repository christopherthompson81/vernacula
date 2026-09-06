# HuggingFace model card sources

One folder per HuggingFace model repo under
[christopherthompson81](https://huggingface.co/christopherthompson81).
Each `<repo-name>/README.md` here is the source of truth for that HF
repo's model card; the file pushed to HF should match the file in this
directory.

## Repo map

| HF repo | Source folder | License | Status |
|---|---|---|---|
| [`sortformer_parakeet_onnx`](https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx) | [`sortformer_parakeet_onnx/`](sortformer_parakeet_onnx/) | mixed (CC-BY-4.0 / NVIDIA OML / MIT / Apache-2.0) | needs upload |
| [`diarizen_onnx`](https://huggingface.co/christopherthompson81/diarizen_onnx) | [`diarizen_onnx/`](diarizen_onnx/) | CC-BY-NC-4.0 | needs upload (current repo has no README) |
| [`cohere-transcribe-03-2026-onnx`](https://huggingface.co/christopherthompson81/cohere-transcribe-03-2026-onnx) | [`cohere-transcribe-03-2026-onnx/`](cohere-transcribe-03-2026-onnx/) | Apache-2.0 | needs upload |
| [`vibevoice-asr-onnx`](https://huggingface.co/christopherthompson81/vibevoice-asr-onnx) | [`vibevoice-asr-onnx/`](vibevoice-asr-onnx/) | MIT | needs upload |
| [`voxlingua107-lid-onnx`](https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx) | [`voxlingua107-lid-onnx/`](voxlingua107-lid-onnx/) | Apache-2.0 | needs upload |
| [`indicconformer-600m-onnx`](https://huggingface.co/christopherthompson81/indicconformer-600m-onnx) | [`indicconformer-600m-onnx/`](indicconformer-600m-onnx/) | MIT | needs upload |
| [`granite-speech-4-1-2b-onnx`](https://huggingface.co/christopherthompson81/granite-speech-4-1-2b-onnx) | [`granite-speech-4-1-2b-onnx/`](granite-speech-4-1-2b-onnx/) | Apache-2.0 | needs upload |
| [`granite-speech-4-1-2b-onnx-bf16`](https://huggingface.co/christopherthompson81/granite-speech-4-1-2b-onnx-bf16) | [`granite-speech-4-1-2b-onnx-bf16/`](granite-speech-4-1-2b-onnx-bf16/) | Apache-2.0 | needs upload |
| [`omnivoice-ipa-onnx`](https://huggingface.co/christopherthompson81/omnivoice-ipa-onnx) | [`omnivoice-ipa-onnx/`](omnivoice-ipa-onnx/) | per file: CC-BY-NC-4.0 (transformer, diff) / Boson Higgs Audio 2 Community (codec) / Apache-2.0 (tokenizer) | live |
| [`kokoro-82m-onnx`](https://huggingface.co/christopherthompson81/kokoro-82m-onnx) | [`kokoro-82m-onnx/`](kokoro-82m-onnx/) | Apache-2.0 | live — see [`scripts/kokoro_export/README.md`](../kokoro_export/README.md#publishing-to-huggingface) |
| [`chatterbox-tts-onnx`](https://huggingface.co/christopherthompson81/chatterbox-tts-onnx) | [`chatterbox-tts-onnx/`](chatterbox-tts-onnx/) | MIT | live — see [`scripts/chatterbox_export/README.md`](../chatterbox_export/README.md#publishing-to-huggingface) |

## What each card includes

- YAML frontmatter — `license`, `license_link` / `license_name` for `other`, `library_name: onnxruntime`, `pipeline_tag`, `tags`, `base_model`, `language`
- A header block with the upstream link and a one-line "for use with Vernacula" statement
- Contents table — every file in the bundle, what it does, which upstream component it derives from
- Export provenance — link to the `scripts/<name>_export/` folder on GitHub and a short paragraph on what the export does
- License section — explicit, with per-component breakdown for mixed-license bundles
- Using these files — minimal `huggingface_hub` + `onnxruntime` snippet for the non-Vernacula path
- Limitations — pointer to upstream model card with a note on what's specific to the ONNX repackaging
- Citation — BibTeX where I have it, link to upstream otherwise
- Acknowledgments — credits original authors, names the repackager
- See also — Vernacula repo, conversion script, upstream model card, the rest of the namespace

## Uploading

The canonical path is [`scripts/upload_to_hf.py`](../upload_to_hf.py),
which auto-resolves the README from this directory by repo basename:

```bash
# Sync the README only (no model artifacts)
python scripts/upload_to_hf.py \
    --model-dir /tmp/empty \
    --repo-id christopherthompson81/<repo-name> \
    --files /dev/null \
    --sync-readme

# Or full bundle + manifest + README
python scripts/make_manifest.py --model-dir ~/models/<bundle> --all
python scripts/upload_to_hf.py \
    --model-dir ~/models/<bundle> \
    --repo-id christopherthompson81/<repo-name> \
    --sync-readme --create-repo
```

For a quick README-only sync, the underlying `huggingface-cli` call also
works:

```bash
huggingface-cli upload christopherthompson81/<repo-name> \
    scripts/hf_readmes/<repo-name>/README.md README.md \
    --commit-message "sync model card from vernacula repo"
```

When updating a card, edit the file here, commit, and re-upload — never
hand-edit the README on the Hub directly, since changes there will be
silently overwritten on the next sync.
