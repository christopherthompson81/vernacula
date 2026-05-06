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

```bash
huggingface-cli upload christopherthompson81/<repo-name> \
    scripts/hf_readmes/<repo-name>/README.md README.md \
    --commit-message "sync model card from vernacula repo"
```

Or use the existing [`scripts/voxlingua107_export/upload_to_hf.py`](../voxlingua107_export/upload_to_hf.py)
pattern, pointing `--write-readme` at the file in this directory instead
of the inline string.

When updating a card, edit the file here, commit, and re-upload — never
hand-edit the README on the Hub directly, since changes there will be
silently overwritten on the next sync.
