# scripts/

Developer scripts for maintaining Vernacula localisation, help files, and ONNX model exports.

The locale and help scripts require the `anthropic` Python package and an Anthropic API key:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...   # or pass --api-key on each command
```

---

## update_locale_keys.py

Translates new or changed UI string keys from `en.json` into all 24 supported language JSON files.

**Before running**, edit the two constants near the top of the file:

- `NEW_KEYS` — the keys to add, in insertion order, with their English values
- `ANCHOR_KEY` — the existing key after which the new keys will be inserted

The script only retranslates keys whose English source has changed since the last run (SQLite cache at `src/Vernacula.Avalonia/Locales/locale_keys.db`). Interrupted runs resume automatically.

```bash
# Translate any new/changed keys and write them into the locale JSON files
python scripts/update_locale_keys.py

# Force retranslation of all keys in NEW_KEYS, ignoring the cache
python scripts/update_locale_keys.py --force

# Pass the API key directly instead of via environment variable
python scripts/update_locale_keys.py --api-key sk-ant-...
```

---

## update_help_files.py

Translates new or changed English help Markdown files (`Help/en/**/*.md`) into all 24 supported language directories (`Help/{lang}/`).

Change detection is based on SHA-256 of the English file contents (cache at `src/Vernacula.Avalonia/Help/help_files.db`). The prompt instructs the model to preserve YAML frontmatter structure, Markdown syntax, relative link hrefs, backtick UI labels, and technical terms — only human-readable prose is translated.

```bash
# Translate all changed files for all languages
python scripts/update_help_files.py

# Force retranslation of everything, ignoring the cache
python scripts/update_help_files.py --force

# Retranslate one specific file for all languages
python scripts/update_help_files.py --files operations/editing_transcripts.md

# Process multiple specific files (comma-separated, relative to Help/en/)
python scripts/update_help_files.py --files index.md,operations/editing_transcripts.md

# Target only specific languages
python scripts/update_help_files.py --lang de,fr

# Combine filters — one file, two languages
python scripts/update_help_files.py --files index.md --lang bg,cs
```

---

## nemo_export/

Scripts for exporting NeMo model checkpoints to ONNX. See [nemo_export/README.md](nemo_export/README.md) for full documentation.

- `export_parakeet_nemo_to_onnx.py` — exports a Parakeet RNNT/TDT `.nemo` checkpoint to the ONNX package used by Vernacula
- `export_sortformer_nemo_to_onnx.py` — exports a streaming Sortformer diarization `.nemo` checkpoint to ONNX
- `export_silero_vad_to_onnx.py` — exports Silero VAD to ONNX
- `setup_nemo_export_env.py` — creates the Python export venv
- `requirements.txt` — export dependencies

---

## deepfilternet3_export/

Exports DeepFilterNet3 as three streaming ONNX models with explicit GRU hidden-state I/O for chunk-by-chunk C# inference. See [deepfilternet3_export/README.md](deepfilternet3_export/README.md) for full documentation.

---

## diarizen_export/

Exports the DiariZen diarization pipeline: segmentation model, WeSpeaker embedding model, and LDA/PLDA transform parameters. See [diarizen_export/README.md](diarizen_export/README.md) for full documentation.

---

## cohere_export/

Exports `CohereLabs/cohere-transcribe-03-2026` to the ONNX package used by Vernacula. See [cohere_export/README.md](cohere_export/README.md) for full documentation.

- `export_cohere_transcribe_to_onnx.py` — exports the base Cohere encoder, mel frontend, config, tokenizer assets, and by default the KV-cache decoder pair
- `requirements.txt` — export dependencies
