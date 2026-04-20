# Language Model Fusion (Parakeet)

Vernacula ships a pure-C# shallow KenLM fusion path for the Parakeet TDT decoder. Selecting a language model in Settings auto-enables beam search and biases decoding toward the chosen domain — fixing the occasional multilingual-drift artifact Parakeet shows on conversational English and preserving specialty vocabulary.

## Published catalogue

Currently published at [christopherthompson81/kenlm-parakeet](https://huggingface.co/christopherthompson81/kenlm-parakeet):

| LM | Size | Target |
|---|---|---|
| `en-general` | ~67 MB | Conversational English (GigaSpeech + People's Speech) |
| `en-medical` | ~17 MB | Medical English — clinical dictation + patient↔doctor dialogue + specialty drug names |

Each domain LM is built from speech-register sources only (spoken transcripts + synthetic dialogue + class-aware template-generated specialty vocabulary).

## Using fusion from the CLI

Pass an ARPA(.gz) file with `--lm` — this auto-bumps beam width to 4. Tune the bias with `--lm-weight` (typical range 0.1–0.5) and `--lm-length-penalty` (default 0.6, offsets the LM's shortening bias).

```bash
vernacula-cli --audio clinic-note.wav --model ~/models/vernacula \
  --lm ~/models/kenlm-parakeet/en-medical.arpa.gz \
  --lm-weight 0.15
```

Full argument documentation lives in [CLI reference → Parakeet decoding](../cli-reference.md#arguments).

## Building your own LM

Building your own is fully scripted under [`scripts/kenlm_build/`](../../scripts/kenlm_build/) — extract corpora from HuggingFace, tokenise with Parakeet's tokenizer, drive KenLM's `lmplz`, validate with the included scispaCy-based harness. See the [README there](../../scripts/kenlm_build/README.md) for the design notes on why each LM is layered the way it is.
