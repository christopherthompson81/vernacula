# Build a subword-level KenLM for Parakeet shallow fusion

## Workflow

1. `extract_hf_corpus.py` — stream transcripts from permissive HF ASR datasets
   into a plain-text corpus file (column-pruned Parquet reads, so audio bytes
   never cross the wire).
2. `build_kenlm_parakeet.py` — tokenise the corpus with Parakeet's tokenizer
   and drive `lmplz` to produce an ARPA keyed by subword IDs.
3. The resulting `.arpa` or `.arpa.gz` can be pointed at via CLI
   `--lm <path>` or in Settings → Speech Recognition → Language model.



Tool output: an ARPA file whose tokens are **Parakeet subword IDs** (integers).
Matches the format that [`KenLmScorer`](../../src/Vernacula.Base/KenLmScorer.cs) expects.

## One-time setup

```bash
pip install tokenizers

# Build KenLM tools (lmplz + build_binary)
git clone https://github.com/kpu/kenlm /tmp/kenlm
cd /tmp/kenlm && mkdir -p build && cd build
cmake .. && make -j"$(nproc)"
export PATH="$PWD/bin:$PATH"

# Fetch the Parakeet tokenizer
huggingface-cli download nvidia/parakeet-tdt-0.6b-v3 tokenizer.json \
  --local-dir /tmp/parakeet-tok
```

## Fetch a corpus

For conversational English (best fit for casual/telephony audio), the
included extractor streams **MLCommons/peoples_speech** (CC-BY-4.0,
ungated) by column-pruning Parquet shards — audio bytes are never
downloaded.

```bash
pip install pyarrow fsspec requests
python3 extract_hf_corpus.py \
  --output    ~/corpora/peoples-speech-en.txt.gz \
  --sources   peoples \
  --max-words 15000000 \
  --workers   6
```

`speechcolab/gigaspeech` (Apache 2.0 data, but gated on HF) can be added
with `--sources peoples,gigaspeech` once you've accepted its terms and
authenticated via `huggingface-cli login`.

For a more general-English LM, an extra Wikipedia or Common Crawl slice
can be concatenated in yourself — the downstream build script accepts any
one-sentence-per-line text file.

## Build the LM

```bash
python3 build_kenlm_parakeet.py \
  --corpus    ~/corpora/peoples-speech-en.txt.gz \
  --tokenizer /tmp/parakeet-tok/tokenizer.json \
  --order     4 \
  --prune     "0 0 1 1" \
  --output    kenlm-parakeet-en.arpa.gz
```

**Size expectations (4-gram, prune "0 0 1 1"):**

| Corpus words | Uncompressed ARPA | Gzipped |
|---|---|---|
| 1M | 3 MB | 1 MB |
| 10M | 25 MB | 7 MB |
| 50M | 100 MB | 30 MB |
| 500M | 800 MB | 220 MB |

## Use

```bash
vernacula --audio sample.wav --model ~/.../parakeet \
  --lm kenlm-parakeet-en.arpa.gz \
  --lm-weight 0.3
```

Passing `--lm` auto-bumps the beam width to 4 (fusion has no effect in greedy
mode, since greedy commits to the argmax before the LM can influence anything).
