# Build a subword-level KenLM for Parakeet shallow fusion

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

## Build

```bash
python3 build_kenlm_parakeet.py \
  --corpus    /path/to/english_corpus.txt \
  --tokenizer /tmp/parakeet-tok/tokenizer.json \
  --order     4 \
  --prune     "0 0 1 1" \
  --output    kenlm-parakeet-en.arpa.gz
```

**Corpus sources that work well:**

- English: Wikipedia dump (e.g. `enwiki-latest-pages-articles`), OpenSubtitles, Common Crawl news slice.
- Spanish / French / etc.: same sources per language.
- Target ~50M words per language for a good general-purpose LM.
- One sentence per line; lowercase is fine but not required.

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
