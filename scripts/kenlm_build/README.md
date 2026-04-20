# Build a subword-level KenLM for Parakeet shallow fusion

## Workflow

1. `extract_hf_corpus.py` — stream general-English transcripts from permissive
   HF ASR datasets (column-pruned Parquet reads, so audio bytes never cross
   the wire).
2. `extract_medical_corpus.py` — stream medical text (MTSamples + HealthCareMagic
   + PubMed abstracts) from text-only HF datasets.
3. **Layer** the domain corpus over the general base by simple concatenation
   with an upweight (`for _ in 1 2 3; do cat medical.txt; done` on top of one
   copy of the general base). See *Design* below.
4. `build_kenlm_parakeet.py` — tokenise the layered corpus with Parakeet's
   tokenizer and drive `lmplz` to produce an ARPA keyed by subword IDs.
5. `evaluate.py` — formal content-preservation metrics against reference
   transcripts (WER / CER / ROUGE-L, scispaCy-tagged entity F1 for medical
   chemicals and diseases).
6. The resulting `.arpa` or `.arpa.gz` can be pointed at via CLI
   `--lm <path>` or in Settings → Speech Recognition → Language model.

## Design: layered LMs per language, not per domain

Shallow fusion applies the LM at every beam expansion regardless of whether
the audio is on-domain — there's no context switch inside the math. A
purely-domain LM penalises general English during the off-domain stretches
that always appear in real speech, so overall transcript quality suffers
*despite* better domain recall. The fix is to build each domain LM as a
superset of general English with domain upweighting, never as a replacement.

Empirically on PriMock57 medical audio with our `evaluate.py` metrics:

- Purely-medical LM: 56% chemical F1, worse WER + ROUGE-L than `en-general`.
- `en-general`:       60% chemical F1 (better than the purely-medical LM
                      because GigaSpeech has enough medical content in
                      spoken context).
- Layered `en-general base + 3× medical`: 60.6% chemical F1 *and* best
                      ROUGE-L of any mode tested.

So each domain file in the HF repo is structured: `en-general corpus ⊕ N ×
domain corpus`. Users pick one LM that matches their primary content type
and always get general fluency plus domain boost.



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

## Recipe: what we ship

The default `vernacula/kenlm-parakeet-en` LM is built from a 3:1 mix of:
- **GigaSpeech** (subset `s`, Apache 2.0) — 2.5M words, cased + punctuated
  (explicit `<COMMA>/<PERIOD>/<QUESTIONMARK>/<EXCLAMATIONPOINT>` tokens restored
  in place). Carries the case/punctuation priors that Parakeet's output expects.
- **MLCommons/peoples_speech** (subset `clean`, CC-BY-4.0) — 15M words,
  lowercase conversational speech. Carries the backchannel and disfluency
  priors ("uh huh", "yeah", "mm-hmm") that fix the multilingual-drift cases
  noted in issue #13.

GigaSpeech is duplicated 3× before concatenation — without this upweighting
the 6:1 raw size asymmetry lets the lowercase People's Speech style dominate
the LM's priors and decapitalize Parakeet's output.

Validation: a 600s en-US sample exhibits a beam=4 regression where English
"Uh uh. Ya." was decoded as Spanish "ajá, ya". With this mixed LM at
fusion weight 0.15, the line recovers to "Uh uh, yeah." while ~500 other
lines stay unchanged vs greedy.

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

`speechcolab/gigaspeech` (Apache 2.0 data, but gated on HF — accept the
terms on https://huggingface.co/datasets/speechcolab/gigaspeech and run
`huggingface-cli login` first) can be added with
`--sources peoples,gigaspeech`.

```bash
python3 extract_hf_corpus.py \
  --output    ~/corpora/gigaspeech-s.txt.gz \
  --sources   gigaspeech \
  --max-words 15000000
```

Then build the mixed corpus the ship LM uses:

```bash
# 3x GigaSpeech (upweight for case/punct priors) + 1x People's Speech
(for i in 1 2 3; do zcat ~/corpora/gigaspeech-s.txt.gz; done;
 zcat ~/corpora/peoples-speech-en.txt.gz) | gzip > ~/corpora/en-mixed.txt.gz
```

For a more general-English LM, an extra Wikipedia or Common Crawl slice
can be concatenated in yourself — the downstream build script accepts any
one-sentence-per-line text file. For lowercase-only corpora (e.g. older
research datasets), a LanguageTool pass is one way to recover casing and
sentence-final punctuation before training; not yet packaged here.

## Build the LM

General (base corpus):

```bash
python3 build_kenlm_parakeet.py \
  --corpus    ~/corpora/en-mixed.txt.gz \
  --tokenizer /tmp/parakeet-tok/tokenizer.json \
  --order     4 --prune "0 0 1 1" \
  --output    kenlm-parakeet-en.arpa.gz
```

Medical (layered over general):

```bash
# 1. Extract medical corpus (~22 M words across 3 sources)
python3 extract_medical_corpus.py --output ~/corpora/en-medical.txt.gz

# 2. Layer: general base + 3x medical
(zcat ~/corpora/en-mixed.txt.gz;
 for _ in 1 2 3; do zcat ~/corpora/en-medical.txt.gz; done) \
  | gzip > ~/corpora/en-medical-layered.txt.gz

# 3. Train. Order 3 is the sweet spot at 88M words — much smaller
#    file with negligible metric delta vs order 4.
python3 build_kenlm_parakeet.py \
  --corpus    ~/corpora/en-medical-layered.txt.gz \
  --tokenizer /tmp/parakeet-tok/tokenizer.json \
  --order     3 --prune "0 0 1" \
  --output    kenlm-parakeet-en-medical.arpa.gz
```

## Validate with formal metrics

```bash
pip install jiwer scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

python3 evaluate.py \
  --ref-dir /path/to/reference-transcripts \
  --hyp-dir /path/to/vernacula-cli-outputs \
  --modes greedy,general,medical
```

Reports per-file and micro-averaged: WER, CER, content-word WER, ROUGE-L,
and scispaCy entity F1 (all-entities / chemicals / diseases). This is the
validation harness the domain LMs are tuned against.

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
