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

## Design: speech-register per-domain corpus, no general base

Shallow fusion is a **style predictor**, not a knowledge predictor: the
LM biases the decoder toward sequences it has seen, uniformly at every
beam expansion, regardless of topic. Training-corpus register therefore
matters more than topic coverage. Written-prose medical text (PubMed
abstracts, patient forum Q&A, DailyMed prescribing prose) biases the
decoder *away* from natural speech patterns even when it has the right
vocabulary — so it hurts overall quality.

Early iterations layered domain corpora on top of an `en-general` base
to compensate for this. On the PriMock57 validation set we discovered
the base was helping purely because GigaSpeech + People's Speech are
spoken transcripts — the speech-register, not the general-English
coverage, was doing all the work. A domain LM built *only* from
spoken-register domain content ties or beats the layered version on
every metric at 3–4× smaller file size:

| LM | WER | Chem F1 | Disease F1 | Size |
|---|---|---|---|---|
| greedy                                | 14.0% | 50.0 | 83.3 | — |
| `en-general`                          | 16.1% | 60.0 | 83.3 | 67 MB |
| v2: gen + 3× written medical (layered) | 15.2% | 60.6 | 82.0 | 60 MB |
| v6: MTSamples 5× + synthetic 2× (NO base) | **15.0%** | 56.2 | **85.2** | **17 MB** |

So each domain file in the HF repo is now speech-register-only:

- Transcripts of actual speech (MTSamples clinical dictation), or
- Speech-register *synthetic* dialogue (`CodCodingCode/cleaned-clinical-conversations`).

Written prose is used only as a gazetteer source (see
`generate_drug_dialogue.py`), never as dominant training text.

## Template-based synthetic dialogue for specialty vocabulary

`generate_drug_dialogue.py` addresses a specific gap: our spoken-register
medical corpora don't mention specialty drug names often enough to give
the LM strong priors on them. Rather than adding back written DailyMed
prose (which biased the decoder toward formal register at a fluency
cost), the script fills clinical-speech templates like

    "I've been taking {drug} for my {cond}."
    "Patient is on {drug} {dose}, {drug2} {dose}."
    "Have you been taking the {drug} as prescribed?"

with fills from a gazetteer extracted from DailyMed via scispaCy. The
resulting corpus reads like real clinical speech but injects specialty
drug names into every utterance.

On PriMock57 the technique hasn't shown a measurable WER/F1 win yet,
because that corpus's drug mentions are dominated by common OTC drugs
the acoustic model already handles well. The approach is expected to
pay off on audio with specialty-drug density (oncology rounds,
psychiatry med reviews, pharmacy consults) which we don't yet have a
held-out benchmark for.



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

Medical (speech-register-only, no general base):

```bash
# 1. Pull MTSamples text (natural clinical dictation) directly
#    (one-shot; see scripts/kenlm_build/evaluate.py for the 'galileo-ai/
#    medical_transcription_40' schema).

# 2. Extract synthetic doctor↔patient dialogue (deduped turns across
#    the LLM-generated corpus; handles overlapping-context rows).
python3 extract_synthetic_dialogue.py \
  --output ~/corpora/en-medical-synthetic.txt.gz

# 3. Build speech-register corpus: 5x MTSamples + 2x synthetic.
#    No en-general base — MTSamples + synthetic already supply the
#    speech-register priors the base used to contribute.
(for _ in 1 2 3 4 5; do zcat ~/corpora/en-mtsamples.txt.gz; done;
 for _ in 1 2; do zcat ~/corpora/en-medical-synthetic.txt.gz; done) \
  | gzip > ~/corpora/en-medical-layered.txt.gz

# 4. Train. Order 3 is plenty at ~60M effective words; 4-gram adds
#    little at much larger file size.
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
