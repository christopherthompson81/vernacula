# audio.cpp as a TTS backend

Vernacula already runs Kokoro-82M through ONNX Runtime. audio.cpp ships its own
Kokoro (`kokoro_tts`, a GGUF package with the voices baked in and eSpeak-ng
phonemization inside the engine). Can that be a second TTS engine beside the
ONNX one — same segmented-synthesis loop, same reader, same export — and what
does the ABI actually give a caller that has to produce word timings?

The ASR side of this question is a separate document,
`audiocpp_backend_investigation.md`. The interesting difference here is that
recognition gets measured word boundaries out of the ABI and synthesis, it turns
out, gets none.

## Run 1 — 2026-09-15 14:05 — what the ABI offers a TTS caller

**Question:** before writing an engine, what does the family declare, and does
`AudioCppResult` carry enough to drive `SegmentedSynthesis`?

Read `AudioCppRequest` / `AudioCppSession` / `AudioCppResult` in the bindings,
`AudioCpp.Server/SpeechRequest.ApplyTo` (the reference construction of a TTS
request) and the `kokoro_tts` case in `tests/AudioCpp.ModelTest`.

The request surface a TTS caller has: `SetText(text, language)`,
`SetVoiceId("af_heart")`, `SetSpeakingRate(x)`, `SetOption("seed", …)`. The
result surface: `Audio` (float PCM + sample rate), `Words`, `Segments`, `Text`.

**Finding:** the shape matches `TtsRequest` almost field for field — voice name,
speed, text — with no reference clip and no diffusion steps. So this is a
voice-list engine, the same capability set as the ONNX Kokoro.

**Implication:** a new `TtsEngine` subclass plus an `ITtsBackend` is the whole
job, if the word timings are there. `Words` exists on the result type, but it is
the same result type ASR uses, so its presence proves nothing. Measure it.

## Run 2 — 2026-09-15 14:10 — probe the real model

A throwaway console project in the scratchpad against the installed package,
`AUDIOCPP_NATIVE_DIR` pointed at the built engine:

```
/mnt/data/models/audiocpp/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf
```

**Question:** does it synthesize, at what sample rate, does it report word
timings, and which voices does the installed package actually render?

Self-description:

```
family=kokoro_tts
desc=Kokoro 82M multilingual text-to-speech with 54 preset voices, optimized
     native CPU inference, and shared eSpeak-ng phonemization.
tts/offline=True  tts/streaming=False
timestamps=False  speakerref=False  style=False
languages=en-us,en-gb,es,fr-fr,hi,it,ja,pt-br,zh
```

One sentence through `af_heart`:

```
voice=af_heart ms=2000 frames=95400 rate=24000 ch=1 sec=3.975 peak=0.3551
  words=0 segments=0 text=null
```

**Finding 1 — no word timings, and the family says so.** `SupportsTimestamps`
is false and `result.Words` comes back empty. This is the opposite of the ASR
result, where the ABI's measured word boundaries were the whole reason to prefer
it over the Cohere path. Synthesis has to estimate.

**Finding 2 — 24 kHz mono**, the same rate as the ONNX Kokoro, so the reader,
the WAV writer and the per-segment files need nothing new.

**Finding 3 — tts/streaming is false.** Offline only. That costs nothing here:
`SegmentedSynthesis` streams at paragraph granularity by calling the engine once
per segment, which is above the level the ABI's streaming mode works at.

**Finding 4 — the voice and the language are coupled, and a mismatch is a hard
error, not a silent fallback:**

```
voice=bm_george lang=en-us →
  Kokoro voice/language mismatch: voice bm_george requires lang_code=b
  but request resolved to a
voice=zz_nosuch →
  unknown Kokoro voice id: zz_nosuch
```

So the request's language is not a free choice beside the voice — it is a
function of the voice's first letter. That is the same rule Vernacula's ONNX
Kokoro already encodes as `IsBritish` (`bf_`/`bm_` → en-GB), generalised to nine
lang codes.

**Implication:** the engine must derive the language from the voice rather than
offer it as a separate control. `UsesLanguage` stays false and there is no way
for a user to produce the mismatch above.

## Run 3 — 2026-09-15 14:14 — which of the 54 voices this package renders

**Question:** the family advertises 54 preset voices across nine languages. The
binding cannot enumerate them, so the list has to be shipped as data — which
means knowing which ones actually work.

All 54 Kokoro v1.0 voice ids, each with the language its prefix implies:

```
ok=41 fail=13
```

41 render. The 13 failures are not scattered — they are two whole languages,
and both fail for a reason that names the package rather than the voice:

```
jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
  Kokoro UniDic resources are not bundled in this GGUF; re-export the model
  with --embed-multilingual-resources to use Japanese voices
zf_xiaobei … zm_yunyang
  Kokoro vocab is missing phoneme symbol: H
```

Everything else — 20 American English, 8 British English, 3 Spanish, 1 French,
4 Hindi, 2 Italian, 3 Brazilian Portuguese — produced audio at 24 kHz with a
plausible peak.

**Finding:** the renderable voice set is a property of the INSTALLED PACKAGE,
not of the family. A list of all 54 would offer thirteen voices that fail
mid-job, after the user has queued a document.

**Decision:** ship the 41 that this package renders, with the other two
languages named in a comment beside the table along with the re-export flag that
would enable Japanese. If a multilingual export is ever what the catalogue
installs, the fix is one edit to that table and it is written down where whoever
makes it will be standing.

**Negative result worth keeping:** I looked for a way to ask the model which
voices it has. There is none — `AudioCppModel` exposes `Languages` but no voice
enumeration, and the server's `/v1/audio/voices` answers from configured
`VoicePresets` plus `*.wav` stems on disk, i.e. from its own configuration
rather than from the model. So the table cannot be derived at run time, only
measured once and shipped.

## Run 4 — 2026-09-15 14:15 — speaking rate and a long paragraph

**Question:** does `SetSpeakingRate` do anything, and does a paragraph longer
than the family's 240-character `text_chunk_size` come back whole or truncated?

```
rate=0.7 sec=4.725
rate=1.0 sec=3.325
rate=1.5 sec=2.175
long chars=687 sec=40.900 ms=19644 words=0
```

**Finding:** speaking rate works and is monotonic, so `UsesSpeed` is true and
the dialog's existing speed control applies unchanged. A 687-character paragraph
came back as one 40.9 s buffer — the engine chunks internally on
`text_chunk_size` and joins, so the backend does not need a chunker of its own
the way the ONNX Kokoro does for its 512-token window.

**Also:** 19.6 s to render 40.9 s of audio on 4 CPU threads, about 2× realtime.
`ggml_cuda_init` reported the RTX 3090 during the run, so the engine in this
tree has CUDA registered and the backend list should be mapped from the app's
execution provider exactly as the ASR path does.

**Implication for the alignment gap (Run 2, Finding 1):** with no measured
timings, the words have to be spread over the segment's duration.
`OmniVoiceIpaAlignment.Proportional` already does this and already has a
no-trace path that weights by word length — it is deliberately separate from the
OmniVoice engine so it can be used without a model. Reuse it rather than write a
second estimator, and name the aligner in the sidecar `audiocpp_proportional` so
a reader of the sidecar can tell these timings are estimated.

## Run 5 — 2026-09-15 14:20 — a whole document through the real loop

**Question:** everything above was one `session.Run` at a time. Does the engine
actually drive `SegmentedSynthesis` — streaming, per-segment files, one sidecar
whose word timings are monotonic across paragraph boundaries rather than within
each one?

A scratch console against the built app, five-block markdown (heading, two
paragraphs, two list items), voice `af_heart`, backends `["cuda","cpu"]`:

```
[5 paragraphs] /5
  chunk 1/5 start=0.00s  samples=42600  words=3   'The Harbour Report.'
  chunk 2/5 start=1.77s  samples=126000 words=15  'The harbour was quiet this morning. Long'
  chunk 3/5 start=7.03s  samples=43200  words=3   'One line item.'
  chunk 4/5 start=8.82s  samples=87600  words=10  'Another line item, rather longer than th'
  chunk 5/5 start=12.47s samples=109200 words=13  'A closing paragraph, to check that the l'
wrote .../render.wav in 4.8s
aligner=audiocpp_proportional words=44
  'The' 0.000..0.313   'Harbour' 0.313..1.044   'Report.' 1.044..1.775
  'The' 1.775..2.003   'harbour' 2.003..2.536   'was' 2.536..2.764
  ... last: 'should.' 16.511..17.025
segment files: 5
wav bytes=1634458 (~17.03s at 24k float32)
monotonic check done
```

**Finding:** it works unmodified. Five segments in, five per-segment WAVs out,
44 words with no backwards step anywhere in the document — the second
paragraph's first word picks up at 1.775 s exactly where the heading's last word
ended, which is `SegmentedSynthesis` shifting each segment's local timings by the
audio before it, not anything this engine does.

The WAV is IEEE float, 1 channel, 24 kHz (`fmt` tag 0x0003, rate 0x5dc0), peak
0.408, RMS 0.047, 23% of samples near zero — a plausible speech envelope with
inter-word silence rather than a buffer of noise or a buffer of nothing.

**Also:** 17.03 s of audio in 4.8 s wall clock, against 2× realtime on 4 CPU
threads in Run 4. So the `["cuda","cpu"]` list did open on CUDA, which is the
first confirmation that the backend mapping matters for synthesis and not only
for recognition.

**Negative result:** the estimator is doing what it can and no better. "The"
(3 characters) gets 0.313 s and "Harbour" (7) gets 0.731 s in the same heading —
proportional to length, which is a reasonable prior and is not a measurement.
A word that is short to write and slow to say will drift. Nothing to do about it
from this side: the ABI reports no timings and the phonemization happens where
we cannot see it. It is why the aligner is NAMED in the sidecar.

## Run 6 — 2026-09-15 14:45 — it cannot say "button"

A real document failed in the reader, mid-job:

```
AudioCpp.AudioCppException: audiocpp_session_run:
  Kokoro vocab is missing phoneme symbol: ̩ (runtime error)
```

U+0329, COMBINING VERTICAL LINE BELOW — the syllabic-consonant mark.

**Question:** what input produces it, and how much of English does that cost?

First, what the symbol is attached to. Diffing `espeak-ng -q --ipa -v en-us`
against Kokoro's own vocab (`KokoroVocab.cs`, 114 entries) over the phonemizer's
200 English golden sentences found exactly one out-of-vocab codepoint, U+0329,
in one sentence — "Rustenburg" → `ɹˈʌsʔn̩bˌɜːɡ`. Confirmed against the engine:

```
FAIL 'Rustenburg'   Kokoro vocab is missing phoneme symbol: ̩
FAIL 'button'       Kokoro vocab is missing phoneme symbol: ̩
FAIL 'kitten'       Kokoro vocab is missing phoneme symbol: ̩
OK   'hidden'       OK 'Wittenberg'
```

**Finding: it cannot say "button".** eSpeak-ng glottalises /t/ before a syllabic
nasal, giving `bˈæʔn̩`, and the engine's own encoder then refuses the mark its own
G2P just produced.

Scale, over the first 40,000 words of `g2p-dict.tsv`:

```
words that cannot be spoken: 100 / 40000  (0.25%)
  U+0329 x99      U+026C 'ɬ' x1
```

0.25% understates it badly, because the failures are not scattered — they are
the `-tten` / `-tton` family, which is common: beaten, bitten, batten, begotten,
written, forgotten, gotten, kitten, button, cotton. And **one is enough to kill
the whole job**: the throw is per paragraph, so a document only has to contain
one such word anywhere. At the golden corpus's rate of one affected sentence in
200, a 300-sentence document fails with probability ~78%.

**Where it is.** `external/audio.cpp/src/models/kokoro_tts/frontend.cpp:149`:

```cpp
const auto it = assets.vocab.find(phonemes.substr(i, width));
if (it == assets.vocab.end()) {
    throw std::runtime_error("Kokoro vocab is missing phoneme symbol: " + ...);
}
```

**This is a divergence from the reference implementation, not a missing
feature.** misaki/KModel tokenizes with `filter(None, map(vocab.get, phonemes))`
— it DROPS phonemes it has no id for. Our own `KokoroVocab` doc comment says so
in as many words, because the ONNX path had to match it. Dropping U+0329 yields
`bˈæʔn`, which is a perfectly good reading of "button"; throwing yields no audio
at all. So the fix upstream is to skip rather than throw, and it is small.

## Run 7 — 2026-09-15 14:52 — can we just hand it our own phonemes?

**Question:** vernacula-phonemizer produces Kokoro-vocab-clean output BY
CONSTRUCTION — that is what `KokoroFormat.Render` is for, and it is why the ONNX
Kokoro never hits this. If the ABI would take phonemes instead of text, the bug
is bypassed entirely and the two engines would also agree on every reading.

Probed every plausible request option:

```
rejected 'phonemes':       unknown Kokoro TTS request option: phonemes
rejected 'phoneme_input':  unknown Kokoro TTS request option: phoneme_input
rejected 'input_phonemes': unknown Kokoro TTS request option: input_phonemes
rejected 'ipa':            unknown Kokoro TTS request option: ipa
rejected 'use_phonemes':   unknown Kokoro TTS request option: use_phonemes
rejected 'g2p':            unknown Kokoro TTS request option: g2p
```

**Finding: there is no phoneme input, and the door is bolted rather than merely
shut.** `frontend.cpp` calls `phonemize_text()` unconditionally, and
`session.cpp` runs `validate_spec_backed_request_options` first, so an
undeclared option is rejected before anything runs. The family's entire declared
request surface is `language`, `seed`, `text_chunk_size`.

**Negative result, recorded because it is tempting and should not be taken:**
their G2P *is* espeak-ng and honours its inline phoneme syntax —
`"hello [[b'Vtn]] world"` renders fine. So a caller could smuggle phonemes past
the G2P by rewriting failing words into Kirshenbaum ASCII inside `[[ ]]`. That
would mean maintaining an IPA→Kirshenbaum converter, against an undocumented
passthrough, to work around a bug whose real fix is four lines in a file we can
see. Not worth it. (Passing IPA as bare text does NOT work: `"bˈʌtn"` renders as
2.52 s of someone reading the characters aloud.)

**Implication:** nothing on Vernacula's side can prevent this failure. What it
CAN do is stop presenting it as a stack trace: the message names a combining
codepoint, which tells the reader nothing, when the useful answer is which word
in their document the engine refused.

## Run 8 — 2026-09-15 15:30 — two changes proposed in a local audio.cpp clone

Filed upstream as 0xShug0/audio.cpp#556. Both changes are on a local branch
`kokoro-drop-unknown-phonemes` in `external/AudioCpp-Bindings/external/audio.cpp`,
unpushed, for review before anything goes upstream.

### (1) Skip unknown phonemes instead of throwing — `frontend.cpp`

The minimal correctness fix, matching `KModel`'s
`filter(None, map(vocab.get, phonemes))`. Rebuilt the engine and re-ran every
case that failed in Run 6:

```
OK sec=1.00 'button'      OK sec=1.10 'kitten'       OK sec=1.18 'written'
OK sec=1.82 'Rustenburg'  OK sec=1.07 'hidden'       OK sec=1.45 'Wittenberg'
OK sec=3.12 'The button had been written over by then.'
```

"button" at 1.00 s beside "hidden" at 1.07 s — comparable, so the word is being
spoken, not truncated to nothing.

**Unexpected finding: the 54-voice sweep went from 41 to 49.** The eight Chinese
voices, which failed in Run 3 with "Kokoro vocab is missing phoneme symbol: H",
now render. That is the same fix reaching a different symbol.

⚠ **Do not read that as "Chinese now works."** Dropping a symbol is the right
call when it is a syllabic diacritic on a consonant that survives; whether it is
right for whatever `H` carries in the Chinese phoneme set is a question for
someone who can judge the output by ear. It matches the reference
implementation's behaviour either way, so this is not a reason to hold the fix —
but the voice table should not gain eight entries on the strength of "it no
longer throws".

The five Japanese voices still fail, and correctly so: "UniDic resources are not
bundled in this GGUF" is a genuinely missing resource, not a vocab gap, and the
fix does not paper over it.

### (2) A supplied phoneme stream — `phonemes` request option

Because the interesting question is not only "stop throwing" but "let a caller
with a better G2P use it". vernacula-phonemizer has a lexicon, heteronym
handling and normalization that eSpeak-ng does not, and `KokoroFormat.Render`
already emits Kokoro's own alphabet — the ONNX engine consumes exactly that
stream today.

Verified against the rebuilt engine, feeding our own reading:

```
declared request options: language, seed, phonemes, text_chunk_size

text      : The button had been written over by then.
our stream: ðə bˈʌTən hæd bɪn ɹˈɪTən Ovəɹ bI ðˈɛn.
  supplied phonemes -> 2.77s peak=0.348
  built-in G2P      -> 2.48s

cache check: same text, two streams -> 34800 vs 35400 samples, identical=False
overlong: Kokoro phoneme string exceeds 510 symbols; supplied phonemes are not
          chunked, so split them across requests
```

The cache check is the one that matters for correctness: the run cache keys on
text, so without adding the supplied stream to the key, two readings of the same
words would return the first one twice. The differing sample counts show the key
change working.

**Finding, and a real constraint on the proposal:** the declared option set is
embedded in the GGUF at conversion time, not read from `model_specs/` in a
normal build (`CMakeLists.txt` only compiles the catalogue in for
`AUDIOCPP_DEPLOYMENT_BUILD`). The probe above only sees `phonemes` because it
passes `ModelSpecOverride`. So even with this merged upstream, an already-
installed `kokoro_82m_q8_0` needs `--model-spec-override` or re-conversion
before it will accept the option — which is something Vernacula would have to
carry, since `ModelConfig.ModelSpecOverride` is per load.

**Not chunked, deliberately.** Chunking splits the TEXT; nothing in the engine
knows where the matching cut points in a caller's phoneme stream are — only
their G2P does. A supplied stream runs whole and the 510-symbol guard tells the
caller to split it. For Vernacula that is not new work: `ChunkForSynthesis`
already cuts paragraphs for the ONNX Kokoro's 512-token window.

### Regression check for (1)

Every distinct English sentence in the phonemizer's golden corpus, through the
patched engine on CUDA:

```
RESULT golden English sentences rendered: 114/114
```

114 distinct rows of the 200 (the file repeats sentences). Nothing that worked
before stopped working, and the one that did NOT work before — the "Rustenburg"
row from Run 6 — now renders. Combined with the voice sweep going 41 → 49, the
fix only ever turns a throw into audio.

## Run 9 — 2026-09-15 14:45 — closing the Japanese resource gap

Run 3 left five Japanese voices failing on "Kokoro UniDic resources are not
bundled in this GGUF". That is a fixable state, not a property of the family, so
it was worth fixing rather than documenting.

**First, a correction to Run 8.** I had said Japanese and Chinese both failed
for missing resources. Wrong: `tests/kokoro_tts/MULTILINGUAL_GGUF.md` says
release packages DO ship `g2p/zh.json`, and Chinese needs nothing else. Chinese
reached the encoder and died on a symbol Kokoro's vocab genuinely lacks — the
Run 8 fix, not a resource. Only Japanese was a real gap, and it needs two
separate things: UniDic INSIDE the GGUF and a MeCab library OUTSIDE it.

### MeCab without root

`g2p_multilingual.cpp` dlopens `libmecab.so.2`; the system has no such package
and `AUDIOCPP_MECAB_LIBRARY` takes an absolute path. The `mecab-python3` wheel
ships a real one (`mecab_python3.libs/libmecab-eada4a80.so.2.0.0`), extracted to
`/mnt/data/models/mecab/libmecab.so.2`. `ldd` resolves it against system
libraries only — no sudo, no build from source.

### Conversion

`.venv-kokoro-gguf` on /mnt/data with the versions the docs name as tested
(misaki 0.9.4, unidic 1.1.0, torch 2.14 CPU — conversion never infers), plus
`hexgrad/Kokoro-82M` (config + 327 MB weights + 54 voice packs) and UniDic 3.1.0
(`sys.dic` 243 MB, `matrix.bin` 481 MB).

```
tools/prepare_kokoro_gguf.py --source .../Kokoro-82M-source
  --output-dir /mnt/data/models/kokoro-multilingual
  --type q8_0 --embed-multilingual-resources --overwrite

{"bytes": 932614784, "tensors": {"F32":360,"Q8_0":111,"BF16":77},
 "resources": 438, "resource_bytes": 781530455}
```

⚠ **This package is not a stock conversion.** `--model-spec` defaults to
`model_specs/kokoro_tts.json`, which in this tree carries the Run 8 `phonemes`
option — so the build bakes it in, and the probe below confirms it
(`declared request options: language, seed, phonemes, text_chunk_size`) with no
`ModelSpecOverride`. That is the answer to the constraint Run 8 raised: a
converted package carries the option, an already-installed one cannot.

### Result

```
OK jf_alpha      ja  sec=2.92 peak=0.440     OK af_heart  en-us sec=2.48
OK jf_gongitsune ja  sec=4.50 peak=0.382     OK bm_george en-gb sec=2.88
OK jf_nezumi     ja  sec=2.42 peak=0.383     OK ef_dora   es    sec=2.10
OK jf_tebukuro   ja  sec=2.95 peak=0.441     OK ff_siwis  fr-fr sec=2.45
OK jm_kumo       ja  sec=3.05 peak=0.595     OK hf_alpha  hi    sec=2.65
                                             OK if_sara   it    sec=2.08
RESULT ok=13 fail=0                          OK pf_dora   pt-br sec=1.68
                                             OK zf_xiaoxiao zh  sec=2.95
```

All five Japanese voices render, and the other seven languages are unchanged, so
bundling disturbed nothing. Duration AND peak are both checked: a package that
loads and emits silence would otherwise pass as working. **54/54 voices are now
renderable** — 41 at Run 3, 49 after the Run 8 vocab fix, 54 here.

⚠ **Rendering is not the same as reading well.** This establishes that the
resource gap is closed and the MeCab path executes, which is exactly what the
error complained about. Whether the Japanese is GOOD Japanese is a judgement by
ear that these numbers cannot make — samples rendered to WAV for that.

### The cost, which is the reason not to just adopt this

**First load is expensive.** The package extracts all 781 MB of bundled
resources to a temp directory, then spends minutes pinned at 99.9% CPU on a
single core before the first request — against roughly two seconds for the
190 MB release package. For a desktop app that loads a backend per job, that is
not a detail.

**The temp directory is shared and outlives the process.** Resources land in
`/tmp/audiocpp-gguf` (994 MB here), reused across loads rather than being
per-process. The docs say bundled data is "removed when its assets are
released"; 23 stale `audiocpp-check-*` directories from earlier days, 182 MB,
were still on disk, so that cleanup does not always happen.

**Implication:** the multilingual package buys 5 voices for ~740 MB of download,
~1 GB of /tmp, and a first-load cost measured in minutes. Worth having proven
possible and recorded; NOT obviously worth making the package Vernacula
installs. And `ResolveKokoro` globs `kokoro-82m*.gguf` and picks ordinal-first
when it finds several, so two packages under one models root would make the
choice depend on the alphabet — a reason to keep this one in its own directory
until somebody decides deliberately.

## Run 10 — 2026-09-15 15:15 — measuring the ENGINE instead of a proxy, and two corrections

The Japanese work (Run 9) was entirely on the G2P-resources side — Japanese
never reached the vocab encoder, it died at `load_ja()`. It closed a gap worth
closing but it was never the diagnostic for "do other languages throw". This is.

**Method:** temporarily restore the throw in the patched engine, then run 40 real
golden sentences per language through the RELEASE package — i.e. ask what a
normal install actually did before the fix.

```
en-us   40 ok /  0 THREW
en-gb   40 ok /  0 THREW
es      35 ok /  5 THREW   missing phoneme symbol: «
fr-fr   34 ok /  6 THREW   missing phoneme symbol: «
hi      40 ok /  0 THREW
it      39 ok /  1 THREW   missing phoneme symbol: ̪ (U+032A)
pt-br   40 ok /  0 THREW
zh  real Chinese    OK 4.05s
zh  Latin in zh     THREW   missing phoneme symbol: H
```

### Correction 1 — Chinese was never broken. The test was.

Run 3 reported eight Chinese voices failing on `missing phoneme symbol: H`, and
Run 8 credited the vocab fix with repairing them. Both wrong. That sweep fed
`"Hello there."` to every voice, and the Chinese G2P passes unmapped Latin
through literally — the `H` was from MY TEST STRING. Real Chinese renders fine
before the fix and after, which the bundled `zh.json` corroborates: its entire
IPA table is in-vocab.

So **the 41 → 49 jump was an artifact of the probe, not a repair.** Japanese is
different and genuinely was broken — it throws at load regardless of text.

**The lesson is about the probe, not the engine.** A voice list verified with one
English sentence tells you the VOICE loads; it says nothing about whether the
LANGUAGE works. Every language should be probed in its own script.

### Correction 2 — the dominant trigger is `«`, not the syllabic mark

Spanish and French die on U+00AB GUILLEMET, which Kokoro's vocab lacks (it
carries `(` and `)` but not the guillemets). 5/40 and 6/40 — 12% and 15% of real
sentences, against English's 0/40 here and roughly 1-in-114 overall. The language
I originally diagnosed is the LEAST affected of the seven.

So the fix is more valuable than Run 6 claimed, and for a different reason: it
repairs ordinary Spanish, French and Italian prose, not just English `button`.

### Correction 3 — the espeak-ng CLI was a bad proxy

Run 6 diffed `espeak-ng -q --ipa` against the vocab and predicted French would
drown in hyphens (3.42%, 507 occurrences). **Zero hyphen throws in practice**,
and the diff entirely missed the guillemets that actually fail. audio.cpp drives
espeak through its own caret-tied IPA integration, so the CLI's output is not
what the engine encodes. Measure the engine.

### Regression gates, against the restored patched engine

| gate | result |
|---|---|
| C# model test (bindings) | ran=4 skipped=1 **failures=0** |
| `audiocpp_c_api_path` / `_exports` | Passed |
| `audiocpp_c_api_model` (C) | **Passed**, 146 s — once given `-DAUDIOCPP_C_API_MODEL_ROOT` |
| `audiocpp_c_api_parity` (C API vs CLI) | **PARITY OK across 5 families** |

```
kokoro_tts  audio  MATCH  cli=79800@24000Hz peak=0.3383  api=79800@24000Hz peak=0.3383
```

79800 frames at 24 kHz is 3.325 s — byte-identical to what the STOCK engine
produced for that sentence in Run 4, and identical between the CLI and the C
API. The fix is inert on text that never had an out-of-vocab symbol, which is
exactly the property it should have.

### ⚠ Every one of these suites is green by default while skipping its real half

Not caused by these changes, but it is why a vocab throw on `«` in Spanish and
French shipped:

- `run-tests.sh` aborted at a missing audio-capture binary and still surfaced
  as exit 0, with the path, model and cross-language stages never running.
- `ctest` reported "100% tests passed" with 2 of 4 skipped — the two that load
  models.
- `audiocpp_c_api_model` skips unless `AUDIOCPP_C_API_MODEL_ROOT` is set at
  configure time.
- `audiocpp_c_api_parity` skips unless `audiocpp_cli` exists, and
  `build-engine.sh` does not build it.
- The server's TTS/ASR/alignment routes skip unless `AUDIOCPP_TTS_MODEL` and
  friends are set.

"The tests pass" was largely a statement about the tests that do not touch a
model.

## Run 11 — 2026-09-15 19:55 — the backend against the merged engine

Upstream merged both engine fixes: #564 (`10f19390`, skip phonemes the vocab has
no id for) and #565 (`31d00b5c`, the English arm of the misaki port). Bumped
`external/AudioCpp-Bindings/external/audio.cpp` from `a4d6ea84` to `31d00b5c`
and rebuilt.

**Question:** does the backend synthesize a real document now, and is it saying
the words or merely returning a buffer?

The local #566 option-array patch was dropped from this checkout first. Nothing
in the Vernacula path calls it — `AudioCppSynthesisService` hands `Speak()` plain
text and lets audio.cpp phonemize — so this measures the merged engine alone.

First, the throw is gone from the binary outright:

```
strings libaudiocpp.so.0.1.0 | grep -c "missing phoneme symbol"   ->  0
```

Synthesis, over the words Run 6 established as fatal:

```
  ok   button      af_heart  1.38s   ok   Rustenburg  af_heart  1.57s
  ok   kitten      af_heart  1.38s   ok   written     af_heart  1.43s
  ok   forgotten   af_heart  1.50s   ok   cotton      af_heart  1.48s
  ok   beaten      af_heart  1.43s   ok   Llanelli    bf_alice  1.35s   (U+026C)
  ok   es ef_dora  ok fr ff_siwis  ok it if_sara  ok pt pf_dora  ok hi hf_alpha
pass=15 fail=0
```

### ⚠ "No exception and not silent" does NOT separate the two fixes

#564 alone drops U+0329 and returns a perfectly healthy buffer — `button` becomes
`bˈʌtn`, which is audio, and which is not the word. Only #565's
`syllabic_to_schwa` restores the `ᵊ`. Distinguishing them needs a readback, so
synthesize at 24 kHz, resample to 16 kHz, transcribe through Parakeet-TDT:

```
  ok   af_heart   heard: The button is here.
  ok   af_heart   heard: The kitten is here.
  ok   af_heart   heard: It was written down.
  ok   af_heart   heard: I had forgotten it.
  ok   af_heart   heard: The cotton is soft.
  ok   af_heart   heard: The dough was beaten.
  ok   af_heart   heard: The garden is hidden.          (control, no syllabic nasal)
  ok   bm_george  heard: The button was forgotten.
  ok   af_heart   heard: The kitten had forgotten the button on the cotton
                         coat and had written a note about it.
pass=9 fail=0
```

**Finding: the `-tten`/`-tton` family reads back intact in both dialects.** The
failure mode from Run 6 — where one such word anywhere in a document killed the
whole job, ~78% of a 300-sentence document — is closed.

ja/zh are not covered here and that is not an omission: this package ships 41
voices and none of them are Japanese or Mandarin. Those need the bundled
multilingual GGUF from Run 9.

**Still open:** #566 (list-valued request options) is unmerged, so the
`SetOptionArray` binding stays uncommitted in AudioCpp-Bindings rather than
shipping a `DllImport` for a symbol upstream does not export.
