# vernacula-tts — investigation log

Standing up `vernacula-tts`: the CLI that joins vernacula-phonemizer (text → canonical IPA) to the
IPA fine-tune of OmniVoice (IPA → speech, optionally voice-cloned). The two halves already existed
and were validated independently; this log covers what joining them exposed.

Related: `docs/omnivoice_onnx_investigation.md` (the export + the C# port),
`docs/omnivoice_ipa_corpus_investigation.md` (the corpus behind the fine-tune),
`scripts/omnivoice_ipa/` (training + diff extraction).

---

## Run 1 — 2026-09-01, wiring survey

**Question:** how much of `vernacula-tts` is new code rather than wiring?

**Finding — almost all wiring.** `Chatterbox.Base` already carries the whole OmniVoice pipeline
(`OmniVoiceTts` diffusion loop, the three ONNX graphs, `Qwen3Tokenizer`, `OmniVoiceTextPrep`,
`OmniVoiceDuration`, `OmniVoiceAudioPost`) and `OmniVoiceDiff`, which folds `ipa_diff_v6.onnx`
(31.6 MB) onto the 2.45 GB base transformer at load via `SessionOptions.AddInitializer` — no merged
file on disk. `chatterbox --backend omnivoice` already drives it, but exposes no `--diff`; only
`tests/OmniVoiceSmoke` did.

Three contract details that had to be read out of the training scripts rather than guessed:

1. **The model is conditioned language-agnostic.** `gen_accept_test.py:40` generates with
   `language=None`, so the style prefix is `<|lang_start|>None<|lang_end|>` and the IPA stream
   carries everything. `--lang` in the CLI therefore selects the PHONEMIZER, never the model.
2. **Duration is estimated on the IPA, not the orthography.** `OmniVoiceDuration` is a ratio
   (target script-weight ÷ reference script-weight × reference tokens), so it is only
   self-consistent when both sides are the same representation. `gen_accept_test.py:36-43` records
   the decision explicitly, including the choice NOT to force ground-truth duration ("the pacing it
   produces is its honest pacing"). Estimating on orthography would have been silently wrong,
   worst for scripts where IPA length diverges hardest from character count (Han, Kanji).
3. **v6 trained on 28 languages, not 66 or 102.** `data_config.json` (the v6 build;
   `data_config_v5.json.bak` beside it) lists 28 `language_id`s — the census-derived greedy cover
   from `sampling_budget.POP_ORDER`, not everything ingested. `build_webdataset.py:144` is emphatic
   that this is the coverage argument and not an `ls`.

**Implication:** the CLI is a thin front end, and the risk is concentrated in the two seams —
the phonemizer dependency and the diff fold. Both bit, below.

---

## Run 2 — 2026-09-01, the assembly-name collision

**Question:** can the new phonemizer simply be added as a submodule alongside the old one?

**Command:** `git submodule add …/vernacula-phonemizer.git external/vernacula-phonemizer`, then
`dotnet build src/Vernacula.Tts.CLI`.

**Finding — no, and it fails silently.** Both phonemizers build an assembly named
**`Vernacula.Phonemizer`** with that root namespace:

- `external/espeak-ng-portable/csharp/src/Vernacula.Phonemizer` — the legacy espeak-ng port, used
  ONLY by `Chatterbox.Base/KokoroTts.cs` for `KokoroFormat.Render`.
- `external/vernacula-phonemizer/csharp/Vernacula.Phonemizer` — the new canonical-IPA engine.

`Chatterbox.Base` referenced the legacy one transitively, so `Vernacula.Tts.CLI` inherited both.
The build **succeeded** and emitted exactly one `Vernacula.Phonemizer.dll` (3.2 MB — the new one
won the copy). No warning, no error. Had the copy gone the other way, `vernacula-tts` would have
thrown `TypeLoadException` at first phonemize; as it stands, `chatterbox`'s Kokoro backend was the
one at risk.

**Fix:** `PrivateAssets="all"` on Chatterbox.Base's reference to the legacy phonemizer, so it stops
flowing to consumers, plus a direct reference in the three projects that actually use Kokoro
(`Chatterbox.CLI`, `Chatterbox.Avalonia`, `tests/KokoroPerf`). Verified after a clean rebuild:
`vernacula-tts` output holds the 3.2 MB new engine, `chatterbox` output holds the 281 KB legacy
one. Full solution builds green.

**Negative result worth keeping:** an `extern alias` on the ProjectReference *does* resolve the
compile-time ambiguity (CS0433) and the build goes green — but it does nothing about two files
racing for one path in the output directory. It looked like the fix and was not one. Cutting the
dependency is.

---

## Run 3 — 2026-09-01, first end-to-end, and the diff evaporates on CUDA

**Question:** does the joined pipeline produce speech?

**Command:** `vernacula-tts --lang en --text "Hello world. This is the vernacula text to speech
pipeline." --ep cuda --print-ipa --verbose`

**Finding (a) — the phonemizer half is clean.** English, Welsh and Icelandic all phonemize
correctly, punctuation preserved as the fine-tune expects:

    en  həlˈoᶷ wˈɝɫd . ðɪs ɪz ðə vɚnˈækjələ tʰˈɛkst tʰuː spˈiːt͡ʃ pʰˈaᶦplaᶦn .
    cy  bˈɔrɛ dˈaː . krˈɔᶤsɔ ˈiː ɡˈəmrɨ .
    is  kˈouðan tˈajɪn .

**Finding (b) — the run died, but on an unrelated cause.** CUDA OOM at session init; `nvidia-smi`
showed `llama-server` holding 22 070 MiB of 24 576. Not our bug. Re-run after it was stopped.

**Finding (c) — THE REAL ONE. The load-time diff fold is silently ignored on CUDA.** Before the
OOM, the log carried **197** instances of:

    Cannot use user supplied initializer with name: (onnx::MatMul_9592) because the ORT planned
    memory location device Device:[DeviceType:1 …] is different from what is supplied
    (OrtMemoryInfo:[name:Cpu …])

`OmniVoiceDiff` supplies folded weights from CPU memory through `AddInitializer`. When ORT plans
the session on CUDA it rejects every one and falls back to the base graph's own initializers.

**Confirmation, not inference** — same text, four runs, comparing waveforms directly:

| pair | corr | max abs sample diff |
|---|---|---|
| `cuda + diff` vs `cuda --no-diff` | **+1.0000** | **0.0000** |
| `cpu + diff` vs `cuda + diff` | +0.0109 | 0.5139 |
| `cpu + diff` vs `cuda --no-diff` | +0.0109 | 0.5139 |

`--ep cuda` with the diff is **bit-identical** to running with no diff at all: stock orthographic
OmniVoice being fed IPA. The failure is inaudible as an error — it just sounds wrong.

This was never caught because the fold's only checks ran on CPU: `OmniVoiceSmoke --fold-selftest`
builds a plain `new SessionOptions()`, and the listen-confirmed clip in the ONNX investigation was
a CPU clip. The Python-side `apply_diff.py` parity check (argmax 100.000%) validates the *diff*,
not the *C# load path on a device*.

**Fix (this CLI):** refuse `--ep cuda` together with a diff, naming both ways out; and downgrade
`--ep auto` to CPU with a printed note rather than letting the fold evaporate.

**Not fixed (open):** `OmniVoiceDiff` itself still fails this way for any caller. `chatterbox
--backend omnivoice` has no `--diff` flag so it cannot currently hit it, and `OmniVoiceSmoke
--diff --ep cuda` would. A real fix means supplying device-side `OrtValue`s, or pre-merging.

---

## Run 4 — 2026-09-01, the CUDA path that does work

**Question:** is CPU (≈0.3× real-time) the only way to get IPA output?

**Finding — no.** `/mnt/data/omnivoice_ipa/onnx_base/omnivoice_transformer_ipa_v6.onnx` is the
PRE-MERGED v6 IPA transformer (1.5 MB graph + 2.45 GB `.onnx.data`). It needs no fold, so nothing
gets rejected, and `Path.Combine` already accepts an absolute `--transformer-file`:

    --no-diff --transformer-file /mnt/data/omnivoice_ipa/onnx_base/omnivoice_transformer_ipa_v6.onnx

**Measured (RTX 3090, 32 steps, ~4.4 s audio):**

| path | time | vs real-time | IPA weights in effect? |
|---|---|---|---|
| CPU + `ipa_diff_v6.onnx` fold | 14.8–18.2 s | 0.2–0.3× | yes |
| CUDA + `ipa_diff_v6.onnx` fold | 1.4 s | 3.1× | **NO — silently base** |
| CUDA + pre-merged IPA transformer | 1.5 s | 2.9× | yes |
| CUDA base (no IPA) | 1.5 s | 3.0× | n/a |

So the merged model on CUDA is ~10× faster than the folded model on CPU and carries the fine-tune.
Confirmed carrying it: `cuda+merged` vs `cuda+base` corr **+0.1506** (not the same rendering),
while `cuda+diff` vs `cuda+base` was +1.0000.

**⚠ Open, needs a listen.** `cpu+diff` vs `cuda+merged` correlate at **−0.0869** — two entirely
different token fields from the same input and nominally the same weights. The precedent says this
is expected and not decidable from here: the diffusion loop is chaotic (one flipped token diverges
the whole field), the ONNX investigation already recorded fp16 landing on "a different but valid,
good-sounding rendering", and commit ff8e19e records that the distance metric cannot adjudicate a
one-token change. **Which of the two is better is a listen test, not a measurement.**

---

## Run 5 — 2026-09-01, output level: a leading transient eats the headroom

**Question:** why is the IPA output so much quieter than the base model's?

**Measured**, same text, after the full post chain:

| output | peak | rms | crest |
|---|---|---|---|
| cpu + diff (IPA) | 0.098 | 0.0315 | 3.1 |
| cuda base | 0.485 | 0.2532 | 1.9 |

Peak 0.098 is impossible if `Normalize(audio, 0.5f)` ran — so where did it go?

**Method:** the post chain is remove-silence → normalize → fade-and-pad, and `FadeAndPad` ramps the
first and last 0.1 s linearly. Divide those edge regions back out by the known ramp and look at
what the peak was *before* the fade:

    cpu + diff    un-faded head peak = 0.500   tail = 0.019   middle = 0.098
    cuda base     un-faded head peak = 0.500   tail = 0.008   middle = 0.485

**Finding:** in BOTH cases the normalized 0.5 peak sits inside the leading 0.1 s. For the base
model the rest of the clip is nearly as loud (middle 0.485) so the fade costs nothing. For the IPA
model the leading sample is a **transient roughly 5× louder than any speech in the clip** — it
absorbs the whole normalization, the fade then removes it, and what is left is 5× too quiet.

This is a faithful port: Python's `_post_process_audio` normalizes before fading too, so the same
would happen upstream. Two candidate fixes if it matters — normalize after fading, or normalize on
a percentile rather than the peak — but both are deliberate deviations from Python parity, so
**not taken here**; recorded for the decision.

**⚠ CORRECTION (Run 6 listening).** The comparison above is against `cuda base`, and the listening
test established that the base output is *noise* — the orthographic model fed IPA. So its "healthy"
peak 0.485 / crest 1.9 is the profile of dense noise, not of good speech, and "the IPA output is 5x
too quiet" does not follow from it: the two clips are not comparable. What survives is the direct
measurement, which stands on its own — the normalized 0.5 peak of the IPA clip sits inside the
leading 0.1 s and the fade removes it, leaving the speech at 0.098. The user reported the clip
sounded fine, so this is a headroom observation, not an established defect.

---

## State

Working, on CPU with the folded diff and on CUDA with the pre-merged transformer: plain synthesis,
voice cloning (`--voice` + `--ref-text`, both sides phonemized, `add_punctuation` on the reference
per `create_voice_clone_prompt`), `--ipa` passthrough, markdown input, the off-corpus notice.

Open, in rough priority order:

1. **`OmniVoiceDiff` on CUDA** — fix the fold (device-side `OrtValue`s) or make the library itself
   refuse, rather than leaving the guard only in this CLI.
2. **The `cpu+diff` vs `cuda+merged` listen test** — Run 4. Decides which becomes the default path.
3. **The leading transient** — Run 5. A fine-tune artifact; worth checking whether it is in the
   training corpus's targets.
4. **Long-form chunking** — the OmniVoice backend still warns past ~1500 tokens (~60 s) and has no
   chunk-and-cross-fade path. `ParagraphChunker` / `ChunkedSynthesizer` exist for Chatterbox.


---

## Run 6 — 2026-09-01, the listening pass, and Welsh

**Method:** user listened to Run 3/4's outputs.

**Verdict:**

| clip | path | verdict |
|---|---|---|
| `re_cpu_diff` | CPU + folded diff, English | fine |
| `re_cuda_merged` | CUDA + pre-merged transformer, English | fine |
| `tts_clone` | CPU + diff, English, voice-cloned | fine |
| `re_cuda_nodiff` | CUDA base model fed IPA | **noise** |
| `tts_cy` | CPU + diff, Welsh, short | **noise** |

**Finding (a) — the CUDA diff bug is now doubly confirmed.** The base model fed IPA produces noise,
and Run 3 measured `cuda + diff` as bit-identical to it. So `--ep cuda` with the fold does not
merely lose the fine-tune, it produces noise. The guard is load-bearing, not cosmetic.

**Finding (b) — both IPA paths are good.** `cpu+diff` and `cuda+merged` both sound fine despite
correlating at −0.087. That settles Run 4's open question the way the precedent predicted: two
valid renderings, chaotic divergence in the diffusion loop, not one being broken. **The pre-merged
CUDA path is therefore the recommended default** — same quality, ~10x faster.

**Finding (c) — Welsh was noise, and the obvious explanations are dead.** Ruled out, in order:

1. **Not data volume.** cy_gb has 2,703 training utterances — MORE than en_us's 2,516, and
   mid-pack across the 28. (Also checked: every language's `repeat` is 1 in `data_config.json`,
   the oversampling being realized as physical shard copies rather than the JSON field.)
2. **Not unseen characters.** Every codepoint the C# emits for the two Welsh test sentences is
   present in the cy_gb training text, at healthy counts: `ɨ` 10,102, `ᶤ` 9,451, `ɬ` 2,615,
   `ɔ` 14,282. The corpus inventory is 50 distinct symbols and our output is a subset.
3. **Not IPA provenance.** `ipa_src` for cy_gb is 3,263 `fleurs_raw` — the same dominant source as
   en_us (2,595) and every other trained language. No Welsh-specific pipeline.
4. **Not a C# port divergence.** This was the strongest candidate — the C# Welsh engine landed
   recently (phonemizer commit cee945c3) while the corpus was phonemized by the TypeScript engine.
   Ran the phonemizer's own parity gate: `dotnet run --project csharp/tools/parity -- cy en es de`
   → **4 languages byte-identical, 0 differ, 800 rows**. The C# Welsh reproduces the TS Welsh
   exactly, so what we feed the model is what trained it.

**Remaining hypotheses, and the experiment that separates them.** The Welsh clip was also the
SHORTEST thing generated (1.8 s, `Bore da. Croeso i Gymru.`) while every clip that sounded fine was
2.4–4.4 s. So "Welsh is weak" and "short utterances are unstable" are both live, and the Welsh test
confounded them.

Generated for a second listening pass:

- `cy_incorpus_1/2` — Welsh IPA taken VERBATIM from `shards/cy_gb/dev.jsonl`, i.e. guaranteed
  in-distribution and long (12.4 s, 7.3 s). Isolates "does the model know Welsh at all" from
  "does it handle my short novel sentence".
- `sweep_cy_short` (1.8 s) vs `sweep_cy_long` (4.9 s) — length, within Welsh.
- `sweep_en_short` (1.5 s) — length, in the language known to work.
- `sweep_es` / `sweep_de` / `sweep_fr` (2.5–3.4 s) — other trained languages at comparable length,
  to test whether this is Welsh-specific at all.

All on the CUDA pre-merged path. Pending listening.


---

## Run 7 — 2026-09-01, the real cause: short input with no speaker anchor

**Second listening pass.** `cy_incorpus_1/2`, `sweep_cy_short`, `sweep_cy_long`, `sweep_en_short`,
`sweep_de` all fine; **`sweep_es` and `sweep_fr` noise/silence**. So Run 6's Welsh framing was
wrong twice over: Welsh is fine (both the in-corpus clips and, on the merged path, the very
sentence that failed on the CPU path), and the failure is not language-specific — it hit Spanish
and French, two of the highest-resource languages in the set.

**"Fluke" is ruled out mechanically.** Re-ran `es` auto with identical arguments: the output is
**bit-identical** (`cmp` clean). `GenConfig` uses temperature 0 throughout — greedy decode — so
generation is a deterministic function of the input. There is no draw to re-roll. Whatever fails,
fails every time for that text.

**A screening metric, and its limits.** Spectral flatness does not separate the labelled clips
(good `tts_clone` 0.154 vs bad `sweep_fr` 0.244). **Envelope modulation in the 2-8 Hz syllable
band does**, on the 13 human-labelled clips: every GOOD ≥ 0.329, every NOISE ≤ 0.293.
⚠ **It does not generalise to cloned audio.** Applied to a paired sweep it flagged `de_clone`
(0.291), `pt_clone` (0.307) and `fr_clone` (0.300) as failures; the user listened and **all ten
cloned clips were fine**. Three false positives in ten. The reference speaker changes the envelope
statistics the threshold was calibrated on. The intermediate conclusion drawn from those flags —
"cloning helps some languages and hurts others" — was an artifact of the proxy and is **retracted**.
The metric is usable as a screen for AUTO-mode output only — but within that scope it has now been
checked in BOTH directions by ear: it called the four short auto clips bad (correct) and the four
lengthened ones good (correct). So for auto-mode triage it is worth keeping; for cloned output it
is not, and no threshold tuning would fix that, since the failure is a calibration mismatch rather
than a noisy boundary.

**The two factors, measured.**

*Length.* Take the four languages that failed short and lengthen the text, nothing else changed.
Metric first, then confirmed by ear — the user listened and the long clips "also tended to work":

| lang | short auto | long auto | long, by ear |
|---|---|---|---|
| es | 0.265 (noise by ear) | 0.372 | works |
| fr | 0.271 (noise by ear) | 0.340 | works |
| ca | 0.235 | 0.417 | works |
| tr | 0.246 | 0.445 | works |

*Speaker anchor.* Take the same short texts and add a reference voice (cross-lingual — an English
reference clip, since the conditioning is acoustic): **all ten languages sound fine**, including
all four that produced noise in auto mode at the same length.

**Cause — it takes BOTH.** The v6 corpus is FLEURS read sentences:

    268,165 utterances   min 1.04s   p1 4.80s   p5 6.48s   median 12.00s   mean 12.78s   max 30.00s
    under 2s: 0.14%   under 3s: 0.21%   under 4s: 0.42%   under 5s: 1.17%

Every clip that failed was under 3.4 s — the bottom half-percent of the training distribution.
Every clip that recovered by lengthening was 7.5-12.4 s, at the median, and all were
listen-confirmed. Both remedies — lengthen, or supply a reference — are therefore established by
ear, not by proxy. But short alone is not
sufficient (`sweep_en_short` 1.5 s and `sweep_cy_short` 1.8 s were both fine), and a reference
voice rescues every short case. The reading that fits all of it: **without a reference the loop
must invent speaker and content together from an all-mask start, and on a span far shorter than
anything it trained on there is not enough context to converge — so it lands in a degenerate
region.** A reference clip supplies the speaker half and the problem goes away.

This also explains Run 6's Welsh result without any Welsh-specific cause: that clip was simply the
shortest thing generated (1.8 s), in auto mode, on the CPU path.

**Fix shipped:** the CLI warns when the duration estimate is under ~5 s AND there is no reference
voice, naming both remedies and stating that re-running is not one. The pre-existing long-form
warning (>1500 tokens) is unchanged.

**Open for the next fine-tune:** the corpus has no short utterances to speak of. If short-form
synthesis matters, that is a corpus gap to fill deliberately (0.21% under 3 s), not an inference
knob. Worth noting alongside it that FLEURS is read news prose — so short *conversational* register
is doubly absent.
