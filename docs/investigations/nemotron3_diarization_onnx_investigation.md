# Nemotron-3-Diarization → ONNX investigation

Issue #246 asks for `nvidia/Nemotron-3-Diarization` (released 2026-09-23). It is a
Streaming-Sortformer successor, so the question is how much of the existing Sortformer
ONNX contract (`export_sortformer_nemo_to_onnx.py` → `SortformerStreamer`) carries over.

## Run 1 — 2026-09-30

**Question:** what actually changed relative to `diar_streaming_sortformer_4spk-v2.1`?

**Commands:** `tar xf Nemotron-3-Diarization.nemo model_config.yaml`; HF `config.json`,
`processor_config.json`, model card.

**Raw findings** (`.nemo` is 199 MB, `nemo_version: 3.0.0`):

| | v2.1 (shipped) | Nemotron-3 |
|---|---|---|
| speakers | 4 | **8** |
| encoder | FastConformer (17 L) + 18-layer 192-d transformer | `TransformerEncoder`, 31 L, d=512, **RoPE**, `subsampling: feature_stacking` ×8, pre-norm |
| mel | 128 bins, n_fft 512, hop 160 | same (preproc `normalize: NA`, preemph 0.97) |
| silence embedding | running mean of popped silence frames | `use_learnable_sil_emb: true` |
| `spkcache_sil_frames_per_spk` | 3 | **1** |
| `spkcache_len` | 188 | **264** |
| output | 80 ms frames | `high_resolution: true`, `output_subsampling_factor: 1` → the card says configurable in multiples of 10 ms, Transformers returns 10 ms frames |
| recommended offline cfg | chunk 124 / fifo 124 / period 124 | chunk 340 / right ctx 40 / fifo 40 / period 300 |

New: **`chunk_right_context`** (look-ahead frames scored by the next chunk) in the recommended
configs — the shipped C# streamer has no right context at all.

**Implications:**
- The installed export venv is NeMo 2.7.1; this needs NeMo 3.0.0 → new venv `.venv-nemo3-export`.
- `SortformerStreamer` hard-codes every constant through `Config`; it must be parameterised
  per model (speakers, cache/fifo/chunk lengths, sil frames, learnable sil emb, frame duration).
- The learnable silence embedding and the 10 ms output path have to be read from NeMo 3.0
  source before deciding the ONNX contract.

## Run 2 — 2026-09-30

**Question:** can the published NeMo load it, and how does the new model produce its outputs?

**Commands:** `uv pip install nemo-toolkit[asr]==3.0.0` into `.venv-nemo3-export`, then read
`sortformer_diar_models.py` / `sortformer_modules.py` from the wheel and from
`NVIDIA-NeMo/Speech@00278b0bd` (main, 2026-09-29); a probe that restores the `.nemo`.

**Raw findings:**
- **PyPI `nemo-toolkit==3.0.0` cannot build this checkpoint.** Its `SortformerEncLabelModel.__init__`
  unconditionally does `from_config_dict(self._cfg.transformer_encoder)`, and the config has no
  `transformer_encoder`. There is no `high_resolution` or `use_learnable_sil_emb` anywhere in it.
  NeMo main has both. Installed `nemo-toolkit @ git+…Speech@00278b0bd` with `--no-deps` over 3.0.0;
  it also needs `lhotse==2.0.0a6` (`ModuleNotFoundError: lhotse.indexing` on 1.33.0).
- Transformers 5.18.0 (PyPI) ships `models/nemotron3_diarization` as well.
- Probe on the restored model: `TransformerEncoder`, `pre_encode = FeatureStacking`,
  `transformer_encoder = None`, `encoder_proj = Linear(512→192)`,
  `subpixel_upsample = Conv1d(192, 1536, k=3, pad=1)` (×8 upsample), `learnable_sil_emb` [512]
  (norm 0.119), an `activity_head` (3-class, training aux only), 99.2 M params.
  Checkpoint's own streaming defaults: chunk 264 / fifo 0 / cache 264 / period 264 / lc=rc=0,
  i.e. *not* the card's offline recommendation.
- How the outputs are formed in NeMo main:
  - `forward_infer` → `upsample_hidden` (subpixel conv over the whole `[cache|fifo|chunk]`
    sequence) → speaker head → sigmoid × mask. **10 ms preds.**
  - `forward_streaming_step` feeds the streaming *state* `downsample_preds(hr, 8)` —
    `avg_pool1d(k=8, ceil_mode)` back to 80 ms — while the *reported* chunk preds are the
    10 ms slice `[(cache+fifo+lc)·8, +chunk·8)`.
  - `forward_for_export` (NeMo's own ONNX path) returns only the 80 ms downsampled preds.
  - Learnable sil emb: `_compress_spkcache` swaps in `learnable_sil_emb` for the disabled
    slots and `streaming_update` skips `_get_silence_profile` entirely. Everything else in
    cache compression is byte-for-byte the v2.1 algorithm the C# port already mirrors.
- Default post-processing, both NeMo (`PostProcessingParams`) and Transformers
  (`extract_speaker_dict`): threshold 0.5, no pads, no min durations, no median filter.

**Implications:**
- ONNX contract = the existing six inputs / three outputs, plus a fourth output
  `spkcache_fifo_chunk_preds_hr` (10 ms). The C# loop keeps driving state from the 80 ms
  output and reports from the 10 ms one.
- `learnable_sil_emb` and the streaming schedule go into the ONNX `metadata_props` so the
  runtime reads them from the artifact rather than from a second hard-coded constant table.
- Chunks must not be zero-padded: the k=3 upsampling conv at the last valid frame would read
  the padded position's hidden state. Feed the exact-length chunk (dynamic axes allow it).

## Run 3 — 2026-09-30

**Question:** does the model trace to ONNX, and does the graph match NeMo?

**Command:** `.venv-nemo3-export/bin/python scripts/nemo_export/export_nemotron3_diarization_to_onnx.py
--nemo …/Nemotron-3-Diarization.nemo --output …/onnx/nemotron-3-diarization.onnx`

**Raw findings:**
- First attempt (wrapping NeMo's `frontend_encoder` directly) dies in
  `transformer_encoder.py:1071 create_block_mask(...)` → `pad_mask: kv_idx < lengths[b]` →
  `RuntimeError: unordered_map::at`. The encoder is **FlexAttention**; neither the TorchScript
  exporter nor ORT has anything for it.
- `FeatureStacking.forward` also computes its pad and `reshape(b, t_new, …)` from Python ints,
  which the tracer would freeze at the example length.
- Fix: an export-mode encoder in the script (SDPA with a `[B,1,1,T]` key-padding mask — the
  mask_mod is padding-only for `attn_mode: full`, `causal_tail_len: 0`), `reshape(b, -1, c·8)`
  for stacking, and the runtime pads each mel chunk to a multiple of 8 with zero rows (what
  FeatureStacking adds anyway). The script refuses any checkpoint whose encoder config differs.
- Three-way parity, max abs diff over valid frames (NeMo modules get the **unpadded** chunk):

  | case | preds wrapper−NeMo | preds ORT−NeMo | preds_hr ORT−NeMo | embs ORT−NeMo |
  |---|---|---|---|---|
  | first chunk (cache 0, fifo 0, 3040 mel) | 1.2e-9 | 5.0e-8 | 1.4e-7 | 6.7e-6 |
  | steady (264 / 40 / 3040) | 1.2e-9 | 4.7e-8 | 1.4e-7 | 6.7e-6 |
  | tail, 1234 mel | 1.6e-9 | 3.7e-8 | 1.3e-7 | 6.7e-6 |
  | tiny tail, 13 mel | 4.2e-9 | 4.6e-8 | 1.2e-7 | 4.8e-6 |

- Artifact: 400 MB fp32 single file (99 M params), opset 17, all time axes dynamic.

**Implications:** the graph is right at the chunk level. What is left to prove is the
*streaming loop* (C#) against NeMo's `forward_streaming` on real multi-speaker audio.

## Run 4 — 2026-09-30

**Question:** does the C# streaming loop (profile-driven `SortformerStreamer`) reproduce NeMo's
`forward_streaming` on real multi-speaker audio?

**Test audio** (public, CC-BY, 16 kHz mono, 5 min each unless noted; RTTM labels from the same
source): the Transformers diarization example (97.6 s, no labels); three VoxConverse dev
recordings (6, 8 and 16 labelled speakers in the excerpt); two AMI SDM test meetings (4 speakers,
excerpt 300–600 s). Written by an inline extraction step to `/mnt/data/models/nemotron3_diarization/audio`.

**Commands:**
```
.venv-nemo3-export/bin/python scripts/nemo_export/nemotron3_diarization_reference.py --nemo … --audio … --out-dir ref
dotnet run --project tests/Nemotron3DiarizationParity -p:EP=Cpu -c Release -- <models-root> csharp --v21 <wavs>
.venv-nemo3-export/bin/python scripts/nemo_export/nemotron3_diarization_fidelity.py --audio-dir … --nemo-dir ref --csharp-dir csharp --stems …
```

**Raw findings (NeMo fed its own preprocessor):**

| file | max\|Δ\| | mean\|Δ\| | flip % | fidelity DER % |
|---|---|---|---|---|
| diarization_example | 0.157 | 2.1e-3 | 0.015 | 0.12 |
| vox_dev_a (16 spk) | **0.872** | 7.5e-3 | 0.33 | **2.63** |
| vox_dev_b | 0.089 | 3.1e-4 | 0.004 | 0.03 |
| vox_dev_c | 0.074 | 9.3e-5 | 0.001 | 0.01 |
| ami_sdm_a | 0.085 | 7.0e-4 | 0.035 | 0.31 |
| ami_sdm_b | 0.163 | 8.7e-5 | 0.007 | 0.10 |

Per-chunk max\|Δ\| on vox_dev_a: `0.019 0.012 0.006 0.034 0.456 0.473 0.125 0.474 0.095 0.058 0.787 0.872`
— chunk 0 (empty cache and FIFO, so no streaming state at all) already differs by 0.019.

**Implication:** a chunk-0 difference can only come from the input. Next: separate the mel
frontend from the loop.

## Run 5 — 2026-09-30

**Question:** is the Run 4 divergence the frontend or the loop?

**Raw findings:**
- NeMo's features vs `benchmark_sortformer_rtf.log_mel_spectrogram` (the Python transcription of
  Vernacula's C# mel, shared with v2.1) on vox_dev_a: mean \|Δ\| 0.0167, p99 0.209, max 2.72
  log-mel units; NeMo gives 30000 frames, Vernacula 30001.
- Added `--mel-source vernacula` to the reference runner (NeMo's loop on Vernacula's mel):

  | file | max\|Δ\| | mean\|Δ\| | flip % | fidelity DER % |
  |---|---|---|---|---|
  | diarization_example | 1.7e-5 | 1.5e-7 | 0 | 0 |
  | vox_dev_a | 1.0e-5 | 8.0e-8 | 0 | 0 |
  | vox_dev_b | 0.061 | 1.2e-4 | 0.001 | 0.010 |
  | vox_dev_c | 0.053 | 3.8e-5 | 0.0004 | 0.003 |
  | ami_sdm_a | 1.4e-5 | 9.3e-8 | 0 | 0 |
  | ami_sdm_b | 2.6e-5 | 4.7e-8 | 0 | 0 |

**Implications:** the ported loop is NeMo's to float noise on four files, and to ≤0.01 % DER on
the other two -- a single late divergence, the cache-selection tie class documented for v2.1 in
`sortformer_topk_ties_investigation.md`. Run 4's vox_dev_a gap is the shared frontend's small
differences amplified by discrete cache choices on a 16-speaker recording (twice the model's
capacity). The frontend is not specific to this model and is out of scope here.

## Run 6 — 2026-09-30

**Question:** accuracy against the labels, v2.1 vs Nemotron-3, and what post-processing to ship.

**Raw findings (C# output, DER %, overlap scored, collar 0.25 / 0):**

| file | Nemotron-3 | v2.1 |
|---|---|---|
| vox_dev_a (16 spk) | 16.1 / 20.2 | 33.7 / 38.9 |
| vox_dev_b | 1.8 / 3.2 | 11.9 / 14.1 |
| vox_dev_c | 0.1 / 0.6 | 18.9 / 20.3 |
| ami_sdm_a | 33.4 / 34.1 | 31.4 / 37.1 |
| ami_sdm_b | 34.7 / 35.2 | 24.2 / 27.1 |

- NeMo with its own frontend scores the same on AMI (33.3 / 34.7), so this is the model, not the port.
- AMI breakdown (collar 0.25): Nemotron-3 miss 32.1 / 34.3, confusion 0.8 / 0.4; v2.1 miss 30.7 /
  24.1, confusion 0.4 / 0.1. Almost all of it is **missed speech**, not attribution. The model card
  scores AMI against *forced-aligned* labels precisely because the original segment annotations
  mark within-segment silence as speech; these labels are the original segments. Nemotron-3's
  tight boundaries are penalised by them; v2.1's pads and gap merging bridge the pauses.
- On VoxConverse, confusion 13.8 vs 20.4 and miss 2.1 vs 12.3 on the 16-speaker clip (v2.1 caps at 4).
- Post-processing sweep on the Nemotron-3 probabilities (segments / DER0 / DER25):

  | on, off, pad_on, pad_off, min_on, min_off | vox_a | ami_a | ami_b | mean DER25 (5 labelled) |
  |---|---|---|---|---|
  | NeMo default 0.5, 0.5, 0, 0, 0, 0 | 74 / 20.2 / 16.1 | 251 / 34.1 / 33.4 | 129 / 35.2 / 34.7 | 17.22 |
  | min_off 0.3 | 52 / 20.4 / 16.1 | 209 / 32.3 / 31.5 | 118 / 34.2 / 33.7 | 16.64 |
  | **min_off 0.5** | 50 / 20.6 / 16.4 | 174 / 28.9 / 28.0 | 99 / 31.6 / 30.8 | 15.41 |
  | min_on 0.25, min_off 0.5 | 53 / 20.5 / 16.3 | 159 / 31.2 / 29.6 | 89 / 32.4 / 31.4 | 15.82 |
  | pads 0.05, min_on 0.1, min_off 0.3 | 51 / 20.5 / 16.3 | 189 / 29.0 / 27.2 | 108 / 30.0 / 29.0 | 14.89 |
  | median window 9 | = default | | | 17.22 |

  vox_dev_b / vox_dev_c are unchanged by every row (17 / 13 segments).

**Decision:** ship NeMo's defaults plus `MinDurOff = 0.5`. It is the single change that reduces
fragmentation (each segment is transcribed on its own) without moving VoxConverse attribution.
The pads row scores slightly better, but only on the loosely-labelled AMI excerpts. Five clips are
too few to tune more finely than this.

## Run 7 — 2026-09-30

**Question:** does the integrated build work end to end, and does anything else regress?

**Commands / raw findings:**
- `dotnet build Vernacula.slnx -p:EP=Cpu` — clean, no warnings.
- Full suite (`Vernacula.Tests`, `Vernacula.Tts.Tests`, `AsrBackendCoverage`, `-p:EP=Cpu`):
  101 / 279 (+4 skipped) / 367 passed. New: `SortformerProfileTests` (6),
  `ModelRepoSelectionTests` +3 cases for the Nemotron-3 download.
- Mutation check: forcing the Nemotron-3 repo append in `ModelManagerService.ActiveRepos` off turns
  `Nemotron3Segmentation_AddsItsModel_OnTopOfTheBackendsFiles` red (1 failed / 9 passed); restored.
- `vernacula-cli --diarization nemotron3 --skip-asr` on vox_dev_b: 12 chunks, 17 segments, 9.1 s
  for 300 s of audio on CPU.
- v2.1 regression: the harness's v2.1 mode run from a worktree at the pre-change commit (7f05969)
  and from the refactor, on vox_dev_a, ami_sdm_a and diarization_example — `cmp` reports the raw
  `.preds.f32` and the `.rttm` **byte-identical** for all three.
- `make_manifest.py` on the export: `nemotron-3-diarization.onnx md5=191e8ec327760d33a400abf856fbf808`,
  381.6 MiB.

**Published** 2026-09-30 with `scripts/upload_to_hf.py --create-repo --sync-readme` to
`christopherthompson81/nemotron3_diarization_onnx` (model, export report, `manifest.json`, card).
Fetched back through the app's download URLs: manifest served, downloaded model md5
`191e8ec327760d33a400abf856fbf808` = manifest. Non-English help/locale text for the new option
falls back to English until the translation scripts are run.

## Environment recipe (for re-running any of the above)

```
uv venv -p 3.12 .venv-nemo3-export
uv pip install -p .venv-nemo3-export/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv-nemo3-export/bin/python 'nemo-toolkit[asr]==3.0.0' onnx onnxruntime onnxscript soundfile pyannote.metrics
uv pip install -p .venv-nemo3-export/bin/python --no-deps \
    "nemo-toolkit @ git+https://github.com/NVIDIA-NeMo/Speech.git@00278b0bd95bb2b8174b88012aa21b003c59d2e9"
uv pip install -p .venv-nemo3-export/bin/python --prerelease=allow 'lhotse==2.0.0a6'
```
(torch 2.14.1+cpu, transformers 5.18.0, onnxruntime 1.26 at the time of these runs.)

## Run 8 — 2026-09-30 (PR #247 review fixes)

**Question:** do the review's findings hold, and do the fixes keep the validated outputs?

**Findings and dispositions:**
- *No optimised-graph cache for Nemotron-3* — held. Switched it to `OrtSessionBuilder.CreateCachedSession`
  (which loads its cache on later runs; v2.1's `OptimizedModelFilePath` only ever writes). CPU load
  1675 ms cold → 1026 ms on a hit; profile still read from the cached graph's metadata; outputs
  byte-identical. Costs a second ~400 MB file beside the model (`*.opt.cpu.<key>.onnx` + `_data`).
  This also makes the startup warm-up worthwhile (it fills the cache).
- *`GetIncrementalSegments` emits early* — held, and wider than reported: besides an open segment
  passing `end <= frontier` at Window 1 / PadOffset 0, an emitted segment could later be merged by
  MinDurOff. New rule: final only if `end + MinDurOff < frontier - PadOnset`. Harness check
  (incremental set == batch set): old rule, Nemotron-3 emitted 9 / 10 / 3 segments not in the batch
  output on vox_dev_a / ami_sdm_a / diarization_example (v2.1 happened to be clean); new rule, equal on
  all six files for both models.
- *Per-job streamer never disposed* — held (pre-existing); now `using`.
- *`n_mels` unchecked*, *malformed metadata throws FormatException* — held; both now
  `InvalidDataException`, with tests.
- *Repo selection matched VibeVoice URLs* — replaced with the backend rule the Settings view uses.
- *Duplicate concurrent model checks on a backend switch* — guarded.
- *Unreachable steady-state guard* — removed; one gate left.
- *Warm-up shape* — warm-up now includes the right context and resets state afterwards.

**Regression:** full suite 107 / 279 (+4 skipped) / 367 passed. Harness rerun on all six files:
Nemotron-3 raw probabilities and all v2.1 output byte-identical to Run 4–7; Nemotron-3 RTTMs differ
only by the MinDurOff = 0.5 decision already recorded in Run 6 (ami_sdm_a 251 → 174 segments).
Settings UI checked headless (Xvfb): option renders, selection persists (1 ↔ 4), and with the file
absent it is listed as missing with the download offered.
