# Sortformer CoreML variant — publish investigation (issue #162)

Goal: produce `diar_streaming_sortformer_4spk-v2.1.coreml.onnx`, verify it against
the shipped dynamic model, and publish it to
`christopherthompson81/sortformer_parakeet_onnx` with a `manifest.json` entry.

The techniques and the M5 measurements are already recorded in
`docs/coreml_onnx_playbook.md` (landed with #164). This log covers only the
*build and publish* of the artifact.

## Environment (Linux workstation, not the Mac)

- repo: the local working tree at `631a88c` (`$WORK` below is a scratch model dir)
- export venv: `.venv-nemo-export` (python 3.12.3) — `nemo-toolkit 2.7.1`,
  `torch 2.11.0+cu128`, `onnx 1.21.0`, `onnxruntime 1.26.0`
- source checkpoint: HF cache snapshot of `nvidia/diar_streaming_sortformer_4spk-v2.1`
- reference dynamic model: `~/.local/share/Vernacula/models/sortformer/diar_streaming_sortformer_4spk-v2.1.onnx`
  (492,268,840 B, md5 `647a22cef31f59dc2c314fa783b2581d` — matches the published manifest)

**Known gap up front:** this box has no CoreML. Partition count and the CoreML-EP
numbers in #162 cannot be re-measured here, and the playbook warns partitioning is
ORT-version dependent. The build ORT here is 1.26.0, not the 1.24.4 used for the
issue's measurements — the constant-folding step (`basic_fold`) runs under ORT, so
the folded graph is not guaranteed byte-identical to the one measured on the Mac.
What *is* verifiable here: numeric parity vs the shipped dynamic model on CPU, the
input/output contract, and the file size.

## Run 1 — 2026-09-09 — export the static CoreML-shaped graph

Question: does `--coreml-static-batch1 --coreml-const-lengths` export cleanly in
this venv, and what does the raw graph look like?

Command:

```bash
./.venv-nemo-export/bin/python scripts/nemo_export/export_sortformer_nemo_to_onnx.py \
  --nemo <hf-cache>/diar_streaming_sortformer_4spk-v2.1.nemo \
  --output $WORK/sortformer_coreml.onnx \
  --opset 17 --coreml-static-batch1 --coreml-const-lengths \
  --chunk-frames 992 --fixed-spkcache-frames 188 --fixed-fifo-frames 124 --overwrite
```

Result: **clean export**, 502,241,105 B, in ~40 s.
`sortformer_coreml.onnx.report.json` records the expected notes, including:

> `spkcache_lengths=188` and `fifo_lengths=124` are baked in as graph constants … **`chunk_lengths` is still a live input.**

That last clause is the thread this whole investigation pulls on.

## Run 2 — 2026-09-09 — graph transforms

Question: do the four transforms reproduce the counts #162/#164 report
(51 `Where` removed, 34 `Pad`, 180 `Gemm`, 3 inputs, 1751 nodes)?

```bash
./.venv-nemo-export/bin/python scripts/nemo_export/coreml_optimize_sortformer.py \
  --input  $WORK/sortformer_coreml.onnx \
  --output $WORK/diar_streaming_sortformer_4spk-v2.1.coreml.onnx \
  --verify --reference ~/.local/share/Vernacula/models/sortformer/diar_streaming_sortformer_4spk-v2.1.onnx
```

Raw output:

```
[1/5] constant-folding at ORT_ENABLE_BASIC
[2/5] removing all-False Where ...        0 removed
[3/5] rewriting Pad -> Concat ...        34 converted
[4/5] pre-transposing Gemm weights ...  180 converted
[5/5] wrote ... (484 MB), 1914 nodes, inputs=['chunk', 'chunk_lengths', 'spkcache', 'fifo']

verifying against reference on CPU EP:
--verify needs a static graph; input chunk has shape ['batch', 'time_chunk', 128]
```

Three findings, in increasing order of importance:

1. `--verify --reference <shipped dynamic model>` **cannot work as documented.**
   `verify()` builds its feed from the *reference* model's signature and bails on any
   non-int dimension, so the one reference the docstring and #162's repro command name
   is the one reference it rejects. It only works against another static graph
   (i.e. `--input`). Cosmetic; parity can be checked with a hand-written feed instead.

2. `Pad` and `Gemm` match (34, 180). `Where` is **0 removed, not 51**, and the graph
   keeps **4 inputs, not 3** (1914 nodes, not 1751).

3. The 51 `Where` nodes are all still there — `drop_allfalse_where` skips them because
   their condition is not a constant initializer. Tracing the ancestry of one:

   ```
   /encoder/layers.0/self_attn/Where.cond
     <- Unsqueeze <- Not <- And <- And <- Tile <- Unsqueeze <- Less
        <- chunk_pre_encode_lengths <- Cast <- ... <- chunk_lengths   [GRAPH INPUT]
   ```

   The attention padding mask is a function of `chunk_lengths`, which is a live input,
   so it cannot fold to a constant, so it is not all-False, so none of the 51 can be
   removed. `chunk_lengths` has exactly one consumer in the folded graph
   (`/pre_encode/conv/Cast`) — that mask chain.

**So the pipeline on `main` cannot produce the graph #162 measured.** The measurement
predates a review fix. `gh pr view 164 --json commits` shows two commits, the second
being "macOS: fix review findings in the CoreML/WebGPU change", and the surviving
comment in `export_sortformer_nemo_to_onnx.py` states the reasoning:

> ⚠ `chunk_lengths` is deliberately NOT baked. […] Baking `chunk_frames` there would let
> the zero-padded tail of that last chunk be attended to as real audio, changing the
> diarization output at the end of every file. It stays a real input; only its mask stays
> data-dependent, **which costs a few partitions, not correctness.**

"A few partitions" is the part that is wrong. Per the playbook's own table, removing the
`Where` nodes is what takes partitions 69 → 35, and `Pad` → `Concat` only reaches 1
partition *after* that (replacing `Where` with `Identity` instead of deleting it made it
worse, 69 → 157 — so the nodes are load-bearing for partitioning, not incidental). The
best available without them is the 69-partition / 166.0 ms row, i.e. **level with the CPU
EP (163.5 ms) and worse than WebGPU (94.0 ms)**. Keeping `chunk_lengths` live does not
cost a few partitions; it costs the entire reason for the variant to exist.

Both `docs/coreml_onnx_playbook.md` and the #164 PR body still advertise the pre-fix
numbers (1 partition / 52.3 ms / "removes 51 all-False `Where` masks") against a pipeline
that no longer produces them.

## Run 3 — 2026-09-09 — is the review fix *right*? Quantify the tail chunk

Question: #162's stated contract is "steady-state only … the final short chunk needs
padding", i.e. bake `chunk_lengths` and let the caller zero-pad. The #164 review called
that a correctness bug. Who is right, and by how much?

This is measurable on the **shipped dynamic model alone** — no graph transform involved.
`Sortformer.cs:330-372` already sends a full-size zero-padded `chunk` buffer and varies
only `chunk_lengths` (`currentLen`, short on the last chunk of every recording), while
`spkcache_lengths`/`fifo_lengths` are always the buffers' own sizes. So run the same
padded input twice, once with the honest `chunk_lengths` and once with 992, and diff the
outputs over the **valid** frames only.

`scratchpad/tail_chunk_effect.py`, random-normal chunk/cache/fifo, seed 7:

| `chunk_lengths` | valid frames | `preds[chunk_valid]` maxAbs | rms | `preds[fifo]` maxAbs | `embs` |
|---|---|---|---|---|---|
| 992 | 124/124 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| 800 | 100/124 | 0.1549 | 0.0626 | 0.0056 | 0.0 |
| 400 |  50/124 | 0.2240 | 0.0792 | 0.0063 | 0.0 |
| 200 |  25/124 | 0.4863 | 0.1859 | 0.0172 | 0.0 |
|  64 |   8/124 | 0.5426 | 0.2364 | 0.0035 | 0.0 |

`preds` are per-speaker probabilities on 0..1. **The review was right and #162's contract
is not viable:** baking `chunk_lengths` moves the final chunk's speaker probabilities by
up to 0.54 and an rms of 0.24 — enough to flip speaker assignments outright, not a
tolerance question. (Random input is a pessimistic stand-in for speech, but two orders of
magnitude of headroom is not going to appear.) `chunk_pre_encode_embs` is unaffected
(0.0 everywhere) because `pre_encode` is a convolution over the padded buffer and never
sees the mask; the damage is entirely in the encoder's self-attention over the packed
436-frame sequence.

Note also that the steady-state assumption for the *other* two buffers is genuinely free:
the warm-up chunks #162 flags as needing padding already work, because the shipped runtime
always passes `spkcache_lengths`/`fifo_lengths` as the full buffer sizes even while those
buffers are still zero-filled.

## Where that leaves issue #162

The artifact #162 asks to publish is reachable — but only as a **steady-state graph that
must not be fed the final chunk of a recording**. It cannot be a whole-recording
replacement for the shipped model at any accuracy anyone would accept. So the publish
needs a decision (see the conversation): the exportable choices are

* **A. Bake all three lengths** (new opt-in flag) → 1 partition / 52.3 ms as measured, and
  the runtime routes the one short tail chunk per recording to the shipped dynamic model
  on CPU or WebGPU. Exact, since every other chunk is full-length; costs a second loaded
  session for one inference per file.
* **B. Ship what `main` produces** (`chunk_lengths` live, 4 inputs, 51 `Where`) → correct
  for every chunk, ~69 partitions, no measurable CoreML win. Publishes a 480 MB file that
  buys the 13% CPU / 10% WebGPU improvement from the static shapes and nothing else.

Result: pending decision.


**Decision (2026-09-09): option A.** Bake all three lengths behind an opt-in flag, publish
the steady-state graph, and make the tail-chunk fallback part of the documented contract.

## Run 4 — 2026-09-09 — build the steady-state graph

Added `--coreml-const-chunk-length` (requires `--coreml-const-lengths`, which requires
`--coreml-static-batch1`) and re-exported. Also fixed `verify()`, which could never run
against the reference its own docstring and #162's repro command name: it built the feed
from the *reference* signature and bailed on any dynamic dimension. It now takes the
buffer shapes from `--input` (the static export) and synthesizes each input by name for
the union of what the two graphs declare — needed because the baked lengths are gone from
`--input`'s signature too.

```
[1/5] constant-folding at ORT_ENABLE_BASIC
[2/5] removing all-False Where ...       51 removed
[3/5] rewriting Pad -> Concat ...        34 converted
[4/5] pre-transposing Gemm weights ...  180 converted
[5/5] wrote diar_streaming_sortformer_4spk-v2.1.coreml.onnx (527 MB), 1788 nodes,
      inputs=['chunk', 'spkcache', 'fifo']
```

Matches #162: 51 / 34 / 180, three inputs, 527 MB. Node count is 1788 vs the 1751 reported
there — expected, since the graph is folded by whatever ORT the build box has (1.26.0 here,
1.24.4 there).

Parity against the shipped dynamic model on the CPU EP, full chunk, seed 7:

```
spkcache_fifo_chunk_preds   maxAbsDiff = 3.278E-07   rms = 6.151E-08
chunk_pre_encode_embs       maxAbsDiff = 3.052E-05   rms = 3.441E-06
```

`preds` reproduces #162 (3.576E-07). `embs` does not — #162 reports exactly 0.0. Chased it:
comparing the reference against the **pre-transform** static export gives the identical
`3.052E-05`, so none of it comes from the four transforms (they are value-preserving to the
bit here). It is the reference session running at ORT's default `ORT_ENABLE_ALL` while the
candidate runs at `ORT_DISABLE_ALL` — ORT's own fusions on the reference side. Not
attributable to the variant.

## Run 5 — 2026-09-09 — the artifact will not load at ORT_ENABLE_ALL

Question (not on anyone's list, which is why it nearly shipped): does the finished file
load the way `OrtSessionBuilder` actually loads models?

```python
ort.InferenceSession(coreml_variant, providers=["CPUExecutionProvider"])   # default = ORT_ENABLE_ALL
```

```
FAIL : graph.cc:3864 AddInitializedOrtValue Attempt to replace the existing tensor
```

| level | loads |
|---|---|
| `ORT_DISABLE_ALL` | yes |
| `ORT_ENABLE_BASIC` | yes |
| `ORT_ENABLE_EXTENDED` | **no** |
| `ORT_ENABLE_ALL` | **no** |

`OrtSessionBuilder.Create` defaults `optLevel` to `ORT_ENABLE_ALL`, and `verify()` happens
to use `ORT_DISABLE_ALL`, so the parity check passes on a file the app cannot open.

Isolated it. Not our transforms — applying **none** of them still fails, and the raw
pre-fold static export loads fine at every level:

| graph | BASIC | EXTENDED | ALL |
|---|---|---|---|
| raw static export | ok | ok | ok |
| BASIC-folded, no transforms | ok | **fail** | **fail** |
| + any subset of the 4 transforms | ok | **fail** | **fail** |
| shipped dynamic model | ok | ok | ok |

So `basic_fold`'s ORT round-trip is what does it: ORT cannot re-optimize its own saved
optimized graph above BASIC. Bisecting with `disabled_optimizers`:

```
disable MatMulAddFusion   -> loads at EXTENDED
disable <11 others>       -> still fails
```

`MatMulAddFusion` reshapes >2D MatMul inputs so it can emit a `Gemm`, and the folded graph
already contains 340 of the initializers it generates (`gemm_input_shape_token_N`,
`gemm_output_reshape_token_N_new_shape`) because the fold pass already ran it. Renaming all
340 out of the way does **not** fix it, so the collision is on something the fusion mints
fresh rather than on those names; it is an ORT re-optimization defect, not something the
graph can be shaped around from here.

**Not a blocker, but it is a hard contract term.** `ORT_ENABLE_BASIC` is the level this
variant has to run at anyway — the playbook already found EXTENDED makes CoreML
partitioning much worse (69 → 191), so the required level and the optimal level are the
same one. It has to be written down in the model card and enforced at the call site, since
the builder's default is the level that throws.

## Run 6 — 2026-09-09 — the 52.3 ms was measured on an ORT the app does not ship

Not a run, a read. Prompted by the reminder that the 1-partition / 52.3 ms result came off
the Apple Silicon MacBook Air, so this issue has to bounce between the two machines
anyway — which raised the question of *which ORT* each side is actually holding.

`Directory.Build.props:21-24`:

```xml
<EP Condition="'$(EP)' == ''">Cuda</EP>
<OnnxRuntimeVersion Condition="'$(EP)' == 'DirectML'">1.24.4</OnnxRuntimeVersion>
<OnnxRuntimeVersion Condition="'$(OnnxRuntimeVersion)' == ''">1.29.0</OnnxRuntimeVersion>
```

Only a **DirectML** build pins 1.24.4. An Apple Silicon build is `-p:EP=Cpu` (per #164 — the
plain `osx-arm64` package is the one carrying the CoreML and WebGPU natives), so it takes
the default: **ORT 1.29.0**.

Every CoreML number in #162, #164 and the playbook was measured on **1.24.4**. And #162's
own caveat says what 1.29.0 does to this graph:

> python ORT 1.29.0 partitions the same graph into **194** and diverges at **~1e-2**.

So the variant's entire benefit is measured on a runtime the macOS build does not use, and
the one data point we have on the runtime it *does* use says the benefit is absent and the
outputs are wrong by 1e-2. That is the blocker for publishing, ahead of anything about the
graph itself. Three ways out, in preference order:

1. **Make it work on 1.29.0.** Re-run the partition survey there and find which op families
   1.29.0 declines that 1.24.4 accepted — same method as the playbook, new ORT. The 1e-2
   divergence needs its own root-cause; at 194 partitions it is probably a CPU↔CoreML fp
   boundary rather than a bad rewrite, but that is a guess until measured.
2. **Pin macOS to 1.24.4** the way DirectML is pinned. Cheap to write, but it forks the
   stack on a third axis and 1.24.4 does not necessarily have the osx-arm64 fixes 1.29.0 has.
3. **Don't ship the variant.** macOS stays on WebGPU (113.0 ms), which #164 already made the
   `Auto` choice there. Costs nothing that exists today.

Nothing here is decidable from this machine.

## What is done, and the Mac-side checklist

Done on this machine (all machine-independent, all committed):

* `--coreml-const-chunk-length`, the opt-in that actually reaches the all-False mask, with
  the correctness obligation it creates written into the flag help and the export report.
* `verify()` fixed so `--reference <the shipped dynamic model>` works — the documented
  usage, previously the one usage that could not run.
* An artifact built here, from the ungated upstream checkpoint, ORT 1.26.0 folding:
  `diar_streaming_sortformer_4spk-v2.1.coreml.onnx`, 526,897,914 B,
  md5 `94cc7392ea9358e22b479fac146e9595`, 1788 nodes, inputs `[chunk, spkcache, fifo]`,
  parity `3.278E-07` on `preds` vs the shipped model at a full chunk.
* Docs corrected to describe the pipeline that exists rather than the pre-review one.

**Not published.** The HF upload and the `manifest.json` entry are deliberately still
pending — publishing a 527 MB file whose only measured win is on an ORT the app does not
ship would be advertising a benefit no user of the shipped build receives.

`scripts/nemo_export/coreml_partition_probe.py` exists so each of these is one command;
the PR body carries the same list with the commands filled in. In order:

1. On **ORT 1.29.0** (what an Apple Silicon build ships), probe the variant on the CoreML EP
   and read the partition count and the CoreML-vs-CPU diffs. **This answers the go/no-go**;
   everything below is moot if 1.29.0 cannot be made to behave.
2. Repeat on 1.24.4, to learn whether a file folded under a *different* ORT (1.26.0, on the
   Linux box) reproduces the 1-partition result at all. If it does not, the shippable
   artifact has to be **built** on the Mac — fold under the ORT it will run on — and the
   Linux box only ever cross-checks numerics.
3. Confirm the load-level constraint holds there too: loads at `ORT_ENABLE_BASIC`, throws at
   `ORT_ENABLE_ALL` (Run 5, traced to `MatMulAddFusion`). If it throws at BASIC as well, the
   artifact needs rebuilding without the ORT fold round-trip and this whole approach needs
   rethinking.
4. Only then publish: `scripts/make_manifest.py` then
   `scripts/upload_to_hf.py --sync-readme`, from whichever machine built the file that
   passed, and add the `manifest.json` entry by extending the published manifest rather than
   regenerating it (the live one deliberately omits `sortformer/`, the int8 variants and
   `silero_vad.onnx`, so `--all` would silently change what the app validates).

## Run 7 — 2026-09-09 — the Mac. The 1.29.0 blocker does not reproduce

Environment: MacBook Air (Apple Silicon), macOS 25.6.0, export venv rebuilt here on
python 3.12.14 / **onnxruntime 1.29.0** (the version an `EP=Cpu` osx-arm64 build takes,
per Run 6). Source checkpoint downloaded fresh from `nvidia/diar_streaming_sortformer_4spk-v2.1`;
reference dynamic model `~/models/diar_streaming_sortformer_4spk-v2.1.onnx`, md5
`647a22cef31f59dc2c314fa783b2581d` — the same file Run 1 used.

### Build reproduces the Linux run

Same two commands as Runs 1 and 4. Transform counts are identical:

```
[2/5] removing all-False Where ...       51 removed
[3/5] rewriting Pad -> Concat ...        34 converted
[4/5] pre-transposing Gemm weights ...  180 converted
[5/5] wrote diar_streaming_sortformer_4spk-v2.1.coreml.onnx (527 MB), 1788 nodes,
      inputs=['chunk', 'spkcache', 'fifo']
```

Artifact: 526,897,872 B, md5 `890c59cc71dedfff9f1c3d7f10924422` — 42 bytes off the Linux
build, as expected from folding under a different ORT.

`verify()` against the shipped dynamic model, CPU EP, seed 7:

| | this Mac (ORT 1.29.0) | Linux (Run 4, ORT 1.26.0) | #162 (ORT 1.24.4) |
|---|---|---|---|
| `preds` maxAbs | 3.576E-07 | 3.278E-07 | 3.576E-07 |
| `embs` maxAbs | **0.000E+00** | 3.052E-05 | 0.0 |

The `embs` figure confirms Run 4's diagnosis: the Linux `3.052E-05` was reference-side
`ORT_ENABLE_ALL` fusion, not the variant. It is exactly zero when both sides match.

### Checklist step 1 — CoreML EP on ORT 1.29.0. **GO.**

```
partitions      1 (1751/1751 nodes on the EP)  <- single partition
All nodes placed on [CoreMLExecutionProvider]
```

| | #162's caveat for 1.29.0 | measured here on 1.29.0 |
|---|---|---|
| partitions | 194 | **1** |
| CoreML-vs-CPU `preds` | ~1e-2 | **4.470E-07** (rms 7.616E-08) |
| inference | — | **51.5 ms** (min/max 51.3/51.7) |
| cold / warm load | — | **2.2 s / 0.16 s** |

CPU EP on this machine, same graph: **154.0 ms**, so **2.99× vs CPU**. WebGPU could not be
compared — it is not in the pip `onnxruntime` wheel, only in the C# package's natives.

**The Run 6 blocker is retired.** Whatever produced "194 partitions / ~1e-2" in #162, it is
not what 1.29.0 does with the artifact this pipeline builds. Pinning macOS to 1.24.4
(Run 6, option 2) is unnecessary.

The 1751 node count also resolves a loose end: `coreml_optimize_sortformer.py` writes 1788,
and ORT's BASIC fold trims it to 1751 at load. #162's "1751 nodes" was the post-load count,
so Run 4's "1788 vs 1751, expected from a different ORT" was explaining a difference that
was never there.

### Checklist step 3 — the load-level constraint holds, and the CoreML EP hides it

**Corrected 2026-09-09, after the first pass got this wrong.** The probe defaults to
`--ep coreml`, so the level matrix was only ever run with the CoreML EP registered:

| level | loads (coreml) | partitions | inference |
|---|---|---|---|
| `ORT_DISABLE_ALL` | yes | 1 (1788/1788) | 83.9 ms |
| `ORT_ENABLE_BASIC` | yes | 1 (1751/1751) | 51.6 ms |
| `ORT_ENABLE_EXTENDED` | yes | 1 (1751/1751) | 52.0 ms |
| `ORT_ENABLE_ALL` | yes | 1 (1751/1751) | 51.9 ms |

That reads as "Run 5's constraint is a 1.26.0 defect". It is not. Loading the same file on
the **CPU EP alone** on the same ORT 1.29.0:

| level | CPU EP | CoreML EP registered |
|---|---|---|
| `ORT_DISABLE_ALL` | loads | loads |
| `ORT_ENABLE_BASIC` | loads | loads |
| `ORT_ENABLE_EXTENDED` | **fails** | loads |
| `ORT_ENABLE_ALL` | **fails** | loads |

Same `AddInitializedOrtValue Attempt to replace the existing tensor` from
`MatMulAddFusion`. CoreML claims the whole graph in `GetCapability` before the CPU-side
fusions get to run, so the fusion never fires and the defect is invisible — on the one EP
this artifact is built for. **Run 5 was right; `ORT_ENABLE_BASIC` or lower is a real,
version-independent contract term**, and it matters most in exactly the case the CoreML EP
cannot cover: a fallback to CPU when CoreML is absent or declines the graph.

The lesson for the probe: `--ep` selects what you measure *and* what you can see. A
load-level matrix is only meaningful per-EP.

(The playbook's separate claim that EXTENDED worsens CoreML partitioning, 69 → 191, was
measured on 1.24.4 against the `chunk_lengths`-live graph and is untested here.)

### The probe could not have reported any of this

`coreml_partition_probe.py` printed `partitions  not reported` on a graph that was at one
partition — a no-go reading on a go. Three defects, all fixed in this commit:

* `make_session` read the captured stderr *inside* the `with` block, while the `seek(0)`
  sat in the context manager's `finally`. `dup2` makes fd 2 share the file offset, so the
  read happened at end-of-writes and returned `""`, always. Reproduced standalone.
* The `GetCapability` summary is WARNING **only when partitions > 1**; at one partition it
  is INFO. Both the session logger (`SessionOptions.log_severity_level`, was 2) and the
  default logger gate it, so both had to drop to INFO. The comment at `PARTITION_RE`
  claiming "logs this at warning level" was the root of it.
* The "requested EP did not take the graph" guard tested `get_providers()[0]`, which is the
  *registration* list, not the assignment list — an EP that registers and claims zero nodes
  still sits at index 0. It now checks the supported-node count, and correctly caught the
  absent WebGPU EP during this run.

(Also `--runs 0` left `got` unbound, crashing the parity block with `NameError`.)

### Where that leaves the publish

Steps 1–3 pass on the runtime the macOS build actually ships. Step 4 — HF upload and the
`manifest.json` entry — is still **not done**, and now has two blockers of its own:

1. `make_manifest.py` has no merge mode (`--files` and `--all` are exclusive and the result
   is written wholesale to `<model-dir>/manifest.json`), so step 4 as written destroys the
   live manifest with a one-entry file — the same failure the step's own warning attributes
   only to `--all`. The published manifest has to be hand-extended, or the tool needs a
   merge flag.
2. **The tail-chunk routing does not exist.** `Sortformer.cs` has no CoreML path at all;
   `ProcessChunk` still passes `min(start + chunkStride, totalFrames) - start` for every
   chunk. Publishing the steady-state artifact before a caller can route the final short
   chunk to the dynamic graph ships a file that silently degrades the end of every
   recording by up to 0.54 on `preds` (Run 3). The routing is the precondition for the
   51.5 ms being usable at all.

## Published — 2026-09-09

`diar_streaming_sortformer_4spk-v2.1.coreml.onnx` is live in
`christopherthompson81/sortformer_parakeet_onnx`: 526,897,872 B, md5
`890c59cc71dedfff9f1c3d7f10924422` — the artifact built and measured in Run 7.

The manifest was **extended, not regenerated**. `make_manifest.py` has no merge mode
(`--files`/`--all` are exclusive and the result is written wholesale to
`<model-dir>/manifest.json`), so running step 4 as originally written would have replaced
the published 9-entry manifest with a 1-entry file. Instead the live manifest was pulled,
the new entry inserted, and every original entry checked byte-identical before upload:
9 → 10 files, `version: 2` preserved, none removed. **`make_manifest.py` still needs a
merge flag before anyone runs step 4 literally.**

The model card was corrected before syncing rather than published as-is. It carried three
claims that Run 7 disproved or superseded:

* the headline and perf table were ORT 1.24.4 (52.3 ms vs 196.3 ms CPU / 113.0 ms WebGPU);
  now leads with the 1.29.0 measurements — 51.5 ms vs 171.8 ms CPU, **3.34×**, single
  partition, `4.470E-07`.
* `ORT_ENABLE_BASIC or lower` was briefly and wrongly scoped to 1.26.0 on the basis of a
  CoreML-EP-only matrix; corrected the same day. It is unconditional — see step 3 above.
* the "1.29.0 splits into 194 partitions and diverges at ~1e-2" caveat is marked as not
  reproducing.

The 1.24.4 CPU/WebGPU baselines (196.3 / 113.0) still disagree with the figures recorded
elsewhere for the same machine and ORT (163.5 / 94.0). That is unresolved and the card now
says so; this machine measured **171.8 ms** for the stock graph on CPU under 1.29.0, which
does not settle which 1.24.4 figure was right. Both the playbook and
`scripts/nemo_export/README.md` still carry the older numbers.

Also confirmed on 1.29.0: the **stock** dynamic graph still fails to compile under CoreML
(`Input: _ConstantOfShape_1_output_0 has unbounded dimension which is not supported`), so
the variant's reason to exist is intact on the shipped runtime.

**Still not consumable from the app.** `Sortformer.cs` gained CoreML/WebGPU selectability
in `b3410d6` (it was the one call site #164 missed, so macOS diarization had been on the
CPU EP regardless of the requested provider), but the tail-chunk routing does not exist:
`ProcessChunk` sends a short `chunk_lengths` on the last chunk of every recording, and this
graph bakes it to 992. Until a caller routes that one chunk to the stock graph, the
published file must not be wired in as a drop-in — see Run 3 for the 0.54 error it causes.
