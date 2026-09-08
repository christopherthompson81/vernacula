# ORT optimization-cache round-trip for Loop-bearing graphs (issue #56)

Background: `conditional_decoder_loop.onnx` (Phase 2 merged graph) can be
written via `OptimizedModelFilePath` but ORT can't load the result back —
body-scope initializers appear duplicated in the outer namespace,
triggering "initializer name is not unique". PR #58 shipped a defensive
sentinel workaround that stops the disk churn but leaves the merged
graph paying ~2 s of optimization on every load.

This investigation tries the two untested approaches from issue #56's
"Possible fixes" list:
- (2) Lower opt level on the WRITE pass — does `ORT_ENABLE_EXTENDED` (or
  lower) skip whichever optimization pass causes the body→outer
  initializer duplication?
- (3) ORT's binary `.ort` format — does its serializer encode subgraphs
  differently than the `.onnx` path?

If either works we ship it; if neither does, we document and the
sentinel stays.

## Run 1 — 2026-05-16 18:07 — Probe matrix: format × opt level

Built a small Python harness (`/tmp/probe_ort_loop_cache.py`) that for
each `GraphOptimizationLevel` × file format tries `InferenceSession`
with `OptimizedModelFilePath`, then attempts to reload the written
file with `ORT_DISABLE_ALL`. Run against ORT 1.23.2 on the merged
graph (`/tmp/cb_dyn5/conditional_decoder_loop.onnx`).

| level             | `.onnx` reload | `.ort` reload |
|---|---|---|
| `DISABLE_ALL`     | RELOAD-FAIL    | **OK** (1.4 s) |
| `ENABLE_BASIC`    | RELOAD-FAIL    | **OK** (1.3 s) |
| `ENABLE_EXTENDED` | RELOAD-FAIL    | **OK** (1.2 s) |
| `ENABLE_ALL`      | RELOAD-FAIL    | **OK** (1.2 s) |

Two findings:

1. **Opt level doesn't matter.** Even `DISABLE_ALL` produces an
   unreadable `.onnx` — so the bug is in the writer itself, not a
   specific optimization pass that's hoisting body initializers
   incorrectly. (This rules out workaround #2 from issue #56 entirely.)
2. **`.ort` round-trips at every level.** The binary serializer is a
   different code path in ORT that doesn't share the bug. Reload comes
   back in ~1.2 s — well under the ~2.1 s we currently pay for cold
   re-optimization on every load. (Workaround #3 wins.)

Bonus side observation: ORT spits hundreds of `Duplicate initializer
'const_transpose_optimizer_token_NN'` warnings while loading the source
graph. These names are *created by ORT's transpose optimizer pass* —
their `const_transpose_optimizer_` prefix is internal ORT naming. I
counted initializer name duplicates in every sub-graph and in the
merged file at every scope (outer + body subgraph) — **zero
duplicates anywhere in the files we produce**. The duplicates are
purely ORT's optimization-pass output naming itself. They're also not
the round-trip cause: the failure happens at `DISABLE_ALL` where the
transpose optimizer never runs. Cosmetic noise.

**Implication:** there is no merge-script change that would help —
the bug lives entirely in ORT's `.onnx` Loop-subgraph serializer.

## Run 2 — 2026-05-16 18:24 — First implementation: always `.ort`

Trivial change to `OrtSessionBuilder`: extension `.onnx` → `.ort`,
add `session.save_model_format = "ORT"` config entry. Wiped cache,
ran cold then warm.

| graph | old `.onnx` HIT | new `.ort` HIT | delta |
|---|---|---|---|
| speech_encoder | 1129 ms | 1872 ms | +743 ms |
| embed_tokens | 32 ms | 50 ms | +18 ms |
| language_model | 875 ms | 2275 ms | **+1400 ms** |
| conditional_decoder_loop | 2098 ms (miss) | 1626 ms (HIT) | −472 ms |
| **total** | **4134 ms** | **5823 ms** | **+1689 ms (NET LOSS)** |

Re-ran for run 3 — same numbers, so it's not page-cache warmup.
The `.ort` format embeds all initializers inline (no `_data`
sidecar), which means ORT can't lazy-load / mmap weights from disk
on demand the way it does with `.onnx` external-data sidecars. For
the 2 GB LM graph that's the difference between 875 ms and 2275 ms.

**Verdict: always-`.ort` is the wrong design.** Need per-graph format
selection — use `.onnx` where it works, `.ort` only as a fallback.

## Run 3 — 2026-05-16 18:28 — Layered cache: `.onnx` → `.ort` → sentinel

Restructured `CreateCachedSession` as a 3-state machine per cache key:

1. `.onnx` (primary). Default for every graph.
2. `.ort` (fallback, marked by a `.use-ort` hint file). Selected after
   a `.onnx` round-trip failure.
3. No cache (last resort, marked by a `.cache-disabled` sentinel).
   Selected after a `.ort` round-trip failure. Untested by any real
   graph but the path exists as defense-in-depth.

Convergence:
- Run 1 (cold): write `.onnx` for all four graphs.
- Run 2 (warm): 3 graphs HIT on `.onnx`; merged Loop graph hits the
  catch, escalates — `[cache-format]` log fires, `.onnx` deleted,
  `.use-ort` hint written, and the fall-through cache-write path
  emits `.ort` *this same call* (so 2-run convergence, not 3).
- Run 3+ (warm): all four HIT on their respective formats.

Measured run 3 (steady-state, all-HIT):

| graph | format | HIT time |
|---|---|---|
| speech_encoder | .onnx | 1117 ms |
| embed_tokens | .onnx | 32 ms |
| language_model | .onnx | 824 ms |
| conditional_decoder_loop | .ort | 1776 ms |
| **total** | | **3752 ms** |

vs the PR #58 baseline of 4134 ms (where the merged Loop graph was
cache=miss at ~2100 ms every run): **382 ms saved per warm load**,
all four graphs genuinely cached.

**Verdict: shipping.** The layered design keeps fast `.onnx` lazy-load
for normal graphs and uses `.ort` only where it has to. The sentinel
infrastructure built in PR #58 is still there as a third tier; it
has not been observed to trigger and probably never will for our
current graphs, but it's cheap defense-in-depth.

## Run 4 — 2026-09-08 09:07 — Upstream re-verification (issue #60)

Question: is the ORT bug PR #59 worked around still live upstream, and is
the root cause in issue #56/#60 stated correctly? Cloned upstream ORT to
`~/Programming/onnxruntime` (shallow, HEAD `bb331b7`) and built a probe
matrix against ORT 1.23.2 / 1.24.4 / 1.29.0 (`onnxruntime-gpu` wheels in
`/mnt/data/ort-loop-repro/venv-gpu-*`).

First attempt reproduced **nothing**. The real
`conditional_decoder_loop.onnx` round-tripped cleanly:

| ORT | EP | `.onnx` reload |
|---|---|---|
| 1.24.4 | CPU | OK (4.6 s) |
| 1.23.2 | CUDA | OK (3.3 s) |

That directly contradicts Run 1, which reported RELOAD-FAIL at every opt
level on 1.23.2. Two synthetic Loop graphs (body-owned initializers;
body capturing outer-scope initializers by name, the shape
`merge_cond_decoder_loop.py` actually emits) also round-tripped fine on
1.29.0.

Negative result worth keeping: **the Loop op alone does not trigger the
bug.** Neither does the CUDA EP, nor the opt level, nor outer-scope
capture. Run 1's "bug is in the `.onnx` writer" conclusion was reached
from a probe that didn't isolate the trigger.

## Run 5 — 2026-09-08 09:09 — The actual trigger: external-initializer sidecar

The probe was missing what `OrtSessionBuilder.CreateCachedSession`
actually sets on the write pass:

```
session.optimized_model_external_initializers_file_name  = <cache>.onnx_data
session.optimized_model_external_initializers_min_size_in_bytes = 1048576
```

Adding those two entries reproduces it verbatim, at every opt level:

```
INVALID_GRAPH : ... In Node, ("", Loop, "", -1) : ... ,
Error t_now_1d_axes initializer name is not unique
```

| ORT | EP | ext-init off | ext-init on |
|---|---|---|---|
| 1.23.2 | CUDA | OK | **RELOAD-FAIL** |
| 1.24.4 | CUDA | OK | **RELOAD-FAIL** |
| 1.29.0 | CUDA | OK | **RELOAD-FAIL** |
| 1.29.0 | CPU  | OK | **RELOAD-FAIL** |

**Still live on 1.29.0, the current release** — which is also the version
`Directory.Build.props` pins for the app, so the `.use-ort` sentinel sitting
in `/mnt/data/models/chatterbox_export` is current, not stale.

The bug is EP-independent (CPU reproduces) and opt-level-independent, but
it needs the external-initializer writer. The `.onnx` serializer per se is
fine.

## Run 6 — 2026-09-08 09:10 — Minimal reproducer

`/mnt/data/ort-loop-repro/make_minimal_extinit.py` builds a ~4 MB graph
with the two features that matter:

- a Loop body owning a **small** initializer (below the 1 MB threshold, so
  it stays inline on write) — mirrors `t_now_1d_axes`, the axes input of
  an Unsqueeze in the real body
- an outer initializer **above** the threshold, which is what makes ORT
  take the external-initializer path at all

Fails identically on 1.29.0 / CPU. So the upstream report needs no 575 MB
attachment and no GPU — a self-contained script is enough.

## Run 7 — 2026-09-08 09:10 — Mechanism: duplication is *within* the body

Dumped the initializer lists of ORT's own output:

```
outer initializers:            ['big_w', 'trip_count_const', 'loop_cond_init']
body of the_loop initializers: ['t_now_1d_axes', 'body_bias',
                                't_now_1d_axes', 'body_bias']
```

Every body initializer is written **twice into the same body subgraph**.

This corrects issue #56's stated root cause ("these end up in *both* the
outer graph's initializer list AND the body subgraph's"). Nothing is
hoisted to the outer scope — the body list is simply appended to without
being cleared first. The renaming and Constant-node workarounds tried in
#56 could never have helped: any body-scope initializer duplicates.

## Run 8 — 2026-09-08 09:11 — Located the upstream defect

`onnxruntime/core/graph/graph.cc` has two sibling serializers that walk
subgraphs the same way. The custom-handler one clears first:

```cpp
// ToGraphProtoWithCustomInitializerHandlingImpl, ~line 5390
// Clear pre-existing initializers from the subgraph proto. The subgraph
// proto was populated by Node::ToProto -> Graph::ToGraphProto() const,
// which already inlined in-memory data. The recursive Impl call below
// will re-add all initializers via the custom handler, so we must clear
// to avoid duplicates.
subgraph_proto->clear_initializer();
subgraph_proto->clear_sparse_initializer();
```

`AddExternalInitializersToGraphProtoImpl` (~line 5191) has the identical
loop and **no clear** before recursing. The subgraph proto arrives already
populated by `Node::ToProto`, then the recursive call appends the same
initializers again.

So the fix upstream looks like the same two `clear_*` calls in the
external-initializer path — the codebase already documents why they're
needed a hundred lines further down.

**Implications for us:**
- The `.ort` fallback shipped in PR #59 stays correct and stays necessary.
- A cheaper workaround now exists that we didn't know about: writing the
  `.onnx` cache *without* the external-initializer entries round-trips
  fine. Only worth taking for graphs that fit under protobuf's 2 GB limit
  — which `conditional_decoder_loop` (575 MB) does. Worth measuring
  against the `.ort` path's load time before changing anything.
- Issue #60's draft upstream body needs rewriting: its affected-versions
  line, its reproducer, and its root-cause paragraph are all wrong in
  ways that would get the report bounced.

## Run 9 — 2026-09-08 09:14 — Blast radius: which graphs actually fell back

Swept `/mnt/data/models` for the sentinels the layered cache writes:

```
chatterbox_export/conditional_decoder_loop.opt.cuda.b21b9e70895b.use-ort
nfa_ctc_onnx/nemo128.opt.cuda.df335d933882.use-ort   (May 17)
nfa_ctc_onnx/nemo128.opt.cuda.912d0b9335e7.use-ort   (Sep 7)
```

No `.cache-disabled` anywhere — the third tier still has never fired.

**`nemo128.onnx` is a second affected graph, and nobody noticed.** It's the
NeMo log-mel preprocessor in the NFA CTC bundle (and the same op contract
the Parakeet export uses). Its failure names a different op:

```
In Node, ("/If", If, "", -1) : ... , Error /Constant_34_output_0
initializer name is not unique
```

Two corrections to the mental model from that:

1. **It isn't a `Loop` bug — it's a subgraph bug.** `If` branches duplicate
   the same way. Any control-flow op with a body qualifies.
2. `nemo128.onnx`'s `If` branches ship with **empty** initializer lists.
   The duplicated `/Constant_34_output_0` is a constant ORT itself folded
   into the branch, and it happens at `DISABLE_ALL` — so ORT inserts
   branch-scope constants even with optimization nominally off. That's the
   detail the `merge_cond_decoder_loop.py` docstring guessed at back in May
   and couldn't confirm.

Issue #60's title and draft body both need to say "subgraph", not "Loop".

## Run 10 — 2026-09-08 09:15 — The perf path that was never measured

With the trigger known, there's a third cache disposition the earlier runs
never tried: write `.onnx` but **without** the external-initializer session
entries. Run 1 ruled out `.onnx` wholesale, so this was never on the table.

`conditional_decoder_loop.onnx`, ORT 1.29.0 / CUDA, best of 3:

| disposition | write | warm load | cache size |
|---|---:|---:|---:|
| miss (no cache) | — | 1906 ms | — |
| `.ort` (**ships today**, PR #59) | 3024 ms | 1550 ms | 577 MB |
| `.onnx`, no ext-init (**untested until now**) | 1980 ms | **1303 ms** | **168 MB** |

**247 ms faster per warm load than the shipping fallback, and 409 MB
smaller on disk.** The size collapse is the mechanism: without the
ext-init entries ORT leaves the optimized graph pointing at the *source*
model's existing `.onnx_data` sidecar rather than copying 577 MB of
weights into a new file, so the warm load mmaps weights it was going to
mmap anyway. `.ort` embeds everything inline, which is exactly the
lazy-load loss Run 2 measured on the LM graph.

`nemo128.onnx` for completeness: miss 16 ms, `.ort` warm 11 ms, `.onnx`
warm 10 ms. Real but worthless — a 1 ms graph. The fallback costs nothing
there and only the Chatterbox graph is worth changing anything for.

Caveat: this measures **load**, not numerical correctness. A
`ChatterboxSmoke` parity run against the no-ext-init cache is required
before this ships.

**Why ext-init can't simply be dropped globally:** `language_model.onnx`
has a 6.1 GB weight sidecar and its optimized cache writes a 2.0 GB one.
Without ext-init that write hits protobuf's 2 GB message limit. The
entries are load-bearing for large graphs and merely harmful for
subgraph-bearing ones.

Suggested shape, if we act on this: insert a tier, so the ladder becomes
`.onnx` + ext-init → `.onnx` no-ext-init → `.ort` → sentinel. The
no-ext-init tier catches every subgraph-bearing graph under 2 GB (both
known cases), and a >2 GB subgraph-bearing graph would fail its write and
fall through to `.ort` as today.

## Run 11 — 2026-09-08 09:24 — Implementation: inline `.onnx` tier

Added a third rung to `OrtSessionBuilder`'s ladder, between the primary
and `.ort`: same `.onnx` format, written *without* the two
external-initializer session entries, marked by a `.no-ext-init` hint.

```
.onnx + sidecar  →  .onnx inline  →  .ort  →  .cache-disabled
```

Two safety properties the new rung needs, both implemented:

- **Write can fail on its own terms.** Without the sidecar, a graph whose
  optimized initializers exceed protobuf's 2 GB message limit can't
  serialize at all — `language_model.onnx` writes a 2.0 GB sidecar and is
  exactly that case. The inline write is wrapped so a failure escalates to
  `.ort` instead of surfacing a cache-write failure as a model-load
  failure. (`language_model` never reaches the tier — it round-trips fine
  on tier 1 — but a future >2 GB subgraph-bearing graph would.)
- **Legacy hint migration.** A `.use-ort` with no `.no-ext-init` beside it
  was written by the old two-tier ladder. Retried once on the inline tier,
  costing one cache miss; the hint left behind stops it repeating. Without
  this, every machine that ran the old code keeps its Chatterbox graph on
  the slow tier forever, since cache keys only invalidate on ORT upgrade
  or model change.

Convergence, from a cleared cache (ChatterboxSmoke, CUDA):

| run | what happened | cond-decoder load |
|---|---|---|
| 1 | write `.onnx` + sidecar | 2143 ms (miss) |
| 2 | reload fails → `[cache-format]` → rewrite inline, same call | 2153 ms (miss) |
| 3+ | **cache HIT** | **1431 ms** |

Seeded a legacy `.use-ort` and confirmed the migration path separately:
run 1 logs the retry and misses (2012 ms), run 2 HITs (1508 ms).

Steady state, all four graphs — no regression on the three that never
left tier 1:

```
speech_encoder.onnx           1057 ms  cache=HIT
embed_tokens.onnx               30 ms  cache=HIT
language_model.onnx            408 ms  cache=HIT
conditional_decoder_loop.onnx 1395 ms  cache=HIT
Loaded sessions in 2896 ms total
```

vs **1776 ms** for the same graph on `.ort` in Run 3. Cache on disk for
that graph: **168 MB, down from 577 MB.**

## Run 12 — 2026-09-08 09:29 — Parity, and the non-determinism floor

Load time proves nothing about numerics, so: 3 runs on the inline cache
vs 3 cache-bypassed baselines, same voice and text, comparing the output
waveforms.

First attempt looked alarming — every pair differed, including two
cache-bypassed runs compared against *each other*. That's the answer,
not the problem: this pipeline is not bitwise reproducible run to run
(CUDA non-determinism, and `ScatterND with reduction=='none'` warns about
exactly this). So the comparison has to be against the floor, not zero.

| comparison | max abs diff | mean abs diff |
|---|---|---|
| no-cache vs no-cache (floor) | 7.265e-02 | 6.624e-04 |
| inline-HIT vs inline-HIT (floor) | 7.395e-02 | 7.757e-04 |
| **inline-HIT vs no-cache** | 9.249e-02 | 8.574e-04 |

Cross-configuration difference sits in the same order as the pipeline's
own run-to-run spread, and waveform RMS agrees to 5 significant figures
(0.046366 / 0.046361 / 0.046363). **The inline cache introduces no
deviation beyond noise the pipeline already has.** It is not a proof of
bitwise equivalence, and no such proof is available here.

`dotnet test tests/Vernacula.Tests`: 40/40 pass.

**Status:** shipped in the working tree, not committed. Issue #60 stays
open — it's about reporting the bug upstream, and the upstream defect
(missing `clear_initializer()` in `AddExternalInitializersToGraphProtoImpl`,
Run 8) is untouched by any of this. What changed is that we no longer pay
for it.
