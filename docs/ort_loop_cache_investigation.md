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
