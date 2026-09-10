# Sortformer fp16 — issue #172

#172: the model card records fp16 as the fastest Sortformer variant measured (22.3 ms/chunk)
but not shipped, because `chunk_pre_encode_embs` diverged to 9.8e-02 and those embeddings
feed back into the speaker cache and FIFO — so it needed an end-to-end check rather than a
single-chunk comparison. That check now exists (`sortformer_fidelity_der.py`, #169). What
did not exist was a **loadable fp16 model to measure**.

## Environment

- `.venv-nemo-export`, python 3.12.3, torch 2.11.0+cu128, onnx 1.21.0, **onnxruntime 1.26.0**,
  onnxconverter-common 1.16.0 (installed for this work), Linux x86-64, RTX 3090
- `main` @ `4e3d13b`

## Run 1 — 2026-09-09 — reproduce the failure

```python
float16.convert_float_to_float16(onnx.load(SRC), keep_io_types=True)
```

Converts in ~3 s, 492 MB → 247 MB, and will not load:

```
Type Error: Type (tensor(float16)) of output arg (/pre_encode/conv/Cast_output_0)
of node (/pre_encode/conv/Cast) does not match expected type (tensor(float))
```

Slightly different node from the one #172 records (`/pre_encode/conv/Mul_3`), same region.

## Run 2 — 2026-09-09 — what is actually in that region?

Question: #172 calls it "mixed precision surviving in the pre-encode convolution stack" and
suggests `op_block_list` tuning. Which ops, and why those?

Walked the graph back from the `chunk_pre_encode_lengths` output:

```
nodes feeding chunk_pre_encode_lengths: 39
  ops: Constant x15, Cast x9, Add x9, Div x3, Sub x3
nodes feeding chunk_pre_encode_embs: 514  (37 shared with the length path)
```

**Not activations at all.** That is `floor((L + 2p - k)/s) + 1` — the conv output-length
formula — emitted by the tracer as float arithmetic, once per pre-encode conv layer. So the
first fix is not a tuning knob but a correctness point: **a frame count must not be rounded
through fp16.** fp16 represents integers exactly only to 2048, and the intermediate divisions
are not integers. Held in fp32 by walking the graph rather than by hardcoding names, so it
survives the exporter renaming nodes.

That alone does not fix it — the next failure is a different `Cast`.

## Run 3 — 2026-09-09 — the converter does not update `Cast` nodes

Question: with the length path protected, why is a `Cast` still wrong?

Compared every `Cast`'s `to` attribute against the converted graph's own `value_info`:

| | Cast nodes | `to=` | disagreements |
|---|---|---|---|
| source | 141 | FLOAT×31, INT64×53, BOOL×57 | **0** |
| naive fp16 | 188 | FLOAT×52, INT64×53, BOOL×57, FLOAT16×26 | **27** |

`onnxconverter-common` rewrites tensor types but leaves each explicit `Cast` declaring
`to=FLOAT`, so 27 of them claim to produce fp32 while the graph types their output fp16.
This is the concrete mechanism behind the playbook's vaguer note that the converter "fights
graphs containing explicit `Cast` nodes". Fixed by trusting the converted types and
reconciling `to`.

## Run 4 — 2026-09-09 — one boundary left, and it is not what it looks like

After the reconciliation, exactly one node has mixed float inputs — and it is the `Mul`
#172 originally reported:

```
nodes with MIXED float input types: 1
  /Mul_7   Mul   inputs=['FLOAT', 'FLOAT16']

node: /Mul_7 -> out FLOAT16
  input chunk_pre_encode_embs   type=FLOAT     produced_by Cast(/pre_encode/out/Add_cast_to_chunk_pre_encode_embs)
  input /Cast_6_output_0        type=FLOAT16
```

The fp32 input is `chunk_pre_encode_embs` — **a graph output**. It is also consumed
internally, feeding onward into the encoder. `keep_io_types=True` appends a `Cast` so the
caller still receives fp32, and the internal consumer then reads that fp32 tensor while
expecting fp16.

So the fix is not to cast the input (an fp16→fp32→fp16 round trip); it is to rewire internal
consumers back to the **pre-cast** fp16 tensor and let the cast serve only the graph output.
Lossless, and it generalizes to any tensor that is both an output and an input.

`keep_io_types` is kept deliberately: `Sortformer.cs` feeds float32 and reads float32, so an
fp32-io fp16 model is a drop-in and the runtime needs no variant path.

## Run 5 — 2026-09-09 — loads, then dies on the first inference

All three passes applied: loads at `ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC` and
`ORT_ENABLE_ALL`. Then:

```
RUNTIME_EXCEPTION : Non-zero status code returned while running ScatterElements node.
CPU execution provider: MLFloat16 data type is not supported with ScatterElements
opset 16 when reduction is 'add'.
```

A model that loads is not a model that runs. `ScatterElements` is held in fp32 too — one
node, cast on each side. Other EPs may have the kernel, but the model has to run on CPU as
well and a per-EP artifact is not worth one op.

Result: `scripts/nemo_export/fp16_convert_sortformer.py`, 492 MB → 247 MB, loading at every
optimization level and running. Single-chunk parity against fp32 on the CPU EP:

```
spkcache_fifo_chunk_preds  maxAbs=9.825E-04  rms=1.970E-04
chunk_pre_encode_embs      maxAbs=9.764E-02  rms=8.629E-03
```

(Both sessions run at `ORT_ENABLE_BASIC`. An earlier `--compare` left the reference at ORT's
default `ENABLE_ALL` while the candidate ran at `DISABLE_ALL`, which folded ORT's own fusions
into the difference and read `preds` as 6.7e-04. Same level on both sides makes the gap
attributable to the dtype — and it is also what lets `--compare` run at all on the CoreML
variant, which cannot be re-optimized above BASIC.)

`embs` at **9.764e-02** reproduces the model card's 9.8e-02 to three digits, so this is the
same artifact the original caveat was about — the card's number is now reproducible rather
than folklore.

## Run 6 — 2026-09-09 — the go/no-go: end-to-end DER

```bash
python scripts/nemo_export/sortformer_fidelity_der.py \
  --audio <three 90 s en-US samples> --max-seconds 90 \
  --nemo <checkpoint> --onnx diar_streaming_sortformer_4spk-v2.1.fp16.onnx
```

```
  en-US_sample_01   90.0s  ref 29 seg / 2 spk   hyp 29 seg / 2 spk
  en-US_sample_02   90.0s  ref 25 seg / 2 spk   hyp 25 seg / 2 spk
  en-US_sample_03   90.0s  ref 22 seg / 2 spk   hyp 22 seg / 2 spk

fidelity DER vs NeMo: DER 0.000 %  confusion 0.000 %  FA 0.000 %  missed 0.000 %
```

**Passes.** Segment and speaker counts match NeMo exactly on all three.

## Run 7 — 2026-09-09 — but does the error compound? (DER is too coarse to say)

DER applies a median filter, a collar and binarization, so 0.000% is consistent with real
per-frame drift being absorbed. The worry on file is specifically *compounding* through the
spkcache/FIFO feedback, which needs the raw posteriors. `fp16_convert_sortformer.py --drift`
runs the ported loop twice on the same audio — fp32 model and fp16 model — and diffs per
chunk:

| sample | per-frame `preds` maxAbs | rms | frames whose binarized speaker set differs |
|---|---|---|---|
| en-US_sample_01 | 2.168E-02 | 1.356E-03 | 0 / 1126 |
| en-US_sample_02 | 1.080E-02 | 4.627E-04 | 0 / 1126 |
| en-US_sample_03 | 3.363E-02 | 1.573E-03 | **3** / 1126 |

Per-chunk maxAbs, sample 01: `7.6E-4  4.7E-3  3.9E-3  1.3E-3  9.3E-3  4.6E-3  1.5E-2  2.2E-2  9.6E-3  8.9E-3`

Two things worth separating. The error **does** grow through the feedback path — 9.8e-04 on
a single chunk becomes 1.1e-02 to 3.4e-02 over a recording, a factor of 10-35. But it
**wanders rather than accumulating**: the per-chunk trace rises and falls, and does not
trend. Bounded, not compounding.

And it is not entirely invisible: 3 frames of 3378 flip their binarized speaker set. All
three are isolated, so the median filter absorbs them and DER stays at 0.000%. That is the
honest form of the result — "0.000%" alone would overstate it.

## Run 8 — 2026-09-09 — a latent bug in the rewiring pass

Self-review before committing: `rewire_output_casts` keyed on "this graph output is produced
by a `Cast`", which is not the same thing as "the converter added this cast". Sortformer's
`chunk_pre_encode_lengths` output is produced by a legitimate `Cast(to=INT64)` the model
itself contains — and it *is* consumed internally, since the attention mask is built from
it. So the pass was rewiring the mask's frame-count input to the pre-cast float tensor: a
silent semantic change dressed as a dtype tidy-up.

Restricted to casts that are fp16 → fp32, which is the `keep_io_types` signature. Rewired
edges drop 4 → 2, i.e. **two of the four were the int64 lengths output.**

Every measured number is byte-identical before and after (`9.8E-04` / `9.76E-02`, DER
0.000%, the same drift table), so the bug was latent — a downstream cast evidently absorbed
it. Recorded because "the numbers didn't change" is exactly the argument that would have let
it ship, and the next graph it ran on might not be so forgiving.

## Run 9 — 2026-09-09 — is it faster? Only on some hardware

Steady-state shapes, `ORT_ENABLE_BASIC`, median of 15:

| model | EP | load | median | min |
|---|---|---|---|---|
| fp32 | CPU | 1.79 s | 333.0 ms | 327.7 ms |
| fp16 | CPU | 1.97 s | **415.3 ms** | 409.7 ms |
| fp32 | CUDA | 1.90 s | 17.6 ms | 16.4 ms |
| fp16 | CUDA | 0.89 s | **10.6 ms** | 10.5 ms |

**fp16 is 1.66× faster on CUDA and 25% SLOWER on CPU.** The CPU result is the expected one —
there are no native fp16 CPU kernels, so ORT casts up and back around every op and pays for
the traffic. It also halves CUDA load time (1.90 s → 0.89 s), which tracks the file being
half the size.

So fp16 is not a replacement for the fp32 model; it is an **execution-provider-gated
variant**. Shipping it as the default would slow down every CPU user.

## Where this leaves #172

1. **A valid fp16 export exists.** `fp16_convert_sortformer.py`, three graph passes plus one
   op held back, loads at every optimization level and runs. ✅
2. **The end-to-end check passes.** DER 0.000% on three 90 s real-speech samples, with the
   caveat from Run 7 stated rather than buried. ✅
3. **The speed claim is hardware-dependent**, and the 22.3 ms on file is unverified from
   here — it is a CoreML number. Measured: CUDA 1.66× faster, CPU 25% slower. ⚠

Not published, and not wired into the runtime. Two things gate that, and both are decisions
rather than work:

* **The CoreML number needs re-measuring on the Apple Silicon machine**, against the fp16
  build of the CoreML variant (the converter handles it — verified, 264 MB, loads at
  `ORT_DISABLE_ALL` and `ORT_ENABLE_BASIC`; its `ORT_ENABLE_ALL` failure is the
  `MatMulAddFusion` one that variant already has, not an fp16 problem). If 22.3 ms holds
  there, fp16-on-CoreML is the fastest path on that platform by a wide margin.
* **Shipping it means EP-gated model selection** — CUDA and possibly CoreML get the fp16
  file, CPU keeps fp32 — which is the same shape of runtime work as the CoreML variant's
  selection path, and should probably be done once for both rather than twice.
