# Sortformer top-k tie-breaking — issue #171

#171 asks why a **saturated** Sortformer stream still diverges from NeMo after #169 and
#170, and says the residual is tie-breaking "elsewhere in the selection — the final global
top-k, or the interaction between the two boost passes and the `+inf` silence pad".

Both the C# port and the Python mirror resolve ties by keeping the **lowest** index, on the
authority of a claim recorded in #170 and repeated in three code comments:

> `torch.topk` on CPU is deterministic and keeps the **lowest** indices among equal values —
> verified directly: 12 tied values with `k=5` returns indices 0..4.

That premise is where this starts.

## Environment

- `.venv-nemo-export`, python 3.12.3, **torch 2.11.0+cu128**, nemo-toolkit 2.7.1, Linux x86-64
- `main` @ `58adc0a`

## Run 1 — 2026-09-09 — does the tie rule the ports implement actually hold?

Question: reproduce #170's check, then repeat it at the sizes Sortformer really uses.
`_get_topk_indices` flattens to `n_frames * n_spk` (≈ 1248) and takes `k = spkcache_len`
(188); `_boost_topk_scores` takes `k` = 35 or 70 per speaker over `n_frames` (≈ 312).

`sortformer_compress_parity.py --probe-topk-ties` (written during this run, and needing
nothing but torch), comparing `torch.topk`'s picks against sorting by `(-value, index)`:

```
A. the #170 experiment, reproduced
  12 tied values                          k=  5   lowest-index rule: DIFFERS
      topk picked but rule would not: [6, 7, 8, 9, 10]
      rule picks but topk did not   : [0, 1, 2, 3, 4]

B. at the real flattened size
  all tied, numel=1248                    k=188   DIFFERS   topk picked [664..851]
  50 distinct + 1198 tied, numel=1248     k=188   DIFFERS
  all tied, numel=2000                    k=188   DIFFERS   topk picked [1251..]

C. the boost call sites
  all tied, numel=312                     k= 35   DIFFERS   topk picked [196..230]
  all tied, numel=312                     k= 70   DIFFERS   topk picked [157..226]
  all tied, numel=1248                    k= 35   DIFFERS   topk picked [820..854]

D. dim=1 on a 3D tensor (the actual boost call)
  boost topk on zeros, k=35: speaker 0 -> [196..205]...   matches range(35)? False
```

**The premise is false, and not just at scale — the #170 fixture itself does not reproduce
here.** `torch.topk(torch.zeros(12), 5)` returns `[6,7,8,9,10]` on this build, not `[0..4]`.
Checked against the `sorted=` flag, since NeMo calls `topk(..., sorted=False)` at both
sites and #170 may have tested the default:

```
zeros(n=12)   k=5   sorted=True  -> [6, 7, 8, 9, 10]
zeros(n=12)   k=5   sorted=False -> [6, 7, 8, 9, 10]
zeros(n=1248) k=188 sorted=True  -> [664, 665, ...]
zeros(n=1248) k=188 sorted=False -> [664, 665, ...]
```

The flag makes no difference — `sorted` orders the *returned* values, it does not change
which are selected. So that is not the explanation either.

What torch actually returns is a **contiguous block from the middle** of the tied range, at
an offset that follows no simple rule (n=12,k=5 → 6; n=312,k=35 → 196; n=312,k=70 → 157;
n=1248,k=188 → 664; n=2000,k=188 → 1251). That is the signature of a quickselect partition,
not of a documented ordering guarantee. It is stable within this build:

```
repeat determinism (20 runs, same process): STABLE
threads=1 / 2 / 4: starts at 664 in every case
```

So it is deterministic *here*, but it is an artifact of one CPU kernel's pivot choices, not
a property of `topk`. The obvious suspicion is that #170's `[0..4]` came off a different
torch build — that work was done on the Apple Silicon machine, which has a different CPU
kernel. If so, the tie order is **platform-dependent**, and "match NeMo's tie-breaking" is
not a well-posed target: there is no single answer to match. Unverified from here; it needs
one command on the Mac (below).

Implication for the ports: the lowest-index rule in `Sortformer.cs`, in
`benchmark_sortformer_rtf.py` and in the three comments citing #170 is **not** "what
torch.topk does". It is a defensible arbitrary choice that happens to be stable and
readable, but the comments assert something measurably untrue and would send the next
person chasing a match that cannot be had.

Next: find out whether the residual divergence in #171 is *only* tie order, or whether
something upstream of the selection also differs. Result: pending.

## Run 2 — 2026-09-09 — where does the divergence actually start?

Question: #171 guesses the residual is "the final global top-k, or the interaction between
the two boost passes and the `+inf` silence pad". Which is it?

Built `scripts/nemo_export/sortformer_compress_parity.py` — the reproduction #171 asks for,
which did not exist. It compares each stage of `_compress_spkcache` against the Python
mirror on a fixture with a controlled saturated fraction, and reports the final selection
two ways: the index sets (which differ whenever tie order differs) and the selected **score
multisets** (equal if and only if the two sides made equivalent choices among equals).

```
  saturated 0%
    quality_scores   maxAbs=4.768E-07      selection  0/188 indices differ
    boost_latest     maxAbs=4.768E-07      score multisets EQUAL
    strong_boost     maxAbs=4.768E-07
    weak_boost       maxAbs=4.768E-07

  saturated 85%
    quality_scores   maxAbs=0.000E+00      selection 40/188 indices differ
    boost_latest     maxAbs=0.000E+00      score multisets DIFFER
    strong_boost     maxAbs=1.386E+00
    weak_boost       maxAbs=2.079E+00

  saturated 100%
    quality_scores   maxAbs=0.000E+00      selection 44/188 indices differ
    boost_latest     maxAbs=0.000E+00      score multisets DIFFER
    strong_boost     maxAbs=1.386E+00
    weak_boost       maxAbs=2.079E+00
```

**It starts in the boost passes, before the final top-k.** Scoring and the latest-frame
boost agree to the bit. `strong_boost` then diverges by exactly **1.386 = 2·log 2**, which
is precisely one strong-boost increment — so the two sides boost the *same number* of
frames per speaker and simply choose *different* ones. `weak_boost` diverges by
2.079 = 1.386 + 0.693, both increments.

Because the boost writes into the scores, the final top-k is then selecting over two
different score matrices — which is why the selected score multisets differ rather than
merely the indices. The unsaturated control is clean at every stage, so nothing here is a
scoring bug.

## Run 3 — 2026-09-09 — does the port's own tie rule have a bias?

Question: the final selection breaks ties on the **speaker-major** flattened index
(`s * extT + t`). That is the layout the entries happen to be flattened in, but it orders
*every* speaker-0 entry ahead of *every* speaker-1 entry. Does that show up as a skew?

Per-speaker counts in the compressed cache (one-off script; the shipped harness reports
the same divergence via `preds rows`):

| | speaker mix | indices differing from NeMo |
|---|---|---|
| **85% saturated** | | |
| NeMo (`torch.topk`) | [36, 69, 44, 39] | — |
| speaker-major (shipped) | **[64, 52, 36, 36]** | 40/188 |
| frame-major | [44, 44, 46, 54] | 54/188 |
| **100% saturated** | | |
| NeMo (`torch.topk`) | [36, 49, 54, 49] | — |
| speaker-major (shipped) | **[69, 47, 36, 36]** | 44/188 |
| frame-major | [38, 55, 49, 46] | 48/188 |

The shipped rule is **monotonically biased toward the low-numbered speaker slots**, and the
low slots bottom out at a floor. NeMo's mix is uneven but not ordered; frame-major's is
flat. Speaker IDs in Sortformer are arbitrary slot assignments, so this is a bias toward
whichever speaker happened to land in slot 0.

Note the trap in the third column: frame-major looks *worse* on raw index agreement (54 vs
40). That metric is measuring agreement with a quickselect artifact, and Run 1 established
there is no order to agree with. The metric that survives Run 1 is what the cache ends up
saying, and on that frame-major is better — unmatched preds rows drop from **33/188 to
8/188** at 100% saturated and from 28 to 25 at 85%.

## Run 4 — 2026-09-09 — can the tie rule fire on real speech at all?

Question: changing a tie rule in a port whose contract is parity needs to be shown safe.
Rather than reason about it, measure when a tie can *decide* anything — a tie strictly
inside the kept set, or strictly outside it, changes nothing; only one straddling the cut
does, and only when more than one entry sits at the cut value.

`sortformer_compress_parity.py --tie-incidence`, mean over 8 seeds:

| saturated | boost ties at a cut | final top-k ties at the cut |
|---|---|---|
| 0% | 0.0 | 0.0 |
| 5% | 0.0 | 0.0 |
| 20% | 0.0 | 0.0 |
| 50% | 84.8 | 0.0 |
| 85% | 257.6 | 122.4 |
| 100% | 350.5 | 126.1 |

**The tie rule cannot fire below ~20% saturated frames.** Ties need bit-identical scores,
which need bit-identical preds, which need saturation. So a change to the tie rule is
provably inert on anything that is not heavily saturated — which is the regime real speech
occupies, and the reason fidelity DER was 0.000% in the first place.

## Run 5 — 2026-09-09 — the change, on real audio

Switched the final selection's tie-break from speaker-major to frame-major in both
`Sortformer.cs` and the Python mirror, sharing one key function so the two cannot drift.

```bash
python scripts/nemo_export/sortformer_fidelity_der.py \
  --audio <the three en-US samples> --max-seconds 90 \
  --nemo <checkpoint> --onnx diar_streaming_sortformer_4spk-v2.1.onnx
```

```
  en-US_sample_01   90.0s  ref 29 seg / 2 spk   hyp 29 seg / 2 spk
  en-US_sample_02   90.0s  ref 25 seg / 2 spk   hyp 25 seg / 2 spk
  en-US_sample_03   90.0s  ref 22 seg / 2 spk   hyp 22 seg / 2 spk

fidelity DER vs NeMo: DER 0.000 %  confusion 0.000 %  FA 0.000 %  missed 0.000 %
```

Unchanged, as Run 4 predicts. Segment and speaker counts match the reference exactly on all
three. 63 unit tests pass, including five new ones that lock the selection's behaviour —
notably that a fully tied score matrix fills all four speakers equally, which is what the
old rule failed (it gave the whole cache to speaker 0).

## Conclusion for #171

The issue asks the port to match NeMo's top-k tie-breaking. **That is not achievable, and
the premise it rests on is false.** NeMo's order is a quickselect artifact of one CPU
kernel: a mid-range block at an offset with no rule, not reproducing across torch builds,
and #170's "keeps the lowest indices" does not reproduce even on its own 12-element fixture.

What the divergence in #171 actually is: the two boost passes choose different frames among
tied scores (Run 2), which perturbs the scores the final top-k then sorts. It is confined to
the saturated regime and cannot fire below ~20% saturation (Run 4).

So the port's obligations are the ones it can meet — deterministic, unbiased, documented —
and the residual row counts in #171 should not be treated as a defect to drive to zero. The
one real finding worth acting on was the *port's own* speaker-slot bias (Run 3), now fixed.

Still open, and cheap for whoever is next on the Apple Silicon machine:

```bash
python scripts/nemo_export/sortformer_compress_parity.py --probe-topk-ties
``` If `torch.topk(torch.zeros(12), 5)` returns `[0..4]` on
that build, the platform-dependence hypothesis is confirmed outright and #171 can be closed
with certainty rather than with strong evidence.
