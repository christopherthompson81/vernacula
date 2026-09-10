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
nothing but torch). Verbatim:

```
torch 2.11.0+cu128

  case                       k             span   gaps  contiguous  lowest-index?
  zeros(n=12    )            5            6..10      0        True  DIFFERS
  zeros(n=312   )           35         196..233      1       False  DIFFERS
  zeros(n=312   )           70         157..233      2       False  DIFFERS
  zeros(n=1248  )          188         664..935      4       False  DIFFERS
  zeros(n=2000  )          188       1251..1499      1       False  DIFFERS

  50 distinct + 1198 tied, k=188: keeps 50/50 of the distinct entries, then 138 tied ones spanning 649..947

  20 repeats in one process: stable
  threads 1/2/4 first index: [664, 664, 664]  (thread-independent)

  If any row says MATCHES, the tie order differs by platform and #171's premise
  is unfixable by construction. If all say DIFFERS, this build agrees with the
  Linux measurement and the ports' rule is arbitrary-but-deterministic, as documented.
```

**The premise is false, and not just at scale — the #170 fixture itself does not reproduce
here.** `torch.topk(torch.zeros(12), 5)` returns `[6,7,8,9,10]` on this build, not `[0..4]`.

Checked the `sorted=` flag too, since NeMo calls `topk(..., sorted=False)` at both sites and
#170 may have tested the default: it makes no difference. `zeros(12)` k=5 gives `6..10` and
`zeros(1248)` k=188 gives `664..935` either way. That flag orders the values that come
*back*; it does not change which are chosen.

What torch returns is a **mid-range subset** of the tied entries, at an offset following no
rule (n=12,k=5 → 6; n=312,k=35 → 196; n=312,k=70 → 157; n=1248,k=188 → 664; n=2000,k=188 →
1251), and — apart from the smallest case — **not contiguous**: 1, 2, 4 and 1 gaps
respectively. That is the signature of a quickselect partition, not of a documented
ordering guarantee. It is stable across 20 repeats and identical at 1, 2 and 4 threads.

> ⚠ The first version of this entry said "a contiguous block from the middle", with spans of
> `196..230` and `664..851`. Both wrong: the sets have holes, and those endpoints were
> `start + k` inferred from a script that printed only the first eight indices, not read off
> the data. Caught in review of the PR — which is a pointed failure, since the whole finding
> is that #170 recorded a `topk` claim it had not checked. The probe now prints `gaps` and
> `contiguous` precisely so a span alone can never be mistaken for a set again.

The obvious suspicion is that #170's `[0..4]` came off a different torch build — that work
was done on the Apple Silicon machine, which has a different CPU kernel. If so, the tie order
is **platform-dependent**, and "match NeMo's tie-breaking" is not a well-posed target: there
is no single answer to match. Unverified from here; it needs one command on the Mac (below).

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
| 50% | 77.2 | 0.0 |
| 85% | 218.8 | 122.4 |
| 100% | 327.4 | 126.1 |

> ⚠ The boost column first read 84.8 / 257.6 / 350.5, because both boost passes were scored
> against `boost_latest`. The weak pass does not see that matrix — it top-k's the
> *strong-boosted* scores, where 33 entries per speaker have already moved up by 2·log 2,
> which shifts the weak cut and its tie multiplicity. Caught in review; the numbers above are
> the corrected ones. The zero rows are unaffected either way — there are no exact ties at
> all below 50% — so the safety argument this run exists to make never depended on it.

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

## Review pass — 2026-09-09

A `/code-review high` pass over the branch found no functional defect in `Sortformer.cs` —
it read `SelectCacheFrames` as a faithful hoist with only the tie comparator changed — but
it found that **the evidence this work ships was itself partly unverified**, which for this
particular change is the worst place to be wrong. Eight findings, all fixed:

* **The replacement `torch.topk` claim was as unchecked as #170's.** "A contiguous block
  from the middle", spans `196..230` and `664..851`. The sets have holes, and those
  endpoints were `start + k` inferred rather than read. Corrected in Run 1 above, in three
  comments, and in the probe, which now prints `gaps` and `contiguous`.
* **`--probe-topk-ties` could not detect the property it exists to establish**: it printed
  only `first..last`, which is identical for a contiguous block and for the scattered set
  torch returns — which is exactly how the wrong claim survived. That is the command the
  Mac is told to run to close #171, so it now prints length, gaps, a contiguous flag, a
  partial-tie case and the thread sweep.
* **The weak-boost tie count used the wrong matrix** (Run 4, corrected above).
* **Run 1's transcript was not the shipped tool's output** — lettered sections and rows the
  probe never produced. It is now pasted verbatim from the probe.
* `boundary_tie`'s `(v > cut).sum() >= k` guard was unreachable, since `cut = v[k-1]` on a
  descending sort means at most `k-1` entries can exceed it. Replaced with the check that
  was meant: the cut is unambiguous when exactly `k` entries are `>=` it.
* `SelectCacheFrames` was documented "Pure and deterministic" while sorting the caller's
  array in place. The one production caller builds it fresh, but this file pools buffers
  elsewhere, so the remark now says so.
* **`HighestScoresWin_AndKeptRowsComeBackSpeakerMajor` asserted a tautology**: it recomputed
  `sIdx * frames + tIdx`, the very key the entries were sorted by, on a fixture with no
  `-inf` and no pad rows, so no entry could take the `max_index` branch and the assertion
  could not fail for any implementation. The fixture now starves the cache (160 live entries
  for 188 rows) so `-inf` picks actually fire — verified by watching the new version fail
  before the fixture was fixed.
* `interchangeable` was annotated `-> tuple[int, int]` and returns `(int, bool)`.

The earlier self-review, before that pass, had already turned up four more:

* `gather_outputs` in the new harness was never called and would have thrown if it were —
  it reads `flat_scores`, bound to `None` on the line above. Deleted; `interchangeable`
  already answers what it was for.
* The harness handed NeMo `torch.from_numpy(preds)`, which **aliases** the fixture that is
  read again afterwards. Now a copy. Checked afterwards whether it had actually mattered —
  it had not, NeMo does not mutate `preds` at either saturation level — so **the row counts
  in Runs 2–3 stand**; the copy is hardening, not a correction.
* The per-speaker mix table in Run 3 was only reproducible from a scratch script. The
  shipped harness prints it for both sides now.
* `SelectCacheFrames` validates `keep` against the array length. `CompressCache` satisfies
  it by construction, but the method is independently reachable now.

Still open, and cheap for whoever is next on the Apple Silicon machine:

```bash
python scripts/nemo_export/sortformer_compress_parity.py --probe-topk-ties
``` If `torch.topk(torch.zeros(12), 5)` returns `[0..4]` on
that build, the platform-dependence hypothesis is confirmed outright and #171 can be closed
with certainty rather than with strong evidence.
