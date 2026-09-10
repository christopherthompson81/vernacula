#!/usr/bin/env python3
"""Compare speaker-cache compression against NeMo's, stage by stage, on a fixture
with a controlled fraction of SATURATED frames.

This is the reproduction issue #171 asks for. Real speech does not saturate hard
enough to move fidelity DER, so the divergence #171 tracks is only visible on a
fixture that forces the regime: Sortformer's float32 sigmoid reaches exactly 1.0
above ~16.6 logits, so confident frames give bit-identical preds rows, hence
bit-identical scores, and the top-k's choice among them is decided by tie order.

Comparing only the final output cannot tell "the ports pick a different tied
frame" (harmless, arbitrary on both sides) apart from "the ports compute a
different score" (a real bug). So each stage of `_compress_spkcache` is compared
separately, and the final selection is reported two ways:

  * the INDEX sets, which differ whenever tie order differs, and
  * the selected SCORE MULTISETS, which are equal if and only if the two sides
    made equivalent choices among equals.

A run where every score stage matches to 0.0 and the score multisets are equal,
but the index sets differ, means the remaining divergence is tie order and
nothing else.

⚠ The reference is NeMo itself, rebuilt at Vernacula's streaming schedule --
never benchmark_sortformer_rtf.py, which is a transcription of the same loop and
shares the port's bugs. Same two constraints as sortformer_fidelity_der.py.

Usage:

    python scripts/nemo_export/sortformer_compress_parity.py \\
        --nemo ~/models/diar_streaming_sortformer_4spk-v2.1.nemo \\
        --saturated 0.0 0.85 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nemo", type=Path,
                   help="Source .nemo checkpoint. Required except with --probe-topk-ties, "
                        "which only needs torch.")
    p.add_argument("--saturated", nargs="+", type=float, default=[0.0, 0.85, 1.0],
                   help="Fractions of frames forced to exactly-1.0 preds (default 0 .85 1).")
    p.add_argument("--frames", type=int, default=None,
                   help="Frames going into compression. Default spkcache_len + update_period, "
                        "the steady state. Pass spkcache_len to model the warm-up cache.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--verbose", action="store_true",
                   help="List the first few differing selections.")
    p.add_argument("--probe-topk-ties", action="store_true",
                   help="Characterize torch.topk's tie behaviour on this build and exit. "
                        "Needs no checkpoint. Run this first on any new machine or torch "
                        "version -- the ports' tie rule was built on a claim about it that "
                        "does not hold (#171).")
    p.add_argument("--tie-incidence", action="store_true",
                   help="Report how often a tie can actually DECIDE a pick, by saturation "
                        "level. Ties inside or outside the kept set change nothing; only "
                        "one straddling the cut does.")
    return p.parse_args()


def probe_topk_ties() -> int:
    """Does torch.topk keep the lowest indices among ties, as #170 recorded?

    On torch 2.11.0/x86-64 it does not, at any size including #170's own fixture. What it
    returns instead is a mid-range SUBSET -- mostly-but-not-always contiguous -- so this
    prints the gap count as well as the span. Reporting only the span is how a wrong
    "contiguous block" claim got into this repo twice; two different sets can share
    endpoints, and a span alone cannot tell them apart.

    The open question is whether some build DOES return 0..k-1. If one does, the tie order
    is platform-dependent and 'match NeMo' is not a well-posed target at all.
    """
    import torch
    print(f"torch {torch.__version__}\n")
    print(f"  {'case':22} {'k':>5} {'span':>16} {'gaps':>6} {'contiguous':>11}  lowest-index?")
    for n, k in ((12, 5), (312, 35), (312, 70), (1248, 188), (2000, 188)):
        idx = sorted(torch.topk(torch.zeros(n), k, sorted=False).indices.tolist())
        gaps = sum(1 for a, b in zip(idx, idx[1:]) if b != a + 1)
        span = f"{idx[0]}..{idx[-1]}"
        print(f"  zeros(n={n:<6}){'':7} {k:>5} {span:>16} {gaps:>6} {str(gaps == 0):>11}  "
              f"{'MATCHES' if idx == list(range(k)) else 'DIFFERS'}")

    # A partial tie, which is the shape that actually occurs: a few distinct scores above a
    # large tied block. Only the tied part is at the tie rule's mercy.
    x = torch.zeros(1248)
    x[:50] = torch.linspace(5, 1, 50)
    idx = sorted(torch.topk(x, 188, sorted=False).indices.tolist())
    kept_distinct = sum(1 for i in idx if i < 50)
    print(f"\n  50 distinct + 1198 tied, k=188: keeps {kept_distinct}/50 of the distinct "
          f"entries, then {len(idx) - kept_distinct} tied ones spanning {idx[50]}..{idx[-1]}")

    x = torch.zeros(1248)
    runs = {tuple(torch.topk(x, 188, sorted=False).indices.tolist()) for _ in range(20)}
    print(f"\n  20 repeats in one process: {'stable' if len(runs) == 1 else 'NOT STABLE'}")
    saved = torch.get_num_threads()
    starts = []
    for t in (1, 2, 4):
        torch.set_num_threads(t)
        starts.append(sorted(torch.topk(x, 188, sorted=False).indices.tolist())[0])
    torch.set_num_threads(saved)
    print(f"  threads 1/2/4 first index: {starts}  "
          f"{'(thread-independent)' if len(set(starts)) == 1 else '(THREAD-DEPENDENT)'}")

    print("\n  If any row says MATCHES, the tie order differs by platform and #171's premise")
    print("  is unfixable by construction. If all say DIFFERS, this build agrees with the")
    print("  Linux measurement and the ports' rule is arbitrary-but-deterministic, as documented.")
    return 0


def make_fixture(n_frames: int, n_spk: int, saturated: float, seed: int):
    """Preds with a controlled saturated fraction.

    A saturated frame is exactly 1.0 for one speaker and 0.0 for the rest -- what the
    float32 sigmoid produces for a confidently single-speaker frame. The remainder are
    ordinary probabilities, which break ties and should select identically on both sides.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    preds = rng.random((1, n_frames, n_spk)).astype(np.float32)
    n_sat = int(round(n_frames * saturated))
    if n_sat:
        rows = rng.choice(n_frames, size=n_sat, replace=False)
        who = rng.integers(0, n_spk, size=n_sat)
        preds[0, rows, :] = 0.0
        preds[0, rows, who] = 1.0
    return preds


def nemo_stages(mods, preds_t):
    """Every intermediate `_compress_spkcache` builds, in order, as numpy."""
    import math
    import torch

    n_spk = preds_t.shape[2]
    per_spk = mods.spkcache_len // n_spk - mods.spkcache_sil_frames_per_spk
    strong = math.floor(per_spk * mods.strong_boost_rate)
    weak = math.floor(per_spk * mods.weak_boost_rate)
    min_pos = math.floor(per_spk * mods.min_pos_scores_rate)

    with torch.no_grad():
        scores = mods._get_log_pred_scores(preds_t)
        scores = mods._disable_low_scores(preds_t, scores, min_pos)
        stages = {"quality_scores": scores.clone()}
        if mods.scores_boost_latest > 0:
            scores[:, mods.spkcache_len:, :] += mods.scores_boost_latest
        stages["boost_latest"] = scores.clone()
        scores = mods._boost_topk_scores(scores, strong, scale_factor=2)
        stages["strong_boost"] = scores.clone()
        scores = mods._boost_topk_scores(scores, weak, scale_factor=1)
        stages["weak_boost"] = scores.clone()
        if mods.spkcache_sil_frames_per_spk > 0:
            pad = torch.full((1, mods.spkcache_sil_frames_per_spk, n_spk), float("inf"))
            scores = torch.cat([scores, pad], dim=1)
        stages["padded"] = scores.clone()
        topk_indices, is_disabled = mods._get_topk_indices(scores)

    flat = scores.permute(0, 2, 1).reshape(1, -1)
    picked = flat.topk(mods.spkcache_len, dim=1, sorted=False).indices[0].tolist()
    return ({k: v[0].numpy() for k, v in stages.items()}, picked,
            topk_indices[0].numpy(), is_disabled[0].numpy(),
            {"strong": strong, "weak": weak, "min_pos": min_pos, "per_spk": per_spk})


def port_stages(B, preds2d, n_frames, ks):
    """The same stages from the Python mirror, which transcribes Sortformer.cs."""
    import numpy as np

    scores = B.SortformerPipelineBase.speaker_quality_scores(None, preds2d, ks["min_pos"])
    stages = {"quality_scores": scores.copy()}
    scores[B.SPEAKER_CACHE_LENGTH:, :] += B.SCORES_BOOST_LATEST
    stages["boost_latest"] = scores.copy()
    B.SortformerPipelineBase.boost(scores, ks["strong"], 2.0)
    stages["strong_boost"] = scores.copy()
    B.SortformerPipelineBase.boost(scores, ks["weak"], 1.0)
    stages["weak_boost"] = scores.copy()

    sil = B.SPKCACHE_SIL_FRAMES
    ext = np.full((n_frames + sil, B.NUM_SPEAKERS), np.inf, dtype=np.float32)
    ext[:n_frames] = scores
    stages["padded"] = ext.copy()

    ext_t = ext.shape[0]
    flat = [(float(ext[t, s]), t, s) for t in range(ext_t) for s in range(B.NUM_SPEAKERS)]
    flat.sort(key=B.cache_tie_key)   # the mirror's own key, so this cannot drift from it
    picked = [s * ext_t + t for _, t, s in flat[: B.SPEAKER_CACHE_LENGTH]]
    return stages, picked


def interchangeable(preds2d, picked_a, picked_b, n_frames, sil) -> tuple[int, bool]:
    """Of the frames the two sides disagree on, how many are INTERCHANGEABLE -- same
    preds row as a frame the other side picked? Saturated frames are identical to each
    other, so swapping them changes which embedding the cache holds but not what the
    cache says about who is speaking."""
    import numpy as np
    ext_t = n_frames + sil

    def rows(picked):
        out = []
        for i in picked:
            t = i % ext_t
            out.append(tuple(preds2d[t]) if t < n_frames else ("SIL",))
        return out

    ra, rb = rows(picked_a), rows(picked_b)
    from collections import Counter
    ca, cb = Counter(ra), Counter(rb)
    same_multiset = ca == cb
    differing = sum((ca - cb).values())
    return differing, same_multiset


def compare(name: str, a, b) -> float:
    """maxAbs over finite entries; +/-inf must land in exactly the same places."""
    import numpy as np
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        n = int((fa != fb).sum())
        print(f"    {name:16} INF PATTERN DIFFERS in {n} entries")
        return float("inf")
    if not np.array_equal(a[~fa], b[~fb]):
        print(f"    {name:16} infinities differ in SIGN")
        return float("inf")
    d = float(np.abs(a[fa] - b[fb]).max()) if fa.any() else 0.0
    print(f"    {name:16} maxAbs={d:.3E}")
    return d


def boundary_tie(values, k: int) -> int:
    """Entries tied AT the k-th largest value, i.e. the ones the tie rule actually decides.
    Zero when the cut is unambiguous -- a tie wholly inside or outside the kept set, or a
    single entry at the cut, changes nothing."""
    import numpy as np
    v = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    if k >= len(v):
        return 0
    cut = v[k - 1]
    if not np.isfinite(cut):
        return 0
    # The cut is unambiguous when exactly k entries are >= it: every tied entry at the cut
    # value is inside the kept set, so which of them is "chosen" decides nothing.
    if int((v >= cut).sum()) == k:
        return 0
    return int((v == cut).sum())


def report_tie_incidence(mods, B, n_frames: int, seeds: int = 8) -> None:
    import torch
    print(f"\n  {'saturated':>10} {'boost ties at a cut':>22} {'final top-k ties at cut':>26}")
    for frac in (0.0, 0.05, 0.2, 0.5, 0.85, 1.0):
        boost_hits = final_hits = 0
        for seed in range(seeds):
            preds = make_fixture(n_frames, B.NUM_SPEAKERS, frac, seed)
            stages, _, _, _, ks = nemo_stages(mods, torch.from_numpy(preds))
            # Each boost pass top-k's the matrix the PREVIOUS pass produced: strong sees
            # boost_latest, weak sees the strong-boosted scores. Measuring both against
            # boost_latest scores the weak cut on a matrix NeMo never top-k's -- the strong
            # pass has already moved 33 entries per speaker up by 2*log 2, which shifts the
            # weak cut and its tie multiplicity.
            for spk in range(B.NUM_SPEAKERS):
                boost_hits += boundary_tie(stages["boost_latest"][:, spk], ks["strong"])
                boost_hits += boundary_tie(stages["strong_boost"][:, spk], ks["weak"])
            final_hits += boundary_tie(stages["padded"].T.reshape(-1), B.SPEAKER_CACHE_LENGTH)
        print(f"  {frac:>9.0%} {boost_hits / seeds:>22.1f} {final_hits / seeds:>26.1f}")
    print("\n  Zero on both columns means the tie rule cannot change the output at that")
    print("  saturation level, whatever rule it is.")


def main() -> int:
    args = parse_args()
    if args.probe_topk_ties:
        return probe_topk_ties()
    if args.nemo is None:
        raise SystemExit("--nemo is required (except with --probe-topk-ties)")
    import numpy as np
    import torch

    import benchmark_sortformer_rtf as B
    from sortformer_fidelity_der import build_reference_model

    model = build_reference_model(
        args.nemo, B.CHUNK_LENGTH, B.FIFO_LENGTH,
        B.SPEAKER_CACHE_LENGTH, B.SPEAKER_CACHE_UPDATE_PERIOD)
    mods = model.sortformer_modules

    n_frames = args.frames or (B.SPEAKER_CACHE_LENGTH + B.SPEAKER_CACHE_UPDATE_PERIOD)
    n_spk = B.NUM_SPEAKERS
    print(f"torch {torch.__version__}   frames={n_frames}  spk={n_spk}  "
          f"spkcache_len={mods.spkcache_len}  sil_per_spk={mods.spkcache_sil_frames_per_spk}")

    if args.tie_incidence:
        report_tie_incidence(mods, B, n_frames)
        return 0

    worst = 0.0
    for frac in args.saturated:
        preds = make_fixture(n_frames, n_spk, frac, args.seed)
        # from_numpy ALIASES; the fixture is read again below, so hand NeMo its own copy
        # rather than audit every callee for an in-place op.
        preds_t = torch.from_numpy(preds.copy())
        nemo_st, nemo_picked, _, _, ks = nemo_stages(mods, preds_t)
        port_st, port_picked = port_stages(B, preds[0], n_frames, ks)

        print(f"\n  saturated {frac:.0%}  (strong={ks['strong']} weak={ks['weak']} "
              f"min_pos={ks['min_pos']})")
        for stage in ("quality_scores", "boost_latest", "strong_boost", "weak_boost", "padded"):
            worst = max(worst, compare(stage, nemo_st[stage], port_st[stage]))

        ns, ps = set(nemo_picked), set(port_picked)
        flat_nemo = nemo_st["padded"].T.reshape(-1)
        sel_n = np.sort(flat_nemo[list(ns)])
        sel_p = np.sort(flat_nemo[list(ps)])
        same_scores = np.array_equal(sel_n, sel_p)
        print(f"    selection       {len(ns - ps)}/{len(ns)} indices differ; "
              f"selected score multisets {'EQUAL' if same_scores else 'DIFFER'}"
              f"{'  -> tie order only' if same_scores and ns != ps else ''}")
        def mix(picked):
            from collections import Counter
            c = Counter(i // (n_frames + mods.spkcache_sil_frames_per_spk) for i in picked)
            return [c[s] for s in range(n_spk)]

        print(f"    speaker mix     NeMo {mix(nemo_picked)}   port {mix(port_picked)}")

        diff_rows, same_pred_multiset = interchangeable(
            preds[0], nemo_picked, port_picked, n_frames, mods.spkcache_sil_frames_per_spk)
        print(f"    preds rows      {diff_rows}/{len(ns)} of the selected rows are not "
              f"matched by an identical row on the other side; "
              f"preds multisets {'EQUAL' if same_pred_multiset else 'DIFFER'}")
        if args.verbose and ns != ps:
            print(f"      NeMo picked, port did not: {sorted(ns - ps)[:8]}")
            print(f"      port picked, NeMo did not: {sorted(ps - ns)[:8]}")

    print(f"\nworst score-stage divergence: {worst:.3E}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
