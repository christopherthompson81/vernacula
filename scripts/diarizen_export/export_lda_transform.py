#!/usr/bin/env python3
"""
Export the LDA/xvec transform and PLDA parameters from the DiariZen model
to flat binary files that can be loaded directly in C#.

Output files (in --output-dir):
  mean1.bin   — float32[256]     — subtract from raw 256-dim WeSpeaker embedding
  lda.bin     — float32[256*128] — LDA projection matrix (256→128)
  mean2.bin   — float32[128]     — subtract after LDA projection
  plda_mu.bin — float32[128]     — PLDA mean (subtract before PLDA transform)
  plda_tr.bin — float32[128*128] — PLDA whitening/eigenspace transform (derived via eigh)
  plda_psi.bin— float32[128]     — PLDA eigenvalues (Phi for VBx)

The full xvec→PLDA pipeline:
  xvec  = sqrt(128) * l2_norm(lda.T @ (sqrt(256) * l2_norm(raw - mean1)) - mean2)
  fea   = (xvec - plda_mu) @ plda_tr.T   (no normalization)

Usage:
    python scripts/diarizen_export/export_lda_transform.py --output-dir ./models/plda
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.linalg import eigh
from huggingface_hub import snapshot_download


def l2_norm(x):
    if x.ndim == 1:
        return x / np.linalg.norm(x)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Export VBx LDA/PLDA transform to C#-readable binaries")
    parser.add_argument("--output-dir", type=Path, default=Path("./models/plda"))
    parser.add_argument("--repo-id", default="BUT-FIT/diarizen-wavlm-large-s80-md")
    args = parser.parse_args()

    # Find the model snapshot
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_name = args.repo_id.replace("/", "--")
    snap_root = cache_dir / f"models--{repo_name}" / "snapshots"

    if snap_root.exists():
        snap = next(snap_root.iterdir())
    else:
        print(f"Downloading {args.repo_id}...")
        snap = Path(snapshot_download(repo_id=args.repo_id))

    # ── xvec_transform ────────────────────────────────────────────────────────
    xvec_path = snap / "plda" / "xvec_transform.npz"
    if not xvec_path.exists():
        raise FileNotFoundError(f"xvec_transform.npz not found at {xvec_path}")

    xvec_data = np.load(xvec_path)
    mean1 = xvec_data["mean1"].astype(np.float32)  # (256,)
    lda   = xvec_data["lda"].astype(np.float32)    # (256, 128)
    mean2 = xvec_data["mean2"].astype(np.float32)  # (128,)
    print(f"mean1: {mean1.shape}, lda: {lda.shape}, mean2: {mean2.shape}")

    # ── PLDA parameters ───────────────────────────────────────────────────────
    plda_path = snap / "plda" / "plda.npz"
    if not plda_path.exists():
        raise FileNotFoundError(f"plda.npz not found at {plda_path}")

    plda_data = np.load(plda_path)
    plda_mu_raw  = plda_data["mu"].astype(np.float64)   # (128,) PLDA mean
    plda_tr_raw  = plda_data["tr"].astype(np.float64)   # (128, 128)
    plda_psi_raw = plda_data["psi"].astype(np.float64)  # (128,)
    print(f"plda mu: {plda_mu_raw.shape}, tr: {plda_tr_raw.shape}, psi: {plda_psi_raw.shape}")

    # Re-derive plda_tr and plda_psi exactly as vbx_setup does (eigh decomposition)
    W = np.linalg.inv(plda_tr_raw.T.dot(plda_tr_raw))
    B = np.linalg.inv((plda_tr_raw.T / plda_psi_raw).dot(plda_tr_raw))
    acvar, wccn = eigh(B, W)
    plda_psi_derived = acvar[::-1].astype(np.float32)    # (128,)
    plda_tr_derived  = wccn.T[::-1].astype(np.float32)   # (128, 128)
    plda_mu          = plda_mu_raw.astype(np.float32)     # (128,)

    print(f"plda_psi derived: min={plda_psi_derived.min():.4f}, max={plda_psi_derived.max():.4f}")
    print(f"plda_tr  derived: shape={plda_tr_derived.shape}")
    print(f"plda_mu:          shape={plda_mu.shape}")

    # ── Write files ───────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mean1.tofile(str(args.output_dir / "mean1.bin"))
    lda.tofile(str(args.output_dir / "lda.bin"))
    mean2.tofile(str(args.output_dir / "mean2.bin"))
    plda_mu.tofile(str(args.output_dir / "plda_mu.bin"))
    plda_tr_derived.tofile(str(args.output_dir / "plda_tr.bin"))
    plda_psi_derived.tofile(str(args.output_dir / "plda_psi.bin"))

    print(f"\nExported to {args.output_dir}/")
    for fname in ["mean1.bin", "lda.bin", "mean2.bin", "plda_mu.bin", "plda_tr.bin", "plda_psi.bin"]:
        p = args.output_dir / fname
        print(f"  {fname:16s} — {p.stat().st_size} bytes")

    # ── Verify pipeline ───────────────────────────────────────────────────────
    np.random.seed(42)
    test_raw = np.random.randn(5, 256).astype(np.float64)

    # xvec_tf
    xvec_out = np.sqrt(lda.shape[1]) * l2_norm(
        lda.T.dot(np.sqrt(lda.shape[0]) * l2_norm(test_raw - mean1).T).T - mean2
    )
    print(f"\nxvec_tf output norms: {np.linalg.norm(xvec_out, axis=1)}")

    # plda_tf
    fea = (xvec_out - plda_mu).dot(plda_tr_derived.T)
    print(f"plda_tf output norms: {np.linalg.norm(fea, axis=1)}")
    print(f"plda_psi (first 5):   {plda_psi_derived[:5]}")


if __name__ == "__main__":
    main()
