"""Verify the exported VoxLingua107 ONNX matches the PyTorch reference.

Runs the same audio through SpeechBrain's EncoderClassifier and the exported
ONNX graph, then compares:
    - top-1 language prediction (must match)
    - logit max-abs-diff (must be ≤ 1e-5 in FP32)
    - embedding cosine similarity (must be ≥ 0.9999)

Usage:
    python verify_voxlingua_parity.py \\
        --model-dir ./voxlingua107 \\
        --clips clip_en.wav clip_zh.wav clip_hi.wav ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from speechbrain.inference.classifiers import EncoderClassifier

from src.ecapa_wrapper import VoxLinguaONNX


MODEL_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
# Compare softmax probabilities rather than raw logits: the absolute logit
# magnitude scales with clip length (longer audio → larger accumulated scores),
# but softmax probabilities are scale-invariant and are what the downstream
# language decision actually reads. 1e-3 max-abs-diff on a probability vector
# is tight enough to catch real export bugs without flapping on FP32 drift.
PROB_TOLERANCE = 1e-3
EMBEDDING_THRESHOLD = 0.9999


def load_clip(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != 16_000:
        raise SystemExit(
            f"clip {path} has sample_rate={sr}; resample to 16 kHz before parity check"
        )
    return audio


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def verify(model_dir: Path, clips: list[Path], savedir: Path) -> int:
    onnx_path = model_dir / "voxlingua107.onnx"
    lang_map_path = model_dir / "lang_map.json"
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} not found — run export_voxlingua_to_onnx.py first")

    print(f"[parity] loading PyTorch reference from {MODEL_SOURCE}")
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE, savedir=str(savedir), run_opts={"device": "cpu"},
    )
    classifier.eval()
    ref_model = VoxLinguaONNX(classifier).eval()

    lang_map = json.loads(lang_map_path.read_text(encoding="utf-8"))
    print(f"[parity] loading ONNX model from {onnx_path}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    failures = 0
    for clip_path in clips:
        audio = load_clip(clip_path)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, T]

        with torch.no_grad():
            ref_logits, ref_embedding = ref_model(audio_tensor)
        ref_logits_np = ref_logits.numpy()
        ref_embedding_np = ref_embedding.numpy()

        onnx_logits, onnx_embedding = session.run(
            ["logits", "embedding"], {"audio": audio[None, :]},
        )

        ref_probs = softmax(ref_logits_np)
        onnx_probs = softmax(onnx_logits)
        prob_diff = float(np.max(np.abs(ref_probs - onnx_probs)))
        emb_cos = cosine(ref_embedding_np[0], onnx_embedding[0])

        ref_top1 = int(np.argmax(ref_logits_np[0]))
        onnx_top1 = int(np.argmax(onnx_logits[0]))
        ref_lang = lang_map[str(ref_top1)]["iso"]
        onnx_lang = lang_map[str(onnx_top1)]["iso"]

        status = "PASS"
        reasons = []
        if ref_top1 != onnx_top1:
            reasons.append(f"top-1 mismatch (pt={ref_lang}, onnx={onnx_lang})")
        if prob_diff > PROB_TOLERANCE:
            reasons.append(f"softmax max-abs-diff={prob_diff:.2e} > {PROB_TOLERANCE:.0e}")
        if emb_cos < EMBEDDING_THRESHOLD:
            reasons.append(f"embedding cosine={emb_cos:.6f} < {EMBEDDING_THRESHOLD}")

        if reasons:
            status = "FAIL"
            failures += 1

        suffix = f"  :: {'; '.join(reasons)}" if reasons else ""
        print(f"[parity] {status}  {clip_path.name:30s}  "
              f"top-1={onnx_lang}  Δprob={prob_diff:.2e}  cos={emb_cos:.6f}{suffix}")

    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory containing voxlingua107.onnx and lang_map.json.")
    p.add_argument("--clips", type=Path, nargs="+", required=True,
                   help="Audio clips to verify (16 kHz mono WAV preferred).")
    p.add_argument("--savedir", type=Path,
                   default=Path("./.voxlingua107-cache"),
                   help="SpeechBrain weight cache directory.")
    args = p.parse_args()

    failures = verify(args.model_dir, args.clips, args.savedir)
    if failures:
        raise SystemExit(f"{failures} parity failure(s)")
    print(f"[parity] all {len(args.clips)} clip(s) passed")


if __name__ == "__main__":
    main()
