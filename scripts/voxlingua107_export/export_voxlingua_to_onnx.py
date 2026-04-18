"""Export speechbrain/lang-id-voxlingua107-ecapa to ONNX.

Produces:
    <out-dir>/voxlingua107.onnx     — end-to-end graph (raw audio → logits, embedding)
    <out-dir>/lang_map.json         — class index → ISO code + English name

Usage:
    python export_voxlingua_to_onnx.py --out-dir ./voxlingua107
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from speechbrain.inference.classifiers import EncoderClassifier

from src.ecapa_wrapper import VoxLinguaONNX
from src.lang_map import write_lang_map


MODEL_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
# 3 seconds of 16 kHz audio — any non-trivial length works for tracing,
# but too short risks hitting ECAPA's internal pooling edge cases.
TRACE_SAMPLES = 16_000 * 3


def load_classifier(savedir: Path) -> EncoderClassifier:
    savedir.mkdir(parents=True, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=str(savedir),
        run_opts={"device": "cpu"},  # export always runs on CPU for determinism
    )


def export(out_dir: Path, savedir: Path) -> None:
    print(f"[voxlingua] loading {MODEL_SOURCE} into {savedir}")
    classifier = load_classifier(savedir)
    classifier.eval()

    model = VoxLinguaONNX(classifier).eval()

    dummy_audio = torch.randn(1, TRACE_SAMPLES, dtype=torch.float32)

    # Sanity-check the forward pass produces the expected shapes before export.
    with torch.no_grad():
        logits, embedding = model(dummy_audio)
    assert logits.shape == (1, 107), f"unexpected logits shape: {logits.shape}"
    assert embedding.dim() == 2 and embedding.size(0) == 1, \
        f"unexpected embedding shape: {embedding.shape}"
    print(f"[voxlingua] forward ok: logits={tuple(logits.shape)}, "
          f"embedding={tuple(embedding.shape)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "voxlingua107.onnx"
    print(f"[voxlingua] exporting to {onnx_path}")

    torch.onnx.export(
        model,
        (dummy_audio,),
        str(onnx_path),
        input_names=["audio"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "audio": {0: "batch", 1: "samples"},
            "logits": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Structural check — catches malformed graphs early.
    onnx.checker.check_model(str(onnx_path))
    print("[voxlingua] onnx.checker passed")

    lang_map_path = out_dir / "lang_map.json"
    write_lang_map(classifier, lang_map_path)
    print(f"[voxlingua] wrote {lang_map_path} with 107 entries")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Destination directory for voxlingua107.onnx and lang_map.json.")
    p.add_argument("--savedir", type=Path,
                   default=Path("./.voxlingua107-cache"),
                   help="Where SpeechBrain caches the downloaded weights.")
    args = p.parse_args()
    export(args.out_dir, args.savedir)


if __name__ == "__main__":
    main()
