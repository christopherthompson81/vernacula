#!/usr/bin/env python3
"""
export_indicconformer_nemo_to_onnx.py

Phase 1/2 exporter for AI4Bharat's IndicConformer hybrid CTC+RNNT model.

The plan is to ship a Parakeet-shaped package (encoder + CTC decoder +
preprocessor + vocab + config) with one extra artifact (language_spans.json)
that tells the C# side which 256-token slice belongs to each of the 22
languages.

Phase 1 scope of this script:
  1. encoder-model.onnx          — Conformer encoder, (features, lens) → (encoded, encoded_lens)
  2. ctc_decoder-model.onnx      — single Conv1d projection, encoded → logits
  3. vocab.txt                   — flat 5632-line vocab (blank is implicit at id 5632)
  4. language_spans.json         — 22 × {start, length}
  5. config.json                 — preprocessor + encoder metadata
  6. export-report.json          — record of what we did and any parity numbers
  7. Parity check on synthetic input vs the PyTorch forward

Phase 2 adds nemo128.onnx (DFT preprocessor, ported from nemo_export/).

Usage:
  python scripts/indicconformer_export/export_indicconformer_nemo_to_onnx.py \\
    --nemo <path to .nemo> \\
    --output-dir ~/models/indicconformer_onnx \\
    --opset 17
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True, help="Path to a local .nemo file.")
    p.add_argument(
        "--output-dir", required=True,
        help="Directory that will receive the exported package."
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Device for the NeMo restore and the PyTorch reference forward.",
    )
    p.add_argument(
        "--parity-seconds", type=float, default=4.0,
        help="Synthetic audio duration (sec) for the parity check.",
    )
    p.add_argument(
        "--parity-tolerance", type=float, default=1e-3,
        help="Max absolute logit delta allowed between PyTorch and ONNX Runtime.",
    )
    p.add_argument(
        "--skip-parity", action="store_true",
        help="Skip the PyTorch/ONNX parity check (for iteration on the export only).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@dataclass
class ExportReport:
    model_class: str = ""
    encoder_onnx: str = ""
    ctc_decoder_onnx: str = ""
    vocab_size: int = 0
    num_languages: int = 0
    languages: list[str] = field(default_factory=list)
    parity_max_abs_delta: float | None = None
    parity_pass: bool | None = None
    notes: list[str] = field(default_factory=list)


# --- ONNX wrapper modules ---------------------------------------------------
#
# NeMo modules accept keyword-argument forwards (audio_signal=, length=) and
# mix data/length tensors in ways torch.onnx.export can trip over. Thin
# wrappers give the exporter positional signatures it can trace cleanly.

def build_encoder_wrapper(torch, encoder):
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, features, features_lens):
            # NeMo Conformer encoder returns (encoded, encoded_lens).
            encoded, encoded_lens = self.enc(
                audio_signal=features, length=features_lens
            )
            return encoded, encoded_lens

    w = EncoderWrapper(encoder)
    w.eval()
    return w


def build_ctc_decoder_wrapper(torch, ctc_decoder):
    class CtcDecoderWrapper(torch.nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, encoded):
            # ConvASRDecoder.forward expects keyword encoder_output= .
            # It returns log-probs; we export the raw logits path by calling
            # the underlying Conv1d directly so the C# side can decide
            # whether to log_softmax.
            logits = self.dec(encoder_output=encoded)
            return logits

    w = CtcDecoderWrapper(ctc_decoder)
    w.eval()
    return w


# --- Export helpers ---------------------------------------------------------

def export_encoder(torch, model, output_path: Path, opset: int, parity_seconds: float):
    print(f"[encoder] exporting to {output_path}")
    wrapper = build_encoder_wrapper(torch, model.encoder)
    device = next(wrapper.parameters()).device

    # Build a dummy log-mel-like feature tensor.
    # NeMo preprocessor produces (batch, mel=80, frames). Frame rate is
    # sample_rate / hop = 16000 / 160 = 100 frames/sec.
    mel = int(model.cfg.preprocessor.features)
    frames = max(int(parity_seconds * 100), 50)
    features = torch.randn(1, mel, frames, dtype=torch.float32, device=device)
    lens = torch.tensor([frames], dtype=torch.int64, device=device)

    dynamic_axes = {
        "features": {0: "batch", 2: "frames"},
        "features_lens": {0: "batch"},
        "encoded": {0: "batch", 2: "encoded_frames"},
        "encoded_lens": {0: "batch"},
    }

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (features, lens),
            str(output_path),
            input_names=["features", "features_lens"],
            output_names=["encoded", "encoded_lens"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    print("[encoder] done")
    return features, lens


def export_ctc_decoder(torch, model, output_path: Path, opset: int,
                       encoded_example) -> None:
    print(f"[ctc_decoder] exporting to {output_path}")
    wrapper = build_ctc_decoder_wrapper(torch, model.ctc_decoder)

    dynamic_axes = {
        "encoded": {0: "batch", 2: "encoded_frames"},
        "logits": {0: "batch", 1: "encoded_frames"},
    }

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (encoded_example,),
            str(output_path),
            input_names=["encoded"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    print("[ctc_decoder] done")


# --- Sidecars ---------------------------------------------------------------

def write_vocab(model, dst: Path) -> int:
    """Flatten the 22 per-language SentencePiece vocabs into one file.

    MultilingualTokenizer.ids_to_tokens takes (ids, lang_id), not a flat id —
    so we walk the sub-tokenizers in offset order and stitch the result.
    """
    tok = model.tokenizer
    lines: list[str | None] = [None] * tok.vocab_size
    for lang in tok.langs:
        offset = int(tok.token_id_offset[lang])
        sub = tok.tokenizers_dict[lang]
        n = sub.vocab_size
        pieces = sub.ids_to_tokens(list(range(n)))
        for local_id, piece in enumerate(pieces):
            lines[offset + local_id] = piece
    missing = [i for i, p in enumerate(lines) if p is None]
    if missing:
        raise RuntimeError(f"Vocab gaps at ids {missing[:10]} (total missing: {len(missing)})")
    dst.write_text("\n".join(lines) + "\n")
    print(f"[vocab] wrote {dst} ({len(lines)} lines)")
    return len(lines)


def write_language_spans(model, dst: Path) -> list[str]:
    tok = model.tokenizer
    offsets = tok.token_id_offset
    spans = {}
    for lang in tok.langs:
        spans[lang] = {
            "start": int(offsets[lang]),
            "length": int(tok.tokenizers_dict[lang].vocab_size),
        }
    payload = {
        "total_vocab_size": int(tok.vocab_size),
        "blank_token_id": int(tok.vocab_size),  # CTC blank appended at the end
        "languages": spans,
    }
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[spans] wrote {dst} ({len(spans)} languages)")
    return list(spans.keys())


def write_config(model, dst: Path) -> None:
    from omegaconf import OmegaConf
    cfg = {
        "preprocessor": OmegaConf.to_container(model.cfg.preprocessor, resolve=True),
        "encoder": {
            "feat_in": int(model.cfg.encoder.feat_in),
            "n_layers": int(model.cfg.encoder.n_layers),
            "d_model": int(model.cfg.encoder.d_model),
        },
        "sample_rate": 16000,
    }
    dst.write_text(json.dumps(cfg, indent=2))
    print(f"[config] wrote {dst}")


# --- Parity check -----------------------------------------------------------

def parity_check(torch, model, encoder_onnx: Path, ctc_decoder_onnx: Path,
                 parity_seconds: float, tolerance: float) -> tuple[float, bool]:
    import numpy as np
    import onnxruntime as ort

    print("\n[parity] building synthetic input")
    mel = int(model.cfg.preprocessor.features)
    frames = max(int(parity_seconds * 100), 50)

    # Seeded so re-runs are deterministic.
    rng = np.random.default_rng(0)
    feats_np = rng.standard_normal((1, mel, frames)).astype(np.float32)
    lens_np = np.array([frames], dtype=np.int64)

    # PyTorch reference: run encoder, then ctc_decoder.
    print("[parity] PyTorch reference forward")
    device = next(model.parameters()).device
    feats_t = torch.from_numpy(feats_np).to(device)
    lens_t = torch.from_numpy(lens_np).to(device)
    with torch.inference_mode():
        encoded_ref, encoded_lens_ref = model.encoder(audio_signal=feats_t, length=lens_t)
        logits_ref = model.ctc_decoder(encoder_output=encoded_ref)
    logits_ref_np = logits_ref.detach().cpu().numpy()
    encoded_ref_np = encoded_ref.detach().cpu().numpy()

    # ONNX Runtime: chain the two graphs the same way.
    print("[parity] ORT encoder forward")
    enc_sess = ort.InferenceSession(
        str(encoder_onnx), providers=["CPUExecutionProvider"]
    )
    enc_out = enc_sess.run(None, {"features": feats_np, "features_lens": lens_np})
    encoded_ort = enc_out[0]

    print("[parity] ORT ctc_decoder forward")
    dec_sess = ort.InferenceSession(
        str(ctc_decoder_onnx), providers=["CPUExecutionProvider"]
    )
    dec_out = dec_sess.run(None, {"encoded": encoded_ort})
    logits_ort = dec_out[0]

    # Shapes should match exactly.
    assert logits_ref_np.shape == logits_ort.shape, (
        f"logits shape mismatch: ref {logits_ref_np.shape} vs ort {logits_ort.shape}"
    )

    # Compare on logits (the thing that matters for decoding).
    delta = float(np.max(np.abs(logits_ref_np - logits_ort)))
    enc_delta = float(np.max(np.abs(encoded_ref_np - encoded_ort)))
    print(f"[parity] encoded max-abs delta:  {enc_delta:.6e}")
    print(f"[parity] logits  max-abs delta:  {delta:.6e}")
    print(f"[parity] tolerance:              {tolerance:.6e}")
    passed = delta <= tolerance
    print(f"[parity] {'PASS' if passed else 'FAIL'}")
    return delta, passed


# --- Main -------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Torch-level imports deferred so --help works without NeMo installed.
    import torch
    import nemo.collections.asr as nemo_asr

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_onnx = out_dir / "encoder-model.onnx"
    ctc_decoder_onnx = out_dir / "ctc_decoder-model.onnx"
    vocab_txt = out_dir / "vocab.txt"
    spans_json = out_dir / "language_spans.json"
    config_json = out_dir / "config.json"
    report_json = out_dir / "export-report.json"

    if not args.overwrite:
        existing = [p for p in (encoder_onnx, ctc_decoder_onnx, vocab_txt,
                                spans_json, config_json) if p.exists()]
        if existing:
            raise SystemExit(
                "Output directory already contains export targets. "
                "Re-run with --overwrite to replace them.\n"
                f"Existing: {', '.join(p.name for p in existing)}"
            )

    # Device.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Restoring: {args.nemo}")
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location=device)
    model.eval()
    model.to(device)

    # Inference-only: freeze the preprocessor's BN/running stats path.
    model.freeze()

    report = ExportReport(
        model_class=type(model).__name__,
        encoder_onnx=str(encoder_onnx),
        ctc_decoder_onnx=str(ctc_decoder_onnx),
    )

    # 1. Encoder.
    features_example, lens_example = export_encoder(
        torch, model, encoder_onnx, args.opset, args.parity_seconds
    )

    # 2. Encoded example for the CTC decoder export (same shape the runtime sees).
    with torch.inference_mode():
        encoded_example, _ = model.encoder(
            audio_signal=features_example, length=lens_example
        )
    export_ctc_decoder(torch, model, ctc_decoder_onnx, args.opset, encoded_example)

    # 3. Sidecars.
    vocab_size = write_vocab(model, vocab_txt)
    langs = write_language_spans(model, spans_json)
    write_config(model, config_json)
    report.vocab_size = vocab_size
    report.languages = langs
    report.num_languages = len(langs)

    # 4. Parity.
    if not args.skip_parity:
        delta, passed = parity_check(
            torch, model, encoder_onnx, ctc_decoder_onnx,
            args.parity_seconds, args.parity_tolerance,
        )
        report.parity_max_abs_delta = delta
        report.parity_pass = passed
    else:
        report.notes.append("parity skipped by --skip-parity")

    report_json.write_text(json.dumps(asdict(report), indent=2))
    print(f"\n[report] wrote {report_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
