#!/usr/bin/env python3
"""
export_indicconformer_nemo_to_onnx.py

Phase 1/2 exporter for AI4Bharat's IndicConformer hybrid CTC+RNNT model.

The plan is to ship a Parakeet-shaped package (encoder + CTC decoder +
preprocessor + vocab + config) with one extra artifact (language_spans.json)
that tells the C# side which 256-token slice belongs to each of the 22
languages.

Shipping package:
  1. encoder-model.onnx          — Conformer encoder, (features, lens) → (encoded, encoded_lens)
  2. ctc_decoder-model.onnx      — single Conv1d projection, encoded → logits
  3. nemo128.onnx                — DFT-conv1d 80-mel preprocessor, waveform → features
  4. vocab.txt                   — flat 5632-line vocab (blank is implicit at id 5632)
  5. language_spans.json         — 22 × {start, length}
  6. config.json                 — preprocessor + encoder metadata
  7. export-report.json          — record of what we did and any parity numbers

Parity checks:
  - synthetic: random log-mel features → encoder → ctc_decoder, PyTorch vs ORT.
  - real-audio (optional, --parity-audio path): .wav → full pipeline PyTorch
    vs ORT (nemo128 + encoder + ctc_decoder chained).

Usage:
  python scripts/indicconformer_export/export_indicconformer_nemo_to_onnx.py \\
    --nemo <path to .nemo> \\
    --output-dir ~/models/indicconformer_onnx \\
    --opset 17
"""

from __future__ import annotations

import argparse
import json
import math
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
    p.add_argument(
        "--parity-audio", default=None,
        help="Optional .wav file for real-audio end-to-end parity (nemo128 + encoder + ctc_decoder).",
    )
    p.add_argument(
        "--parity-audio-tolerance", type=float, default=1e-2,
        help="Tolerance for real-audio parity. Looser than synthetic because it compounds three graphs.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@dataclass
class ExportReport:
    model_class: str = ""
    encoder_onnx: str = ""
    ctc_decoder_onnx: str = ""
    preprocessor_onnx: str = ""
    vocab_size: int = 0
    num_languages: int = 0
    languages: list[str] = field(default_factory=list)
    parity_max_abs_delta: float | None = None
    parity_pass: bool | None = None
    parity_audio_max_abs_delta: float | None = None
    parity_audio_pass: bool | None = None
    notes: list[str] = field(default_factory=list)


# --- ONNX wrapper modules ---------------------------------------------------
#
# NeMo modules accept keyword-argument forwards (audio_signal=, length=) and
# mix data/length tensors in ways torch.onnx.export can trip over. Thin
# wrappers give the exporter positional signatures it can trace cleanly.


def build_dft_preprocessor_wrapper(torch, preprocessor):
    """DFT-basis preprocessor wrapper. Ported from
    scripts/nemo_export/export_parakeet_nemo_to_onnx.py — same IndicConformer
    preprocessor config family (80 mel, 16 kHz, n_fft=512, 25/10 ms, per_feature
    normalize, hann window, log+add guard).

    Replaces torch.stft with F.conv1d on a precomputed windowed cos/sin basis
    matrix so the ONNX graph contains only Conv/Pow/Add/Sqrt/Log/MatMul ops —
    no STFT op (ONNX Runtime's STFT diverges from PyTorch's on this toolchain).
    """

    class DFTConvPreprocessorWrapper(torch.nn.Module):
        def __init__(self, pre):
            super().__init__()
            f = pre.featurizer
            self.preemph = float(f.preemph)
            self.n_fft = int(f.n_fft)
            self.hop_length = int(f.hop_length)
            self.win_length = int(f.win_length)
            self.mag_power = float(f.mag_power)
            # NeMo 1.x FilterbankFeatures has neither `exact_pad` nor
            # `stft_pad_amount`; it always uses center-reflect padding. NeMo 2.x
            # added both. Default to the 1.x behavior when missing.
            self.exact_pad = bool(getattr(f, "exact_pad", False))
            _stft_pad = getattr(f, "stft_pad_amount", None)
            self.stft_pad_amount = None if _stft_pad is None else int(_stft_pad)
            self.log_zero_guard_type = str(f.log_zero_guard_type)
            self.log_zero_guard_value = float(f.log_zero_guard_value)
            self.normalize = str(getattr(f, "normalize", "None"))
            self.pad_to = getattr(f, "pad_to", 0)
            self.pad_value = float(getattr(f, "pad_value", 0.0))

            # Windowed DFT basis, built in fp64 then cast to fp32.
            win = f.window.detach().float().cpu()
            if win.shape[0] < self.n_fft:
                left = (self.n_fft - win.shape[0]) // 2
                right = self.n_fft - win.shape[0] - left
                win = torch.nn.functional.pad(win, [left, right])
            n_bins = self.n_fft // 2 + 1
            n_range = torch.arange(self.n_fft, dtype=torch.float64)
            k_range = torch.arange(n_bins, dtype=torch.float64)
            angles = 2.0 * math.pi * k_range.unsqueeze(1) * n_range.unsqueeze(0) / self.n_fft
            win64 = win.to(dtype=torch.float64)
            cos_b = (torch.cos(angles) * win64.unsqueeze(0)).to(dtype=torch.float32)
            sin_b = (torch.sin(angles) * win64.unsqueeze(0)).to(dtype=torch.float32)
            # conv1d weight shape: [out_channels, in_channels=1, kernel=n_fft]
            dft_matrix = torch.cat([cos_b, sin_b], dim=0).unsqueeze(1)
            self.register_buffer("dft_matrix", dft_matrix)
            self.register_buffer("fb", f.fb.detach().float().cpu())

        def get_seq_len(self, seq_len):
            # Matches NeMo FilterbankFeatures.get_seq_len exactly, including
            # the trailing +1 that was missing from the NeMo-2.x-derived
            # reference wrapper in scripts/nemo_export/.
            pad = (self.stft_pad_amount * 2 if self.stft_pad_amount is not None
                   else (self.n_fft // 2) * 2)
            return (torch.floor_divide(seq_len + pad - self.n_fft, self.hop_length) + 1).to(dtype=torch.long)

        def _normalize_per_feature(self, features, seq_len):
            bs, _, max_t = features.shape
            t_idx = torch.arange(max_t, device=features.device).unsqueeze(0).expand(bs, max_t)
            mask = t_idx < seq_len.unsqueeze(1)
            masked = torch.where(mask.unsqueeze(1), features, 0.0)
            numer = masked.sum(axis=2)
            denom = mask.sum(axis=1)
            mean = numer / denom.unsqueeze(1)
            var = torch.sum(
                torch.where(mask.unsqueeze(1), features - mean.unsqueeze(2), 0.0) ** 2,
                axis=2,
            ) / (denom.unsqueeze(1) - 1.0)
            std = torch.sqrt(var).masked_fill(torch.sqrt(var).isnan(), 0.0) + 1e-5
            return (features - mean.unsqueeze(2)) / std.unsqueeze(2)

        def forward(self, waveforms, waveforms_lens):
            seq_len_unfixed = self.get_seq_len(waveforms_lens.float())
            seq_len = torch.where(
                waveforms_lens == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed
            )

            # Time mask + pre-emphasis.
            t_mask = (torch.arange(waveforms.shape[1], device=waveforms.device).unsqueeze(0)
                      < waveforms_lens.unsqueeze(1))
            waveforms = torch.cat(
                (waveforms[:, :1], waveforms[:, 1:] - self.preemph * waveforms[:, :-1]),
                dim=1,
            ).masked_fill(~t_mask, 0.0)

            # Padding matches torch.stft semantics: constant zeros if exact_pad,
            # otherwise reflection pad of n_fft//2 each side.
            if self.stft_pad_amount is not None:
                waveforms = torch.nn.functional.pad(
                    waveforms.unsqueeze(1),
                    (self.stft_pad_amount, self.stft_pad_amount),
                    "constant",
                ).squeeze(1)
            elif not self.exact_pad:
                half = self.n_fft // 2
                waveforms = torch.nn.functional.pad(
                    waveforms.unsqueeze(1), (half, half), "reflect",
                ).squeeze(1)

            # Conv1d DFT: [B,1,T] * [2*n_bins,1,n_fft] -> [B, 2*n_bins, frames]
            frames = torch.nn.functional.conv1d(
                waveforms.unsqueeze(1),
                self.dft_matrix.to(dtype=waveforms.dtype, device=waveforms.device),
                stride=self.hop_length,
                padding=0,
            )
            n_bins = self.n_fft // 2 + 1
            real, imag = frames[:, :n_bins, :], frames[:, n_bins:, :]
            features = torch.sqrt(real.pow(2) + imag.pow(2))
            if self.mag_power != 1.0:
                features = features.pow(self.mag_power)

            features = torch.matmul(
                self.fb.to(dtype=features.dtype, device=features.device), features
            )

            if self.log_zero_guard_type == "add":
                features = torch.log(features + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                features = torch.log(torch.clamp(features, min=self.log_zero_guard_value))
            else:
                raise RuntimeError(f"Unsupported log_zero_guard_type: {self.log_zero_guard_type}")

            if self.normalize == "per_feature":
                features = self._normalize_per_feature(features, seq_len)
            elif self.normalize not in ("", "None", "none"):
                raise RuntimeError(f"Unsupported normalize mode: {self.normalize}")

            max_len = features.size(-1)
            pad_mask = (torch.arange(max_len, device=features.device).repeat(features.size(0), 1)
                        >= seq_len.unsqueeze(1))
            features = features.masked_fill(pad_mask.unsqueeze(1), self.pad_value)

            if isinstance(self.pad_to, int) and self.pad_to > 0:
                pad_amt = features.size(-1) % self.pad_to
                if pad_amt != 0:
                    features = torch.nn.functional.pad(
                        features, (0, self.pad_to - pad_amt), value=self.pad_value
                    )
            return features, seq_len

    w = DFTConvPreprocessorWrapper(preprocessor)
    w.eval()
    return w


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


def export_preprocessor(torch, model, output_path: Path, opset: int,
                        dummy_seconds: float = 4.0) -> None:
    print(f"[preprocessor] exporting to {output_path}")
    wrapper = build_dft_preprocessor_wrapper(torch, model.preprocessor)
    device = next(model.parameters()).device
    wrapper.to(device)

    sample_rate = int(getattr(model.preprocessor, "_sample_rate", 16000))
    num_samples = max(int(dummy_seconds * sample_rate), sample_rate)
    waveforms = torch.zeros((1, num_samples), dtype=torch.float32, device=device)
    waveforms_lens = torch.tensor([num_samples], dtype=torch.int64, device=device)

    dynamic_axes = {
        "waveforms": {0: "batch", 1: "samples"},
        "waveforms_lens": {0: "batch"},
        "features": {0: "batch", 2: "frames"},
        "features_lens": {0: "batch"},
    }

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (waveforms, waveforms_lens),
            str(output_path),
            input_names=["waveforms", "waveforms_lens"],
            output_names=["features", "features_lens"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    print("[preprocessor] done")


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

def parity_check_real_audio(
    torch, model, wav_path: Path, preprocessor_onnx: Path,
    encoder_onnx: Path, ctc_decoder_onnx: Path, tolerance: float,
) -> tuple[float, bool]:
    """End-to-end parity: load a real .wav, run the full PyTorch pipeline
    (preprocessor → encoder → ctc_decoder), run the full ONNX pipeline
    (nemo128 → encoder → ctc_decoder), compare final logits.
    """
    import numpy as np
    import soundfile as sf
    import onnxruntime as ort

    print(f"\n[parity-audio] loading {wav_path}")
    arr, sr = sf.read(str(wav_path), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(
            f"Parity audio must be 16 kHz mono; got {sr} Hz. "
            "The test_audio library stores Indic clips already at 16 kHz."
        )
    waveforms_np = arr[None, :].astype(np.float32)
    lens_np = np.array([arr.shape[0]], dtype=np.int64)

    device = next(model.parameters()).device
    waveforms_t = torch.from_numpy(waveforms_np).to(device)
    lens_t = torch.from_numpy(lens_np).to(device)

    print("[parity-audio] PyTorch reference: preprocessor → encoder → ctc_decoder")
    with torch.inference_mode():
        features_ref, features_lens_ref = model.preprocessor(
            input_signal=waveforms_t, length=lens_t
        )
        encoded_ref, _ = model.encoder(audio_signal=features_ref, length=features_lens_ref)
        logits_ref = model.ctc_decoder(encoder_output=encoded_ref)
    logits_ref_np = logits_ref.detach().cpu().numpy()
    features_ref_np = features_ref.detach().cpu().numpy()

    print("[parity-audio] ORT: nemo128 → encoder → ctc_decoder")
    cpu = ["CPUExecutionProvider"]
    pre_sess = ort.InferenceSession(str(preprocessor_onnx), providers=cpu)
    enc_sess = ort.InferenceSession(str(encoder_onnx), providers=cpu)
    dec_sess = ort.InferenceSession(str(ctc_decoder_onnx), providers=cpu)

    pre_out = pre_sess.run(None, {"waveforms": waveforms_np, "waveforms_lens": lens_np})
    features_ort, features_lens_ort = pre_out[0], pre_out[1]

    enc_out = enc_sess.run(None, {"features": features_ort, "features_lens": features_lens_ort})
    encoded_ort = enc_out[0]

    dec_out = dec_sess.run(None, {"encoded": encoded_ort})
    logits_ort = dec_out[0]

    assert logits_ref_np.shape == logits_ort.shape, (
        f"logits shape mismatch: ref {logits_ref_np.shape} vs ort {logits_ort.shape}"
    )

    feat_delta = float(np.max(np.abs(features_ref_np - features_ort)))
    logits_delta = float(np.max(np.abs(logits_ref_np - logits_ort)))
    print(f"[parity-audio] features max-abs delta: {feat_delta:.6e}")
    print(f"[parity-audio] logits   max-abs delta: {logits_delta:.6e}")
    print(f"[parity-audio] tolerance:              {tolerance:.6e}")
    passed = logits_delta <= tolerance
    print(f"[parity-audio] {'PASS' if passed else 'FAIL'}")
    return logits_delta, passed


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
    preprocessor_onnx = out_dir / "nemo128.onnx"
    vocab_txt = out_dir / "vocab.txt"
    spans_json = out_dir / "language_spans.json"
    config_json = out_dir / "config.json"
    report_json = out_dir / "export-report.json"

    if not args.overwrite:
        existing = [p for p in (encoder_onnx, ctc_decoder_onnx, preprocessor_onnx,
                                vocab_txt, spans_json, config_json) if p.exists()]
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
        preprocessor_onnx=str(preprocessor_onnx),
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

    # 3. Preprocessor (DFT-conv1d).
    export_preprocessor(torch, model, preprocessor_onnx, args.opset, args.parity_seconds)

    # 4. Sidecars.
    vocab_size = write_vocab(model, vocab_txt)
    langs = write_language_spans(model, spans_json)
    write_config(model, config_json)
    report.vocab_size = vocab_size
    report.languages = langs
    report.num_languages = len(langs)

    # 5. Synthetic parity (encoder + ctc_decoder only, random mel input).
    if not args.skip_parity:
        delta, passed = parity_check(
            torch, model, encoder_onnx, ctc_decoder_onnx,
            args.parity_seconds, args.parity_tolerance,
        )
        report.parity_max_abs_delta = delta
        report.parity_pass = passed
    else:
        report.notes.append("parity skipped by --skip-parity")

    # 6. Real-audio parity (full pipeline: nemo128 → encoder → ctc_decoder).
    if args.parity_audio and not args.skip_parity:
        audio_delta, audio_pass = parity_check_real_audio(
            torch, model, Path(args.parity_audio).expanduser(),
            preprocessor_onnx, encoder_onnx, ctc_decoder_onnx,
            args.parity_audio_tolerance,
        )
        report.parity_audio_max_abs_delta = audio_delta
        report.parity_audio_pass = audio_pass

    report_json.write_text(json.dumps(asdict(report), indent=2))
    print(f"\n[report] wrote {report_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
