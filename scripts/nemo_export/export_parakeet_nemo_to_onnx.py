from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a NeMo Parakeet RNNT/TDT .nemo checkpoint into the ONNX package "
            "shape used by this repository."
        )
    )
    parser.add_argument(
        "--nemo",
        required=True,
        help="Path to a local .nemo file, for example parakeet-tdt-0.6b-v3.nemo.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive nemo128.onnx, encoder-model.onnx, decoder_joint-model.onnx, vocab.txt, and config.json.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset to request during export.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use for model restore/export.",
    )
    parser.add_argument(
        "--preprocessor-seconds",
        type=float,
        default=20.0,
        help="Dummy waveform length used when exporting the preprocessor wrapper.",
    )
    parser.add_argument(
        "--skip-preprocessor",
        action="store_true",
        help="Skip nemo128.onnx export if NeMo or torch cannot export the preprocessor cleanly.",
    )
    parser.add_argument(
        "--preprocessor-opset",
        type=int,
        default=None,
        help="Optional opset override for nemo128.onnx. Defaults to --opset.",
    )
    parser.add_argument(
        "--preprocessor-mode",
        choices=("wrapper", "custom", "dft"),
        default="wrapper",
        help=(
            "How to build nemo128.onnx. "
            "'wrapper' exports the live NeMo preprocessor module directly (usually fails). "
            "'custom' rebuilds the feature extractor using torch.stft (exports but ONNX Runtime diverges). "
            "'dft' rebuilds the feature extractor using a conv1d DFT basis matrix — no STFT op, "
            "ONNX-safe, and should match NeMo numerically."
        ),
    )
    parser.add_argument(
        "--preprocessor-dynamo",
        choices=("auto", "true", "false"),
        default="auto",
        help="Control whether torch.onnx.export uses the newer dynamo exporter path for nemo128.onnx.",
    )
    parser.add_argument(
        "--preprocessor-optimize",
        choices=("auto", "true", "false"),
        default="auto",
        help="Control exporter graph optimization for nemo128.onnx.",
    )
    parser.add_argument(
        "--preprocessor-verify",
        action="store_true",
        help="Enable torch.onnx.export verification for nemo128.onnx.",
    )
    parser.add_argument(
        "--preprocessor-no-constant-folding",
        action="store_true",
        help="Disable constant folding during nemo128.onnx export.",
    )
    parser.add_argument(
        "--preprocessor-static-shape",
        action="store_true",
        help="Export nemo128.onnx without dynamic axes as a candidate for closer frame-count behavior.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf
    except Exception as exc:  # pragma: no cover - import error path is intentional
        raise SystemExit(
            "Missing export dependencies. Create a Python 3.11/3.12 environment and "
            "install the packages from scripts/nemo_export/requirements.txt.\n"
            f"Original import error: {exc}"
        ) from exc

    return torch, ASRModel, OmegaConf, exc_to_string


def exc_to_string(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    return requested


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        collisions = [
            name
            for name in (
                "nemo128.onnx",
                "encoder-model.onnx",
                "encoder-model.onnx.data",
                "decoder_joint-model.onnx",
                "decoder_joint-model.onnx.data",
                "vocab.txt",
                "config.json",
                "export-report.json",
            )
            if (path / name).exists()
        ]
        if collisions:
            raise SystemExit(
                "Output directory already contains export targets. "
                "Re-run with --overwrite to replace them.\n"
                f"Existing files: {', '.join(collisions)}"
            )


def cleanup_target(path: Path) -> None:
    if path.exists():
        path.unlink()


def replace_file(src: Path, dst: Path) -> None:
    cleanup_target(dst)
    shutil.move(str(src), str(dst))


def replace_if_present(src: Path, dst: Path) -> None:
    if src.exists():
        replace_file(src, dst)


def resolve_export_bool(value: str, default: bool) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    return default


def consolidate_external_data(onnx_path: Path) -> None:
    """Re-save an ONNX model so all external tensors live in a single sidecar file.

    NeMo's exporter sometimes writes one file per tensor.  This collapses them
    into ``<stem>.onnx.data`` so the output directory stays clean.
    Has no effect when the model already uses a single-file or embedded format.
    """
    import onnx  # heavy import — defer until needed

    model = onnx.load(str(onnx_path))
    locations: set[str] = set()
    for t in model.graph.initializer:
        for kv in t.external_data:
            if kv.key == "location":
                locations.add(kv.value)

    sidecar_name = onnx_path.name + ".data"
    already_clean = len(locations) == 0 or (len(locations) == 1 and sidecar_name in locations)
    if already_clean:
        return

    print(f"  Consolidating {len(locations)} external data files → {sidecar_name} …")
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar_name,
        size_threshold=0,
    )

    # Remove stale per-tensor files that are no longer referenced
    parent = onnx_path.parent
    for loc in locations:
        stale = parent / loc
        if stale.exists() and stale.name != sidecar_name:
            stale.unlink()


def export_rnnt_pair(model: Any, output_dir: Path, opset: int) -> None:
    export_stub = output_dir / "parakeet_rnnt.onnx"
    cleanup_target(export_stub)

    model.export(str(export_stub), onnx_opset_version=opset)

    encoder_src = output_dir / "encoder-parakeet_rnnt.onnx"
    decoder_src = output_dir / "decoder_joint-parakeet_rnnt.onnx"

    if not encoder_src.exists() or not decoder_src.exists():
        raise RuntimeError(
            "NeMo export did not produce the expected split RNNT artifacts "
            "(encoder-parakeet_rnnt.onnx and decoder_joint-parakeet_rnnt.onnx)."
        )

    replace_file(encoder_src, output_dir / "encoder-model.onnx")
    replace_file(decoder_src, output_dir / "decoder_joint-model.onnx")

    replace_if_present(
        output_dir / "encoder-parakeet_rnnt.onnx.data",
        output_dir / "encoder-model.onnx.data",
    )
    replace_if_present(
        output_dir / "decoder_joint-parakeet_rnnt.onnx.data",
        output_dir / "decoder_joint-model.onnx.data",
    )
    cleanup_target(export_stub)

    # Ensure output uses single-file external data regardless of what NeMo produced
    consolidate_external_data(output_dir / "encoder-model.onnx")
    consolidate_external_data(output_dir / "decoder_joint-model.onnx")


def export_preprocessor(
    model: Any,
    torch: Any,
    output_dir: Path,
    opset: int,
    dummy_seconds: float,
    *,
    mode: str,
    dynamo: bool,
    optimize: bool,
    verify: bool,
    do_constant_folding: bool,
    dynamic_axes_enabled: bool,
) -> None:
    class CustomPreprocessorWrapper(torch.nn.Module):
        def __init__(self, preprocessor: Any) -> None:
            super().__init__()
            featurizer = preprocessor.featurizer
            self.preemph = float(featurizer.preemph)
            self.n_fft = int(featurizer.n_fft)
            self.hop_length = int(featurizer.hop_length)
            self.win_length = int(featurizer.win_length)
            self.mag_power = float(featurizer.mag_power)
            self.exact_pad = bool(featurizer.exact_pad)
            self.stft_pad_amount = (
                None if getattr(featurizer, "stft_pad_amount", None) is None else int(featurizer.stft_pad_amount)
            )
            self.log_zero_guard_type = str(featurizer.log_zero_guard_type)
            self.log_zero_guard_value = float(featurizer.log_zero_guard_value)
            self.normalize = str(getattr(featurizer, "normalize", "None"))
            self.pad_to = getattr(featurizer, "pad_to", 0)
            self.pad_value = float(getattr(featurizer, "pad_value", 0.0))
            self.register_buffer("window", featurizer.window.detach().float().cpu())
            self.register_buffer("fb", featurizer.fb.detach().float().cpu())

        def get_seq_len(self, seq_len: Any) -> Any:
            pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
            seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
            return seq_len.to(dtype=torch.long)

        def normalize_per_feature(self, features: Any, seq_len: Any) -> Any:
            batch_size = features.shape[0]
            max_time = features.shape[2]
            time_steps = torch.arange(max_time, device=features.device).unsqueeze(0).expand(batch_size, max_time)
            valid_mask = time_steps < seq_len.unsqueeze(1)
            masked = torch.where(valid_mask.unsqueeze(1), features, 0.0)
            mean_numerator = masked.sum(axis=2)
            denominator = valid_mask.sum(axis=1)
            mean = mean_numerator / denominator.unsqueeze(1)
            variance = torch.sum(
                torch.where(valid_mask.unsqueeze(1), features - mean.unsqueeze(2), 0.0) ** 2,
                axis=2,
            ) / (denominator.unsqueeze(1) - 1.0)
            std = torch.sqrt(variance)
            std = std.masked_fill(std.isnan(), 0.0)
            std = std + 1e-5
            return (features - mean.unsqueeze(2)) / std.unsqueeze(2)

        def forward(self, waveforms: Any, waveforms_lens: Any) -> tuple[Any, Any]:
            seq_len_time = waveforms_lens
            seq_len_unfixed = self.get_seq_len(waveforms_lens)
            seq_len = torch.where(waveforms_lens == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)

            time_mask = torch.arange(waveforms.shape[1], device=waveforms.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            waveforms = torch.cat(
                (waveforms[:, :1], waveforms[:, 1:] - self.preemph * waveforms[:, :-1]),
                dim=1,
            )
            waveforms = waveforms.masked_fill(~time_mask, 0.0)
            if self.stft_pad_amount is not None:
                waveforms = torch.nn.functional.pad(
                    waveforms.unsqueeze(1),
                    (self.stft_pad_amount, self.stft_pad_amount),
                    "constant",
                ).squeeze(1)

            features = torch.stft(
                waveforms,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=not self.exact_pad,
                window=self.window.to(dtype=torch.float32, device=waveforms.device),
                return_complex=False,
                pad_mode="constant",
            )
            features = torch.sqrt(features.pow(2).sum(-1))

            if self.mag_power != 1.0:
                features = features.pow(self.mag_power)

            features = torch.matmul(self.fb.to(dtype=features.dtype, device=features.device), features)
            if self.log_zero_guard_type == "add":
                features = torch.log(features + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                features = torch.log(torch.clamp(features, min=self.log_zero_guard_value))
            else:
                raise RuntimeError(f"Unsupported log_zero_guard_type: {self.log_zero_guard_type}")

            if self.normalize == "per_feature":
                features = self.normalize_per_feature(features, seq_len)
            elif self.normalize not in ("", "None", "none"):
                raise RuntimeError(f"Unsupported normalize mode for custom preprocessor: {self.normalize}")

            max_len = features.size(-1)
            mask = torch.arange(max_len, device=features.device).repeat(features.size(0), 1) >= seq_len.unsqueeze(1)
            features = features.masked_fill(mask.unsqueeze(1), self.pad_value)

            if self.pad_to == "max":
                raise RuntimeError("pad_to='max' is not supported by the custom preprocessor export path.")
            if isinstance(self.pad_to, int) and self.pad_to > 0:
                pad_amt = features.size(-1) % self.pad_to
                if pad_amt != 0:
                    features = torch.nn.functional.pad(features, (0, self.pad_to - pad_amt), value=self.pad_value)

            return features, seq_len

    class DFTConvPreprocessorWrapper(torch.nn.Module):
        """Mel feature extractor using a conv1d DFT basis matrix.

        Replaces torch.stft with F.conv1d on a precomputed windowed cos/sin basis.
        The STFT op in ONNX opset 17 produces diverged results in ONNX Runtime on
        this toolchain; using only Conv/Pow/Add/Sqrt avoids that entirely.

        The get_seq_len formula matches NeMo's FilterbankFeatures exactly, including
        the +1 that the torch.stft-based custom wrapper was missing.
        """

        def __init__(self, preprocessor: Any) -> None:
            super().__init__()
            featurizer = preprocessor.featurizer
            self.preemph = float(featurizer.preemph)
            self.n_fft = int(featurizer.n_fft)
            self.hop_length = int(featurizer.hop_length)
            self.win_length = int(featurizer.win_length)
            self.mag_power = float(featurizer.mag_power)
            self.exact_pad = bool(featurizer.exact_pad)
            self.stft_pad_amount = (
                None if getattr(featurizer, "stft_pad_amount", None) is None else int(featurizer.stft_pad_amount)
            )
            self.log_zero_guard_type = str(featurizer.log_zero_guard_type)
            self.log_zero_guard_value = float(featurizer.log_zero_guard_value)
            self.normalize = str(getattr(featurizer, "normalize", "None"))
            self.pad_to = getattr(featurizer, "pad_to", 0)
            self.pad_value = float(getattr(featurizer, "pad_value", 0.0))

            # Build windowed DFT basis as float64 to preserve precision, then cast to float32.
            # The window is zero-padded from win_length to n_fft using the same center-pad
            # convention that torch.stft uses internally.
            win = featurizer.window.detach().float().cpu()
            if win.shape[0] < self.n_fft:
                left = (self.n_fft - win.shape[0]) // 2
                right = self.n_fft - win.shape[0] - left
                win = torch.nn.functional.pad(win, [left, right])

            n_bins = self.n_fft // 2 + 1
            n_range = torch.arange(self.n_fft, dtype=torch.float64)
            k_range = torch.arange(n_bins, dtype=torch.float64)
            angles = 2.0 * math.pi * k_range.unsqueeze(1) * n_range.unsqueeze(0) / self.n_fft
            win64 = win.to(dtype=torch.float64)
            cos_basis = (torch.cos(angles) * win64.unsqueeze(0)).to(dtype=torch.float32)
            sin_basis = (torch.sin(angles) * win64.unsqueeze(0)).to(dtype=torch.float32)
            # Shape [2*n_bins, 1, n_fft]: conv1d weight (out_channels, in_channels, kernel)
            dft_matrix = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1)
            self.register_buffer("dft_matrix", dft_matrix)
            self.register_buffer("fb", featurizer.fb.detach().float().cpu())

        def get_seq_len(self, seq_len: Any) -> Any:
            # Matches NeMo FilterbankFeatures.get_seq_len exactly.
            pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
            return torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length).to(dtype=torch.long)

        def normalize_per_feature(self, features: Any, seq_len: Any) -> Any:
            batch_size = features.shape[0]
            max_time = features.shape[2]
            time_steps = torch.arange(max_time, device=features.device).unsqueeze(0).expand(batch_size, max_time)
            valid_mask = time_steps < seq_len.unsqueeze(1)
            masked = torch.where(valid_mask.unsqueeze(1), features, 0.0)
            mean_numerator = masked.sum(axis=2)
            denominator = valid_mask.sum(axis=1)
            mean = mean_numerator / denominator.unsqueeze(1)
            variance = torch.sum(
                torch.where(valid_mask.unsqueeze(1), features - mean.unsqueeze(2), 0.0) ** 2,
                axis=2,
            ) / (denominator.unsqueeze(1) - 1.0)
            std = torch.sqrt(variance)
            std = std.masked_fill(std.isnan(), 0.0)
            std = std + 1e-5
            return (features - mean.unsqueeze(2)) / std.unsqueeze(2)

        def forward(self, waveforms: Any, waveforms_lens: Any) -> tuple[Any, Any]:
            seq_len_time = waveforms_lens
            seq_len_unfixed = self.get_seq_len(waveforms_lens.float())
            seq_len = torch.where(waveforms_lens == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)

            # Pre-emphasis and time masking
            time_mask = torch.arange(waveforms.shape[1], device=waveforms.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            waveforms = torch.cat(
                (waveforms[:, :1], waveforms[:, 1:] - self.preemph * waveforms[:, :-1]),
                dim=1,
            )
            waveforms = waveforms.masked_fill(~time_mask, 0.0)

            # Padding: exact_pad pads with zeros; center pads with reflect (matching torch.stft behavior)
            if self.stft_pad_amount is not None:
                waveforms = torch.nn.functional.pad(
                    waveforms.unsqueeze(1),
                    (self.stft_pad_amount, self.stft_pad_amount),
                    "constant",
                ).squeeze(1)
            elif not self.exact_pad:
                half = self.n_fft // 2
                waveforms = torch.nn.functional.pad(
                    waveforms.unsqueeze(1),
                    (half, half),
                    "reflect",
                ).squeeze(1)

            # Conv1d DFT: [batch, 1, T] * [2*n_bins, 1, n_fft] -> [batch, 2*n_bins, frames]
            frames = torch.nn.functional.conv1d(
                waveforms.unsqueeze(1),
                self.dft_matrix.to(dtype=waveforms.dtype, device=waveforms.device),
                stride=self.hop_length,
                padding=0,
            )
            n_bins = self.n_fft // 2 + 1
            real = frames[:, :n_bins, :]
            imag = frames[:, n_bins:, :]
            features = torch.sqrt(real.pow(2) + imag.pow(2))  # magnitude spectrum

            if self.mag_power != 1.0:
                features = features.pow(self.mag_power)

            # Mel filterbank
            features = torch.matmul(self.fb.to(dtype=features.dtype, device=features.device), features)

            # Log compression
            if self.log_zero_guard_type == "add":
                features = torch.log(features + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                features = torch.log(torch.clamp(features, min=self.log_zero_guard_value))
            else:
                raise RuntimeError(f"Unsupported log_zero_guard_type: {self.log_zero_guard_type}")

            # Per-feature normalization
            if self.normalize == "per_feature":
                features = self.normalize_per_feature(features, seq_len)
            elif self.normalize not in ("", "None", "none"):
                raise RuntimeError(f"Unsupported normalize mode for DFT preprocessor: {self.normalize}")

            # Mask padding frames
            max_len = features.size(-1)
            mask = torch.arange(max_len, device=features.device).repeat(features.size(0), 1) >= seq_len.unsqueeze(1)
            features = features.masked_fill(mask.unsqueeze(1), self.pad_value)

            if self.pad_to == "max":
                raise RuntimeError("pad_to='max' is not supported by the DFT preprocessor export path.")
            if isinstance(self.pad_to, int) and self.pad_to > 0:
                pad_amt = features.size(-1) % self.pad_to
                if pad_amt != 0:
                    features = torch.nn.functional.pad(features, (0, self.pad_to - pad_amt), value=self.pad_value)

            return features, seq_len

    class PreprocessorWrapper(torch.nn.Module):
        def __init__(self, preprocessor: Any) -> None:
            super().__init__()
            self.preprocessor = preprocessor

        def forward(self, waveforms: Any, waveforms_lens: Any) -> tuple[Any, Any]:
            features, features_lens = self.preprocessor(
                input_signal=waveforms,
                length=waveforms_lens,
            )
            return features, features_lens

    if mode == "dft":
        wrapper = DFTConvPreprocessorWrapper(model.preprocessor)
    elif mode == "custom":
        wrapper = CustomPreprocessorWrapper(model.preprocessor)
    else:
        wrapper = PreprocessorWrapper(model.preprocessor)
    wrapper.eval()
    wrapper.to(model.device)

    sample_rate = int(getattr(model.preprocessor, "_sample_rate", 16000))
    num_samples = max(int(dummy_seconds * sample_rate), sample_rate)
    batch = 1

    waveforms = torch.zeros((batch, num_samples), dtype=torch.float32, device=model.device)
    waveforms_lens = torch.tensor([num_samples], dtype=torch.int64, device=model.device)

    dst = output_dir / "nemo128.onnx"
    cleanup_target(dst)

    with torch.inference_mode():
        dynamic_axes = None
        if dynamic_axes_enabled:
            dynamic_axes = {
                "waveforms": {0: "batch", 1: "samples"},
                "waveforms_lens": {0: "batch"},
                "features": {0: "batch", 2: "frames"},
                "features_lens": {0: "batch"},
            }

        torch.onnx.export(
            wrapper,
            (waveforms, waveforms_lens),
            str(dst),
            input_names=["waveforms", "waveforms_lens"],
            output_names=["features", "features_lens"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=dynamo,
            optimize=optimize,
            verify=verify,
            do_constant_folding=do_constant_folding,
        )


def extract_vocab_entries(model: Any) -> list[tuple[int, str]]:
    vocab_by_id: dict[int, str] = {}

    decoding_vocab = getattr(getattr(model, "decoding", None), "vocabulary", None)
    if decoding_vocab:
        for idx, token in enumerate(decoding_vocab):
            vocab_by_id[idx] = str(token)

    tokenizer = getattr(model, "tokenizer", None)
    tokenizer_impl = getattr(tokenizer, "tokenizer", None)

    if hasattr(tokenizer_impl, "get_vocab"):
        raw_vocab = tokenizer_impl.get_vocab()
        if isinstance(raw_vocab, dict):
            for token, idx in raw_vocab.items():
                normalized_idx = int(idx) - 1 if int(idx) > 0 else int(idx)
                vocab_by_id[normalized_idx] = str(token)

    if not vocab_by_id:
        raise RuntimeError("Could not extract a tokenizer vocabulary from the NeMo model.")

    return sorted(vocab_by_id.items(), key=lambda item: item[0])


def write_vocab(model: Any, output_dir: Path) -> None:
    entries = extract_vocab_entries(model)
    blank_id = len(entries)
    blank_token = "<blk>"

    dst = output_dir / "vocab.txt"
    with dst.open("w", encoding="utf-8", newline="\n") as handle:
        for idx, token in entries:
            handle.write(f"{token} {idx}\n")

        if not any(token == blank_token for _, token in entries):
            handle.write(f"{blank_token} {blank_id}\n")


@dataclass
class ExportMetadata:
    nemo_file: str
    nemo_model_class: str
    nemo_model_name: str | None
    device: str
    opset: int
    sample_rate: int | None
    preprocessor_exported: bool
    notes: list[str]


def write_config(model: Any, omega_conf: Any, output_dir: Path, metadata: ExportMetadata) -> None:
    cfg = getattr(model, "cfg", None)
    payload: dict[str, Any] = {
        "export": asdict(metadata),
    }

    if cfg is not None:
        try:
            payload["nemo_cfg"] = omega_conf.to_container(cfg, resolve=True)
        except Exception:
            payload["nemo_cfg"] = str(cfg)

    with (output_dir / "config.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_report(output_dir: Path, metadata: ExportMetadata) -> None:
    with (output_dir / "export-report.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(metadata), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    nemo_path = Path(args.nemo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not nemo_path.exists():
        raise SystemExit(f".nemo file not found: {nemo_path}")

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    torch, ASRModel, OmegaConf, format_exc = import_dependencies()
    device = resolve_device(torch, args.device)

    model = ASRModel.restore_from(str(nemo_path), map_location=device)
    model.freeze()
    model.eval()
    model.to(device)

    notes: list[str] = []
    export_rnnt_pair(model, output_dir, args.opset)

    preprocessor_exported = False
    if args.skip_preprocessor:
        notes.append("Skipped preprocessor export because --skip-preprocessor was provided.")
    else:
        try:
            preprocessor_opset = args.preprocessor_opset or args.opset
            _is_custom_path = args.preprocessor_mode in ("custom", "dft")
            preprocessor_dynamo = resolve_export_bool(
                args.preprocessor_dynamo,
                default=False if _is_custom_path else True,
            )
            preprocessor_optimize = resolve_export_bool(
                args.preprocessor_optimize,
                default=False if _is_custom_path else True,
            )
            export_preprocessor(
                model,
                torch,
                output_dir,
                preprocessor_opset,
                args.preprocessor_seconds,
                mode=args.preprocessor_mode,
                dynamo=preprocessor_dynamo,
                optimize=preprocessor_optimize,
                verify=args.preprocessor_verify,
                do_constant_folding=not args.preprocessor_no_constant_folding,
                dynamic_axes_enabled=not args.preprocessor_static_shape,
            )
            preprocessor_exported = True
            notes.append(
                "Preprocessor export settings: "
                f"mode={args.preprocessor_mode}, opset={preprocessor_opset}, dynamo={preprocessor_dynamo}, optimize={preprocessor_optimize}, "
                f"verify={args.preprocessor_verify}, constant_folding={not args.preprocessor_no_constant_folding}, "
                f"dynamic_axes={not args.preprocessor_static_shape}"
            )
        except Exception as exc:
            notes.append(
                "Preprocessor ONNX export failed. encoder-model.onnx and "
                f"decoder_joint-model.onnx were still exported successfully. Failure: {format_exc(exc)}"
            )

    write_vocab(model, output_dir)

    cfg_target = getattr(model, "cfg", None)
    metadata = ExportMetadata(
        nemo_file=str(nemo_path),
        nemo_model_class=type(model).__name__,
        nemo_model_name=getattr(cfg_target, "name", None) if cfg_target is not None else None,
        device=device,
        opset=args.opset,
        sample_rate=int(getattr(getattr(model, "preprocessor", None), "_sample_rate", 0) or 0) or None,
        preprocessor_exported=preprocessor_exported,
        notes=notes,
    )

    write_config(model, OmegaConf, output_dir, metadata)
    write_report(output_dir, metadata)

    print("Export complete:")
    for name in (
        "encoder-model.onnx",
        "encoder-model.onnx.data",
        "decoder_joint-model.onnx",
        "decoder_joint-model.onnx.data",
        "nemo128.onnx",
        "vocab.txt",
        "config.json",
        "export-report.json",
    ):
        path = output_dir / name
        if path.exists():
            print(f"  - {path}")

    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
