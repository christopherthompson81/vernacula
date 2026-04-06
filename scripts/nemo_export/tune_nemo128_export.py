from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf


@dataclass
class CandidateResult:
    name: str
    command: list[str]
    success: bool
    note: str | None
    feature_lengths: dict[str, list[int]] | None
    feature_diff: dict[str, Any] | None
    encoder_diff: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and score multiple nemo128.onnx candidates against a legacy reference."
    )
    parser.add_argument("--nemo", required=True, help="Path to parakeet-tdt-0.6b-v3.nemo")
    parser.add_argument("--audio", required=True, help="Wave file used for candidate scoring.")
    parser.add_argument("--legacy-model-dir", required=True, help="Directory containing the legacy nemo128/encoder bundle.")
    parser.add_argument("--output-root", required=True, help="Directory where candidate exports will be created.")
    parser.add_argument("--report", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--export-script",
        default=str(Path(__file__).resolve().parent / "export_parakeet_nemo_to_onnx.py"),
        help="Path to export_parakeet_nemo_to_onnx.py",
    )
    return parser.parse_args()


def load_audio(path: Path, max_seconds: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    wave, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if int(sample_rate) != 16000:
        raise SystemExit(f"Expected 16 kHz audio, got {sample_rate} for {path}")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    max_samples = int(max_seconds * sample_rate)
    wave = np.asarray(wave[:max_samples], dtype=np.float32)[None, :]
    lengths = np.asarray([wave.shape[1]], dtype=np.int64)
    return wave, lengths


def compare(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    common_shape = [min(int(a), int(b)) for a, b in zip(lhs.shape, rhs.shape)]
    slices = tuple(slice(0, dim) for dim in common_shape)
    lhs_view = lhs[slices].astype(np.float64)
    rhs_view = rhs[slices].astype(np.float64)
    delta = lhs_view - rhs_view
    lhs_flat = lhs_view.reshape(-1)
    rhs_flat = rhs_view.reshape(-1)
    lhs_norm = np.linalg.norm(lhs_flat)
    rhs_norm = np.linalg.norm(rhs_flat)
    cosine = 1.0
    if lhs_norm == 0.0 and rhs_norm == 0.0:
        cosine = 1.0
    elif lhs_norm == 0.0 or rhs_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))
    return {
        "lhs_shape": [int(x) for x in lhs.shape],
        "rhs_shape": [int(x) for x in rhs.shape],
        "compared_shape": common_shape,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(np.max(np.abs(delta))),
        "cosine_similarity": cosine,
    }


def run_preprocessor(session: ort.InferenceSession, wave: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features, feature_lengths = session.run(
        None,
        {
            session.get_inputs()[0].name: wave,
            session.get_inputs()[1].name: lengths,
        },
    )
    return np.asarray(features), np.asarray(feature_lengths)


def run_encoder(session: ort.InferenceSession, features: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded, encoded_lengths = session.run(
        None,
        {
            session.get_inputs()[0].name: features.astype(np.float32),
            session.get_inputs()[1].name: lengths.astype(np.int64),
        },
    )
    return np.asarray(encoded), np.asarray(encoded_lengths)


def candidate_commands(python_exe: str, export_script: Path, nemo_path: Path, output_root: Path) -> list[tuple[str, list[str]]]:
    base = [
        python_exe,
        str(export_script),
        "--nemo",
        str(nemo_path),
        "--overwrite",
        "--skip-preprocessor",
    ]
    pre_base = [
        python_exe,
        str(export_script),
        "--nemo",
        str(nemo_path),
        "--overwrite",
    ]
    return [
        (
            "legacyish_opset17_dynamo_false",
            [
                *pre_base,
                "--output-dir",
                str(output_root / "legacyish_opset17_dynamo_false"),
                "--opset",
                "17",
                "--preprocessor-opset",
                "17",
                "--preprocessor-dynamo",
                "false",
                "--preprocessor-optimize",
                "false",
            ],
        ),
        (
            "legacyish_opset17_dynamo_false_static",
            [
                *pre_base,
                "--output-dir",
                str(output_root / "legacyish_opset17_dynamo_false_static"),
                "--opset",
                "17",
                "--preprocessor-opset",
                "17",
                "--preprocessor-dynamo",
                "false",
                "--preprocessor-optimize",
                "false",
                "--preprocessor-static-shape",
            ],
        ),
        (
            "opset17_dynamo_true",
            [
                *pre_base,
                "--output-dir",
                str(output_root / "opset17_dynamo_true"),
                "--opset",
                "17",
                "--preprocessor-opset",
                "17",
                "--preprocessor-dynamo",
                "true",
            ],
        ),
        (
            "opset17_dynamo_false_no_fold",
            [
                *pre_base,
                "--output-dir",
                str(output_root / "opset17_dynamo_false_no_fold"),
                "--opset",
                "17",
                "--preprocessor-opset",
                "17",
                "--preprocessor-dynamo",
                "false",
                "--preprocessor-optimize",
                "false",
                "--preprocessor-no-constant-folding",
            ],
        ),
        (
            "current_style_opset18",
            [
                *pre_base,
                "--output-dir",
                str(output_root / "current_style_opset18"),
                "--opset",
                "17",
                "--preprocessor-opset",
                "18",
            ],
        ),
        (
            "custom_config_opset17",
            [
                *pre_base,
                "--output-dir",
                str(output_root / "custom_config_opset17"),
                "--opset",
                "17",
                "--preprocessor-opset",
                "17",
                "--preprocessor-mode",
                "custom",
                "--preprocessor-dynamo",
                "false",
                "--preprocessor-optimize",
                "false",
            ],
        ),
        (
            "dft_conv_opset17",
            [
                *pre_base,
                "--output-dir",
                str(output_root / "dft_conv_opset17"),
                "--opset",
                "17",
                "--preprocessor-opset",
                "17",
                "--preprocessor-mode",
                "dft",
                "--preprocessor-dynamo",
                "false",
                "--preprocessor-optimize",
                "false",
            ],
        ),
        (
            "encoder_only_baseline",
            [
                *base,
                "--output-dir",
                str(output_root / "encoder_only_baseline"),
                "--opset",
                "17",
            ],
        ),
    ]


def main() -> None:
    args = parse_args()
    nemo_path = Path(args.nemo).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    legacy_dir = Path(args.legacy_model_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    export_script = Path(args.export_script).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    venv_root = Path(__file__).resolve().parents[2] / ".venv-nemo-export"
    python_exe = str(venv_root / "bin" / "python")
    wave, lengths = load_audio(audio_path)

    legacy_pre = ort.InferenceSession(str(legacy_dir / "nemo128.onnx"), providers=["CPUExecutionProvider"])
    legacy_enc = ort.InferenceSession(str(legacy_dir / "encoder-model.onnx"), providers=["CPUExecutionProvider"])
    legacy_features, legacy_feature_lengths = run_preprocessor(legacy_pre, wave, lengths)
    legacy_encoded, _ = run_encoder(legacy_enc, legacy_features, legacy_feature_lengths)

    results: list[CandidateResult] = []

    for name, command in candidate_commands(python_exe, export_script, nemo_path, output_root):
        candidate_dir = output_root / name
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            results.append(
                CandidateResult(
                    name=name,
                    command=command,
                    success=False,
                    note=f"export failed: {proc.stderr or proc.stdout}",
                    feature_lengths=None,
                    feature_diff=None,
                    encoder_diff=None,
                )
            )
            continue

        pre_path = candidate_dir / "nemo128.onnx"
        enc_path = candidate_dir / "encoder-model.onnx"
        if not pre_path.exists():
            results.append(
                CandidateResult(
                    name=name,
                    command=command,
                    success=False,
                    note="candidate did not produce nemo128.onnx",
                    feature_lengths=None,
                    feature_diff=None,
                    encoder_diff=None,
                )
            )
            continue

        candidate_pre = ort.InferenceSession(str(pre_path), providers=["CPUExecutionProvider"])
        candidate_enc = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
        candidate_features, candidate_feature_lengths = run_preprocessor(candidate_pre, wave, lengths)
        candidate_encoded, _ = run_encoder(candidate_enc, candidate_features, candidate_feature_lengths)
        results.append(
            CandidateResult(
                name=name,
                command=command,
                success=True,
                note=None,
                feature_lengths={
                    "legacy": legacy_feature_lengths.tolist(),
                    "candidate": candidate_feature_lengths.tolist(),
                },
                feature_diff=compare(legacy_features, candidate_features),
                encoder_diff=compare(legacy_encoded, candidate_encoded),
            )
        )

    payload = {
        "audio": str(audio_path),
        "nemo": str(nemo_path),
        "legacy_model_dir": str(legacy_dir),
        "output_root": str(output_root),
        "results": [asdict(item) for item in results],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {report_path}")


if __name__ == "__main__":
    main()
