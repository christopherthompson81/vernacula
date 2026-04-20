#!/usr/bin/env python3
"""
setup_indicconformer_env.py

Creates a Python virtual environment for IndicConformer ONNX export and
installs AI4Bharat's NeMo fork (the nemo-v2 branch implements the
multi-softmax ASR models used by IndicConformer).

This env is deliberately separate from `.venv-nemo-export`: the AI4Bharat
fork pins different versions of NeMo/PyTorch internals, and mixing them
would break Parakeet export.

Usage:
    python scripts/indicconformer_export/setup_indicconformer_env.py
    python scripts/indicconformer_export/setup_indicconformer_env.py --cuda-version cu121
    python scripts/indicconformer_export/setup_indicconformer_env.py --cuda-version ""
"""

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"

AI4BHARAT_NEMO_REPO = "https://github.com/AI4Bharat/NeMo.git"
AI4BHARAT_NEMO_BRANCH = "nemo-v2"


def run(*cmd: str) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(list(cmd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the IndicConformer export venv and install dependencies."
    )
    parser.add_argument(
        "--python", default="python3",
        help="Python interpreter to use for venv creation (default: python3)",
    )
    parser.add_argument(
        "--venv-dir", default=str(REPO_ROOT / ".venv-indicconformer-export"),
        help="Path to the virtual environment directory",
    )
    parser.add_argument(
        "--cuda-version", default="cu128",
        help="PyTorch CUDA build tag, e.g. cu121, cu128 (pass '' for CPU wheel)",
    )
    parser.add_argument(
        "--nemo-clone-dir", default=str(REPO_ROOT / ".venv-indicconformer-export" / "_src" / "NeMo"),
        help="Where to clone the AI4Bharat NeMo fork.",
    )
    args = parser.parse_args()

    venv = Path(args.venv_dir)
    python = str(venv / "bin" / "python")

    run(args.python, "-m", "venv", str(venv))
    run(python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

    if args.cuda_version:
        torch_index = f"https://download.pytorch.org/whl/{args.cuda_version}"
        print(f"\nInstalling PyTorch with CUDA ({args.cuda_version}) from {torch_index}")
        run(python, "-m", "pip", "install", "torch", "--index-url", torch_index)
        run(python, "-m", "pip", "install", "onnxruntime-gpu>=1.20,<2")
    else:
        run(python, "-m", "pip", "install", "onnxruntime>=1.20,<2")

    run(python, "-m", "pip", "install", "-r", str(REQUIREMENTS))

    # Clone AI4Bharat's NeMo fork (nemo-v2 branch) and install it editable.
    clone_dir = Path(args.nemo_clone_dir)
    if not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--branch", AI4BHARAT_NEMO_BRANCH, "--depth", "1",
            AI4BHARAT_NEMO_REPO, str(clone_dir))
    else:
        print(f"\nAI4Bharat NeMo clone already present at {clone_dir}; skipping clone.")

    # The fork's `./reinstall.sh` pulls NVIDIA Apex, Megatron, and other heavy
    # training deps we don't need for inference+export. Use the documented
    # lightweight path instead.
    print("\nInstalling AI4Bharat NeMo fork (editable, ASR extras)")
    run(python, "-m", "pip", "install", "-e", f"{clone_dir}[asr]")

    print(f"\nEnvironment ready. Activate with:")
    print(f"  source {venv}/bin/activate")
    print(f"\nPhase 1 discovery:")
    print(f"  python scripts/indicconformer_export/inspect_indicconformer.py \\")
    print(f"    --hf-repo ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large")


if __name__ == "__main__":
    main()
