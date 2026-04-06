#!/usr/bin/env python3
"""
setup_nemo_export_env.py

Creates a Python virtual environment for NeMo ONNX export and installs
the required dependencies.

PyTorch is installed from the CUDA-specific index by default so that GPU
acceleration is available during export.  Pass --cuda-version "" to install
the CPU wheel instead.

Usage:
    python scripts/nemo_export/setup_nemo_export_env.py
    python scripts/nemo_export/setup_nemo_export_env.py --python python3.12
    python scripts/nemo_export/setup_nemo_export_env.py --cuda-version cu121
    python scripts/nemo_export/setup_nemo_export_env.py --cuda-version ""
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"


def run(*cmd: str) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(list(cmd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the NeMo export venv and install dependencies."
    )
    parser.add_argument(
        "--python", default="python3",
        help="Python interpreter to use for venv creation (default: python3)",
    )
    parser.add_argument(
        "--venv-dir", default=str(REPO_ROOT / ".venv-nemo-export"),
        help="Path to the virtual environment directory",
    )
    parser.add_argument(
        "--cuda-version", default="cu128",
        help="PyTorch CUDA build tag, e.g. cu121, cu128 (pass '' for CPU wheel)",
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

    run(python, "-m", "pip", "install", "-r", str(REQUIREMENTS))

    print(f"\nEnvironment ready. Activate with:")
    print(f"  source {venv}/bin/activate")
    print(f"\nExample export:")
    print(f"  python scripts/nemo_export/export_parakeet_nemo_to_onnx.py \\")
    print(f"    --nemo ~/models/parakeet-tdt-0.6b-v3.nemo \\")
    print(f"    --output-dir ~/models/parakeet_onnx")


if __name__ == "__main__":
    main()
