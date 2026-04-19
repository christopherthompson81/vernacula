"""Build a manifest.json for the shipping artifacts.

Matches the schema ModelManagerService uses for integrity checks:

    {
      "files": {
        "<remote_relative_path>": { "md5": "<lowercase hex>" },
        ...
      }
    }

Run from the model artifacts dir after a successful export:

    python make_manifest.py --model-dir ../../../artifacts/voxlingua107

Writes <model-dir>/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory containing voxlingua107.onnx + lang_map.json.")
    p.add_argument("--files", nargs="+",
                   default=["voxlingua107.onnx", "lang_map.json"],
                   help="File names (relative to --model-dir) to include in the manifest.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: <model-dir>/manifest.json).")
    args = p.parse_args()

    files_entry: dict[str, dict[str, str]] = {}
    for rel in args.files:
        path = args.model_dir / rel
        if not path.exists():
            raise SystemExit(f"missing: {path}")
        digest = md5_of(path)
        size_mb = path.stat().st_size / 1024 / 1024
        files_entry[rel] = {"md5": digest}
        print(f"  {rel:<30s}  md5={digest}  size={size_mb:6.2f} MiB", file=sys.stderr)

    out_path = args.out or (args.model_dir / "manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"files": files_entry}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
