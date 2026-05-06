"""Build a `manifest.json` for any Vernacula ONNX shipping bundle.

Single-source-of-truth manifest builder shared across every
`scripts/*_export/` pipeline. The output schema is the one
`Vernacula.Avalonia` reads at runtime:

    { "files": { "<filename>": { "md5": "<lowercase hex>" } } }

Usage:

    # Hash a known list of files
    python scripts/make_manifest.py --model-dir ~/models/voxlingua107 \\
        --files voxlingua107.onnx lang_map.json

    # Or hash every non-hidden file in the dir, recursive
    python scripts/make_manifest.py --model-dir ~/models/voxlingua107 --all

Writes `<model-dir>/manifest.json` (or `--out` if given).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing scripts/_export_utils when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _export_utils.manifest import build_manifest  # noqa: E402


def discover_files(model_dir: Path) -> list[str]:
    """Sorted list of non-hidden files under model_dir, excluding manifest.json itself."""
    rel = []
    for p in sorted(model_dir.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(model_dir).parts):
            continue
        relpath = p.relative_to(model_dir).as_posix()
        if relpath == "manifest.json":
            continue
        rel.append(relpath)
    return rel


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory containing the bundle artifacts.")
    p.add_argument("--files", nargs="+", default=None,
                   help="Files (relative to --model-dir) to include. Mutually exclusive with --all.")
    p.add_argument("--all", action="store_true",
                   help="Include every non-hidden file under --model-dir (recursive).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: <model-dir>/manifest.json).")
    args = p.parse_args()

    if not args.files and not args.all:
        p.error("specify --files <names> or --all")
    if args.files and args.all:
        p.error("--files and --all are mutually exclusive")

    files = args.files if args.files else discover_files(args.model_dir)
    if not files:
        p.error(f"no files found under {args.model_dir}")

    try:
        manifest = build_manifest(args.model_dir, files)
    except FileNotFoundError as e:
        sys.exit(f"missing: {e}")

    for rel, entry in manifest["files"].items():
        size_mb = (args.model_dir / rel).stat().st_size / 1024 / 1024
        print(f"  {rel:<40s}  md5={entry['md5']}  size={size_mb:7.2f} MiB",
              file=sys.stderr)

    out = args.out or (args.model_dir / "manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
