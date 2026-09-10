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

    # --all, minus local-only files (ONNX Runtime optimisation caches)
    python scripts/make_manifest.py --model-dir ~/models/kokoro --all \\
        --exclude '*.ort' '*.use-ort'

    # Add ONE file to an already-published bundle without dropping the rest
    python scripts/make_manifest.py --model-dir ~/models \\
        --files new_variant.onnx --merge-into ./published-manifest.json \\
        --out ~/models/manifest.json

Writes `<model-dir>/manifest.json` (or `--out` if given).

⚠ Without `--merge-into` the output contains ONLY the files named, and is written
wholesale. Pointing `--files` at a live bundle's manifest therefore deletes every
entry you did not name.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

# Allow importing scripts/_export_utils when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _export_utils.manifest import build_manifest, dump_manifest  # noqa: E402


def discover_files(model_dir: Path, exclude: list[str] | None = None) -> list[str]:
    """Sorted list of non-hidden files under model_dir, excluding manifest.json
    itself, README.md (synced separately by upload_to_hf.py) and anything
    matching an `exclude` glob (relative path or bare file name)."""
    rel = []
    for p in sorted(model_dir.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(model_dir).parts):
            continue
        relpath = p.relative_to(model_dir).as_posix()
        if relpath in ("manifest.json", "README.md"):
            continue
        name = relpath.rsplit("/", 1)[-1]
        if exclude and any(fnmatch.fnmatch(relpath, g) or fnmatch.fnmatch(name, g) for g in exclude):
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
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: <model-dir>/manifest.json).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", nargs="+", default=None,
                     help="Files (relative to --model-dir) to include.")
    src.add_argument("--all", action="store_true",
                     help="Include every non-hidden file under --model-dir (recursive).")
    p.add_argument("--exclude", nargs="+", default=None, metavar="GLOB",
                   help="With --all: skip files matching these globs, e.g. '*.ort' '*.use-ort'.")
    p.add_argument("--merge-into", type=Path, default=None, metavar="MANIFEST",
                   help="Start from an existing manifest and add/replace only the entries "
                        "built here, instead of writing a manifest containing only them. "
                        "Use a published manifest.json when adding one file to a live "
                        "bundle -- without this the other entries are dropped.")
    args = p.parse_args()

    files = args.files if args.files else discover_files(args.model_dir, args.exclude)
    if not files:
        p.error(f"no files found under {args.model_dir}")

    try:
        manifest = build_manifest(args.model_dir, files)
    except FileNotFoundError as e:
        sys.exit(f"missing: {e}")

    built = manifest["files"]          # only these were hashed here; report on these

    if args.merge_into is not None:
        # ⚠ WITHOUT THIS, PUBLISHING ONE NEW FILE DELETES THE REST OF THE BUNDLE.
        # The result is written wholesale, so `--files <one-new-file>` against a live
        # manifest replaces nine entries with one and the app then fails to validate
        # every other asset. Merging keeps unrelated entries untouched, including any
        # extra keys (e.g. "version", per-entry "size") the published file carries that
        # build_manifest does not emit.
        if not args.merge_into.exists():
            sys.exit(f"--merge-into: no such file: {args.merge_into}")
        try:
            base = json.loads(args.merge_into.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            sys.exit(f"--merge-into: cannot read {args.merge_into}: {e}")
        if not isinstance(base.get("files"), dict):
            sys.exit(f"--merge-into: {args.merge_into} has no \"files\" object")

        added = [k for k in manifest["files"] if k not in base["files"]]
        replaced = [k for k in manifest["files"] if k in base["files"]]
        for rel, entry in manifest["files"].items():
            # Preserve any extra keys already on an entry being replaced (e.g. "size").
            merged = dict(base["files"].get(rel, {}))
            merged.update(entry)
            base["files"][rel] = merged
        base["files"] = dict(sorted(base["files"].items()))
        manifest = base
        print(f"  merged into {args.merge_into}: "
              f"{len(added)} added, {len(replaced)} replaced, "
              f"{len(base['files'])} total", file=sys.stderr)

    for rel, entry in built.items():
        size_mb = (args.model_dir / rel).stat().st_size / 1024 / 1024
        print(f"  {rel:<40s}  md5={entry['md5']}  size={size_mb:7.2f} MiB",
              file=sys.stderr)

    out = dump_manifest(manifest, args.out or (args.model_dir / "manifest.json"))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
