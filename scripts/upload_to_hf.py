"""Upload a Vernacula ONNX bundle to a HuggingFace Hub repo.

Generic uploader replacing the per-export `upload_to_hf.py` scripts.
Defaults to syncing the model card from
`scripts/hf_readmes/<repo-basename>/README.md` so the source repo stays
the single source of truth for what's on HF.

One-time setup (outside this script):
  1. Create the model repo at https://huggingface.co/new (or pass
     `--create-repo`).
  2. `huggingface-cli login` with a write token.

Usage:

    # Upload artifacts only (manifest.json must already exist — run
    # scripts/make_manifest.py first if not).
    python scripts/upload_to_hf.py \\
        --model-dir ~/models/voxlingua107 \\
        --repo-id christopherthompson81/voxlingua107-lid-onnx

    # Sync the model card too (from scripts/hf_readmes/<basename>/README.md)
    python scripts/upload_to_hf.py \\
        --model-dir ~/models/voxlingua107 \\
        --repo-id christopherthompson81/voxlingua107-lid-onnx \\
        --sync-readme

    # Override the README source path
    python scripts/upload_to_hf.py \\
        --model-dir ~/models/voxlingua107 \\
        --repo-id christopherthompson81/voxlingua107-lid-onnx \\
        --sync-readme --readme path/to/README.md

Each file is uploaded individually so progress is visible and re-runs
resume cleanly (HF skips identical uploads).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


HF_README_DIR = Path(__file__).resolve().parent / "hf_readmes"


def default_readme_for_repo(repo_id: str) -> Path:
    """Canonical `scripts/hf_readmes/<basename>/README.md` for a repo id."""
    basename = repo_id.split("/", 1)[-1]
    return HF_README_DIR / basename / "README.md"


def discover_files(model_dir: Path) -> list[str]:
    """Sorted list of non-hidden files under model_dir, excluding README.md."""
    rel = []
    for p in sorted(model_dir.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(model_dir).parts):
            continue
        relpath = p.relative_to(model_dir).as_posix()
        if relpath == "README.md":
            # README is uploaded via --sync-readme from a separate source.
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
    p.add_argument("--repo-id", required=True,
                   help="e.g. christopherthompson81/voxlingua107-lid-onnx")
    p.add_argument("--files", nargs="+", default=None,
                   help="File names (relative to --model-dir) to upload. "
                        "Default: every non-hidden file under --model-dir.")
    p.add_argument("--create-repo", action="store_true",
                   help="Run create_repo(exist_ok=True) before uploading.")
    p.add_argument("--private", action="store_true",
                   help="With --create-repo: make the repo private (default public).")
    p.add_argument("--sync-readme", action="store_true",
                   help="Upload the model card from scripts/hf_readmes/<basename>/README.md "
                        "(or --readme PATH).")
    p.add_argument("--readme", type=Path, default=None,
                   help="Override the README source for --sync-readme.")
    p.add_argument("--commit-message", default="sync from vernacula repo")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without uploading.")
    args = p.parse_args()

    api = HfApi()

    if args.create_repo and not args.dry_run:
        print(f"[upload] ensuring repo {args.repo_id} exists", file=sys.stderr)
        create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    files = args.files if args.files else discover_files(args.model_dir)
    if not files:
        sys.exit(f"no files found under {args.model_dir}")

    for rel in files:
        path = args.model_dir / rel
        if not path.exists():
            sys.exit(f"missing: {path}")
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"[upload] {rel} ({size_mb:.2f} MiB) -> {args.repo_id}/{rel}",
              file=sys.stderr)
        if args.dry_run:
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    if args.sync_readme:
        readme = args.readme or default_readme_for_repo(args.repo_id)
        if not readme.exists():
            sys.exit(f"README source not found: {readme}\n"
                     f"Either create it under scripts/hf_readmes/ or pass --readme PATH.")
        print(f"[upload] {readme} -> {args.repo_id}/README.md", file=sys.stderr)
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(readme),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=args.commit_message,
            )

    print(f"[upload] done: https://huggingface.co/{args.repo_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
