"""Upload the VoxLingua107 ONNX + lang map + manifest to a HuggingFace Hub repo.

One-time setup (outside this script):
    1. Create a model repo at https://huggingface.co/new — public, name
       'voxlingua107-lid-onnx' under your namespace.
    2. `huggingface-cli login` with a write token.

Then from the repo root:

    python make_manifest.py --model-dir ../../../artifacts/voxlingua107
    python upload_to_hf.py  --model-dir ../../../artifacts/voxlingua107 \\
                            --repo-id christopherthompson81/voxlingua107-lid-onnx

The script uploads each tracked file individually so progress is visible
and re-runs can resume (HF skips identical uploads).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_FILES = ["voxlingua107.onnx", "lang_map.json", "manifest.json"]
README_DEFAULT = """\
# VoxLingua107 ECAPA-TDNN — ONNX

Re-packaged ONNX export of
[speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
for use as the language-identification backend in the
[Vernacula](https://github.com/christopherthompson81/vernacula) app.

## Contents

| File                     | Purpose                                                   |
|--------------------------|-----------------------------------------------------------|
| `voxlingua107.onnx`      | End-to-end graph: raw 16 kHz audio → 107-class logits + 256-dim embedding |
| `lang_map.json`          | Class index → `{ iso, name }` lookup                     |
| `manifest.json`          | Per-file MD5 hashes for integrity checks                 |

Preprocessing (FBANK via Conv1D, per-utterance mean-variance norm) is
folded into the graph, so consumers just send raw PCM.

## Export provenance

Exported via
[public/scripts/voxlingua107_export/](https://github.com/christopherthompson81/vernacula/tree/main/scripts/voxlingua107_export).
The `STFT` op is replaced with two `Conv1D` passes (cos + sin basis,
windowed) so the preprocessing path has CUDA kernels end-to-end —
roughly a 27× speedup on CUDA vs the stock SpeechBrain export.

## License

Inherits Apache 2.0 from the SpeechBrain source model.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory containing the artifacts to upload.")
    p.add_argument("--repo-id", required=True,
                   help="e.g. christopherthompson81/voxlingua107-lid-onnx")
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                   help="File names (relative to --model-dir) to upload.")
    p.add_argument("--write-readme", action="store_true",
                   help="Upload a boilerplate README.md at the repo root "
                        "(safe to run once; don't run repeatedly if you've "
                        "hand-edited the README on the Hub).")
    p.add_argument("--create-repo", action="store_true",
                   help="Attempt create_repo(exist_ok=True) before uploading.")
    p.add_argument("--private", action="store_true",
                   help="With --create-repo: make the repo private (default public).")
    p.add_argument("--commit-message", default="voxlingua107 onnx artifacts")
    args = p.parse_args()

    api = HfApi()

    if args.create_repo:
        print(f"[upload] ensuring repo {args.repo_id} exists", file=sys.stderr)
        create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    for rel in args.files:
        path = args.model_dir / rel
        if not path.exists():
            raise SystemExit(f"missing: {path}")
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"[upload] {rel} ({size_mb:.2f} MiB) → {args.repo_id}/{rel}",
              file=sys.stderr)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    if args.write_readme:
        readme_path = args.model_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(README_DEFAULT, encoding="utf-8")
            print(f"[upload] wrote boilerplate {readme_path}", file=sys.stderr)
        print(f"[upload] README.md → {args.repo_id}/README.md", file=sys.stderr)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="README",
        )

    print(f"[upload] done: https://huggingface.co/{args.repo_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
