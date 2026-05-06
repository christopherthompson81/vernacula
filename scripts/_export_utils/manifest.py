"""Single source of truth for `manifest.json` shape and MD5 hashing.

The output matches the schema that
`Vernacula.Avalonia/Services/ModelManagerService.cs::ParseManifestHashes`
reads:

    {
      "files": {
        "<filename>": { "md5": "<lowercase hex>" },
        ...
      }
    }

Conventions: chunked 1 MiB read (matches existing C# `ComputeMd5`),
lowercase hex digest, `json.dumps(..., indent=2)` plus a trailing
newline. The C# parser is case-insensitive, so the case convention is
informational; the trailing newline keeps the file POSIX-clean.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHUNK_BYTES = 1 << 20  # 1 MiB — matches the existing chunked reads


def md5_of_file(path: Path) -> str:
    """Lowercase hex MD5 of a file, read in 1 MiB chunks."""
    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def build_manifest(model_dir: Path, files: list[str]) -> dict:
    """Build a manifest dict for `files` (paths relative to `model_dir`).

    Raises `FileNotFoundError` if any listed file is missing — caller
    decides how to handle it. Order of keys in the output preserves the
    order of `files`, since dict insertion order is stable in Python 3.7+.
    """
    model_dir = Path(model_dir)
    files_entry: dict[str, dict[str, str]] = {}
    for rel in files:
        path = model_dir / rel
        if not path.exists():
            raise FileNotFoundError(path)
        files_entry[rel] = {"md5": md5_of_file(path)}
    return {"files": files_entry}


def dump_manifest(manifest: dict, out_path: Path) -> Path:
    """Serialize a manifest dict to disk in the canonical format.

    The serialization conventions (`indent=2`, trailing newline, UTF-8)
    live here. Callers that already have a `manifest` dict from
    `build_manifest()` should use this; one-shot callers should use
    `write_manifest()`. Returns the path that was written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def write_manifest(
    model_dir: Path,
    files: list[str],
    out_path: Path | None = None,
) -> Path:
    """Build and write `manifest.json` under `model_dir` (or `out_path`).

    Convenience wrapper around `build_manifest()` + `dump_manifest()`.
    Returns the path that was written.
    """
    model_dir = Path(model_dir)
    manifest = build_manifest(model_dir, files)
    out = Path(out_path) if out_path else model_dir / "manifest.json"
    return dump_manifest(manifest, out)
