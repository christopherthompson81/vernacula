"""Shared utilities for Vernacula's ONNX export pipelines.

These helpers exist to prevent format drift across exports — most
importantly the `manifest.json` shape read by Vernacula.Avalonia's
`ModelManagerService.ParseManifestHashes`.

Currently provides:
  - `manifest`: build / write `manifest.json` for a model directory

See https://github.com/christopherthompson81/vernacula/issues/29 for
scope and rationale.

## Importing from a per-export script

The `_export_utils` package lives at `scripts/_export_utils/`, so a
script under `scripts/<some_export>/foo.py` needs to put `scripts/` on
`sys.path` before importing. Canonical idiom:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _export_utils.manifest import build_manifest, write_manifest

Scripts at the `scripts/` root (e.g. `make_manifest.py`) use
`parents[0]` instead of `parents[1]` since they're already a sibling
of the package.
"""

from . import manifest  # noqa: F401
