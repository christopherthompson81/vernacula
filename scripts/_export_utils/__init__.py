"""Shared utilities for Vernacula's ONNX export pipelines.

These helpers exist to prevent format drift across exports — most
importantly the `manifest.json` shape read by Vernacula.Avalonia's
`ModelManagerService.ParseManifestHashes`.

Currently provides:
  - `manifest`: build / write `manifest.json` for a model directory

See https://github.com/christopherthompson81/vernacula/issues/29 for
scope and rationale.
"""

from . import manifest  # noqa: F401
