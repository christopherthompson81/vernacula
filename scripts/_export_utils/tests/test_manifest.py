"""Regression test for the manifest contract.

Locks the `manifest.json` schema and serialization conventions read by
`Vernacula.Avalonia/Services/ModelManagerService.cs::ParseManifestHashes`
— if any of these assertions fail, the C# download verifier might
silently reject packages built with the new helpers.

Run as:

    python scripts/_export_utils/tests/test_manifest.py

Exits 0 on success, 1 on failure. No external dependencies.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# scripts/_export_utils/tests/test_manifest.py — `scripts/` is parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _export_utils.manifest import (  # noqa: E402
    build_manifest,
    dump_manifest,
    md5_of_file,
    write_manifest,
)


# Pre-computed: md5("hello\n") and md5("hello") respectively.
HELLO_NL_MD5 = "b1946ac92492d2347c6235b4d2611184"
HELLO_MD5 = "5d41402abc4b2a76b9719d911017c592"

# Locked golden output. Any drift here indicates a contract change that
# the C# verifier might silently tolerate but the manifest format has
# diverged from prior bundles.
GOLDEN_MANIFEST = (
    '{\n'
    '  "files": {\n'
    '    "a.bin": {\n'
    '      "md5": "b1946ac92492d2347c6235b4d2611184"\n'
    '    },\n'
    '    "subdir/b.bin": {\n'
    '      "md5": "5d41402abc4b2a76b9719d911017c592"\n'
    '    }\n'
    '  }\n'
    '}\n'
)


def test_md5_of_file_matches_known_digest() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x"
        path.write_bytes(b"hello\n")
        assert md5_of_file(path) == HELLO_NL_MD5, (
            f"expected {HELLO_NL_MD5}, got {md5_of_file(path)}"
        )


def test_md5_of_file_lowercase() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x"
        path.write_bytes(b"hello")
        digest = md5_of_file(path)
        assert digest == digest.lower(), f"digest must be lowercase: {digest}"


def test_build_manifest_preserves_input_order() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "z.bin").write_bytes(b"hello\n")
        (root / "a.bin").write_bytes(b"hello\n")
        m = build_manifest(root, ["z.bin", "a.bin"])
        keys = list(m["files"].keys())
        assert keys == ["z.bin", "a.bin"], (
            f"build_manifest reordered files: {keys}"
        )


def test_build_manifest_missing_file_raises() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            build_manifest(Path(d), ["nonexistent.bin"])
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError for missing file")


def test_serialized_manifest_matches_golden() -> None:
    """The full byte-for-byte output is the actual contract."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.bin").write_bytes(b"hello\n")
        (root / "subdir").mkdir()
        (root / "subdir" / "b.bin").write_bytes(b"hello")

        out = root / "manifest.json"
        write_manifest(root, ["a.bin", "subdir/b.bin"], out)
        actual = out.read_text(encoding="utf-8")

        if actual != GOLDEN_MANIFEST:
            raise AssertionError(
                "manifest serialization drifted.\n"
                f"--- expected ---\n{GOLDEN_MANIFEST}"
                f"--- actual ---\n{actual}"
            )


def test_dump_manifest_round_trips() -> None:
    """`build_manifest` + `dump_manifest` produces same bytes as `write_manifest`."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.bin").write_bytes(b"hello\n")

        via_write = root / "via-write.json"
        write_manifest(root, ["a.bin"], via_write)

        via_dump = root / "via-dump.json"
        dump_manifest(build_manifest(root, ["a.bin"]), via_dump)

        a = via_write.read_bytes()
        b = via_dump.read_bytes()
        assert a == b, "write_manifest and build+dump produced different bytes"


def main() -> None:
    tests = [
        ("md5_of_file matches known digest", test_md5_of_file_matches_known_digest),
        ("md5_of_file is lowercase", test_md5_of_file_lowercase),
        ("build_manifest preserves input order", test_build_manifest_preserves_input_order),
        ("build_manifest raises on missing file", test_build_manifest_missing_file_raises),
        ("serialized manifest matches golden", test_serialized_manifest_matches_golden),
        ("dump_manifest round-trips", test_dump_manifest_round_trips),
    ]

    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL  {name}\n  {e}", file=sys.stderr)
            failures += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
        else:
            print(f"ok    {name}")

    if failures:
        print(f"\n{failures} of {len(tests)} tests failed", file=sys.stderr)
        sys.exit(1)
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
