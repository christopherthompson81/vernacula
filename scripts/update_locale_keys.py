#!/usr/bin/env python3
"""
scripts/update_locale_keys.py

Detects keys present in en.json but missing from any other locale file,
translates them via the Anthropic API, and inserts them in the correct
position (matching the key order in en.json).

Features:
  - Auto-detects missing keys by diffing en.json against each locale file
  - Inserts each missing key immediately after its nearest preceding neighbour
    that already exists in the target file
  - SQLite translation cache: cached results are reused across runs
  - --force   : retranslate everything regardless of cache
  - --lang    : comma-separated language codes to target (e.g. bg,de)
  - --dry-run : report missing keys without translating or writing

Usage:
    python scripts/update_locale_keys.py
    python scripts/update_locale_keys.py --lang de,fr
    python scripts/update_locale_keys.py --force
    python scripts/update_locale_keys.py --dry-run
    ANTHROPIC_API_KEY=sk-ant-... python scripts/update_locale_keys.py
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import anthropic

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
LOCALES = ROOT / "src" / "Vernacula.Avalonia" / "Locales"
EN_JSON = LOCALES / "en.json"
DB_PATH = LOCALES / "locale_keys.db"

# ---------------------------------------------------------------------------
# Language table: (locale_code, display_name)
# ---------------------------------------------------------------------------
LANGUAGES = [
    ("bg", "Bulgarian"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("it", "Italian"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mt", "Maltese"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sv", "Swedish"),
    ("uk", "Ukrainian"),
]

TECH_TERMS = "ASR, SQLite, ONNX, GPU, CUDA, TensorRT"

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def db_init(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS translations (
            key_name        TEXT NOT NULL,
            lang_code       TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            source_english  TEXT NOT NULL,
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (key_name, lang_code)
        );
    """)
    con.commit()


def db_get(con: sqlite3.Connection, key: str, lang: str) -> tuple[str | None, str | None]:
    row = con.execute(
        "SELECT translated_text, source_english FROM translations "
        "WHERE key_name=? AND lang_code=?",
        (key, lang),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def db_upsert(
    con: sqlite3.Connection, key: str, lang: str, text: str, source_english: str
) -> None:
    con.execute("""
        INSERT INTO translations(key_name, lang_code, translated_text, source_english)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key_name, lang_code) DO UPDATE SET
            translated_text = excluded.translated_text,
            source_english  = excluded.source_english,
            updated_at      = datetime('now')
    """, (key, lang, text, source_english))
    con.commit()


# ---------------------------------------------------------------------------
# Key detection
# ---------------------------------------------------------------------------

def load_json_keys(path: Path) -> dict[str, str]:
    """Load a JSON locale file and return its key→value dict (preserving order)."""
    return json.loads(path.read_text(encoding="utf-8"))


def find_missing_keys(en_data: dict[str, str], lang_path: Path) -> list[str]:
    """Return en.json keys that are absent from the given locale file, in en.json order."""
    if not lang_path.exists():
        return list(en_data.keys())
    lang_data = load_json_keys(lang_path)
    return [k for k in en_data if k not in lang_data]


def anchor_for_key(en_keys: list[str], target_key: str, present_keys: set[str]) -> str | None:
    """
    Find the closest preceding key in en_keys that is already present in the
    target locale file — this becomes the insertion anchor.
    Returns None if no preceding key exists in the target file.
    """
    idx = en_keys.index(target_key)
    for i in range(idx - 1, -1, -1):
        if en_keys[i] in present_keys:
            return en_keys[i]
    return None


# ---------------------------------------------------------------------------
# JSON file editing — regex-based to preserve file formatting
# ---------------------------------------------------------------------------

def insert_keys_after_anchor(
    json_path: Path,
    anchor_key: str | None,
    keys_to_insert: dict[str, str],
) -> bool:
    """
    Insert key/value pairs into a JSON file.

    If anchor_key is given, the pairs are inserted immediately after that key's line.
    If anchor_key is None, the pairs are inserted at the very start of the object
    (after the opening '{').

    Uses line-based insertion to preserve formatting.
    Returns True if the file was modified.
    """
    text  = json_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Skip keys that already exist
    existing    = {k for k in keys_to_insert if re.search(r'"' + re.escape(k) + r'"\s*:', text)}
    keys_to_add = {k: v for k, v in keys_to_insert.items() if k not in existing}
    if not keys_to_add:
        return False

    indent    = "  "
    new_lines = [
        f'{indent}"{k}": "{json.dumps(v, ensure_ascii=False)[1:-1]}",\n'
        for k, v in keys_to_add.items()
    ]

    if anchor_key is not None:
        anchor_pat = re.compile(r'^\s*"' + re.escape(anchor_key) + r'"\s*:')
        anchor_idx = next(
            (i for i, ln in enumerate(lines) if anchor_pat.match(ln)), None
        )
        if anchor_idx is None:
            print(f"    WARNING: anchor '{anchor_key}' not found in {json_path.name}, skipping group")
            return False
        # Ensure the anchor line ends with a comma
        if not lines[anchor_idx].rstrip().endswith(","):
            lines[anchor_idx] = lines[anchor_idx].rstrip("\n\r") + ",\n"
        insert_at = anchor_idx + 1
    else:
        # Insert after the opening '{'
        insert_at = next((i + 1 for i, ln in enumerate(lines) if ln.strip() == "{"), 0)

    for i, nl in enumerate(new_lines):
        lines.insert(insert_at + i, nl)

    json_path.write_text("".join(lines), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_one(client: anthropic.Anthropic, key_name: str, eng_text: str, lang_name: str) -> str:
    system = (
        f"You are a professional software localisation editor translating UI strings "
        f"into {lang_name} for a cross-platform desktop transcription application. "
        f"Always preserve these technical terms unchanged: {TECH_TERMS}. "
        f"Use natural, concise phrasing appropriate for a button label or tooltip. "
        f"Match the register and brevity of the English source."
    )
    user = (
        f"Translate this UI string into {lang_name}:\n\n"
        f'"{eng_text}"\n\n'
        f"Return ONLY the translated text, no surrounding quotes, no explanation."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and translate locale keys missing from non-English JSON files."
    )
    parser.add_argument("--api-key",  help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-translate all missing keys even if cached translations exist")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Report missing keys without translating or writing files")
    parser.add_argument("--lang",     default="",
                        help="Comma-separated language codes to target (default: all)")
    args = parser.parse_args()

    # ── Resolve target languages ──────────────────────────────────────────────
    lang_filter  = {c.strip() for c in args.lang.split(",") if c.strip()}
    target_langs = [(c, n) for c, n in LANGUAGES if not lang_filter or c in lang_filter]
    if not target_langs:
        print("ERROR: no matching languages found", file=sys.stderr)
        sys.exit(1)

    # ── Load en.json ──────────────────────────────────────────────────────────
    en_data = load_json_keys(EN_JSON)
    en_keys = list(en_data.keys())
    print(f"en.json: {len(en_keys)} keys")

    # ── Detect missing keys per language ─────────────────────────────────────
    # missing_by_key[key] = list of lang_codes that need it
    missing_by_key: dict[str, list[str]] = defaultdict(list)
    lang_missing:   dict[str, list[str]] = {}

    for lang_code, _ in target_langs:
        lang_path = LOCALES / f"{lang_code}.json"
        missing   = find_missing_keys(en_data, lang_path)
        lang_missing[lang_code] = missing
        for k in missing:
            missing_by_key[k].append(lang_code)

    total_missing = sum(len(v) for v in lang_missing.values())
    if total_missing == 0:
        print("All locale files are up-to-date — nothing to do.")
        return

    print(f"\nMissing key × language combinations: {total_missing}")
    for key, langs in sorted(missing_by_key.items(), key=lambda x: -len(x[1])):
        print(f"  {key!r:50s}  missing in {len(langs)} language(s): {', '.join(langs)}")

    if args.dry_run:
        print("\n(dry-run — no files written)")
        return

    # ── Require API key for actual translation ────────────────────────────────
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: supply --api-key or set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    con = sqlite3.connect(DB_PATH)
    db_init(con)

    # ── Translate missing key × language pairs ────────────────────────────────
    all_pairs = [(k, lc) for k, langs in missing_by_key.items() for lc in langs]
    total_ops = len(all_pairs)
    done = errors = 0

    lang_name_map = dict(LANGUAGES)

    for key, langs in missing_by_key.items():
        eng = en_data[key]
        print(f"\n{'─'*70}")
        print(f"  Key     : {key}")
        print(f"  English : {eng[:72]}{'...' if len(eng) > 72 else ''}")
        print(f"{'─'*70}")

        for lang_code in langs:
            lang_name = lang_name_map[lang_code]
            done += 1
            pct = done * 100 // total_ops

            if not args.force:
                cached_text, cached_src = db_get(con, key, lang_code)
                if cached_text is not None and cached_src == eng:
                    print(f"  [{pct:3d}%] {lang_code:<4}  {lang_name:<26} cached")
                    continue

            print(
                f"  [{pct:3d}%] {lang_code:<4}  {lang_name:<26} translating...",
                end="", flush=True,
            )
            try:
                result = translate_one(client, key, eng, lang_name)
                db_upsert(con, key, lang_code, result, eng)
                print(f" ok  ({len(result)} chars)")
            except Exception as exc:
                errors += 1
                print(f" FAILED: {exc}")

    print(f"\nTranslation pass complete.  Errors: {errors}")
    if errors:
        print("  Re-run the script to retry any failed translations.")

    # ── Write to JSON files ───────────────────────────────────────────────────
    print("\nUpdating locale JSON files...")

    for lang_code, lang_name in target_langs:
        missing = lang_missing[lang_code]
        if not missing:
            print(f"  {lang_code}.json  already up-to-date")
            continue

        lang_path    = LOCALES / f"{lang_code}.json"
        present_keys = set(load_json_keys(lang_path).keys()) if lang_path.exists() else set()

        # Group missing keys by their anchor (insertion point), preserving en.json order.
        # Process groups in the order their anchor appears so that earlier insertions
        # don't invalidate later anchor line numbers — we reload after each group.
        groups: dict[str | None, dict[str, str]] = defaultdict(dict)
        for key in missing:
            cached_text, _ = db_get(con, key, lang_code)
            value  = cached_text if cached_text is not None else en_data[key]
            anchor = anchor_for_key(en_keys, key, present_keys)
            groups[anchor][key] = value
            if cached_text is None:
                print(f"  {lang_code}.json  WARNING: no translation for '{key}', using English")

        modified = False
        for anchor, keys_to_insert in groups.items():
            if insert_keys_after_anchor(lang_path, anchor, keys_to_insert):
                modified = True
            # Update present_keys so subsequent groups see the newly inserted keys
            present_keys.update(keys_to_insert.keys())

        print(f"  {lang_code}.json  {'updated' if modified else 'already up-to-date'}")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
