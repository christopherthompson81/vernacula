#!/usr/bin/env python3
"""
scripts/update_help_files.py

Translates new or changed English help Markdown files into all supported
language directories via the Anthropic API.

Features:
  - Detects which files changed since the last run (SQLite cache, keyed by SHA-256)
  - Resumes automatically if interrupted — cached translations are reused
  - Preserves YAML frontmatter structure; translates only title/description
  - Preserves all Markdown syntax, relative link hrefs, code spans, and
    technical terms — only human-readable prose is translated
  - --force  : retranslate everything regardless of cache
  - --files  : comma-separated relative paths to retranslate (e.g. index.md)
  - --lang   : comma-separated language codes to target (e.g. bg,de)

Usage:
    python scripts/update_help_files.py
    python scripts/update_help_files.py --force
    python scripts/update_help_files.py --files operations/editing_transcripts.md
    python scripts/update_help_files.py --lang de,fr
    ANTHROPIC_API_KEY=sk-ant-... python scripts/update_help_files.py
"""

import argparse
import hashlib
import os
import sqlite3
import sys
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
ROOT     = Path(__file__).resolve().parent.parent
HELP_DIR = ROOT / "src" / "Vernacula.Avalonia" / "Help"
EN_DIR   = HELP_DIR / "en"
DB_PATH  = HELP_DIR / "help_files.db"

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

TECH_TERMS = (
    "ASR, ONNX, GPU, CUDA, cuDNN, TensorRT, SQLite, FLAC, WAV, MP3, M4A, "
    "OGG, AAC, WMA, AIFF, WebM, MP4, MOV, MKV, AVI, WMV, FLV, MTS, OPUS, "
    "SRT, DOCX, XLSX, CSV, JSON, MD, Parakeet, SortFormer, FFmpeg, NVIDIA"
)

SYSTEM_PROMPT = """\
You are a professional technical writer translating cross-platform desktop application
help documentation into {lang_name}.

Rules — follow these exactly:
1. Translate all human-readable prose into {lang_name}. Use natural, clear
   phrasing appropriate for end-user documentation.
2. PRESERVE unchanged:
   - The YAML front matter keys (title, description, topic_id) — translate
     the VALUES of `title` and `description` but leave `topic_id` as-is.
   - All Markdown syntax: headings (#), bold (**), tables (|), bullet lists
     (- and *), numbered lists, horizontal rules, and code fences.
   - All relative link hrefs (the part inside parentheses, e.g.
     `loading_completed_jobs.md`) — DO NOT translate these file paths.
     Translate only the link text (the part inside square brackets).
   - All inline code spans delimited by backticks — these are UI button
     labels and must NOT be translated.
   - These technical terms: {tech_terms}.
3. Output ONLY the translated Markdown file — no explanation, no commentary.
"""

USER_PROMPT = """\
Translate the following help file into {lang_name}.

---
{content}
---
"""


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def db_init(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS translations (
            file_path       TEXT NOT NULL,
            lang_code       TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            source_hash     TEXT NOT NULL,
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (file_path, lang_code)
        );
    """)
    con.commit()


def db_get(con: sqlite3.Connection, file_path: str, lang: str) -> tuple[str | None, str | None]:
    row = con.execute(
        "SELECT translated_text, source_hash FROM translations "
        "WHERE file_path=? AND lang_code=?",
        (file_path, lang),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def db_upsert(
    con: sqlite3.Connection, file_path: str, lang: str, text: str, source_hash: str
) -> None:
    con.execute("""
        INSERT INTO translations(file_path, lang_code, translated_text, source_hash)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_path, lang_code) DO UPDATE SET
            translated_text = excluded.translated_text,
            source_hash     = excluded.source_hash,
            updated_at      = datetime('now')
    """, (file_path, lang, text, source_hash))
    con.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_en_files() -> list[Path]:
    """Return all .md files under Help/en/, sorted for stable ordering."""
    return sorted(EN_DIR.rglob("*.md"))


def relative_key(en_file: Path) -> str:
    """Relative path string used as the DB key, e.g. 'operations/editing_transcripts.md'."""
    return en_file.relative_to(EN_DIR).as_posix()


def out_path(lang_code: str, rel: str) -> Path:
    return HELP_DIR / lang_code / rel


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_file(
    client: anthropic.Anthropic,
    content: str,
    lang_name: str,
) -> str:
    system = SYSTEM_PROMPT.format(lang_name=lang_name, tech_terms=TECH_TERMS)
    user   = USER_PROMPT.format(lang_name=lang_name, content=content)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate English help Markdown files into all language directories."
    )
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-translate all files even if cached translations are up-to-date"
    )
    parser.add_argument(
        "--files", default="",
        help="Comma-separated relative file paths to process (default: all changed)"
    )
    parser.add_argument(
        "--lang", default="",
        help="Comma-separated language codes to target (default: all)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: supply --api-key or set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    con = sqlite3.connect(DB_PATH)
    db_init(con)

    # ── Resolve target languages ──────────────────────────────────────────────
    lang_filter = {c.strip() for c in args.lang.split(",") if c.strip()}
    target_langs = [(c, n) for c, n in LANGUAGES if not lang_filter or c in lang_filter]
    if not target_langs:
        print("ERROR: no matching languages found", file=sys.stderr)
        sys.exit(1)

    # ── Collect source files ──────────────────────────────────────────────────
    all_files = collect_en_files()
    file_filter = {p.strip() for p in args.files.split(",") if p.strip()}

    if file_filter:
        en_files = [f for f in all_files if relative_key(f) in file_filter]
        if not en_files:
            print(f"ERROR: none of the specified files found under {EN_DIR}", file=sys.stderr)
            sys.exit(1)
    else:
        en_files = all_files

    # ── Detect changed files ──────────────────────────────────────────────────
    changed:   list[tuple[Path, str, str]] = []   # (path, rel_key, hash)
    unchanged: list[str]                  = []

    for en_file in en_files:
        content = en_file.read_text(encoding="utf-8")
        h       = sha256(content)
        rel     = relative_key(en_file)
        # A file is "changed" if ANY language is missing or has a stale hash
        needs_work = args.force or file_filter
        if not needs_work:
            for lang_code, _ in target_langs:
                _, cached_hash = db_get(con, rel, lang_code)
                if cached_hash != h:
                    needs_work = True
                    break
        if needs_work:
            changed.append((en_file, rel, h))
        else:
            unchanged.append(rel)

    print(f"Files unchanged (all langs cached) : {len(unchanged)}")
    print(f"Files to process                   : {len(changed)}")
    if changed:
        for _, rel, _ in changed:
            print(f"  {rel}")

    if not changed:
        print("\nNothing to translate. Use --force to retranslate everything.")
        con.close()
        return

    # ── Translate ─────────────────────────────────────────────────────────────
    total_ops = len(changed) * len(target_langs)
    done = errors = 0

    for en_file, rel, h in changed:
        content = en_file.read_text(encoding="utf-8")
        print(f"\n{'─'*70}")
        print(f"  File    : {rel}")
        print(f"  Size    : {len(content)} chars")
        print(f"{'─'*70}")

        for lang_code, lang_name in target_langs:
            done += 1
            pct = done * 100 // total_ops

            if not args.force:
                cached_text, cached_hash = db_get(con, rel, lang_code)
                if cached_text is not None and cached_hash == h:
                    print(f"  [{pct:3d}%] {lang_code:<4}  {lang_name:<26} cached")
                    continue

            print(
                f"  [{pct:3d}%] {lang_code:<4}  {lang_name:<26} translating...",
                end="", flush=True,
            )
            try:
                result = translate_file(client, content, lang_name)
                db_upsert(con, rel, lang_code, result, h)
                print(f" ok  ({len(result)} chars)")
            except Exception as exc:
                errors += 1
                print(f" FAILED: {exc}")

    print(f"\nTranslation pass complete.  Errors: {errors}")
    if errors:
        print("  Re-run the script to retry any failed translations.")

    # ── Write output files ────────────────────────────────────────────────────
    print("\nWriting translated files...")
    written = skipped = 0

    for en_file, rel, h in changed:
        for lang_code, _ in target_langs:
            cached_text, cached_hash = db_get(con, rel, lang_code)
            if cached_text is None:
                print(f"  WARNING: no translation in cache for {rel} [{lang_code}], skipping")
                skipped += 1
                continue

            dest = out_path(lang_code, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(cached_text, encoding="utf-8")
            written += 1

    print(f"  Written : {written} file(s)")
    if skipped:
        print(f"  Skipped : {skipped} file(s) (missing translations — re-run to retry)")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
