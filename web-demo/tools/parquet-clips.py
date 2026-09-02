#!/usr/bin/env python3
"""Extract audio clips + transcripts from a HuggingFace parquet shard into a directory.

The dataset-rows API serves ungated datasets only, and the gated ones (Vaani, Afrivoice) are
exactly where several remaining languages live. Their repos ARE reachable with a token once the
terms are accepted, so this reads the parquet directly instead.

    python3 tools/parquet-clips.py --repo ARTPARK-IISc/Vaani-transcription-part \\
        --file audio/Awadhi/test-00000-of-00001.parquet --out /tmp/vaani/awa [--limit 60]

Writes <out>/<n>.<ext> plus <out>/index.tsv of "<file>\t<transcript>", which
tools/make-voice-from-openslr.mjs --dir consumes.

⚠ The token is read from the environment or ~/.cache/huggingface/token and sent only to
huggingface.co. It is never printed, and never passed on a command line where `ps` would show it.
"""
import argparse, io, os, sys, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--file", required=True)
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=80)
p.add_argument("--scan", type=int, default=4000, help="rows to look at while collecting --limit usable ones")
p.add_argument("--min-sec", type=float, default=0.0)
p.add_argument("--max-sec", type=float, default=0.0)
p.add_argument("--text-field", default=None)
p.add_argument("--audio-field", default=None)
a = p.parse_args()

token = os.environ.get("HF_TOKEN") or ""
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except OSError:
        token = ""

url = f"https://huggingface.co/datasets/{a.repo}/resolve/main/{a.file}"
local = os.path.join("/tmp", a.repo.replace("/", "_") + "_" + os.path.basename(a.file))
if not os.path.exists(local):
    # ⚠ curl, not urllib: HuggingFace redirects LFS files to a signed CDN URL, and a client that
    # re-sends the Authorization header across that hop gets a 403 from the CDN. curl drops it.
    import subprocess
    cfg = local + ".curlrc"
    with open(os.open(cfg, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "w") as f:
        f.write(f'header = "Authorization: Bearer {token}"\n' if token else "")
    try:
        subprocess.run(["curl", "-sfL", "--max-time", "3600", "--config", cfg, url, "-o", local],
                       check=True)
    finally:
        os.unlink(cfg)
print(f"  {os.path.getsize(local)/1e6:.1f} MB parquet")

import pyarrow.parquet as pq
# ⚠ Row groups, not the whole table: these shards run to a gigabyte and the audio column expands
# several times over in memory. Only `limit` rows are ever needed.
pf = pq.ParquetFile(local)
cols = pf.schema_arrow.names
print("  columns:", cols)

def pick(cands, names):
    for c in cands:
        if c in names:
            return c
    return None

text_col = a.text_field or pick(["text", "transcript", "transcription", "sentence", "raw_text",
                                 "normalized_text", "verbatim_transcript"], cols)
audio_col = a.audio_field or pick(["audio", "audio_filepath", "wav", "speech"], cols)
if not text_col or not audio_col:
    sys.exit(f"could not find text/audio columns in {cols}")
print(f"  using text='{text_col}' audio='{audio_col}'")

os.makedirs(a.out, exist_ok=True)
# ⚠ COUNT USABLE ROWS, NOT ROWS. Vaani's transcript column is mostly empty — it flags each row with
# `isTranscriptionAvailable` — so taking the first N rows yielded one clip out of forty. Keep reading
# until `limit` rows have both audio and text (or `scan` rows have gone by).
dur_col = "duration" if "duration" in cols else None
cols_read = [c for c in (text_col, audio_col, dur_col) if c]
rows, seen = [], 0
for batch in pf.iter_batches(batch_size=64, columns=cols_read):
    for r in batch.to_pylist():
        seen += 1
        tx = (r.get(text_col) or "").strip()
        au = r.get(audio_col)
        # The audio cell is either an HF Audio struct {"bytes","path"} or, in some corpora, the raw
        # file bytes — WenetSpeech-Wu stores a bare WAV per row.
        has_audio = (isinstance(au, dict) and au.get("bytes")) or isinstance(au, (bytes, bytearray))
        if not tx or not has_audio:
            continue
        if dur_col and (a.min_sec or a.max_sec):
            d = float(r.get(dur_col) or 0)
            if d and ((a.min_sec and d < a.min_sec) or (a.max_sec and d > a.max_sec)):
                continue
        rows.append(r)
        if len(rows) >= a.limit:
            break
    if len(rows) >= a.limit or seen >= a.scan:
        break
print(f"  {len(rows)} usable of {seen} rows scanned")
index = []
for i, r in enumerate(rows[: a.limit]):
    au, tx = r[audio_col], (r[text_col] or "").strip()
    if isinstance(au, (bytes, bytearray)):
        data, name = bytes(au), f"{i}.wav"
    else:
        data = au.get("bytes") if isinstance(au, dict) else None
        name = (au.get("path") if isinstance(au, dict) else str(au)) or f"{i}.wav"
    if not data or not tx:
        continue
    ext = os.path.splitext(name)[1] or ".wav"
    fn = f"{i:05d}{ext}"
    with open(os.path.join(a.out, fn), "wb") as f:
        f.write(data)
    index.append(f"{fn}\t{tx}")
with open(os.path.join(a.out, "index.tsv"), "w", encoding="utf8") as f:
    f.write("\n".join(index) + "\n")
print(f"  wrote {len(index)} clips -> {a.out}")
