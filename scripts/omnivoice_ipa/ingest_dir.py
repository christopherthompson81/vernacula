#!/usr/bin/env python3
"""Ingest a DIRECTORY of audio + transcripts into the corpus — the non-FLEURS path.

`ingest_fleurs.py` is bound to FLEURS in three places: it downloads `data/<lang>/audio/train.tar.gz`,
it reads a FLEURS TSV for sentence ids and gender, and it looks IPA up in the alignment DB keyed by
the FLEURS wav name. A corpus that is not FLEURS has none of those, so this is a sibling rather than
a flag on that script — the two share only the encode step, and entangling them would put a second
"where does IPA come from" branch in the file whose header already warns that two IPA sources is the
drift this pipeline has been bitten by before.

Written for OpenSLR SLR83 (UK/Ireland English dialects, CC BY-SA 4.0) so `en_gb` can enter the
fine-tune, but it takes any `<id>\t<transcript>` index beside audio files.

⚠ IPA COMES FROM THE PHONEMIZER, PER UTTERANCE, and the rows are marked `ipa_src="phonemizer"`.
That is IPA PROVENANCE and it stays true regardless of QC — it is not a quality verdict. Freshly
ingested rows have never been checked against what the reader actually said, which is a weaker
guarantee than the FLEURS path gives, and they say so by carrying an EMPTY `status`: no verdict, not
"clean". Run the alignment pass (`asr_align_dir.py` in vernacula-phonemizer) and propagate its labels
into `status` before treating these rows as QC'd.

  python3 ingest_dir.py --dir /path/to/slr83/southern_english_female --lang en_gb --phon-lang en-GB
"""
from __future__ import annotations

import argparse, glob, json, os, subprocess, sys, tempfile, time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa

SR_OUT = 24000
MIN_SECONDS, MAX_SECONDS = 1.0, 30.0
OUT = "/mnt/data/omnivoice_ipa/corpus/tokens"
ENCODER = "/mnt/data/omnivoice_ipa/onnx/higgs_encoder.onnx"
# Override with VERNACULA_PHONEMIZER so the default is not a statement about one machine --
# the same convention the asr-align tools use for ASR_ALIGN_ROOT.
PHONEMIZER = os.environ.get("VERNACULA_PHONEMIZER",
                            os.path.expanduser("~/Programming/vernacula-phonemizer"))

IDENT = __import__("re").compile(r"^[0-9a-f]{16,}$|^[A-Za-z0-9_+/=-]{24,}$")


def read_index(root: str, audio: dict[str, str]) -> dict[str, str]:
    """{id: transcript} from whatever index files the corpus ships.

    ⚠ THE TRANSCRIPT IS NOT THE LONGEST COLUMN. That rule put Common Voice's 128-character `client_id`
    into 21 reference voices as their transcript (web-demo #86). Prefer a declared header; otherwise
    take the longest column that is not an opaque identifier.
    """
    TEXT_COLS = ("sentence", "transcript", "transcription", "text", "raw_text", "normalized_text")
    out: dict[str, str] = {}
    idx = [f for f in glob.glob(f"{root}/**/*", recursive=True)
           if f.lower().endswith((".tsv", ".csv", ".txt")) and "readme" not in f.lower()]
    for path in idx:
        lines = open(path, encoding="utf8", errors="replace").read().split("\n")
        first = next((l for l in lines if l.strip()), "")
        sep = "\t" if "\t" in first else ","
        head = [h.strip().strip('"').lower() for h in first.split(sep)]
        tcol = next((i for i, h in enumerate(head) if h in TEXT_COLS), -1)
        icol = next((i for i, h in enumerate(head) if h in ("path", "file", "filename", "id", "utt_id")), -1)
        for n, line in enumerate(lines):
            if not line.strip() or (tcol >= 0 and n == 0):
                continue
            parts = [p.strip().strip('"') for p in line.split(sep)]
            uid = parts[icol] if icol >= 0 and icol < len(parts) and parts[icol] in audio else \
                next((p for p in parts if p in audio), None)
            if uid is None:
                continue
            if tcol >= 0 and tcol < len(parts):
                t = parts[tcol]
            else:
                t = next(iter(sorted((p for p in parts if p != uid and not IDENT.match(p)),
                                     key=len, reverse=True)), "")
            if t and len(t) > 3 and not IDENT.match(t):
                out.setdefault(uid, t)
    return out


def phonemize(texts: list[str], lang: str) -> list[str]:
    """One phonemizer process for the whole corpus — a per-utterance spawn costs more than the encode."""
    tmp_in = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf8")
    tmp_in.write("\n".join(t.replace("\n", " ") for t in texts)); tmp_in.close()
    tmp_out = tempfile.mktemp(suffix=".txt")
    probe = os.path.join(PHONEMIZER, f"ingest-phonemize.{os.getpid()}.tmp.mts")
    with open(probe, "w", encoding="utf8") as f:
        f.write(
            'import { readFileSync, writeFileSync } from "node:fs";\n'
            'import { phonemizeAsync } from "./src/index.ts";\n'
            f'const lines = readFileSync({json.dumps(tmp_in.name)}, "utf8").split("\\n");\n'
            "const out = [];\n"
            "for (const l of lines) {\n"
            f'  try {{ out.push((await phonemizeAsync(l, {json.dumps(lang)})).replace(/[\\r\\n]+/g, " ").trim()); }}\n'
            '  catch { out.push(""); }\n'
            "}\n"
            f'writeFileSync({json.dumps(tmp_out)}, out.join("\\n"), "utf8");\n')
    try:
        subprocess.run(["npx", "tsx", os.path.basename(probe)], cwd=PHONEMIZER, check=True,
                       stdout=subprocess.DEVNULL)
        return open(tmp_out, encoding="utf8").read().split("\n")
    finally:
        for p in (probe, tmp_in.name, tmp_out):
            try: os.remove(p)
            except OSError: pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of audio + an index file")
    ap.add_argument("--lang", required=True, help="corpus language key, e.g. en_gb")
    ap.add_argument("--phon-lang", help="phonemizer code (default: --lang with _ -> -)")
    ap.add_argument("--provider", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", help="JSON list of utterance ids; ingest only these. en_gb is a coverage "
                                 "patch, not a corpus — it ships the utterances that carry the vowel "
                                 "units en_us never trained, not every utterance in the archive.")
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing codes/manifest pair rather than replacing it "
                         "(SLR83 ships one archive per dialect; en_gb is their union)")
    a = ap.parse_args()
    phon = a.phon_lang or a.lang.replace("_", "-")

    audio = {}
    for f in glob.glob(f"{a.dir}/**/*", recursive=True):
        if f.lower().endswith((".wav", ".flac", ".mp3", ".opus")):
            b = os.path.basename(f)
            audio[b] = f
            audio[os.path.splitext(b)[0]] = f
    text = read_index(a.dir, audio)
    print(f"  {len(set(audio.values())):,} audio files, {len(text):,} transcripts matched")
    if not text:
        print("  no transcripts matched — check the index format", file=sys.stderr); return 1

    if a.ids:
        keep = set(json.load(open(a.ids, encoding="utf8")))
        text = {k: v for k, v in text.items() if k in keep or os.path.splitext(k)[0] in keep}
        print(f"  {len(text):,} of {len(keep):,} requested ids present here")
    items = sorted(text.items())[: a.limit or None]
    t0 = time.time()
    ipas = phonemize([t for _, t in items], phon)
    print(f"  phonemized {sum(1 for i in ipas if i):,}/{len(items):,} in {time.time()-t0:.0f}s")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if a.provider == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(ENCODER, providers=providers)
    # ⚠ ORT FALLS BACK TO CPU SILENTLY. Requesting CUDAExecutionProvider when the installed wheel is
    # plain `onnxruntime` rather than `onnxruntime-gpu` logs a warning and runs on CPU anyway -- and
    # this script's output is normally piped through a `grep -v Warning`, so the flag looked honoured.
    # Measured cost of not noticing: 5.14 s/utterance instead of 0.10, 110 min instead of 2.
    active = sess.get_providers()[0]
    if a.provider == "cuda" and active != "CUDAExecutionProvider":
        print(f"  ⚠ --provider cuda REQUESTED BUT RUNNING ON {active} -- expect ~50x slower.\n"
              f"    pip install onnxruntime-gpu (uninstall onnxruntime first; they collide).",
              file=sys.stderr, flush=True)
    in_name = sess.get_inputs()[0].name
    codes_out, manifest = {}, []
    n_skip = 0
    for (uid, t), ipa in zip(items, ipas):
        if not ipa:
            n_skip += 1; continue
        wav, sr = sf.read(audio[uid], dtype="float32")
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if not (MIN_SECONDS <= len(wav) / sr <= MAX_SECONDS):
            n_skip += 1; continue
        if sr != SR_OUT:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR_OUT, res_type="soxr_hq")
        pad = (-len(wav)) % 960          # the encoder's two paths disagree by a frame otherwise
        x = np.pad(wav, (0, pad)).reshape(1, 1, -1).astype(np.float32)
        codes = sess.run(["audio_codes"], {in_name: x})[0][0]
        key = os.path.splitext(os.path.basename(audio[uid]))[0]
        if key in codes_out:      # two files, one stem: silently keeping the last would be a lie
            print(f"  ⚠ duplicate id {key}, keeping the first", file=sys.stderr)
            n_skip += 1
            continue
        codes_out[key] = codes.astype(np.int16)
        manifest.append(dict(id=key, sentence_id=None, lang=a.lang, ipa=ipa, gender=None,
                             dur_s=round(len(wav) / SR_OUT, 2), n_frames=int(codes.shape[-1]),
                             # ⚠ NOT "hand", NOT "" — this row's IPA has never been checked against
                             # the audio. A consumer can tell it apart from a DB-derived row.
                             ipa_src="phonemizer", status="", text=t))
        if len(manifest) % 500 == 0:
            print(f"    {len(manifest):,} encoded…", flush=True)

    cp, mp = f"{OUT}/codes_{a.lang}.npz", f"{OUT}/manifest_{a.lang}.jsonl"
    # ⚠ BOTH files, not just the npz: guarding on `cp` alone crashed in `open(mp)` below when a
    # previous run had been interrupted between the two writes.
    if a.append and os.path.exists(cp) and os.path.exists(mp):
        prev = np.load(cp)
        merged = {k: prev[k] for k in prev.files}
        dup = sum(1 for k in codes_out if k in merged)
        merged.update(codes_out)
        codes_out = merged
        old_rows = [json.loads(l) for l in open(mp, encoding="utf8") if l.strip()]
        old_by_id = {r["id"]: r for r in old_rows}
        # ⚠ RE-INGESTING A ROW MUST NOT DISCARD ITS VERDICT. `status` is a review decision that came
        # from the alignment pass or from a person; `ipa`, `codes` and `n_frames` are derivations of
        # the audio. Replacing the whole row reset 1,265 `verified` labels to "" on a re-run, and an
        # empty status reads as NO VERDICT, so nothing downstream would have flagged the loss.
        # Same lesson corpus_filter records: label at build, decide later, never fuse the two.
        kept = 0
        for r in manifest:
            was = old_by_id.get(r["id"])
            if was and was.get("status") and not r.get("status"):
                r["status"] = was["status"]
                kept += 1
        seen = {r["id"] for r in manifest}
        manifest = [r for r in old_rows if r["id"] not in seen] + manifest
        print(f"  appended to {len(old_rows):,} existing rows ({dup} ids replaced, "
              f"{kept} verdicts carried forward)")
    # ⚠ Write both, THEN publish both. These two files are one artifact -- a manifest row without its
    # codes is unusable and codes without a row are invisible -- and this is a long GPU job that has
    # been interrupted before. Writing in place left them disagreeing on exactly that failure.
    np.savez_compressed(cp + ".tmp.npz", **codes_out)
    with open(mp + ".tmp", "w", encoding="utf8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(cp + ".tmp.npz", cp)
    os.replace(mp + ".tmp", mp)
    print(f"  {len(manifest):,} rows, {n_skip:,} skipped -> {os.path.basename(cp)} "
          f"({os.path.getsize(cp)/1e6:.0f} MB), {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
