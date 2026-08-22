#!/usr/bin/env python3
"""
Encode the rows an earlier policy skipped and APPEND them to the existing codes. Never rewrites a vector.

⚠ THE npz IS WRITE-ONCE AND THIS KEEPS IT THAT WAY. `codes_<lang>.npz` is `{utterance_id: (8, T) int16}`,
a pure function of the waveform and the encoder. Exclusion used to be applied at ENCODE time, so a row
the align DB marked `defective_audio` was never encoded at all — a revisable judgement fused to a GPU
artifact. This adds the missing ids back; `build_manifest.py` then labels every id, and
`corpus_filter.load_manifest` applies the policy at load, where it can change for free.

⚠ APPENDING IS ONLY SAFE BECAUSE THE ENCODER IS REPRODUCIBLE, and that was measured, not assumed. These
npz span 2026-07-01 to 2026-08-22 — different driver, different onnxruntime, and `arena_extend_strategy`
changed on the last day. Re-encoding 40 utterances already stored in `en_us` (written 07-01) reproduced
them BIT-IDENTICALLY, as did 40 in `mi_nz` (written 08-22). Without that, appending would silently mix
two encoders inside one file. ⚠ RE-RUN THAT CHECK (`--verify N`) IF THE RUNTIME MOVES AGAIN.

⚠ WHAT THIS DELIBERATELY DOES NOT ADD. Rows outside MIN/MAX_SECONDS are skipped by the ENCODER's own
constraint, not by a judgement about training data, so they are not this script's business:

    3254  verified, over 30s   — 32.6 hours; the window is doing real work here
     258  other,    over 30s
     139  under 1s

Changing the window is a decision about the codec (a 25 Hz tokenizer on a 108 s utterance), and it should
be taken on its own terms rather than smuggled in as a "restore".

  python3 topup_codes.py --check           # report, write nothing
  python3 topup_codes.py
  python3 topup_codes.py --verify 40 en_us # re-encode stored ids and compare bit-for-bit
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sqlite3
import sys
import tarfile

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = os.environ.get("OMNIVOICE_ROOT", "/mnt/data/omnivoice_ipa")
TOKENS = f"{ROOT}/corpus/tokens"
AUDIO = f"{ROOT}/corpus/audio_cache/data"
ALIGN_DB = f"{ROOT}/work/asr_align/align.sqlite"
MIN_SECONDS, MAX_SECONDS, SR_OUT = 1.0, 30.0, 24000


def session():
    # Same options as ingest_fleurs.session() — see the notes there on use_tf32 and the arena.
    return ort.InferenceSession(f"{ROOT}/onnx/higgs_encoder.onnx", providers=[
        ("CUDAExecutionProvider", {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT",
                                   "arena_extend_strategy": "kSameAsRequested"}),
        "CPUExecutionProvider"])


def wanted(lang: str, have: set[str]) -> set[str]:
    if not os.path.exists(ALIGN_DB):
        return set()
    db = sqlite3.connect(f"file:{ALIGN_DB}?mode=ro", uri=True)
    rows = db.execute("SELECT wav FROM utt WHERE lang=? AND ipa IS NOT NULL AND TRIM(ipa) <> ''",
                      (lang,)).fetchall()
    db.close()
    return {(w[:-4] if w.endswith(".wav") else w) for (w,) in rows} - have


def encode(sess, wav, sr):
    if sr != SR_OUT:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR_OUT, res_type="soxr_hq")
    x = np.pad(wav, (0, (-len(wav)) % 960)).reshape(1, 1, -1).astype(np.float32)
    return sess.run(["audio_codes"], {sess.get_inputs()[0].name: x})[0][0].astype(np.int16)


def run(lang: str, check: bool, verify: int) -> tuple[int, int, int]:
    npz = f"{TOKENS}/codes_{lang}.npz"
    tar = f"{AUDIO}/{lang}/audio/train.tar.gz"
    if not (os.path.exists(npz) and os.path.exists(tar)):
        return 0, 0, 0
    z = dict(np.load(npz))
    todo = wanted(lang, set(z))
    if not todo and not verify:
        return 0, 0, 0
    if check:
        return len(todo), 0, 0
    sess = session()
    added = oow = 0
    same = mism = 0
    with tarfile.open(tar) as t:
        for m in t:
            if not m.name.endswith(".wav"):
                continue
            uid = m.name.split("/")[-1][:-4]
            checking = verify and uid in z and same + mism < verify
            if uid not in todo and not checking:
                continue
            wav, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32")
            if not (MIN_SECONDS <= len(wav) / sr <= MAX_SECONDS):
                oow += 1
                continue
            codes = encode(sess, wav, sr)
            if checking:
                # ⚠ the whole append is unsound if this ever fails; see the module note
                if codes.shape == z[uid].shape and np.array_equal(codes, z[uid]):
                    same += 1
                else:
                    mism += 1
                continue
            z[uid] = codes
            added += 1
    if verify:
        print(f"  {lang:14}verify: {same} bit-identical, {mism} MISMATCH")
        if mism:
            print("  ⚠ ENCODER IS NOT REPRODUCIBLE HERE — do not append to this file.", file=sys.stderr)
            return 0, 0, mism
    if added:
        np.savez(npz, **z)
    return len(todo), added, oow


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("langs", nargs="*")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", type=int, default=0, help="re-encode N stored ids and compare")
    a = ap.parse_args()
    langs = a.langs or sorted(os.path.basename(f)[len("codes_"):-len(".npz")]
                              for f in glob.glob(f"{TOKENS}/codes_*.npz"))
    t_todo = t_add = t_oow = 0
    for lang in langs:
        todo, added, oow = run(lang, a.check, a.verify)
        t_todo += todo; t_add += added; t_oow += oow
        if (added or oow) and not a.check:
            print(f"  {lang:14}+{added} codes"
                  + (f"   ({oow} outside the {MIN_SECONDS:.0f}-{MAX_SECONDS:.0f}s window)" if oow else ""))
    verb = "would encode" if a.check else "appended"
    print(f"\n{verb} {t_todo if a.check else t_add} codes across {len(langs)} languages"
          + (f"; {t_oow} were outside the duration window and left alone" if t_oow else ""))
    if not a.check and t_add:
        print("⚠ Now re-run build_manifest.py so the new ids get rows and labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
