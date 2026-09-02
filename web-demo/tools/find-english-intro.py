#!/usr/bin/env python3
"""Locate where a LibriVox recording's English announcement ends and the reading begins.

LibriVox files open with a spoken English boilerplate ("This is a LibriVox recording. All LibriVox
recordings are in the public domain...") before the text itself. For English books that hardly
matters; for LATIN or ANCIENT GREEK it decides whether a reference clip is the language you wanted
or an announcer reading a copyright notice — and the silence structure alone cannot tell them apart,
because the boilerplate's phrases are the same length as a line of Caesar.

So this runs the ENGLISH CTC model over the opening minutes and reports, per speech run, how
confidently it decodes as English. Boilerplate decodes cleanly; Latin decodes as noise. The boundary
is where the confidence collapses.

    python3 tools/find-english-intro.py --wav bg01.wav --bundle /mnt/data/models/nfa_ctc_onnx

⚠ This finds the boundary. It does NOT transcribe the Latin — the transcript for a reference clip
still has to come from the printed text, matched by hand.
"""
import argparse, json, os
import numpy as np, onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument("--wav", required=True)
p.add_argument("--bundle", default="/mnt/data/models/nfa_ctc_onnx")
p.add_argument("--seconds", type=float, default=180.0)
p.add_argument("--gap-ms", type=int, default=700)
a = p.parse_args()

import wave
with wave.open(a.wav) as w:
    sr, n = w.getframerate(), w.getnframes()
    pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
if sr != 16000:                       # the NeMo bundle is a 16 kHz model
    import subprocess, tempfile
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.wav, "-ac", "1", "-ar", "16000",
                    "-sample_fmt", "s16", tmp], check=True)
    with wave.open(tmp) as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    os.unlink(tmp)
pcm = pcm[: int(a.seconds * sr)]

# ⚠ vocab.txt is "<token> <id>" per line, not one token per line. Splitting on whitespace and
# taking the first field matters: keeping the whole line pastes the numeric id into every decode.
vocab = [l.rstrip("\n").rsplit(" ", 1)[0] for l in open(os.path.join(a.bundle, "vocab.txt"), encoding="utf8")]
pre = ort.InferenceSession(os.path.join(a.bundle, "nemo128.onnx"), providers=["CPUExecutionProvider"])
ctc = ort.InferenceSession(os.path.join(a.bundle, "ctc-model.onnx"), providers=["CPUExecutionProvider"])

def decode(seg):
    """Greedy CTC over one segment -> (text, mean per-frame max probability)."""
    x = seg[None, :].astype(np.float32)
    feats, flen = pre.run(None, {"waveforms": x, "waveforms_lens": np.array([x.shape[1]], np.int64)})
    logp = ctc.run(None, {"audio_signal": feats, "length": flen})[0][0]     # [T, V+1] log-softmax
    ids = logp.argmax(-1)
    conf = float(np.exp(logp.max(-1)).mean())
    out, prev = [], -1
    for i in ids:
        if i != prev and i < len(vocab):
            out.append(vocab[i])
        prev = i
    return "".join(out).replace("▁", " ").strip(), conf

# Speech runs, so each decode covers one utterance rather than straddling a pause.
frame = sr // 100
m = len(pcm) // frame
db = 10 * np.log10(np.array([np.mean(pcm[i * frame:(i + 1) * frame] ** 2) for i in range(m)]) + 1e-12)
thr, gap = db.max() - 30, a.gap_ms // 10
runs, start, sil = [], -1, 0
for i in range(m):
    if db[i] >= thr:
        if start < 0: start = i
        sil = 0
    elif start >= 0:
        sil += 1
        if sil >= gap:
            runs.append((start, i - sil + 1)); start, sil = -1, 0
if start >= 0: runs.append((start, m))

print(f"{len(pcm)/sr:.0f}s analysed, {len(runs)} runs")
for i, (s, e) in enumerate(runs[:20]):
    text, conf = decode(pcm[s * frame:e * frame])
    print(f"  run {i:2d}  {s/100:6.1f}-{e/100:6.1f}s  conf={conf:.2f}  {text[:70]}")
