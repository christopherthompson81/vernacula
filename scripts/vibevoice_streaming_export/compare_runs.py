#!/usr/bin/env python
"""Compare transcripts from run_reference.py records: word error rate of each against the
first, per-chunk exact-match count, and timing/VRAM columns. Used to size the effect of the
acoustic sampling noise (seeds) and, later, ONNX-vs-reference parity."""
import json, re, sys


def words(t):
    # Speaker markers are structure, not transcript; drop them so a consumer that turns
    # them into fields (the C# CLI) compares on words alone.
    return re.findall(r"[\w']+", re.sub(r"Speaker \d+:", " ", t).lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def speakers(t):
    return [int(m) for m in re.findall(r"Speaker (\d+):", t)]


def load(p):
    r = json.load(open(p, encoding="utf-8-sig"))
    if isinstance(r, list):   # Vernacula CLI --export-format json: [{speaker,start,end,text}]
        return dict(chunks=[s["text"] for s in r], n_chunks=len(r), rtf=float("nan"),
                    peak_gib=float("nan"), weights_gib=float("nan"),
                    text=" ".join(f"Speaker {s['speaker'].split('_')[-1]}: {s['text']}" for s in r))
    return r


if __name__ != "__main__":
    sys.exit(0)
recs = [load(p) for p in sys.argv[1:]]
base = recs[0]
print(f"{'file':40} {'WER vs first':>12} {'chunks==':>9} {'spk turns':>9} {'distinct':>8} "
      f"{'RTF':>6} {'peak GiB':>8} {'wts GiB':>7}")
for p, r in zip(sys.argv[1:], recs):
    same = sum(a == b for a, b in zip(base["chunks"], r["chunks"]))
    sp = speakers(r["text"])
    print(f"{p.split('/')[-1]:40} {wer(base['text'], r['text']):12.3f} "
          f"{same:4d}/{r['n_chunks']:<4d} {len(sp):9d} {len(set(sp)):8d} "
          f"{r['rtf']:6.3f} {r['peak_gib']:8.2f} {r['weights_gib']:7.2f}")
