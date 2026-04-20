#!/usr/bin/env python3
"""
validate_indicconformer_package.py

End-to-end smoke test for an IndicConformer ONNX package (either the
120M home-exported shape or the repackaged 600M shape — both share the
same on-disk contract now). Runs the full pipeline:

  wav -> nemo128.onnx -> encoder-model.onnx -> ctc_decoder-model.onnx
      -> language-mask argmax + dedup (drop shared blank)
      -> lookup in flat vocab.txt via language_spans.json
      -> SentencePiece-style detokenize ('▁' -> ' ')

Also doubles as the Python reference for the C# decode path in Phase 3.

Usage:
  python scripts/indicconformer_export/validate_indicconformer_package.py \\
    --package ~/models/indicconformer_600m_onnx \\
    --wav ~/Programming/test_audio/hi-IN/hi-IN_fleurs_01.wav \\
    --lang hi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True,
                   help="Directory with encoder-model.onnx + ctc_decoder-model.onnx + nemo128.onnx + vocab.txt + language_spans.json")
    p.add_argument("--wav", required=True, help="16 kHz mono .wav")
    p.add_argument("--lang", required=True, help="BCP-47-ish code without region, e.g. 'hi', 'ta'")
    p.add_argument("--expected", default=None,
                   help="Optional expected reference transcript for informational diff.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf

    pkg = Path(args.package).expanduser()
    preproc = pkg / "nemo128.onnx"
    encoder = pkg / "encoder-model.onnx"
    ctc = pkg / "ctc_decoder-model.onnx"
    vocab_path = pkg / "vocab.txt"
    spans_path = pkg / "language_spans.json"
    for p in (preproc, encoder, ctc, vocab_path, spans_path):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    spans = json.loads(spans_path.read_text())
    langs = spans["languages"]
    if args.lang not in langs:
        raise SystemExit(f"language {args.lang!r} not in {sorted(langs)}")
    lang = langs[args.lang]
    start, length = int(lang["start"]), int(lang["length"])
    blank_id = int(spans["blank_token_id"])  # 5632, shared

    # Flat vocab (5632 lines, one token per id across all 22 langs).
    vocab = vocab_path.read_text().splitlines()
    assert len(vocab) == spans["total_vocab_size"], (
        f"vocab.txt has {len(vocab)} lines, spans says {spans['total_vocab_size']}"
    )

    # Load audio.
    arr, sr = sf.read(args.wav, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"wav must be 16 kHz; got {sr}")
    waveforms = arr[None, :].astype(np.float32)
    wav_lens = np.array([arr.shape[0]], dtype=np.int64)

    cpu = ["CPUExecutionProvider"]
    pre_sess = ort.InferenceSession(str(preproc), providers=cpu)
    enc_sess = ort.InferenceSession(str(encoder), providers=cpu)
    dec_sess = ort.InferenceSession(str(ctc), providers=cpu)

    # 1. Preprocessor: wav -> features, features_lens.
    feats, feats_lens = pre_sess.run(
        None, {"waveforms": waveforms, "waveforms_lens": wav_lens}
    )

    # 2. Encoder: features -> encoded, encoded_lens.
    encoded, encoded_lens = enc_sess.run(
        None, {"features": feats, "features_lens": feats_lens}
    )
    T = int(encoded_lens[0])

    # 3. CTC decoder: encoded -> logits [B, T, 5633] (or logprobs; either way argmax is equivalent).
    logits = dec_sess.run(None, {"encoded": encoded})[0]

    # 4. Language-masked argmax + dedup.
    # Build the argmax over the 257 valid positions (256 lang tokens + 1 shared blank).
    # Doing it as a gather on the sub-span keeps the blank at local position 256.
    lang_slice = logits[0, :T, start:start + length]         # [T, 256]
    blank_logit = logits[0, :T, blank_id:blank_id + 1]       # [T, 1]
    sub = np.concatenate([lang_slice, blank_logit], axis=1)  # [T, 257]
    local_blank_id = length  # = 256

    indices = sub.argmax(axis=-1)
    # Greedy CTC collapse: drop consecutive dupes, then drop blanks.
    pieces: list[str] = []
    prev = None
    for idx in indices.tolist():
        if idx == prev:
            continue
        prev = idx
        if idx == local_blank_id:
            continue
        # Lookup: local id 0..255 -> global id start..start+255.
        global_id = start + idx
        piece = vocab[global_id]
        pieces.append(piece)

    # SentencePiece-style detokenize: '▁' at the head of a piece marks a word boundary.
    text = "".join(pieces).replace("\u2581", " ").strip()

    print(f"decoded ({args.lang}): {text}")
    if args.expected is not None:
        print(f"expected:         {args.expected}")


if __name__ == "__main__":
    main()
