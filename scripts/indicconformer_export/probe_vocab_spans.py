#!/usr/bin/env python3
"""
probe_vocab_spans.py

Phase 1 follow-up. The checkpoint uses an AggregateTokenizer that unions 22
per-language SentencePiece vocabs (256 tokens each) into a flat 5632-entry
vocab. The "multi-softmax" design masks 5376 of those logits at decode time
based on language_id. To decide what lives in the shipping package and what
lives in C#, we need:

  1. Are the 22 language blocks contiguous in the flat vocab?
  2. Is there a per-language [start, length] table we can extract?
  3. Does the model carry a prebuilt mask tensor (so we can bake it into
     the exported artifact) or is the mask built on the fly?

Outputs:
  - /tmp/indicconformer_vocab.txt  (flat 5632-line vocab, one token per line)
  - /tmp/indicconformer_language_spans.json  (22 entries, [start, length])

Usage:
  python scripts/indicconformer_export/probe_vocab_spans.py \
    --nemo <path to downloaded .nemo>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True, help="Path to a local .nemo file.")
    p.add_argument("--vocab-out", default="/tmp/indicconformer_vocab.txt")
    p.add_argument("--spans-out", default="/tmp/indicconformer_language_spans.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import nemo.collections.asr as nemo_asr

    print(f"Restoring: {args.nemo}")
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cpu")
    model.eval()

    tok = model.tokenizer
    print(f"\nTokenizer class: {type(tok).__name__}")
    print(f"Languages: {list(tok.langs)}")
    print(f"Total vocab size: {tok.vocab_size}")

    # Offsets table — how the MultilingualTokenizer assigns a contiguous
    # range in the flat id-space to each per-language SentencePiece.
    offsets = getattr(tok, "token_id_offset", None)
    print(f"\ntoken_id_offset keys: {list(offsets.keys()) if isinstance(offsets, dict) else offsets}")

    tokenizers_dict = getattr(tok, "tokenizers_dict", None)
    if tokenizers_dict is None:
        raise RuntimeError("Expected `tokenizers_dict` on MultilingualTokenizer; layout changed?")

    # Build per-language [start, length] from each sub-tokenizer's vocab_size
    spans: dict[str, dict] = {}
    for lang in tok.langs:
        sub = tokenizers_dict[lang]
        start = offsets[lang] if isinstance(offsets, dict) else None
        length = sub.vocab_size
        spans[lang] = {"start": start, "length": length}
        print(f"  {lang:4s}  start={start:>5}  length={length:>4}")

    # Contiguity check.
    sorted_by_start = sorted(spans.items(), key=lambda kv: kv[1]["start"])
    contiguous = True
    expected = sorted_by_start[0][1]["start"]
    for lang, s in sorted_by_start:
        if s["start"] != expected:
            contiguous = False
            print(f"  GAP: {lang} starts at {s['start']}, expected {expected}")
        expected = s["start"] + s["length"]
    total = sum(s["length"] for s in spans.values())
    print(f"\nContiguous: {contiguous}")
    print(f"Sum of per-language lengths: {total}")
    print(f"Matches vocab_size?: {total == tok.vocab_size}")

    # Flat vocab dump. Each id -> its token string. For SentencePiece inside
    # an aggregate this requires asking the per-language sub-tokenizer.
    print("\nDumping flat vocab...")
    vocab_lines: list[str] = []
    for token_id in range(tok.vocab_size):
        try:
            piece = tok.ids_to_tokens([token_id])[0]
        except Exception as exc:
            piece = f"<err:{exc}>"
        vocab_lines.append(piece)
    Path(args.vocab_out).write_text("\n".join(vocab_lines) + "\n")
    print(f"  wrote {args.vocab_out}  ({len(vocab_lines)} lines)")

    # Spans JSON.
    Path(args.spans_out).write_text(json.dumps(
        {
            "total_vocab_size": tok.vocab_size,
            "languages": spans,
        },
        indent=2,
        ensure_ascii=False,
    ))
    print(f"  wrote {args.spans_out}")

    # Is there a prebuilt mask tensor on the ctc_decoder we can bake?
    print("\nctc_decoder buffers / parameters:")
    ctc = getattr(model, "ctc_decoder", None) or getattr(model, "decoder", None)
    for name, buf in ctc.named_buffers():
        print(f"  buf  {name}  shape={tuple(buf.shape)}  dtype={buf.dtype}")
    for name, param in ctc.named_parameters():
        print(f"  parm {name}  shape={tuple(param.shape)}  dtype={param.dtype}")

    # Look for a language-mask attr on the model itself.
    for attr in ("language_masks", "lang_masks", "masks", "softmax_masks"):
        obj = getattr(model, attr, None)
        if obj is not None:
            print(f"\nmodel.{attr}: {type(obj).__name__}  "
                  f"repr-preview={repr(obj)[:200]}")


if __name__ == "__main__":
    main()
