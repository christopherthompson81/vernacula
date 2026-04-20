#!/usr/bin/env python3
"""
repackage_600m_indicconformer.py

AI4Bharat ships the 600M IndicConformer as a pre-exported ONNX package at
`ai4bharat/indic-conformer-600m-multilingual`. The architecture is identical
to the 120M we exported in Phase 1/2 (same 22 × 256 CTC layout, same 80-mel
Conformer frontend — verified byte-for-byte from their preprocessor.ts) but
the shipping shape differs:

  - Input/output tensor names: encoder uses `audio_signal/length` → `outputs/
    encoded_lengths`, CTC uses `encoder_output → logprobs`. We rename to
    `features/features_lens → encoded/encoded_lens` and `encoded → logits`
    so one C# backend handles both 120M and 600M without per-model branches.
  - Vocab: per-language dict of 257 tokens (first is `<unk>`, last is blank
    at local id 256). We flatten to a 5632-line flat vocab.txt (drop each
    language's `<unk>` and blank — shared CTC blank is implicit at id 5632).
  - Masks: per-language 5633-long bool arrays. We convert to 22 × {start,
    length} spans, matching our `language_spans.json` shape.
  - Preprocessor: they ship TorchScript. We reuse our existing DFT-conv1d
    `nemo128.onnx` from the 120M export because the frontend config is
    identical (verified: sample_rate 16k, 80 mel, n_fft 512, hop 160, win
    400, hann, preemph 0.97, log+add guard, per-feature normalize, mag² ).

Usage:
  python scripts/indicconformer_export/repackage_600m_indicconformer.py \\
    --nemo128-from ~/models/indicconformer_onnx/nemo128.onnx \\
    --output-dir ~/models/indicconformer_600m_onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# IO rename maps. Keys are AI4Bharat's names; values are ours.
ENCODER_IO_RENAME = {
    "audio_signal": "features",
    "length": "features_lens",
    "outputs": "encoded",
    "encoded_lengths": "encoded_lens",
}
CTC_IO_RENAME = {
    "encoder_output": "encoded",
    "logprobs": "logits",
}

# BCP-47 ordering used on the 120M side; the AI4Bharat dict uses the same
# keys but unordered. We keep this list so the span table and flat vocab
# come out in a stable deterministic order.
LANGS = [
    "as", "bn", "brx", "doi", "kok", "gu", "hi", "kn", "ks", "mai",
    "ml", "mr", "mni", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
]

REPO_ID = "ai4bharat/indic-conformer-600m-multilingual"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", required=True,
        help="Destination for the repackaged IndicConformer 600M package.",
    )
    p.add_argument(
        "--nemo128-from", required=True,
        help="Path to an existing nemo128.onnx from the 120M export (preprocessor is identical).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
    )
    return p.parse_args()


def snapshot_600m_assets():
    """Downloads (or reuses cached) 600M repo assets. Returns the snapshot root Path."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[
            "assets/encoder.onnx",
            "assets/ctc_decoder.onnx",
            "assets/vocab.json",
            "assets/language_masks.json",
            "assets/layers.*",
            "assets/onnx__*",
            "assets/pre_encode*",
            "assets/Constant_*",
        ],
    )
    return Path(p)


def rename_and_consolidate_onnx(
    src_path: Path, dst_path: Path, io_rename: dict[str, str],
) -> None:
    """Load ONNX with (possibly external) data, rename IO tensors, and
    resave with single-file external data sidecar `<dst>.data`.

    Note on symlinks: HF Hub stores blobs under cache/blobs/ and creates
    relative symlinks in the snapshot dir pointing back to them. The
    official onnx Python loader rejects symlinked external data as a
    safety measure, so we load the graph without external data first and
    manually populate each initializer's raw_data by reading the realpath
    of its referenced external file."""
    import onnx
    from onnx import external_data_helper

    print(f"  loading {src_path.name} (graph only; resolving external data manually)")
    model = onnx.load(str(src_path), load_external_data=False)

    base_dir = src_path.parent

    def _resolve_tensor(t) -> bool:
        if not external_data_helper.uses_external_data(t):
            return False
        loc = None
        for kv in t.external_data:
            if kv.key == "location":
                loc = kv.value
                break
        if loc is None:
            raise RuntimeError(f"Tensor {t.name!r} uses external data but has no 'location' key")
        t.raw_data = (base_dir / loc).resolve().read_bytes()
        t.data_location = onnx.TensorProto.DEFAULT
        del t.external_data[:]
        return True

    resolved_tensors = 0
    # Initializers.
    for t in model.graph.initializer:
        if _resolve_tensor(t):
            resolved_tensors += 1
    # Constant node attributes can also carry external data (AI4Bharat's
    # encoder has a Constant_1970_attr__value of this form). Walk all nodes
    # recursively in case any subgraphs exist (if/loop branches).
    def _walk(graph):
        nonlocal resolved_tensors
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t:
                    if _resolve_tensor(attr.t):
                        resolved_tensors += 1
                if attr.type == onnx.AttributeProto.TENSORS:
                    for t in attr.tensors:
                        if _resolve_tensor(t):
                            resolved_tensors += 1
                if attr.type == onnx.AttributeProto.GRAPH and attr.g:
                    _walk(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        _walk(g)
    _walk(model.graph)
    print(f"    resolved {resolved_tensors} external tensors (initializers + attrs)")

    # Rename graph inputs.
    for inp in model.graph.input:
        new = io_rename.get(inp.name)
        if new is not None:
            print(f"    rename input {inp.name} -> {new}")
            inp.name = new
    # Rename graph outputs.
    for out in model.graph.output:
        new = io_rename.get(out.name)
        if new is not None:
            print(f"    rename output {out.name} -> {new}")
            out.name = new
    # Rename references in node inputs/outputs.
    renamed = 0
    for node in model.graph.node:
        for i, n in enumerate(node.input):
            if n in io_rename:
                node.input[i] = io_rename[n]
                renamed += 1
        for i, n in enumerate(node.output):
            if n in io_rename:
                node.output[i] = io_rename[n]
                renamed += 1
    print(f"    rewrote {renamed} node IO references")

    sidecar = dst_path.name + ".data"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    stale_sidecar = dst_path.parent / sidecar
    if stale_sidecar.exists():
        stale_sidecar.unlink()

    # Decide whether to use external data: only if the model has large
    # initializers. For ctc_decoder (~23 MB) we can inline; for encoder
    # (~2.5 GB) we must externalize.
    total_init_bytes = sum(len(t.raw_data) for t in model.graph.initializer)
    if total_init_bytes > 500 * 1024 * 1024:
        print(f"  saving {dst_path.name} with external data -> {sidecar}")
        onnx.save_model(
            model, str(dst_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=sidecar,
            size_threshold=0,
        )
    else:
        print(f"  saving {dst_path.name} (inline, {total_init_bytes/1e6:.1f} MB)")
        onnx.save_model(model, str(dst_path))


def flatten_vocab(
    per_lang_vocab: dict[str, list[str]], out_path: Path,
) -> tuple[int, list[dict]]:
    """Turn {lang: [<unk>, t1, t2, ..., t255, <blank>]} (length 257) into a flat
    5632-line vocab.txt covering 22 × 256 real tokens. `<unk>` (local id 0) and
    the per-language blank (local id 256) are both dropped — the CTC graph has
    one shared blank at id 5632, implicit.

    Returns (total_vocab_size, spans) where spans is a list of {language,
    start, length} entries in LANGS order.
    """
    # AI4Bharat's per-language vocab has 257 entries: [<unk>, t1, ..., t256].
    # The CTC softmax head emits 256 logits per language (the first 256 of
    # the 257-dim per-lang output); the 257th output slot is the shared CTC
    # blank, not a real token. So vocab[lang][0..255] are the 256 real
    # entries (with <unk> at index 0) and vocab[lang][256] is unused padding
    # (mirrors the RNNT head's 257-dim layout).
    lines: list[str] = []
    spans = []
    cursor = 0
    for lang in LANGS:
        subvocab = per_lang_vocab[lang]
        assert len(subvocab) == 257, f"{lang}: expected 257 entries, got {len(subvocab)}"
        assert subvocab[0] in ("<unk>", "<UNK>", "unk"), f"{lang}[0]={subvocab[0]!r}"
        real = subvocab[:256]
        assert len(real) == 256, f"{lang}: first-256 slice is {len(real)} tokens"
        lines.extend(real)
        spans.append({"language": lang, "start": cursor, "length": 256})
        cursor += 256
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[vocab] wrote {out_path} ({len(lines)} lines)")
    return len(lines), spans


def write_language_spans(spans: list[dict], vocab_size: int, out_path: Path) -> None:
    # Match the 120M schema: top-level has total_vocab_size, blank_token_id,
    # and languages = {lang: {start, length}}.
    payload = {
        "total_vocab_size": int(vocab_size),
        "blank_token_id": int(vocab_size),  # shared CTC blank at 5632
        "languages": {s["language"]: {"start": s["start"], "length": s["length"]}
                      for s in spans},
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[spans] wrote {out_path} ({len(spans)} languages)")


def verify_masks_match_spans(
    per_lang_masks: dict[str, list[bool]], spans: list[dict], vocab_size: int,
) -> None:
    """Cross-check: AI4Bharat's pre-materialized language_masks.json should
    agree with the contiguous spans we derived from the per-language vocab
    sizes. If they don't, our flattening is wrong."""
    for s in spans:
        lang = s["language"]
        mask = per_lang_masks[lang]
        expected_true = list(range(s["start"], s["start"] + s["length"])) + [vocab_size]
        actual_true = [i for i, v in enumerate(mask) if v]
        if actual_true != expected_true:
            raise RuntimeError(
                f"[{lang}] mask disagrees with span. expected True at "
                f"{expected_true[:3]}..{expected_true[-3:]}, got "
                f"{actual_true[:3]}..{actual_true[-3:]}"
            )
    print("[verify] masks match derived spans for all 22 languages")


def write_config(out_path: Path) -> None:
    # Minimal config — enough for the C# side to know the preprocessor
    # contract and the blank id. Preprocessor params are the same as 120M.
    cfg = {
        "sample_rate": 16000,
        "preprocessor": {
            "features": 80,
            "n_fft": 512,
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "normalize": "per_feature",
            "preemph": 0.97,
            "mag_power": 2.0,
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 5.960464477539063e-08,
        },
        "encoder": {"d_model": 1024, "family": "indicconformer-600m"},
        "ctc": {"blank_token_id": 5632, "logits_dim": 5633},
    }
    out_path.write_text(json.dumps(cfg, indent=2))
    print(f"[config] wrote {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "encoder": out_dir / "encoder-model.onnx",
        "ctc_decoder": out_dir / "ctc_decoder-model.onnx",
        "nemo128": out_dir / "nemo128.onnx",
        "vocab": out_dir / "vocab.txt",
        "spans": out_dir / "language_spans.json",
        "config": out_dir / "config.json",
    }

    if not args.overwrite:
        existing = [p for p in targets.values() if p.exists()]
        if existing:
            raise SystemExit(
                "Output directory already contains target files. Re-run with "
                "--overwrite to replace them.\n"
                f"Existing: {', '.join(p.name for p in existing)}"
            )

    print("Snapshotting 600M ONNX assets (may be cached already) …")
    snap = snapshot_600m_assets()
    print(f"  {snap}")

    # 1. Encoder: rename IO, consolidate external data into a single sidecar.
    print("\n[encoder] repackaging")
    rename_and_consolidate_onnx(
        snap / "assets" / "encoder.onnx",
        targets["encoder"],
        ENCODER_IO_RENAME,
    )

    # 2. CTC decoder: rename IO, inline weights (small enough).
    print("\n[ctc_decoder] repackaging")
    rename_and_consolidate_onnx(
        snap / "assets" / "ctc_decoder.onnx",
        targets["ctc_decoder"],
        CTC_IO_RENAME,
    )

    # 3. Preprocessor: reuse our 120M-era nemo128.onnx verbatim.
    src_nemo128 = Path(args.nemo128_from).expanduser()
    if not src_nemo128.exists():
        raise SystemExit(f"nemo128 source not found: {src_nemo128}")
    print(f"\n[nemo128] copying from {src_nemo128}")
    shutil.copyfile(src_nemo128, targets["nemo128"])
    print(f"[nemo128] wrote {targets['nemo128']}")

    # 4. Vocab + spans (with cross-check against published masks).
    per_lang_vocab = json.loads((snap / "assets" / "vocab.json").read_text())
    per_lang_masks = json.loads((snap / "assets" / "language_masks.json").read_text())

    vocab_size, spans = flatten_vocab(per_lang_vocab, targets["vocab"])
    verify_masks_match_spans(per_lang_masks, spans, vocab_size)
    write_language_spans(spans, vocab_size, targets["spans"])

    # 5. Config.
    write_config(targets["config"])

    print("\nDone.")


if __name__ == "__main__":
    main()
