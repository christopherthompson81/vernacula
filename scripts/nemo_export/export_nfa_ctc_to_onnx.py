"""Export a NeMo CTC ASR .nemo checkpoint into the ONNX bundle the
C# Viterbi forced aligner will consume (issue #36 / Stage 1 #9).

This script is the export side of NeMo Forced Aligner (NFA): NFA itself
is a CTC ASR model + Viterbi DP, and the model side is what we export
here. Viterbi over the model's log-probs is a follow-up PR — runs in
C# against this bundle's outputs.

Output bundle (mirrors the Parakeet export's file layout where possible):

  - nemo128.onnx                — log-mel preprocessor (same op contract
                                  as the Parakeet bundle; reusable
                                  across CTC and RNNT/TDT exports)
  - ctc-model.onnx[.data]       — encoder + CTC projection in one
                                  graph. Input: features [B, F, T]
                                  + features_lens [B] from nemo128.
                                  Output: log_probs [B, T, V+1] and
                                  output_lens [B].
  - vocab.txt                   — sentencepiece vocab, one (token, id)
                                  per line, plus the CTC blank token
                                  at id == len(vocab).
  - config.json                 — export metadata + full NeMo cfg dump
                                  for reference
  - export-report.json          — concise summary (what got exported,
                                  what failed, what notes apply)

Supported source models:

- Pure CTC checkpoints (`EncDecCTCModel`, `EncDecCTCModelBPE`).
- Hybrid RNNT+CTC checkpoints (`EncDecHybridRNNTCTCModel`,
  `EncDecHybridRNNTCTCBPEModel`) — the script switches the model's
  decoding strategy to CTC before export so the exported graph only
  carries the CTC head. Recommended default per NFA's own quickstart:
  `stt_en_fastconformer_hybrid_large_pc` (CTC head used).

Pattern intentionally matches `export_parakeet_nemo_to_onnx.py` so a
single shared-helpers refactor can pull both scripts together when a
third NeMo export consumer arrives. Some helpers duplicate the
Parakeet script's bodies verbatim — keep them in sync until that
refactor.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a NeMo CTC ASR .nemo checkpoint into the ONNX bundle "
            "used by Vernacula's forced-aligner (issue #36)."
        )
    )
    parser.add_argument(
        "--nemo",
        required=True,
        help=(
            "Path to a local .nemo file. Recommended defaults: "
            "stt_en_fastconformer_hybrid_large_pc (use CTC head — script handles "
            "the switch); stt_en_fastconformer_ctc_large (pure CTC); "
            "stt_en_conformer_ctc_large (older, smaller)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive nemo128.onnx, ctc-model.onnx, vocab.txt, and config.json.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset to request during export.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use for model restore/export.",
    )
    parser.add_argument(
        "--preprocessor-seconds",
        type=float,
        default=20.0,
        help="Dummy waveform length used when exporting the preprocessor wrapper.",
    )
    parser.add_argument(
        "--skip-preprocessor",
        action="store_true",
        help=(
            "Skip nemo128.onnx export if you already have a compatible preprocessor bundle "
            "(e.g. from the Parakeet export — the log-mel frontend is the same)."
        ),
    )
    parser.add_argument(
        "--preprocessor-opset",
        type=int,
        default=None,
        help="Optional opset override for nemo128.onnx. Defaults to --opset.",
    )
    parser.add_argument(
        "--preprocessor-mode",
        choices=("wrapper", "custom", "dft"),
        default="dft",
        help=(
            "How to build nemo128.onnx. 'wrapper' exports NeMo's preprocessor module "
            "directly (often fails). 'custom' rebuilds it with explicit STFT. 'dft' uses "
            "the Conv1d-DFT variant that tends to match NeMo's frame counts most reliably. "
            "Defaults match the Parakeet export's preferred mode."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in --output-dir.",
    )
    return parser.parse_args()


# ── Generic helpers (mirrors export_parakeet_nemo_to_onnx.py) ────────────────


def import_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf
    except Exception as exc:
        raise SystemExit(
            "Missing export dependencies. Create a Python 3.11/3.12 environment and "
            "install the packages from scripts/nemo_export/requirements.txt.\n"
            f"Original import error: {exc}"
        ) from exc
    return torch, ASRModel, OmegaConf, exc_to_string


def exc_to_string(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    return requested


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        collisions = [
            name
            for name in (
                "nemo128.onnx",
                "ctc-model.onnx",
                "ctc-model.onnx.data",
                "vocab.txt",
                "tokenizer.model",
                "config.json",
                "export-report.json",
            )
            if (path / name).exists()
        ]
        if collisions:
            raise SystemExit(
                "Output directory already contains export targets. "
                "Re-run with --overwrite to replace them.\n"
                f"Existing files: {', '.join(collisions)}"
            )


def cleanup_target(path: Path) -> None:
    if path.exists():
        path.unlink()


def replace_file(src: Path, dst: Path) -> None:
    cleanup_target(dst)
    shutil.move(str(src), str(dst))


def replace_if_present(src: Path, dst: Path) -> None:
    if src.exists():
        replace_file(src, dst)


def consolidate_external_data(onnx_path: Path) -> None:
    """Re-save an ONNX model so all external tensors live in a single sidecar file.

    Behavior mirrors export_parakeet_nemo_to_onnx.consolidate_external_data:
    no-op when already clean (single sidecar), full re-save + stale per-tensor
    cleanup otherwise. size_threshold=0 so every initializer goes external
    (matches Parakeet) — keeps the .onnx file small and the .data file the
    sole weight carrier.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    locations: set[str] = set()
    for t in model.graph.initializer:
        for kv in t.external_data:
            if kv.key == "location":
                locations.add(kv.value)

    sidecar_name = onnx_path.name + ".data"
    already_clean = len(locations) == 0 or (len(locations) == 1 and sidecar_name in locations)
    if already_clean:
        return

    print(f"  Consolidating {len(locations)} external data files → {sidecar_name} …")
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar_name,
        size_threshold=0,
    )

    parent = onnx_path.parent
    for loc in locations:
        stale = parent / loc
        if stale.exists() and stale.name != sidecar_name:
            stale.unlink()


# ── Hybrid-vs-pure CTC handling ─────────────────────────────────────────────


def switch_to_ctc_decoding_if_hybrid(model: Any) -> bool:
    """If the loaded model is a hybrid RNNT+CTC checkpoint, switch its
    decoding strategy to CTC so model.export() emits only the CTC head.
    Returns True if a switch happened."""
    model_class = type(model).__name__
    is_hybrid = "Hybrid" in model_class
    if not is_hybrid:
        return False
    # NeMo's hybrid models expose change_decoding_strategy(...) that gates
    # which head is active. We need the CTC head for forced alignment.
    if not hasattr(model, "change_decoding_strategy"):
        raise RuntimeError(
            f"Loaded model is hybrid ({model_class}) but doesn't expose "
            "change_decoding_strategy. NeMo API changed?"
        )
    try:
        model.change_decoding_strategy(decoder_type="ctc")
    except TypeError:
        # Older NeMo API used a positional or differently-named arg.
        model.change_decoding_strategy("ctc")
    return True


# ── CTC encoder + head export ───────────────────────────────────────────────


def export_ctc_graph(model: Any, output_dir: Path, opset: int) -> None:
    """Export the encoder + CTC projection as a single ONNX.

    NeMo's `model.export()` for a CTC model writes one ONNX (and possibly
    a sidecar .data) containing the encoder followed by the CTC head's
    linear projection + log-softmax. Inputs: (features [B, F, T],
    features_lens [B]). Outputs: (log_probs [B, T, V+1], log_probs_lens [B])
    — the trailing +1 in vocab dim is the CTC blank.

    The encoder for hybrid models is shared with the RNNT head; switching
    decoding strategy to 'ctc' before export should make NeMo only wire
    the CTC head into the exported graph."""
    export_stub = output_dir / "nfa_ctc.onnx"
    cleanup_target(export_stub)

    model.export(str(export_stub), onnx_opset_version=opset)

    # NeMo may emit the file at the stub path directly, OR under a
    # prefixed name like "encoder-nfa_ctc.onnx" for split exports. CTC
    # models are usually single-graph (no split), but check both.
    target = output_dir / "ctc-model.onnx"
    if export_stub.exists():
        # Single-file export: rename to the target name.
        replace_file(export_stub, target)
        replace_if_present(
            output_dir / "nfa_ctc.onnx.data",
            output_dir / "ctc-model.onnx.data",
        )
    else:
        # NeMo didn't write to the stub path. Most common failure: hybrid
        # model didn't honor the CTC switch and emitted a SPLIT export
        # (encoder + decoder_joint). List what actually appeared so the
        # next person diagnosing this doesn't have to ls the dir.
        artifacts = sorted(p.name for p in output_dir.glob("*nfa_ctc*"))
        raise RuntimeError(
            f"NeMo export did not produce nfa_ctc.onnx. Found instead: {artifacts or '(nothing matching *nfa_ctc*)'}. "
            "If you see encoder-nfa_ctc.onnx + decoder_joint-nfa_ctc.onnx, the hybrid-model "
            "CTC switch didn't take — try a pure-CTC checkpoint (e.g. stt_en_fastconformer_ctc_large) "
            "or report the NeMo version that produced this output."
        )

    consolidate_external_data(target)


# ── Preprocessor export (verbatim from export_parakeet_nemo_to_onnx) ────────
#
# The CTC and RNNT models share the same NeMo preprocessor (log-mel
# frontend). Rather than re-write the preprocessor wrappers, we import
# them from the Parakeet export module. The shared bits are at module
# scope and don't have side effects.


def export_preprocessor_via_parakeet(
    model: Any,
    torch: Any,
    output_dir: Path,
    opset: int,
    dummy_seconds: float,
    mode: str,
) -> None:
    """Delegate to the Parakeet export's preprocessor function, which
    handles all three wrapper modes (wrapper/custom/dft) for any NeMo
    AudioToMelSpectrogramPreprocessor."""
    import sys

    helper_dir = str(Path(__file__).parent)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    from export_parakeet_nemo_to_onnx import export_preprocessor  # type: ignore[import-not-found]

    export_preprocessor(
        model,
        torch,
        output_dir,
        opset,
        dummy_seconds,
        mode=mode,
        dynamo=mode not in ("custom", "dft"),
        optimize=mode not in ("custom", "dft"),
        verify=False,
        do_constant_folding=True,
        dynamic_axes_enabled=True,
    )


# ── Vocab + config + report (mirrors Parakeet export) ───────────────────────


def extract_vocab_entries(model: Any) -> list[tuple[int, str]]:
    """Same logic as Parakeet's vocab extraction. CTC and RNN-T share
    the tokenizer surface."""
    vocab_by_id: dict[int, str] = {}

    decoding_vocab = getattr(getattr(model, "decoding", None), "vocabulary", None)
    if decoding_vocab:
        for idx, token in enumerate(decoding_vocab):
            vocab_by_id[idx] = str(token)

    tokenizer = getattr(model, "tokenizer", None)
    tokenizer_impl = getattr(tokenizer, "tokenizer", None)
    if hasattr(tokenizer_impl, "get_vocab"):
        raw_vocab = tokenizer_impl.get_vocab()
        if isinstance(raw_vocab, dict):
            for token, idx in raw_vocab.items():
                normalized_idx = int(idx) - 1 if int(idx) > 0 else int(idx)
                vocab_by_id[normalized_idx] = str(token)

    if not vocab_by_id:
        raise RuntimeError("Could not extract a tokenizer vocabulary from the NeMo model.")

    return sorted(vocab_by_id.items(), key=lambda item: item[0])


def write_vocab(model: Any, output_dir: Path) -> None:
    entries = extract_vocab_entries(model)
    blank_id = len(entries)
    blank_token = "<blk>"
    dst = output_dir / "vocab.txt"
    with dst.open("w", encoding="utf-8", newline="\n") as handle:
        for idx, token in entries:
            handle.write(f"{token} {idx}\n")
        if not any(token == blank_token for _, token in entries):
            handle.write(f"{blank_token} {blank_id}\n")


@dataclass
class ExportMetadata:
    nemo_file: str
    nemo_model_class: str
    nemo_model_name: str | None
    hybrid_switched_to_ctc: bool
    device: str
    opset: int
    sample_rate: int | None
    # The Viterbi DP in the C# follow-up converts frame indices to wall-clock
    # timestamps as `frame_index * frame_shift_seconds * encoder_subsampling`.
    # Captured at export time from the live NeMo model so the C# side doesn't
    # have to re-derive them by probing the preprocessor / encoder ONNX shapes.
    frame_shift_seconds: float | None
    encoder_subsampling: int | None
    preprocessor_exported: bool
    tokenizer_model_exported: bool
    notes: list[str]


def extract_frame_timing(model: Any) -> tuple[float | None, int | None]:
    """Read frame shift (seconds) + encoder subsampling factor from the
    live NeMo model. Returns (None, None) when either can't be determined —
    the C# side will warn rather than crash in that case."""
    frame_shift = None
    subsampling = None
    preproc = getattr(model, "preprocessor", None)
    if preproc is not None:
        featurizer = getattr(preproc, "featurizer", None)
        sr = getattr(preproc, "_sample_rate", None)
        hop = getattr(featurizer, "hop_length", None) if featurizer is not None else None
        if sr and hop:
            frame_shift = float(hop) / float(sr)
    encoder_cfg = getattr(getattr(model, "cfg", None), "encoder", None)
    if encoder_cfg is not None:
        # FastConformer / Conformer typically expose subsampling_factor;
        # sometimes named differently (e.g. "subsampling" or "stride_factor").
        for attr in ("subsampling_factor", "subsampling", "stride_factor"):
            val = getattr(encoder_cfg, attr, None)
            if val is not None:
                try:
                    subsampling = int(val)
                    break
                except (TypeError, ValueError):
                    pass
    return frame_shift, subsampling


def export_tokenizer_model(model: Any, output_dir: Path) -> bool:
    """Copy the sentencepiece tokenizer model file alongside vocab.txt so
    the C# Viterbi side can tokenize reference transcripts with the exact
    same encoding the CTC model was trained on. vocab.txt alone is
    insufficient — it lists tokens but not the byte→token encoding rules.

    Returns True if a tokenizer.model file was written."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return False
    # NeMo's BPE tokenizer wraps a SentencePieceProcessor at .tokenizer
    # and exposes the source .model path at one of these attrs depending
    # on NeMo version.
    candidates = []
    for attr in ("tokenizer_model_path", "model_path", "vocab_path"):
        path = getattr(tokenizer, attr, None)
        if path:
            candidates.append(Path(str(path)))
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        for attr in ("model_file", "model_path"):
            path = getattr(inner, attr, None)
            if path:
                candidates.append(Path(str(path)))
    for src in candidates:
        if src.exists() and src.suffix == ".model":
            dst = output_dir / "tokenizer.model"
            cleanup_target(dst)
            shutil.copyfile(str(src), str(dst))
            return True
    return False


def write_config(model: Any, omega_conf: Any, output_dir: Path, metadata: ExportMetadata) -> None:
    cfg = getattr(model, "cfg", None)
    payload: dict[str, Any] = {"export": asdict(metadata)}
    if cfg is not None:
        try:
            payload["nemo_cfg"] = omega_conf.to_container(cfg, resolve=True)
        except Exception:
            payload["nemo_cfg"] = str(cfg)
    with (output_dir / "config.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_report(output_dir: Path, metadata: ExportMetadata) -> None:
    with (output_dir / "export-report.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(metadata), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    nemo_path = Path(args.nemo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not nemo_path.exists():
        raise SystemExit(f".nemo file not found: {nemo_path}")

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    torch, ASRModel, OmegaConf, format_exc = import_dependencies()
    device = resolve_device(torch, args.device)

    model = ASRModel.restore_from(str(nemo_path), map_location=device)
    model.freeze()
    model.eval()
    model.to(device)

    notes: list[str] = []
    hybrid_switched = switch_to_ctc_decoding_if_hybrid(model)
    if hybrid_switched:
        notes.append(
            f"Loaded {type(model).__name__} (hybrid); switched decoding to CTC "
            "before export so the graph only carries the CTC head."
        )

    export_ctc_graph(model, output_dir, args.opset)

    preprocessor_exported = False
    if args.skip_preprocessor:
        notes.append("Skipped preprocessor export because --skip-preprocessor was provided.")
    else:
        try:
            export_preprocessor_via_parakeet(
                model,
                torch,
                output_dir,
                args.preprocessor_opset or args.opset,
                args.preprocessor_seconds,
                args.preprocessor_mode,
            )
            preprocessor_exported = True
            notes.append(f"Preprocessor exported in mode={args.preprocessor_mode}.")
        except Exception as exc:
            notes.append(
                "Preprocessor export failed. ctc-model.onnx is still usable if you bring your "
                f"own preprocessor (e.g. from the Parakeet bundle). Failure: {format_exc(exc)}"
            )

    write_vocab(model, output_dir)
    tokenizer_model_exported = export_tokenizer_model(model, output_dir)
    if not tokenizer_model_exported:
        notes.append(
            "Could not locate the sentencepiece tokenizer .model file inside the loaded "
            "NeMo model. The C# Viterbi side will need this to tokenize reference "
            "transcripts with the same encoding the CTC model was trained on — without "
            "it, only vocab.txt's token strings are available. Check NeMo version / "
            "tokenizer attrs if this matters for your downstream consumer."
        )

    frame_shift_seconds, encoder_subsampling = extract_frame_timing(model)
    if frame_shift_seconds is None or encoder_subsampling is None:
        notes.append(
            f"Could not fully determine frame timing (frame_shift={frame_shift_seconds}, "
            f"subsampling={encoder_subsampling}). The C# Viterbi will need both to convert "
            "frame indices to wall-clock timestamps."
        )

    cfg_target = getattr(model, "cfg", None)
    metadata = ExportMetadata(
        nemo_file=str(nemo_path),
        nemo_model_class=type(model).__name__,
        nemo_model_name=getattr(cfg_target, "name", None) if cfg_target is not None else None,
        hybrid_switched_to_ctc=hybrid_switched,
        device=device,
        opset=args.opset,
        sample_rate=int(getattr(getattr(model, "preprocessor", None), "_sample_rate", 0) or 0) or None,
        frame_shift_seconds=frame_shift_seconds,
        encoder_subsampling=encoder_subsampling,
        preprocessor_exported=preprocessor_exported,
        tokenizer_model_exported=tokenizer_model_exported,
        notes=notes,
    )

    write_config(model, OmegaConf, output_dir, metadata)
    write_report(output_dir, metadata)

    print("Export complete:")
    for name in (
        "ctc-model.onnx",
        "ctc-model.onnx.data",
        "nemo128.onnx",
        "vocab.txt",
        "tokenizer.model",
        "config.json",
        "export-report.json",
    ):
        path = output_dir / name
        if path.exists():
            print(f"  - {path}")

    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
