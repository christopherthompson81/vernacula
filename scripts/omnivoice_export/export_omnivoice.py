#!/usr/bin/env python3
"""Export k2-fsa/OmniVoice to ONNX (Phase 1: faithful fp32).

OmniVoice's generate() splits into three neural graphs (the rest — tokenizer, duration
estimate, the diffusion masking schedule, CFG/top-k/gumbel scoring — stays host-side):

  1. omnivoice_transformer.onnx  embeds + Qwen3 backbone + audio_heads (the denoiser
                                 called num_step times per generation)
  2. higgs_encoder.onnx          HiggsAudioV2TokenizerModel.encode  (ref wav -> codes)
  3. higgs_decoder.onnx          HiggsAudioV2TokenizerModel.decode  (codes -> 24k wav)

We drive each export with the REAL captured tensors from capture_reference.py so the
trace sees in-distribution shapes/values. Parity is checked separately (parity_check.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx.external_data_helper import _get_all_tensors

from omnivoice.models.omnivoice import OmniVoice


def _consolidate_external_data(path: Path) -> None:
    """Merge a graph's scattered external-weight files into one ``<name>.onnx.data``.

    The legacy exporter spills weights of models >2GB as many loose files in the output
    dir (``model.llm.embed_tokens.weight``, ``onnx__MatMul_*``, ...) with no extension —
    functional but unportable (move the .onnx alone and it breaks). Reload, re-save all
    tensors to a single sidecar, then delete the old loose files.
    """
    # Collect external locations BEFORE loading data — loading clears the location refs.
    meta = onnx.load(str(path), load_external_data=False)
    old_locs = set()
    for t in _get_all_tensors(meta):
        for ed in t.external_data:
            if ed.key == "location":
                old_locs.add(ed.value)
    model = onnx.load(str(path), load_external_data=True)
    data_name = path.name + ".data"
    # remove a stale consolidated sidecar so save starts clean
    (path.parent / data_name).unlink(missing_ok=True)
    onnx.save_model(
        model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
        location=data_name, size_threshold=1024,
    )
    removed = 0
    for loc in old_locs:
        if loc == data_name:
            continue
        f = path.parent / loc
        if f.exists():
            f.unlink()
            removed += 1
    print(f"  consolidated {len(old_locs)} external files -> {data_name} "
          f"(removed {removed} loose files)")


# ---------------------------------------------------------------------------
# ONNX-friendly wrappers: unwrap the ModelOutput dataclasses, return raw tensors.
# ---------------------------------------------------------------------------
class TransformerWrapper(nn.Module):
    """input_ids[2B,8,S] int64, audio_mask[2B,S] bool, attention_mask[2B,1,S,S] bool
    -> logits[2B,8,S,1025] float32."""

    def __init__(self, model: OmniVoice):
        super().__init__()
        self.model = model

    def forward(self, input_ids, audio_mask, attention_mask):
        return self.model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
        ).logits


class EncoderWrapper(nn.Module):
    """input_values[B,1,T] float32 @24k -> audio_codes[B,8,Tc] int64."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tok = tokenizer

    def forward(self, input_values):
        return self.tok.encode(input_values).audio_codes


class DecoderWrapper(nn.Module):
    """audio_codes[B,8,Tc] int64 -> audio_values[B,1,Tsamp] float32 @24k."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tok = tokenizer

    def forward(self, audio_codes):
        return self.tok.decode(audio_codes).audio_values


def _export(module, inputs, input_names, output_names, dynamic_axes, path, opset,
            try_dynamo):
    """Export one graph, preferring dynamo, falling back to the legacy exporter."""
    errors = []
    if try_dynamo:
        try:
            torch.onnx.export(
                module, inputs, str(path),
                input_names=input_names, output_names=output_names,
                dynamic_axes=dynamic_axes, opset_version=opset,
                dynamo=True,
            )
            print(f"  [dynamo] exported -> {path}")
            return "dynamo"
        except Exception as e:  # noqa: BLE001
            errors.append(f"dynamo: {type(e).__name__}: {e}")
            print(f"  [dynamo] failed: {type(e).__name__}: {e}\n  -> falling back to legacy")
    try:
        torch.onnx.export(
            module, inputs, str(path),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=opset,
            dynamo=False,
        )
        print(f"  [legacy] exported -> {path}")
        return "legacy"
    except Exception as e:  # noqa: BLE001
        errors.append(f"legacy: {type(e).__name__}: {e}")
        raise RuntimeError("Both exporters failed:\n" + "\n".join(errors))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data/models/omnivoice/k2-fsa-OmniVoice")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--capture", default=str(Path(__file__).resolve().parent / "capture/reference.npz")
    )
    p.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent / "onnx")
    )
    p.add_argument("--opset", type=int, default=18)
    p.add_argument(
        "--components", default="transformer,encoder,decoder",
        help="Comma-separated subset to export.",
    )
    p.add_argument("--no-dynamo", action="store_true", help="Skip dynamo, legacy only.")
    p.add_argument("--adapter", default=None,
                   help="Optional peft LoRA dir to merge into the model before export "
                        "(produces a STANDALONE fine-tuned graph, not base+adapter).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(args.components.split(","))
    try_dynamo = not args.no_dynamo

    cap = np.load(args.capture)
    dev = torch.device(args.device)

    print(f"Loading {args.model} (fp32) on {dev} ...")
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=torch.float32)
    if args.adapter:
        # Merge the LoRA (+ full embed_tokens from modules_to_save) into the base weights
        # so the exported graph is a self-standing fine-tuned model, not base+delta.
        from peft import PeftModel
        print(f"Merging adapter {args.adapter} ...")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    # Force a plain attention impl (the flex_attention path is guarded and only used
    # when document_ids is passed; inference passes an explicit attention_mask).
    try:
        model.llm.config._attn_implementation = "sdpa"
    except Exception:  # noqa: BLE001
        pass

    report: dict[str, str] = {}

    if "transformer" in want:
        print("Exporting transformer ...")
        wrap = TransformerWrapper(model).eval()
        inp = (
            torch.from_numpy(cap["tf_input_ids"]).to(dev),
            torch.from_numpy(cap["tf_audio_mask"]).to(dev),
            torch.from_numpy(cap["tf_attention_mask"]).to(dev),
        )
        report["transformer"] = _export(
            wrap, inp,
            input_names=["input_ids", "audio_mask", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "two_b", 2: "seq"},
                "audio_mask": {0: "two_b", 1: "seq"},
                "attention_mask": {0: "two_b", 2: "seq", 3: "seq"},
                "logits": {0: "two_b", 2: "seq"},
            },
            path=out_dir / "omnivoice_transformer.onnx", opset=args.opset,
            try_dynamo=try_dynamo,
        )
        # legacy export spills the 0.6B weights as loose files -> single sidecar
        _consolidate_external_data(out_dir / "omnivoice_transformer.onnx")

    if "encoder" in want:
        print("Exporting higgs encoder ...")
        wrap = EncoderWrapper(model.audio_tokenizer).eval()
        inp = (torch.from_numpy(cap["enc_input_values"]).to(model.audio_tokenizer.device),)
        report["encoder"] = _export(
            wrap, inp,
            input_names=["input_values"], output_names=["audio_codes"],
            dynamic_axes={
                "input_values": {0: "batch", 2: "samples"},
                "audio_codes": {0: "batch", 2: "codes"},
            },
            path=out_dir / "higgs_encoder.onnx", opset=args.opset,
            try_dynamo=try_dynamo,
        )

    if "decoder" in want:
        print("Exporting higgs decoder ...")
        wrap = DecoderWrapper(model.audio_tokenizer).eval()
        inp = (torch.from_numpy(cap["dec_audio_codes"]).to(model.audio_tokenizer.device),)
        report["decoder"] = _export(
            wrap, inp,
            input_names=["audio_codes"], output_names=["audio_values"],
            dynamic_axes={
                "audio_codes": {0: "batch", 2: "codes"},
                "audio_values": {0: "batch", 2: "samples"},
            },
            path=out_dir / "higgs_decoder.onnx", opset=args.opset,
            try_dynamo=try_dynamo,
        )

    (out_dir / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nExport report -> {out_dir/'export-report.json'}: {report}")


if __name__ == "__main__":
    main()
