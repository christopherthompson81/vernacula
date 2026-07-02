#!/usr/bin/env python3
"""Parity for the MERGED (IPA-fine-tuned) transformer export: merged-PyTorch vs ONNX.

The stock parity_check compares ONNX against capture/reference.npz's tf_logits, which are
the BASE model's logits — invalid for a fine-tuned export (weights changed by design).
This runs the same captured inputs through both the merged PyTorch model and the exported
ONNX and compares, validating the EXPORT fidelity (not the fine-tune).
"""
import numpy as np
import torch
import onnxruntime as ort
from omnivoice.models.omnivoice import OmniVoice
from export_omnivoice import TransformerWrapper

BASE = "/mnt/data/models/omnivoice/k2-fsa-OmniVoice"
ADAPTER = "/mnt/data/omnivoice_ipa/train/checkpoints_v2/checkpoint-4000"
ONNX = "/mnt/data/omnivoice_ipa/onnx/omnivoice_transformer.onnx"
CAP = "/home/chris/Programming/vernacula/scripts/omnivoice_export/capture/reference.npz"

cap = np.load(CAP)
print("merging model (cpu, fp32) ...")
model = OmniVoice.from_pretrained(BASE, device_map="cpu", dtype=torch.float32)
from peft import PeftModel
model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
try:
    model.llm.config._attn_implementation = "sdpa"
except Exception:
    pass
wrap = TransformerWrapper(model).eval()

with torch.no_grad():
    pt = wrap(
        torch.from_numpy(cap["tf_input_ids"]),
        torch.from_numpy(cap["tf_audio_mask"]),
        torch.from_numpy(cap["tf_attention_mask"]),
    ).numpy()

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
onx = sess.run(["logits"], {
    "input_ids": cap["tf_input_ids"].astype(np.int64),
    "audio_mask": cap["tf_audio_mask"].astype(bool),
    "attention_mask": cap["tf_attention_mask"].astype(bool),
})[0]

agree = float(np.mean(onx.argmax(-1) == pt.argmax(-1)))
max_abs = float(np.max(np.abs(onx - pt)))
mse = float(np.mean((onx - pt) ** 2))
print(f"shapes: pytorch {pt.shape}  onnx {onx.shape}")
print(f"argmax agreement : {agree*100:.3f}%")
print(f"max abs logit diff: {max_abs:.3e}")
print(f"logit MSE        : {mse:.3e}")
print("PASS" if agree > 0.999 and max_abs < 1e-2 else "CHECK")
