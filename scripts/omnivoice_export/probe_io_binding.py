#!/usr/bin/env python3
"""Characterise ONNX Runtime IO-binding modes on CUDA for the OmniVoice transformer, to find
a correct mitigation for the corruption seen when CPU-pinned OrtValues were bound to a CUDA
session in C#.

Compares logits from, all on CUDA (use_tf32=0 for a stable reference):
  A. plain session.run (known-correct path)
  B. CPU-bound REUSED output OrtValue (the C# pattern that corrupted), run repeatedly
  C. device-bound output + copy_outputs_to_cpu (the canonical CUDA IO-binding pattern)
  D. CPU-bound output + explicit SynchronizeBoundOutputs() before reading

Reports max-abs vs A for each, after several reuse iterations (the bug showed up in a loop).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", default=str(Path(__file__).resolve().parent / "onnx/omnivoice_transformer.onnx"))
    p.add_argument("--capture", default=str(Path(__file__).resolve().parent / "capture/reference.npz"))
    args = p.parse_args()

    cap = np.load(args.capture)
    ids = np.ascontiguousarray(cap["tf_input_ids"].astype(np.int64))
    am = np.ascontiguousarray(cap["tf_audio_mask"].astype(bool))
    at = np.ascontiguousarray(cap["tf_attention_mask"].astype(bool))
    B, _, S = ids.shape
    out_shape = [B, 8, S, 1025]

    prov = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=prov)
    feeds = {"input_ids": ids, "audio_mask": am, "attention_mask": at}

    # A. plain run (reference)
    refA = sess.run(["logits"], feeds)[0]

    def maxabs(x):
        return float(np.max(np.abs(x - refA)))

    # B. CPU-bound reused output OrtValue, several iterations (mimics the C# loop)
    out_cpu = ort.OrtValue.ortvalue_from_numpy(np.zeros(out_shape, np.float32))  # CPU
    ioB = sess.io_binding()
    ioB.bind_cpu_input("input_ids", ids)
    ioB.bind_cpu_input("audio_mask", am)
    ioB.bind_cpu_input("attention_mask", at)
    ioB.bind_ortvalue_output("logits", out_cpu)
    for _ in range(5):
        sess.run_with_iobinding(ioB)
    B_out = out_cpu.numpy().copy()
    print(f"B  cpu-bound reused output     : max-abs vs plain = {maxabs(B_out):.3e}")

    # D. same as B but explicit output sync before reading
    out_cpu2 = ort.OrtValue.ortvalue_from_numpy(np.zeros(out_shape, np.float32))
    ioD = sess.io_binding()
    ioD.bind_cpu_input("input_ids", ids)
    ioD.bind_cpu_input("audio_mask", am)
    ioD.bind_cpu_input("attention_mask", at)
    ioD.bind_ortvalue_output("logits", out_cpu2)
    for _ in range(5):
        sess.run_with_iobinding(ioD)
        ioD.synchronize_outputs()
    D_out = out_cpu2.numpy().copy()
    print(f"D  cpu-bound + sync_outputs    : max-abs vs plain = {maxabs(D_out):.3e}")

    # C. device-bound output, copy back
    ioC = sess.io_binding()
    ioC.bind_cpu_input("input_ids", ids)
    ioC.bind_cpu_input("audio_mask", am)
    ioC.bind_cpu_input("attention_mask", at)
    ioC.bind_output("logits", "cuda")
    for _ in range(5):
        sess.run_with_iobinding(ioC)
    C_out = ioC.copy_outputs_to_cpu()[0]
    print(f"C  device-bound output         : max-abs vs plain = {maxabs(C_out):.3e}")

    # E. device-bound input OrtValues + device output (fully on device)
    d_ids = ort.OrtValue.ortvalue_from_numpy(ids, "cuda", 0)
    d_am = ort.OrtValue.ortvalue_from_numpy(am, "cuda", 0)
    d_at = ort.OrtValue.ortvalue_from_numpy(at, "cuda", 0)
    d_out = ort.OrtValue.ortvalue_from_shape_and_type(out_shape, np.float32, "cuda", 0)
    ioE = sess.io_binding()
    ioE.bind_ortvalue_input("input_ids", d_ids)
    ioE.bind_ortvalue_input("audio_mask", d_am)
    ioE.bind_ortvalue_input("attention_mask", d_at)
    ioE.bind_ortvalue_output("logits", d_out)
    for _ in range(5):
        sess.run_with_iobinding(ioE)
    E_out = d_out.numpy()
    print(f"E  device in + device out      : max-abs vs plain = {maxabs(E_out):.3e}")


if __name__ == "__main__":
    main()
