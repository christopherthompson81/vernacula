#!/usr/bin/env python
"""Same loop as test_streaming_parity.py but with ORT IO binding: KV cache tensors stay on
the CUDA device between steps and only logits (last row) come back. This is what the C#
backend does, so its numbers are the ones that matter for the app. Writes the same record
shape so compare_runs.py can check it is still parity-clean."""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vibevoice_export"))
from test_static_kv_parity import _decoder_fp_inputs, _ort_value_to_f32  # noqa: E402
from test_streaming_parity import windows  # noqa: E402


def to_ort(name, arr, fp_casts, device):
    if name in fp_casts:
        import ml_dtypes
        if fp_casts[name] == ml_dtypes.bfloat16:
            u16 = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16).view(np.uint16))
            return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(u16, 16)
        arr = arr.astype(fp_casts[name])
    return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), device, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--opt-level", default="extended", choices=["basic", "extended", "all"])
    args = ap.parse_args()

    md = Path(args.model_dir)
    rep = json.loads((md / "export-report.json").read_text())
    st, tk = rep["streaming"], rep["tokenizer"]
    L, KH, HD, H = rep["num_layers"], rep["num_kv_heads"], rep["head_dim"], rep["hidden_size"]
    so = ort.SessionOptions()
    so.graph_optimization_level = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                   "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                                   "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[args.opt_level]
    prov = [("CUDAExecutionProvider", {"device_id": 0})]
    t0 = time.time()
    enc = ort.InferenceSession(str(md / "audio_encoder.onnx"), so, providers=prov)
    dec = ort.InferenceSession(str(md / "decoder_single.onnx"), so, providers=prov)
    load_s = time.time() - t0
    fp = _decoder_fp_inputs(dec)
    kv_np = np.float32 if rep["f32_kv_cache"] else fp["past_key_0"]
    kv_names_in = [n for i in range(L) for n in (f"past_key_{i}", f"past_value_{i}")]
    kv_names_out = [n for i in range(L) for n in (f"present_key_{i}", f"present_value_{i}")]

    from vibevoice.processor.audio_utils import load_audio_use_ffmpeg
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(md))
    audio, _ = load_audio_use_ffmpeg(args.audio, resample=True, target_sr=st["sample_rate"])
    audio = audio.astype(np.float32)
    dur = len(audio) / st["sample_rate"]

    # Empty tensors start on the host (a 0-byte device allocation segfaults in ORT 1.29);
    # every present_* output after the first step is device-resident.
    past = [to_ort(n, np.zeros((1, KH, 0, HD), dtype=kv_np), {}, "cpu") for n in kv_names_in]
    empty_audio = np.zeros((0, H), dtype=np.float32)
    empty_ids = np.zeros((1, 0), dtype=np.int64)

    def step(prefix, audio_emb, suffix):
        nonlocal past
        # A fresh binding per step: IoBinding.clear_binding_inputs()/outputs() segfaults in
        # ORT 1.29's Python wrapper (the C# ClearBoundInputs is fine). Cheap to create.
        binding = dec.io_binding()
        keep = [to_ort("prefix_input_ids", np.asarray(prefix, np.int64).reshape(1, -1), fp, "cpu"),
                to_ort("audio_embeddings", audio_emb, fp, "cpu"),
                to_ort("suffix_input_ids", np.asarray(suffix, np.int64).reshape(1, -1), fp, "cpu")]
        for n, v in zip(("prefix_input_ids", "audio_embeddings", "suffix_input_ids"), keep):
            binding.bind_ortvalue_input(n, v)
        for n, v in zip(kv_names_in, past):
            binding.bind_ortvalue_input(n, v)
        binding.bind_output("logits", "cpu")
        for n in kv_names_out:
            binding.bind_output(n, "cuda", 0)
        dec.run_with_iobinding(binding)
        outs = binding.get_outputs()
        past = outs[1:]
        logits = _ort_value_to_f32(outs[0])   # lm_head is BF16, so logits come back BF16
        return int(np.argmax(logits[0, -1]))

    t0 = time.time()
    step(tk["prompt_token_ids"], empty_audio, empty_ids)
    chunks, times, ntok, enc_s = [], [], 0, 0.0
    for w in windows(audio, st["window_samples"], st["hop_samples"]):
        te = time.time()
        emb = enc.run(None, {"input_values": w[None]})[0]
        enc_s += time.time() - te
        nxt = step([tk["speech_start_id"]], emb, [tk["speech_end_id"]])
        ids = []
        for _ in range(args.max_new_tokens):
            if nxt in (tk["text_chunk_end_id"], tk["eos_token_id"]):
                break
            ids.append(nxt)
            nxt = step([nxt], empty_audio, empty_ids)
        step([tk["text_chunk_end_id"]], empty_audio, empty_ids)
        ntok += len(ids)
        text = tok.decode(ids, skip_special_tokens=True)
        for s in tk["strip_tokens"]:
            text = text.replace(s, "")
        chunks.append(text); times.append(time.time() - t0)
    gen_s = time.time() - t0
    import subprocess
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    rec = dict(audio=os.path.basename(args.audio), runtime="ort-cuda-iobound", opt_level=args.opt_level,
               model_dir=str(md), duration_s=dur, load_s=load_s, gen_s=gen_s, rtf=gen_s / dur,
               enc_s=enc_s, tokens=ntok, n_chunks=len(chunks), kv_len=int(past[0].shape()[2]),
               chunk_done_s=times, chunks=chunks, text="".join(chunks),
               peak_gib=float(used) / 1024 if used else float("nan"), weights_gib=float("nan"))
    print(f"--- {rec['audio']}: {dur:.1f}s audio, {gen_s:.1f}s gen, RTF {rec['rtf']:.3f}, "
          f"encoder {enc_s:.1f}s, {ntok} tokens ({ntok / gen_s:.1f} tok/s), final kv {rec['kv_len']}, "
          f"gpu used at end {rec['peak_gib']:.2f} GiB")
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
