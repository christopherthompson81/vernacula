#!/usr/bin/env python
"""Run the exported streaming package with ONNX Runtime, mirroring upstream's
streaming_generate loop exactly, and write a record compare_runs.py can diff against a
run_reference.py --deterministic record.

Loop (from modeling_vibevoice_asr.py:429):
  prefill prompt; per window: [speech_start] + frames + [speech_end], greedy until
  text_chunk_end/EOS (cap max_new_tokens), then feed text_chunk_end itself.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vibevoice_export"))
from test_static_kv_parity import _decoder_fp_inputs, run_decoder_session  # noqa: E402


def windows(audio, win, hop):
    starts = range(0, len(audio), hop)
    for s in starts:
        seg = audio[s:s + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        yield seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--opt-level", default="extended", choices=["disable", "basic", "extended", "all"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    md = Path(args.model_dir)
    rep = json.loads((md / "export-report.json").read_text())
    st, tk = rep["streaming"], rep["tokenizer"]
    L, KH, HD = rep["num_layers"], rep["num_kv_heads"], rep["head_dim"]
    kv_dtype = np.float32 if rep["f32_kv_cache"] else None  # None -> decided from session dtype

    so = ort.SessionOptions()
    so.graph_optimization_level = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[args.opt_level]
    providers = ["CPUExecutionProvider"] if args.cpu else ["CUDAExecutionProvider"]
    t0 = time.time()
    enc = ort.InferenceSession(str(md / "audio_encoder.onnx"), so, providers=providers)
    dec = ort.InferenceSession(str(md / "decoder_single.onnx"), so, providers=providers)
    load_s = time.time() - t0
    fp_casts = _decoder_fp_inputs(dec)
    if kv_dtype is None:
        kv_dtype = fp_casts.get("past_key_0", np.float32)
    out_names = ["logits"] + [f"present_key_{i}" for i in range(L)] + [f"present_value_{i}" for i in range(L)]

    from vibevoice.processor.audio_utils import load_audio_use_ffmpeg
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(md))
    audio, _ = load_audio_use_ffmpeg(args.audio, resample=True, target_sr=st["sample_rate"])
    audio = audio.astype(np.float32)
    dur = len(audio) / st["sample_rate"]

    past = [np.zeros((1, KH, 0, HD), dtype=kv_dtype) for _ in range(2 * L)]
    hidden = rep["hidden_size"]
    empty_audio = np.zeros((0, hidden), dtype=np.float32)
    empty_ids = np.zeros((1, 0), dtype=np.int64)

    def step(prefix, audio_emb, suffix):
        nonlocal past
        feed = {"prefix_input_ids": np.asarray(prefix, dtype=np.int64).reshape(1, -1),
                "audio_embeddings": audio_emb, "suffix_input_ids": np.asarray(suffix, dtype=np.int64).reshape(1, -1)}
        for i in range(L):
            feed[f"past_key_{i}"], feed[f"past_value_{i}"] = past[2 * i], past[2 * i + 1]
        res = run_decoder_session(dec, feed, fp_casts, out_names)
        past = [res[1 + i // 2] if i % 2 == 0 else res[1 + L + i // 2] for i in range(2 * L)]
        return int(np.argmax(res[0][0, -1]))

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
        chunks.append(text)
        times.append(time.time() - t0)
        print(f"[{len(chunks)}] {text}", flush=True)
    gen_s = time.time() - t0

    rec = dict(audio=os.path.basename(args.audio), runtime="ort-cpu" if args.cpu else "ort-cuda",
               opt_level=args.opt_level, model_dir=str(md), duration_s=dur, load_s=load_s,
               gen_s=gen_s, rtf=gen_s / dur, enc_s=enc_s, tokens=ntok, n_chunks=len(chunks),
               kv_len=int(past[0].shape[2]), chunk_done_s=times, chunks=chunks, text="".join(chunks),
               peak_gib=float("nan"), weights_gib=float("nan"))
    print(f"\n--- {rec['audio']}: {dur:.1f}s audio, {gen_s:.1f}s gen, RTF {rec['rtf']:.3f}, "
          f"encoder {enc_s:.1f}s, {ntok} tokens, final kv {rec['kv_len']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
