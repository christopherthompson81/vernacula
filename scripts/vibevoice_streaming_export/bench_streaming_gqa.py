#!/usr/bin/env python
"""Drive the GQA package: one KV buffer per layer, allocated once, bound as BOTH past_* and
present_* so GroupQueryAttention updates it in place. Writes the usual record so
compare_runs.py can score it against the torch reference."""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_streaming_parity import windows  # noqa: E402


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
    MAX = rep["static_kv_max_tokens"]

    so = ort.SessionOptions()
    so.graph_optimization_level = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                   "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                                   "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[args.opt_level]
    prov = [("CUDAExecutionProvider", {"device_id": 0})]
    t0 = time.time()
    enc = ort.InferenceSession(str(md / "audio_encoder.onnx"), so, providers=prov)
    dec = ort.InferenceSession(str(md / "decoder_gqa.onnx"), so, providers=prov)
    load_s = time.time() - t0

    enc_dtype = np.float16 if "float16" in enc.get_inputs()[0].type else np.float32
    from vibevoice.processor.audio_utils import load_audio_use_ffmpeg
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(md))
    audio, _ = load_audio_use_ffmpeg(args.audio, resample=True, target_sr=st["sample_rate"])
    audio = audio.astype(np.float32)
    dur = len(audio) / st["sample_rate"]

    # One allocation for the whole run. past_key_i and present_key_i are the same tensor.
    kv = [ort.OrtValue.ortvalue_from_numpy(np.zeros((1, KH, MAX, HD), np.float16), "cuda", 0)
          for _ in range(2 * L)]
    kv_bytes = 2 * L * KH * MAX * HD * 2
    empty_audio = np.zeros((0, H), np.float16)
    empty_ids = np.zeros((1, 0), np.int64)
    kv_pos = 0

    def step(prefix, audio_emb, suffix):
        nonlocal kv_pos
        n = np.asarray(prefix).size + audio_emb.shape[0] + np.asarray(suffix).size
        total = kv_pos + n
        if total > MAX:
            raise RuntimeError(f"KV buffer full: {total} > {MAX}")
        b = dec.io_binding()
        keep = [ort.OrtValue.ortvalue_from_numpy(np.asarray(prefix, np.int64).reshape(1, -1)),
                ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(audio_emb)),
                ort.OrtValue.ortvalue_from_numpy(np.asarray(suffix, np.int64).reshape(1, -1)),
                ort.OrtValue.ortvalue_from_numpy(np.array([total - 1], np.int32)),
                ort.OrtValue.ortvalue_from_numpy(np.array([total], np.int32))]
        for name, val in zip(("prefix_input_ids", "audio_embeddings", "suffix_input_ids",
                              "seqlens_k", "total_sequence_length"), keep):
            b.bind_ortvalue_input(name, val)
        # get_outputs() returns values in BINDING order, not model order, so logits is bound
        # first and read as [0]. Binding it last silently yields present_key_0 instead.
        b.bind_output("logits", "cpu")
        for i in range(L):
            b.bind_ortvalue_input(f"past_key_{i}", kv[2 * i])
            b.bind_ortvalue_input(f"past_value_{i}", kv[2 * i + 1])
            b.bind_ortvalue_output(f"present_key_{i}", kv[2 * i])      # same buffer, in place
            b.bind_ortvalue_output(f"present_value_{i}", kv[2 * i + 1])
        dec.run_with_iobinding(b)
        logits = b.get_outputs()[0].numpy()
        kv_pos = total
        return int(np.argmax(logits[0, -1]))

    t0 = time.time()
    step(tk["prompt_token_ids"], empty_audio, empty_ids)
    chunks, times, ntok, enc_s = [], [], 0, 0.0
    for w in windows(audio, st["window_samples"], st["hop_samples"]):
        te = time.time()
        # The encoder may be exported fp32 or fp16; feed it what it declares.
        emb = enc.run(None, {"input_values": w[None].astype(enc_dtype)})[0].astype(np.float16)
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
                          capture_output=True, text=True).stdout.strip().splitlines()[0]
    rec = dict(audio=os.path.basename(args.audio), runtime="ort-cuda-gqa", opt_level=args.opt_level,
               model_dir=str(md), duration_s=dur, load_s=load_s, gen_s=gen_s, rtf=gen_s / dur,
               enc_s=enc_s, tokens=ntok, n_chunks=len(chunks), kv_len=kv_pos, kv_max=MAX,
               kv_buffer_gib=kv_bytes / 2**30, chunk_done_s=times, chunks=chunks,
               text="".join(chunks), peak_gib=float(used) / 1024, weights_gib=float("nan"))
    print(f"--- {rec['audio']}: {dur:.1f}s audio, {gen_s:.1f}s gen, RTF {rec['rtf']:.3f}, "
          f"encoder {enc_s:.1f}s, {ntok} tokens ({ntok / gen_s:.1f} tok/s), kv {kv_pos}/{MAX} "
          f"(buffer {kv_bytes / 2**30:.2f} GiB), gpu {rec['peak_gib']:.2f} GiB")
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
