#!/usr/bin/env python
"""Run upstream VibeVoice-ASR-Streaming exactly as distributed, with the measurements the
port needs on record: per-chunk text, wall time, RTF, peak VRAM, and an optional seed so the
stochastic acoustic sampling can be compared across runs.

This is upstream's demo/vibevoice_asr_streaming_inference_from_file.py with instrumentation
around the unchanged `streaming_generate` call. Nothing about the model or its inputs differs.
"""
import argparse, json, os, time
import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", help="write a JSON record here")
    ap.add_argument("--seed", type=int, help="torch.manual_seed before generation")
    ap.add_argument("--context_info")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--encode_mode", default="split_then_encode",
                    choices=["split_then_encode", "encode_then_split"])
    ap.add_argument("--deterministic", action="store_true",
                    help="use the acoustic latent mean instead of sampling around it")
    args = ap.parse_args()

    with open(os.path.join(args.model_path, "preprocessor_config.json")) as f:
        pc = json.load(f)
    sr = pc["target_sample_rate"]
    frame = pc["speech_tok_compress_ratio"] / sr
    chunk_s, delay_s = pc["chunk_frames"] * frame, pc["lookahead_frames"] * frame

    if args.deterministic:
        # encode_speech calls VibeVoiceTokenizerEncoderOutput.sample(); make it return the
        # mean. This is the only change from upstream behaviour and it is opt-in.
        from vibevoice.modular import modular_vibevoice_tokenizer as mt
        mt.VibeVoiceTokenizerEncoderOutput.sample = lambda self, dist_type="fix": (self.mean, self.std)

    processor = VibeVoiceASRProcessor.from_pretrained(args.model_path)
    assert processor.tokenizer.text_chunk_end_id is not None
    t0 = time.time()
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation=args.attn).to("cuda").eval()
    load_s = time.time() - t0
    torch.cuda.synchronize()
    weights_gib = torch.cuda.memory_allocated() / 2**30

    audio, _ = load_audio_use_ffmpeg(args.audio, resample=True, target_sr=sr)
    dur = len(audio) / sr
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    chunks, times = [], []
    t0 = time.time()
    for idx, total, text in model.streaming_generate(
            audio_tensor=torch.from_numpy(audio), tokenizer=processor.tokenizer,
            chunk_duration=chunk_s, text_audio_delay=delay_s, sample_rate=sr,
            max_new_tokens_per_chunk=args.max_new_tokens, temperature=0.0,
            context_info=args.context_info, encode_mode=args.encode_mode):
        times.append(time.time() - t0)
        chunks.append(text)
        print(f"[{idx + 1}/{total}] {text}", flush=True)
    torch.cuda.synchronize()
    gen_s = time.time() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    rec = dict(audio=os.path.basename(args.audio), seed=args.seed, attn=args.attn,
               deterministic=args.deterministic,
               encode_mode=args.encode_mode, context_info=args.context_info,
               chunk_s=chunk_s, lookahead_s=delay_s, duration_s=dur, load_s=load_s,
               weights_gib=weights_gib, peak_gib=peak_gib, gen_s=gen_s, rtf=gen_s / dur,
               n_chunks=len(chunks), chunk_done_s=times, chunks=chunks, text="".join(chunks))
    print(f"\n--- {rec['audio']}: {dur:.1f}s audio, {gen_s:.1f}s gen, RTF {rec['rtf']:.3f}, "
          f"load {load_s:.1f}s, weights {weights_gib:.2f} GiB, peak {peak_gib:.2f} GiB")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
