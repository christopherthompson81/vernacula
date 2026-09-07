#!/usr/bin/env python
"""Export a VibeVoice-ASR-Streaming checkpoint (upstream microsoft/VibeVoice layout) to the
ONNX package shape Vernacula's VibeVoice backend consumes.

Graphs
  audio_encoder.onnx   input_values [1, num_samples] (float32)
                       -> audio_embeddings [num_frames, hidden] (float32)
                       Acoustic + semantic tokenizer encoders, latent *mean* (no sampling),
                       both connectors, summed. Run once per 83,200-sample window; this is
                       upstream's default per-window cold encoding (see the investigation
                       doc, Run 3), so no streaming conv cache is exported.
  decoder_single.onnx  prefix_input_ids [1,P] + audio_embeddings [N,hidden] +
                       suffix_input_ids [1,S] + past_key/value_i [1,kv_heads,C,head_dim]
                       -> logits [1,P+N+S,vocab] + present_key/value_i [1,kv_heads,C+P+N+S,head_dim]
                       Same contract as the non-streaming decoder_single.onnx. The streaming
                       loop is: prefill prompt (prefix=prompt, N=0, S=0); per window
                       prefix=[speech_start], audio=window frames, suffix=[speech_end]; decode
                       prefix=[token] until <|text_chunk_end|>/EOS; then prefix=[text_chunk_end].

Reuses the decoder helpers from scripts/vibevoice_export (f32 KV cache patch, mask builder,
exporter) so the numerical recipe matches the non-streaming package.
"""
from __future__ import annotations

import argparse, gc, json, shutil, sys, time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "vibevoice_export"
sys.path.insert(0, str(OLD))
from _common import flatten_past_key_values, kv_input_names, kv_output_names  # noqa: E402
from export_vibevoice_asr_to_onnx import (  # noqa: E402
    build_full_attention_mask, export_onnx_graph, f32_kv_cache_context,
)
from transformers.cache_utils import DynamicCache  # noqa: E402

PROMPT = ("You are a helpful assistant that transcribes audio input into text output. "
          "Please transcribe the following audios streamingly with these keys: speaker, content")
PROMPT_HOTWORDS_HEAD = PROMPT + " and extra info: "
PROMPT_TAIL = "\n"
METADATA_FILES = ["config.json", "preprocessor_config.json", "tokenizer.json",
                  "tokenizer_config.json", "added_tokens.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json"]


class StreamingAudioEncoder(nn.Module):
    """Mean-only acoustic + semantic features for one waveform, upstream module layout."""

    def __init__(self, inner, tower_dtype, connector_dtype):
        super().__init__()
        self.acoustic = inner.model.acoustic_tokenizer
        self.semantic = inner.model.semantic_tokenizer
        self.ac_conn = inner.model.acoustic_connector
        self.se_conn = inner.model.semantic_connector
        self.tower_dtype, self.connector_dtype = tower_dtype, connector_dtype

    def forward(self, input_values):
        x = input_values.to(self.tower_dtype).unsqueeze(1)
        a = self.acoustic.encode(x).mean          # [1, N, 64]
        s = self.semantic.encode(x).mean          # [1, N, 128]
        f = self.ac_conn(a.to(self.connector_dtype)) + self.se_conn(s.to(self.connector_dtype))
        return f[0].to(torch.float32)             # [N, hidden]


class DecoderSingle(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.lm = inner.model.language_model      # transformers Qwen2Model
        self.lm_head = inner.lm_head
        self.cfg = inner.config.decoder_config

    def forward(self, prefix_input_ids, audio_embeddings, suffix_input_ids, *past_key_values):
        emb = self.lm.embed_tokens
        x = torch.cat((emb(prefix_input_ids),
                       audio_embeddings.to(emb.weight.dtype).unsqueeze(0),
                       emb(suffix_input_ids)), dim=1)
        past = past_key_values[0].shape[2]
        q = x.shape[1]
        position_ids = torch.arange(q, device=x.device).unsqueeze(0) + past
        mask = build_full_attention_mask(attention_mask=None, query_length=q, kv_length=past + q,
                                         past_length=past, dtype=x.dtype, device=x.device)
        cache = DynamicCache(
            ddp_cache_data=tuple((past_key_values[i], past_key_values[i + 1])
                                 for i in range(0, len(past_key_values), 2)),
            config=self.cfg)
        out = self.lm(inputs_embeds=x, position_ids=position_ids, attention_mask=mask,
                      past_key_values=cache, use_cache=True)
        logits = self.lm_head(out.last_hidden_state)
        return (logits, *flatten_past_key_values(out.past_key_values))


def load(model_path, dtype, device, attn):
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    proc = VibeVoiceASRProcessor.from_pretrained(model_path)
    m = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, attn_implementation=attn).to(device).eval()
    return m, proc


def tokenizer_extras(tok, pc):
    ids = lambda s: tok.encode(s, add_special_tokens=False)
    need = {"speech_start_id": tok.speech_start_id, "speech_end_id": tok.speech_end_id,
            "text_chunk_end_id": tok.text_chunk_end_id, "eos_token_id": tok.eos_token_id}
    for k, v in need.items():
        if v is None:
            raise SystemExit(f"tokenizer has no {k}; not a streaming checkpoint")
    frame = pc["speech_tok_compress_ratio"]
    return {
        **need,
        "prompt_token_ids": ids(PROMPT + PROMPT_TAIL),
        "prompt_hotwords_head_token_ids": ids(PROMPT_HOTWORDS_HEAD),
        "prompt_tail_token_ids": ids(PROMPT_TAIL),
        "strip_tokens": ["<|text_chunk_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
                         "<|box_start|>", "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>"],
        "sample_rate": pc["target_sample_rate"],
        "samples_per_frame": frame,
        "chunk_frames": pc["chunk_frames"],
        "lookahead_frames": pc["lookahead_frames"],
        "window_samples": (pc["chunk_frames"] + pc["lookahead_frames"]) * frame,
        "hop_samples": pc["chunk_frames"] * frame,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--f32-kv-cache", action="store_true", default=True)
    ap.add_argument("--bf16-kv-cache", dest="f32_kv_cache", action="store_false")
    ap.add_argument("--skip-audio-encoder", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    mp = Path(args.model_path)
    pc = json.loads((mp / "preprocessor_config.json").read_text())

    # Eager attention: the f32 KV patch replaces Qwen2Attention.forward with the eager path,
    # and the legacy exporter traces eager attention cleanly.
    t0 = time.time()
    model, proc = load(args.model_path, dtype, args.device, "eager")
    print(f"loaded in {time.time() - t0:.1f}s")
    dc = model.config.decoder_config
    num_layers, kv_heads = dc.num_hidden_layers, dc.num_key_value_heads
    head_dim = dc.hidden_size // dc.num_attention_heads
    hidden = dc.hidden_size
    extras = tokenizer_extras(proc.tokenizer, pc)
    for f in METADATA_FILES:
        if (mp / f).exists():
            shutil.copy(mp / f, out / f)

    if not args.skip_audio_encoder:
        # The acoustic decoder half is never used for ASR; drop it so it is not traced or
        # kept resident. Conv towers run in float32 (ORT has no bf16 Conv); connectors too,
        # since they are tiny and the decoder casts the result to its own dtype anyway.
        if hasattr(model.model.acoustic_tokenizer, "decoder"):
            del model.model.acoustic_tokenizer.decoder
        enc = StreamingAudioEncoder(model, torch.float32, torch.float32)
        enc.acoustic.to(torch.float32); enc.semantic.to(torch.float32)
        enc.ac_conn.to(torch.float32); enc.se_conn.to(torch.float32)
        enc = enc.to(args.device).eval()
        dummy = torch.zeros((1, extras["window_samples"]), dtype=torch.float32, device=args.device)
        with torch.no_grad():
            n_frames = enc(dummy).shape[0]
        assert n_frames == extras["chunk_frames"] + extras["lookahead_frames"], n_frames
        print(f"exporting audio_encoder.onnx ({extras['window_samples']} samples -> {n_frames} frames) ...")
        export_onnx_graph(model=enc, args=(dummy,), output_path=out / "audio_encoder.onnx",
                          input_names=["input_values"], output_names=["audio_embeddings"],
                          opset=args.opset,
                          dynamic_axes={"input_values": {1: "num_samples"},
                                        "audio_embeddings": {0: "num_frames"}},
                          exporter="legacy")
        del enc; gc.collect(); torch.cuda.empty_cache()

    if not args.skip_decoder:
        dec = DecoderSingle(model).eval()
        kv_dtype = torch.float32 if args.f32_kv_cache else dtype
        pfx = torch.tensor([extras["prompt_token_ids"][:5]], dtype=torch.long, device=args.device)
        aud = torch.zeros((4, hidden), dtype=dtype, device=args.device)
        sfx = torch.tensor([[extras["speech_end_id"]]], dtype=torch.long, device=args.device)
        kv = [torch.zeros((1, kv_heads, 0, head_dim), dtype=kv_dtype, device=args.device)
              for _ in range(2 * num_layers)]
        print(f"exporting decoder_single.onnx (kv dtype {kv_dtype}) ...")
        import contextlib
        with (f32_kv_cache_context() if args.f32_kv_cache else contextlib.nullcontext()):
            export_onnx_graph(
                model=dec, args=(pfx, aud, sfx, *kv), output_path=out / "decoder_single.onnx",
                input_names=["prefix_input_ids", "audio_embeddings", "suffix_input_ids",
                             *kv_input_names(num_layers)],
                output_names=["logits", *kv_output_names(num_layers)], opset=args.opset,
                dynamic_axes={"prefix_input_ids": {1: "prefix_len"},
                              "audio_embeddings": {0: "num_audio_tokens"},
                              "suffix_input_ids": {1: "suffix_len"},
                              **{n: {2: "cache_len"} for n in kv_input_names(num_layers)},
                              "logits": {1: "seq_len"},
                              **{n: {2: "cache_len_out"} for n in kv_output_names(num_layers)}},
                exporter="legacy")

    report = {
        "model": "vibevoice_asr_streaming", "repo_id": mp.name, "dtype": args.dtype,
        "opset": args.opset, "device": args.device, "deterministic_audio": True,
        "f32_kv_cache": bool(args.f32_kv_cache), "static_kv_cache": False,
        "audio_embeddings_dtype": "float32",
        "num_layers": num_layers, "num_kv_heads": kv_heads, "head_dim": head_dim,
        "hidden_size": hidden, "vocab_size": dc.vocab_size,
        "acoustic_tokenizer_chunk_size": extras["window_samples"],
        "streaming": {k: extras[k] for k in ("sample_rate", "samples_per_frame", "chunk_frames",
                                             "lookahead_frames", "window_samples", "hop_samples")},
        "tokenizer": {k: extras[k] for k in ("speech_start_id", "speech_end_id", "text_chunk_end_id",
                                             "eos_token_id", "prompt_token_ids",
                                             "prompt_hotwords_head_token_ids",
                                             "prompt_tail_token_ids", "strip_tokens")},
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "export-report.json").write_text(json.dumps(report, indent=1))
    print("wrote", out / "export-report.json")


if __name__ == "__main__":
    main()
