"""Phase-1 spike for Chatterbox internal cross-attention alignment.

Validates the hypothesis (from the investigation doc) that the LM's own
self-attention contains a clean text→speech alignment we can use in place
of NFA forced alignment.

Mechanism (copied from chatterbox.models.t3.inference.alignment_stream_analyzer
which Resemble AI uses for hallucination detection):
  - LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)] are the three
    (layer, head) pairs they empirically found carry alignment signal.
  - Set tfmr.config.output_attentions=True so HF returns attention
    weights from every layer's forward (note: forces fall-back from
    SDPA to eager attention; slower but exposes the matrix).
  - Hook the three layers' self_attn modules; capture output[1] which
    is the attention weights tensor of shape (B, H, T_q, T_kv).
  - Mean-average the three head slices to get a (T_speech, T_text)
    alignment matrix.

Output:
  - alignment.npy: the (T_speech, T_text) matrix
  - alignment.png: heatmap visualization
  - meta.json: chunk text + timing + LM step count

Usage:
  python extract_alignment.py \\
    --text "Hello world. This is a test of the alignment." \\
    --voice /path/to/voice.wav \\
    --out /tmp/spike_chunk_1

Or use --text-file to read from a file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Three (layer, head) pairs Resemble AI identified as alignment-bearing.
# Source: chatterbox/models/t3/inference/alignment_stream_analyzer.py
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="text to synthesize (single chunk)")
    src.add_argument("--text-file", help="path to text file to synthesize (whole file = one chunk)")
    p.add_argument("--voice", required=True, help="voice reference WAV")
    p.add_argument("--out", required=True, help="output directory (created if missing)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default: cuda)")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="max LM steps to generate (default: 1024 to match CLI; the constant in chatterbox is 1000)")
    args = p.parse_args()

    text = args.text if args.text else Path(args.text_file).read_text()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] ChatterboxTTS on {args.device}...")
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=args.device)

    # ── Hook the alignment-bearing layers ──────────────────────────────
    # Per-layer list of (B, H, T_q, T_kv) tensors, one per LM forward
    # pass (1 prefill + N decode steps).
    captured: dict[int, list[torch.Tensor]] = {layer: [] for layer, _ in LLAMA_ALIGNED_HEADS}

    def make_hook(layer_idx: int):
        def hook(_module, _input, output):
            # LlamaAttention.forward returns (attn_output, attn_weights, past_key_value).
            # attn_weights is None when output_attentions=False; populated otherwise.
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                captured[layer_idx].append(output[1].detach().cpu())
        return hook

    tfmr = model.t3.tfmr
    handles = []
    for layer_idx, _head in LLAMA_ALIGNED_HEADS:
        handles.append(tfmr.layers[layer_idx].self_attn.register_forward_hook(make_hook(layer_idx)))
    tfmr.config.output_attentions = True

    # ── Synthesize ────────────────────────────────────────────────────
    print(f"[synth] {len(text)} chars (first 80: {text[:80]!r})")
    t0 = time.perf_counter()
    wav = model.generate(text, audio_prompt_path=args.voice)
    synth_sec = time.perf_counter() - t0
    print(f"[synth] done in {synth_sec:.2f}s → {wav.shape[-1]} samples")

    # Detach hooks (don't leak them across runs in this process).
    for h in handles:
        h.remove()

    # ── Process captured attentions ────────────────────────────────────
    # captured[layer] is [step0_attn, step1_attn, ...] for that layer.
    # Step 0 is the prefill pass: (1, H, T_total, T_total) where T_total
    #   covers conditioning + text + initial-speech-token.
    # Step i>0 is one decode step: (1, H, 1, T_total + i) — only the new
    #   query row, attending back to all previous keys.
    # We need to:
    #   (a) Slice each head's row to the text-token columns.
    #   (b) Concat the per-step single-row outputs (step 1..N) underneath
    #       the prefill's last text row(s).
    #   (c) Mean-average the three heads to get a single alignment map.

    # Locate the text-tokens span. ChatterboxTTS.generate doesn't expose
    # the slice publicly, but we can recover it from the prefill shape:
    # the conditioning prefix has fixed length per t3.prepare_input_embeds,
    # and the text tokens follow. For a clean spike we DUMP THE FULL
    # prefill attention map (no text-only slice) and let the plotter show
    # the whole grid. If the diagonal stripe is visible against the text
    # columns we'll know the slice empirically.

    # Sanity: same number of steps per layer.
    step_counts = {layer: len(a) for layer, a in captured.items()}
    print(f"[attn ] steps per layer: {step_counts}")
    n_steps = next(iter(step_counts.values()))
    assert all(c == n_steps for c in step_counts.values()), \
        f"step count mismatch across layers: {step_counts}"

    if n_steps == 0:
        print("[ERROR] no attention captured — output_attentions flag may not have taken effect.", file=sys.stderr)
        sys.exit(1)

    # Build per-head alignment matrices then average.
    # For each layer/head: collect per-step rows. The prefill's last row
    # is the first speech token's attention; subsequent steps give one
    # row each. Stacked → (T_speech, T_kv_at_end).
    # We pad each step's row to the final width with zeros so they stack.
    final_kv_len = captured[LLAMA_ALIGNED_HEADS[0][0]][-1].shape[-1]
    print(f"[attn ] final KV length: {final_kv_len}")

    per_head_align = []
    for layer_idx, head_idx in LLAMA_ALIGNED_HEADS:
        rows = []
        for step_attn in captured[layer_idx]:
            # step_attn: (1, H, T_q, T_kv)
            head_slice = step_attn[0, head_idx]  # (T_q, T_kv)
            for q in range(head_slice.shape[0]):
                row = head_slice[q].numpy()
                # Right-pad with zeros so all rows have final_kv_len cols.
                if row.shape[0] < final_kv_len:
                    row = np.pad(row, (0, final_kv_len - row.shape[0]))
                rows.append(row)
        per_head_align.append(np.stack(rows, axis=0))  # (T_speech_total, T_kv_final)

    # Mean across heads. All three should have the same T_speech rows.
    align_shapes = [a.shape for a in per_head_align]
    assert all(s == align_shapes[0] for s in align_shapes), \
        f"per-head alignment shape mismatch: {align_shapes}"
    alignment_full = np.mean(per_head_align, axis=0)  # (T_speech, T_kv)
    print(f"[attn ] full alignment shape (speech_rows, kv_cols): {alignment_full.shape}")

    # Save raw + meta.
    np.save(out_dir / "alignment_full.npy", alignment_full)
    meta = {
        "text_preview": text[:200],
        "text_length_chars": len(text),
        "synthesis_seconds": synth_sec,
        "lm_steps_captured": int(n_steps),
        "speech_rows": int(alignment_full.shape[0]),
        "kv_cols_final": int(alignment_full.shape[1]),
        "aligned_heads": [list(p) for p in LLAMA_ALIGNED_HEADS],
        "wav_samples": int(wav.shape[-1]),
        "wav_sample_rate": int(model.sr),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[save ] {out_dir/'alignment_full.npy'}, {out_dir/'meta.json'}")

    # ── Heatmap ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn ] matplotlib not available; skipping PNG heatmap")
        return

    fig, ax = plt.subplots(figsize=(
        min(20, alignment_full.shape[1] * 0.04 + 4),
        min(20, alignment_full.shape[0] * 0.02 + 4),
    ))
    im = ax.imshow(alignment_full, aspect="auto", origin="upper", cmap="magma")
    ax.set_xlabel("KV position (conditioning + text + speech tokens)")
    ax.set_ylabel("Speech token (output row)")
    layers = [layer for layer, _ in LLAMA_ALIGNED_HEADS]
    heads = [head for _, head in LLAMA_ALIGNED_HEADS]
    ax.set_title(
        f"Chatterbox cross-attention, mean of layers {layers}\n"
        f"heads {heads} — {alignment_full.shape[0]} steps "
        f"× {alignment_full.shape[1]} kv"
    )
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    png_path = out_dir / "alignment_full.png"
    plt.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"[save ] {png_path}")

    # Also save the WAV so we can listen back and correlate.
    try:
        import torchaudio
        torchaudio.save(str(out_dir / "synth.wav"), wav, model.sr)
        print(f"[save ] {out_dir/'synth.wav'}")
    except Exception as e:
        print(f"[warn ] could not save WAV ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
