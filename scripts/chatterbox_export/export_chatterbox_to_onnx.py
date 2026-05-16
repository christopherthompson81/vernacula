#!/usr/bin/env python3
"""Export Chatterbox TTS to ONNX.

Stage 0 implements all four graphs:

  * `embed_tokens.onnx`        — text token embedding + position handling
  * `speech_encoder.onnx`      — speech encoder + S3 tokenizer + cond prep
  * `language_model.onnx`      — Llama backbone + speech_head, KV-cache I/O
  * `conditional_decoder.onnx` — speech tokens + conditioning → waveform

Graphs 1, 2, 4 were adapted from VladOS95-cyber's MIT-licensed reference
and vendored in `_chatterbox_internals.py`. Graph 3 (the LM) is fully
ours — Vlad's script never exported it.

Run:

    python scripts/chatterbox_export/export_chatterbox_to_onnx.py \\
        --output-dir ./models/chatterbox_export \\
        --device cuda --dtype float32
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from _common import (
    add_local_script_path,
    ensure_output_dir,
    kv_input_names,
    kv_output_names,
    nvidia_smi_query,
    read_ort_available_providers,
    resolve_device,
    resolve_dtype,
    save_export_report,
    EXAGGERATION_TOKEN,
    START_SPEECH_TOKEN,
    S3GEN_SR,
    LLM_HIDDEN_SIZE,
    LLM_NUM_LAYERS,
    LLM_NUM_KV_HEADS,
    LLM_HEAD_DIM,
)

add_local_script_path()


DEFAULT_REPO_ID = "ResembleAI/chatterbox"
EXPORT_FILES = [
    "embed_tokens.onnx",
    "speech_encoder.onnx",
    "language_model.onnx",
    "conditional_decoder.onnx",
    "export-report.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--revision", default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--opset-embed-tokens", type=int, default=20)
    p.add_argument("--opset-speech-encoder", type=int, default=20)
    p.add_argument("--opset-language-model", type=int, default=18)
    p.add_argument("--opset-conditional-decoder", type=int, default=18,
                   help="Cond decoder needs opset 18+ for the Col2Im op "
                        "(F.fold-based window_sumsquare in our ISTFT). "
                        "Lower opset triggers a scatter_add path that has "
                        "ONNX duplicate-index correctness issues.")
    p.add_argument("--lm-graph-mode", default="unified", choices=["unified", "prefill+step"])
    p.add_argument("--skip-embed-tokens", action="store_true")
    p.add_argument("--skip-speech-encoder", action="store_true")
    p.add_argument("--skip-language-model", action="store_true",
                   help="Skip the LM graph (it's the heaviest — ~2 GB fp32)")
    p.add_argument("--skip-conditional-decoder", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--audio-prompt", type=Path, default=None,
                   help="Reference audio at any sample rate. If omitted, a 13 s torch.randn dummy is used "
                        "(fine for smoke tests, not for parity).")
    p.add_argument("--no-onnxslim", action="store_true",
                   help="Skip the onnxslim + external-data pass after export")
    p.add_argument("--with-item-patch", action="store_true",
                   help="Apply Vlad's `torch.Tensor.item = lambda x: x` monkeypatch around "
                        "torch.onnx.export. Off by default; turn on if tracing fails on .item() calls.")
    return p.parse_args()


@contextmanager
def item_no_op_patch(enabled: bool):
    """Context manager for Vlad's `.item()` no-op trick. Scoped, not global."""
    import torch
    if not enabled:
        yield
        return
    original = torch.Tensor.item
    torch.Tensor.item = lambda self: self  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.Tensor.item = original  # type: ignore[method-assign]


def stage_environment() -> dict:
    """Print + collect environment info for export-report.json."""
    import torch
    import onnxruntime as ort
    env = {
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "onnxruntime": ort.__version__,
        "ort_providers": read_ort_available_providers(),
    }
    print(f"torch: {env['torch']}  cuda_available={env['torch_cuda_available']}")
    print(f"onnxruntime: {env['onnxruntime']}")
    print(f"  providers available: {env['ort_providers']}")
    smi = nvidia_smi_query()
    if smi:
        env["nvidia_smi"] = smi
        print(f"GPU: {smi['name']} (driver {smi['driver_version']}, "
              f"{smi['memory_used_mib']}/{smi['memory_total_mib']} MiB)")
    return env


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


DUMMY_AUDIO_SAMPLES = 312_936  # 13.039 s at 24 kHz, matches Vlad's dummy
DUMMY_AUDIO_SEED = 0


def load_audio_prompt(path: Path | None, target_sr: int, device, dtype):
    """Return a 1xN audio tensor at target_sr.

    If `path` is None, generate a *deterministic* random-noise prompt
    by seeding a fresh Generator. Re-running the same command without
    --audio-prompt produces byte-identical audio bytes → byte-identical
    speech tokens → byte-identical conditional_decoder.onnx hashes.
    Required for the parity test in E4 and for the artifact_hashes in
    export-report.json to be reproducible across runs.
    """
    import torch
    if path is None:
        gen = torch.Generator(device=device).manual_seed(DUMMY_AUDIO_SEED)
        return torch.randn(1, DUMMY_AUDIO_SAMPLES, device=device, dtype=dtype, generator=gen)
    import librosa
    waveform, _sr = librosa.load(str(path), sr=target_sr, mono=True)
    return torch.from_numpy(waveform).unsqueeze(0).to(device=device, dtype=dtype)


def build_reference_input_ids(device):
    """Vlad's hardcoded 79-token reference prompt with the trailing
    `START_SPEECH_TOKEN, START_SPEECH_TOKEN` pair retained per his
    compatibility note ('most likely by accident, but we keep it').
    """
    import torch
    return torch.tensor([[
        EXAGGERATION_TOKEN, 255, 281,  39,  46,  56,   2,  53,   2, 286,  41,  37,   2, 136, 122,
        49,   2, 152,   2, 103,   2, 277,  21, 101,   7,   2, 301,  55,  34,
        28,   7,   2,  53,   2, 296,  18,  18, 115,   2,  51,   2,  33, 245,
        2,  17, 190,   2,  42,   2,  50,  18, 125,   4,  32,   2, 290, 169,
        142,   2,  41,   2,  43,   2,  18,  29,  91,   2,  25, 186,   8,  20,
        14,  80,   2,  29,  86, 213, 216,   9,   0, START_SPEECH_TOKEN, START_SPEECH_TOKEN
    ]], dtype=torch.long, device=device)


def apply_safe_dense_patch(chatterbox_model, SafeDenseLayer):
    """REMOVED — DO NOT USE.

    Vlad's `SafeDenseLayer` (BatchNorm1d→LayerNorm substitution) was
    introduced to work around an apparent ONNX export issue with the
    upstream `DenseLayer`. Direct testing (probe_dense.py) showed that
    the upstream layer ONNX-exports cleanly on CPU; the original
    failure was the same CUDA-side `torch.jit.trace` bug we hit on
    the cond decoder, not a symbolic-conversion problem.

    The substitution drops BatchNorm1d's running mean/var (which
    encode learned activation statistics) and replaces with a
    randomly-initialized LayerNorm. Parity test E4 confirmed this
    drifts speaker_embeddings by 93% of dynamic range (cosine sim
    0.81 instead of 1.0), silently degrading voice-clone quality.

    Function preserved as a stub so callers fail loudly if it's ever
    re-introduced. The export script now skips this call entirely.
    """
    raise RuntimeError(
        "SafeDenseLayer substitution was removed after parity test "
        "showed it drifts speaker_embeddings by ~93%. Use upstream "
        "DenseLayer directly (export speech_encoder on CPU)."
    )


def export_embed_tokens(embed_tokens_mod, input_ids, position_ids, exaggeration,
                        out_path: Path, opset: int, item_patch: bool):
    import torch
    print(f"  exporting embed_tokens.onnx (opset {opset}) ...")
    t0 = time.perf_counter()
    with item_no_op_patch(item_patch):
        torch.onnx.export(
            embed_tokens_mod,
            (input_ids, position_ids, exaggeration),
            str(out_path),
            export_params=True,
            opset_version=opset,
            input_names=["input_ids", "position_ids", "exaggeration"],
            output_names=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "position_ids": {0: "batch_size", 1: "sequence_length"},
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
                "exaggeration": {0: "batch_size"},
            },
        )
    print(f"    done ({time.perf_counter() - t0:.1f}s)  {out_path.stat().st_size / 1e6:.2f} MB")


def export_speech_encoder(prepare_conditionals_mod, audio_values, out_path: Path,
                          opset: int, item_patch: bool, chatterbox_model):
    """Export the speech encoder on CPU.

    On CUDA, `torch.jit.trace` hits the same kind of spurious
    cuda:0/cpu device mismatch as the cond decoder (probe via
    `probe_dense.py`). Eager mode works on CUDA. Workaround: move the
    module + inputs to CPU just for the export. ONNX graph is
    device-independent; ORT can still run it on CUDA EP at session load.

    Also applies a scoped patch to `DenseLayer.forward` to specialize
    away the `if len(x.shape) == 2` branch, which makes ONNX BatchNorm
    fail with "unknown channel size" (the if-node hides the channel
    dim from the symbolic checker). The patched forward assumes 2D
    input; only the 2D branch is ever exercised by the pipeline
    (verified via probe_dense_shape.py).
    """
    import torch
    from _export_patches import (
        patched_dense_layer_for_export,
        patched_s3tokenizer_for_export,
        patched_rotary_for_export,
    )
    print(f"  exporting speech_encoder.onnx (opset {opset}) ...")
    t0 = time.perf_counter()
    orig_device = next(prepare_conditionals_mod.parameters()).device
    prepare_conditionals_mod.cpu()
    audio_values_cpu = audio_values.cpu()
    prev_default = torch.get_default_device() if hasattr(torch, "get_default_device") else None
    torch.set_default_device("cpu")
    s3_tokenizer = chatterbox_model.s3gen.tokenizer
    try:
        with item_no_op_patch(item_patch), \
             patched_dense_layer_for_export(chatterbox_model.s3gen.speaker_encoder), \
             patched_s3tokenizer_for_export(s3_tokenizer), \
             patched_rotary_for_export(s3_tokenizer):
            torch.onnx.export(
                prepare_conditionals_mod,
                (audio_values_cpu,),
                str(out_path),
                export_params=True,
                opset_version=opset,
                input_names=["audio_values"],
                output_names=["audio_features", "audio_tokens", "speaker_embeddings", "speaker_features"],
                dynamic_axes={
                    "audio_values": {0: "batch_size", 1: "num_samples"},
                    "audio_features": {0: "batch_size", 1: "sequence_length"},
                    "audio_tokens": {0: "batch_size", 1: "audio_sequence_length"},
                    "speaker_embeddings": {0: "batch_size"},
                    "speaker_features": {0: "batch_size", 1: "feature_dim"},
                },
            )
    finally:
        if prev_default is not None:
            torch.set_default_device(prev_default)
        prepare_conditionals_mod.to(orig_device)
    print(f"    done ({time.perf_counter() - t0:.1f}s)  {out_path.stat().st_size / 1e6:.2f} MB")


def export_conditional_decoder(cond_decoder_mod, speech_tokens, speaker_embeddings,
                               speaker_features, out_path: Path, opset: int, item_patch: bool,
                               chatterbox_model):
    """Export the conditional decoder.

    The new ConditionalDecoder is a thin wrapper that delegates to
    upstream `chatterbox.s3gen.flow.inference()` + `mel2wav.inference()`.
    We apply `patched_cond_decoder_for_export` to strip
    `@torch.inference_mode()` from those upstream methods (poisoned
    tensors break the JIT trace) and to swap `mel2wav._stft` /
    `_istft` for ONNX-friendly real-format implementations. The iSTFT
    is rerouted through our `ISTFT` class.

    Still runs on CPU — same JIT-trace-on-CUDA bug as before
    (CausalBlock1D `x * mask` device mismatch, confirmed outside ONNX).
    Resulting ONNX is device-independent.
    """
    import torch
    from _export_patches import patched_cond_decoder_for_export
    print(f"  exporting conditional_decoder.onnx (opset {opset}) ...")
    t0 = time.perf_counter()
    cond_decoder_mod = cond_decoder_mod.cpu()
    speech_tokens_cpu = speech_tokens.cpu()
    speaker_embeddings_cpu = speaker_embeddings.cpu()
    speaker_features_cpu = speaker_features.cpu()
    import _chatterbox_internals as ci
    ci.istft.cpu()
    # Move upstream s3gen submodules to cpu too — patched _istft
    # references our istft singleton, but the upstream
    # mel2wav/flow modules need to be on cpu for the trace.
    chatterbox_model.s3gen.cpu()
    prev_default = torch.get_default_device() if hasattr(torch, "get_default_device") else None
    torch.set_default_device("cpu")
    try:
        with item_no_op_patch(item_patch), \
             patched_cond_decoder_for_export(chatterbox_model.s3gen, ci.istft):
            torch.onnx.export(
                cond_decoder_mod,
                (speech_tokens_cpu, speaker_embeddings_cpu, speaker_features_cpu),
                str(out_path),
                export_params=True,
                opset_version=opset,
                input_names=["speech_tokens", "speaker_embeddings", "speaker_features"],
                output_names=["waveform"],
                dynamic_axes={
                    "speech_tokens": {0: "batch_size", 1: "num_speech_tokens"},
                    "speaker_embeddings": {0: "batch_size"},
                    "speaker_features": {0: "batch_size", 1: "feature_dim"},
                    "waveform": {0: "batch_size", 1: "num_samples"},
                },
            )
    finally:
        if prev_default is not None:
            torch.set_default_device(prev_default)
    print(f"    done ({time.perf_counter() - t0:.1f}s)  {out_path.stat().st_size / 1e6:.2f} MB")


def build_lm_wrapper(chatterbox_model, device):
    """Wrap chatterbox.t3.tfmr + .speech_head into a single nn.Module
    that exposes the HF KV-cache I/O schema.

    Inputs (one positional arg per ONNX input):
      - inputs_embeds: (B, S, LLM_HIDDEN_SIZE=1024)
      - attention_mask: (B, S_total) where S_total = past_kv_len + S
      - past_key_values.{N}.key, past_key_values.{N}.value  for N in 0..29
            each: (B, LLM_NUM_KV_HEADS=16, past_kv_len, LLM_HEAD_DIM=64)

    Outputs:
      - logits: (B, S, SPEECH_HEAD_OUTPUT_DIM=8194)
      - present.{N}.key, present.{N}.value  for N in 0..29
            each: (B, 16, past_kv_len + S, 64)

    Schema matches the published `onnx-community/chatterbox-ONNX`
    `language_model.onnx` so consumers (and our own C# orchestrator)
    can swap our export for theirs.
    """
    import torch.nn as nn

    class LMWithSpeechHead(nn.Module):
        def __init__(self, tfmr, speech_head):
            super().__init__()
            self.tfmr = tfmr
            self.speech_head = speech_head

        def forward(self, inputs_embeds, attention_mask, *past_kv_flat):
            # Reshape flat (key0, value0, key1, value1, ...) into HF's
            # legacy tuple-of-tuples format. transformers 4.46.3 accepts
            # both this and the new Cache class; we use the legacy form
            # to match Vlad's flow and the published bundle. Transformers
            # 4.47+ removes legacy; that's a follow-up.
            past_kv = tuple(
                (past_kv_flat[2 * i], past_kv_flat[2 * i + 1])
                for i in range(LLM_NUM_LAYERS)
            )
            out = self.tfmr(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = self.speech_head(out.last_hidden_state)
            # Flatten present_key_values to a positional output tuple.
            present_flat = []
            for layer_kv in out.past_key_values:
                present_flat.append(layer_kv[0])
                present_flat.append(layer_kv[1])
            return (logits, *present_flat)

    return LMWithSpeechHead(chatterbox_model.t3.tfmr, chatterbox_model.t3.speech_head).eval().to(device)


def export_language_model(lm_mod, out_path: Path, opset: int, device, dtype, item_patch: bool):
    """Export the Llama-with-speech-head graph with KV-cache I/O.

    Uses small dummy shapes for the trace (batch=1, seq=4, past=0). The
    dynamic_axes spec lets ORT consume arbitrary batch / sequence /
    past_length at runtime; the published HF bundle uses the same shape
    pattern.
    """
    import torch
    print(f"  exporting language_model.onnx (opset {opset}) ...")
    t0 = time.perf_counter()

    # Dummy inputs. seq=4 / past=0 is the "prefill" config; the graph
    # supports growing past_kv_len at runtime via the dynamic axis.
    B, S, past_len = 1, 4, 0
    inputs_embeds = torch.randn(B, S, LLM_HIDDEN_SIZE, device=device, dtype=dtype)
    attention_mask = torch.ones(B, past_len + S, dtype=torch.int64, device=device)
    past_kv_flat = []
    for _ in range(LLM_NUM_LAYERS):
        past_kv_flat.append(torch.zeros(B, LLM_NUM_KV_HEADS, past_len, LLM_HEAD_DIM, device=device, dtype=dtype))
        past_kv_flat.append(torch.zeros(B, LLM_NUM_KV_HEADS, past_len, LLM_HEAD_DIM, device=device, dtype=dtype))

    input_names = ["inputs_embeds", "attention_mask"] + kv_input_names(LLM_NUM_LAYERS)
    output_names = ["logits"] + kv_output_names(LLM_NUM_LAYERS)

    dynamic_axes = {
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    for name in kv_input_names(LLM_NUM_LAYERS):
        dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
    for name in kv_output_names(LLM_NUM_LAYERS):
        dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}

    with item_no_op_patch(item_patch):
        torch.onnx.export(
            lm_mod,
            (inputs_embeds, attention_mask, *past_kv_flat),
            str(out_path),
            export_params=True,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    print(f"    done ({time.perf_counter() - t0:.1f}s)  {out_path.stat().st_size / 1e6:.2f} MB")


def slim_and_externalize(output_dir: Path, filenames: list[str]) -> None:
    """Post-export onnxslim pass + external data save.

    Matches Vlad's post-processing block (`ONNXSLIM_THRESHOLD=1e10` to
    disable size-threshold pruning, then save all tensors into a single
    sidecar). If onnxslim's shape inference raises — typically on large
    graphs whose proto-serialized size exceeds protobuf's 2 GB limit —
    we fall back to externalizing the raw export. The slim is an
    optimization, not load-bearing; correctness comes from torch.onnx.export.
    """
    import onnx
    import onnxslim
    os.environ["ONNXSLIM_THRESHOLD"] = "10000000000"
    for fn in filenames:
        path = output_dir / fn
        if not path.exists():
            continue
        slimmed = None
        try:
            print(f"  slimming {fn} ...")
            slimmed = onnxslim.slim(str(path))
        except Exception as e:
            print(f"    slim failed for {fn}: {type(e).__name__}: {e}")
            print("    falling back to raw externalization")
        if slimmed is None:
            # Re-load the raw export (torch.onnx.export wrote it inline)
            # and rewrite with external data.
            slimmed = onnx.load(str(path))
        onnx.save_model(
            slimmed, str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{fn}_data",
        )


def main() -> None:
    args = parse_args()

    print("Chatterbox ONNX export — Stage 0 / E2")
    print(f"  repo_id: {args.repo_id}")
    print(f"  revision: {args.revision or '(latest)'}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  device: {args.device}  dtype: {args.dtype}")
    print()

    env = stage_environment()
    print()

    ensure_output_dir(args.output_dir, args.overwrite, EXPORT_FILES)

    # Lazy imports so --help works without paying the chatterbox/torch load cost
    import torch
    from chatterbox.tts import ChatterboxTTS
    import _chatterbox_internals as ci

    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)
    if dtype is not torch.float32:
        # Vlad's reference exports fp32. Float16/bfloat16 paths are an E2 stretch
        # — leave fp32 as the smoke-test default; the 3090 ship target casts the
        # exported fp32 graph at session load time via ORT.
        print(f"  WARN: dtype={args.dtype} not yet validated; export proceeds but may fail.")

    print(f"Loading ChatterboxTTS from {args.repo_id} (revision={args.revision or 'latest'}) ...")
    t0 = time.perf_counter()
    chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    param_count = (
        sum(p.numel() for p in chatterbox_model.s3gen.parameters())
        + sum(p.numel() for p in chatterbox_model.t3.parameters())
    )
    print(f"  s3gen + t3 parameter count: {param_count:,}")

    # SafeDenseLayer monkey-patch removed — see apply_safe_dense_patch
    # docstring. Upstream DenseLayer with BatchNorm1d ONNX-exports
    # cleanly on CPU. The substitution silently degraded voice cloning.
    # tracked in export-report.json so old reports stay comparable;
    # future cleanup can drop the field.
    patched = False

    print("Building export wrappers ...")
    prepare_conditionals = ci.PrepareConditionalsModel(chatterbox_model).eval().to(device)
    embed_tokens = ci.InputsEmbeds(chatterbox_model).eval().to(device)
    cond_decoder = ci.ConditionalDecoder(chatterbox_model).eval().to(device)
    # The vendored `istft` is a module-level singleton; its hann_window
    # buffer must also follow the export device or torch.stft crashes
    # during trace.
    ci.istft.to(device)
    print("  PrepareConditionalsModel, InputsEmbeds, ConditionalDecoder built")

    # Build canonical inputs
    audio_values = load_audio_prompt(args.audio_prompt, target_sr=S3GEN_SR,
                                     device=device, dtype=torch.float32)
    prompt_label = "random noise" if args.audio_prompt is None else args.audio_prompt
    print(f"  audio_values: shape={tuple(audio_values.shape)} (prompt={prompt_label})")

    input_ids = build_reference_input_ids(device)
    position_ids = torch.where(
        input_ids >= START_SPEECH_TOKEN,
        torch.zeros_like(input_ids),
        torch.arange(input_ids.shape[1], device=device).unsqueeze(0) - 1,
    )
    exaggeration = torch.tensor([0.5], device=device)
    print(f"  input_ids: shape={tuple(input_ids.shape)}  position_ids: {tuple(position_ids.shape)}")

    graphs_exported: list[str] = []

    if not args.skip_embed_tokens:
        export_embed_tokens(
            embed_tokens, input_ids, position_ids, exaggeration,
            args.output_dir / "embed_tokens.onnx",
            opset=args.opset_embed_tokens, item_patch=args.with_item_patch,
        )
        graphs_exported.append("embed_tokens.onnx")

    if not args.skip_speech_encoder:
        export_speech_encoder(
            prepare_conditionals, audio_values,
            args.output_dir / "speech_encoder.onnx",
            opset=args.opset_speech_encoder, item_patch=args.with_item_patch,
            chatterbox_model=chatterbox_model,
        )
        graphs_exported.append("speech_encoder.onnx")

    if not args.skip_language_model:
        # LM export uses tiny dummy inputs (B=1, S=4, past=0); shape
        # generality comes from dynamic_axes. Runs on the same device as
        # the rest of the chatterbox model — t3.tfmr stays on cuda
        # because cond_decoder.cpu() in the next step only touches s3gen.
        lm_mod = build_lm_wrapper(chatterbox_model, device)
        export_language_model(
            lm_mod, args.output_dir / "language_model.onnx",
            opset=args.opset_language_model, device=device, dtype=torch.float32,
            item_patch=args.with_item_patch,
        )
        graphs_exported.append("language_model.onnx")
        del lm_mod  # release the wrapper; tfmr/speech_head are still owned by chatterbox_model

    if not args.skip_conditional_decoder:
        # Cond decoder needs real speech tokens + speaker conditioning. Run
        # PrepareConditionalsModel + InputsEmbeds + (eager Llama LM) to get
        # them, then feed to the cond decoder export. Mirrors Vlad's flow.
        print("Running PyTorch reference pipeline to build cond decoder inputs ...")
        with torch.no_grad():
            cond_emb, prompt_token, speaker_embeddings, speaker_features = \
                prepare_conditionals(audio_values=audio_values)
            text_emb = embed_tokens(input_ids=input_ids, position_ids=position_ids, exaggeration=exaggeration)
            inputs_embeds = torch.cat((cond_emb, text_emb), dim=1)

            # chatterbox.t3 uses a split LlamaModel + speech_head layout:
            # `tfmr` is the bare backbone (returns BaseModelOutputWithPast
            # with `last_hidden_state` only), and `speech_head` projects the
            # last hidden state to the speech vocab. Vlad sidestepped this
            # by loading his own `vladislavbro/llama_backbone_0.5`
            # LlamaForCausalLM mirror; we keep provenance at ResembleAI by
            # composing the backbone + head ourselves.
            llm = chatterbox_model.t3.tfmr
            speech_head = chatterbox_model.t3.speech_head
            llm.eval()
            speech_head.eval()
            from transformers import RepetitionPenaltyLogitsProcessor
            rep_proc = RepetitionPenaltyLogitsProcessor(penalty=1.2)
            generate_tokens = torch.tensor([[START_SPEECH_TOKEN]], dtype=torch.long, device=device)
            past_key_values = None
            max_new_tokens = 256
            from tqdm import tqdm
            for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
                out = llm(inputs_embeds=inputs_embeds, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                next_logits = speech_head(out.last_hidden_state[:, -1, :])
                next_logits = rep_proc(generate_tokens, next_logits)
                next_token = torch.argmax(next_logits, dim=-1).unsqueeze(-1)
                generate_tokens = torch.cat((generate_tokens, next_token), dim=-1)
                if (next_token.view(-1) == ci.STOP_SPEECH_TOKEN).all():
                    break
                pos = torch.full((input_ids.shape[0], 1), i + 1, dtype=torch.long, device=device)
                inputs_embeds = embed_tokens(next_token, pos, exaggeration)
            speech_tokens = torch.cat([prompt_token, generate_tokens[:, 1:-1]], dim=1)
            print(f"  speech_tokens: shape={tuple(speech_tokens.shape)}")

        export_conditional_decoder(
            cond_decoder, speech_tokens, speaker_embeddings, speaker_features,
            args.output_dir / "conditional_decoder.onnx",
            opset=args.opset_conditional_decoder, item_patch=args.with_item_patch,
            chatterbox_model=chatterbox_model,
        )
        graphs_exported.append("conditional_decoder.onnx")

    if graphs_exported and not args.no_onnxslim:
        print("\nPost-export: onnxslim + external data ...")
        slim_and_externalize(args.output_dir, graphs_exported)

    # Provenance: hash everything we emitted
    hashes = {}
    for fn in graphs_exported:
        p = args.output_dir / fn
        if p.exists():
            hashes[fn] = {"sha256": hash_file(p), "size_bytes": p.stat().st_size}
        data_path = args.output_dir / f"{fn}_data"
        if data_path.exists():
            hashes[f"{fn}_data"] = {"sha256": hash_file(data_path), "size_bytes": data_path.stat().st_size}

    report_path = args.output_dir / "export-report.json"
    save_export_report(
        report_path,
        repo_id=args.repo_id,
        revision=args.revision,
        device=device,
        dtype=args.dtype,
        opset_embed_tokens=args.opset_embed_tokens,
        opset_speech_encoder=args.opset_speech_encoder,
        opset_language_model=args.opset_language_model,
        opset_conditional_decoder=args.opset_conditional_decoder,
        lm_graph_mode=args.lm_graph_mode,
        safe_dense_layer_patched=patched,
        extra={
            "stage": "E3-all-four-graphs",
            "graphs_exported": graphs_exported,
            "artifact_hashes": hashes,
            "environment": env,
            "audio_prompt": str(args.audio_prompt) if args.audio_prompt else None,
            "audio_prompt_samples": int(audio_values.shape[1]),
            "input_ids_length": int(input_ids.shape[1]),
            "max_lm_steps_used": 256,  # for the cond-decoder input gen
        },
    )
    print(f"\nWrote {report_path}")
    print(f"Graphs emitted: {graphs_exported}")
    if not graphs_exported:
        print("(no graphs requested — passed --skip-* for all four)")


if __name__ == "__main__":
    sys.exit(main())
