#!/usr/bin/env python3
"""Export Chatterbox TTS to ONNX.

Stage 0 step E2: implements three of the four graphs by orchestrating
the wrapper modules vendored in `_chatterbox_internals.py`:

  * `embed_tokens.onnx`        — text token embedding + position handling
  * `speech_encoder.onnx`      — speech encoder + S3 tokenizer + cond prep
  * `conditional_decoder.onnx` — speech tokens + conditioning → waveform

The Llama language model graph lands in step E3 (separate work — we own
that export from scratch, not adapted from Vlad's reference).

Run:

    python scripts/chatterbox_export/export_chatterbox_to_onnx.py \\
        --output-dir ./models/chatterbox_export \\
        --device cuda --dtype float32

`--skip-language-model` is set by default at the CLI layer until E3 lands.
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
    nvidia_smi_query,
    read_ort_available_providers,
    resolve_device,
    resolve_dtype,
    save_export_report,
    EXAGGERATION_TOKEN,
    START_SPEECH_TOKEN,
    S3GEN_SR,
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
    p.add_argument("--opset-conditional-decoder", type=int, default=17)
    p.add_argument("--lm-graph-mode", default="unified", choices=["unified", "prefill+step"])
    p.add_argument("--skip-embed-tokens", action="store_true")
    p.add_argument("--skip-speech-encoder", action="store_true")
    p.add_argument("--skip-language-model", action="store_true", default=True,
                   help="(default: skipped until E3 lands)")
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


def load_audio_prompt(path: Path | None, target_sr: int, device, dtype):
    """Return a 1xN audio tensor at target_sr. If path is None, return
    random noise sized for the dummy export (~13 s at 24 kHz, matching
    Vlad's reference dummy size for trace shape stability).
    """
    import torch
    if path is None:
        # 312_936 samples == 13.039 s at 24 kHz, matching Vlad's dummy
        return torch.randn(1, 312_936, device=device, dtype=dtype)
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
    """Replace `s3gen.speaker_encoder.xvector.dense` (DenseLayer wrapping
    BatchNorm1d) with `SafeDenseLayer` (LayerNorm). Required for the
    speech_encoder graph to export. Vlad asserts this is inference-equivalent;
    E2 parity test must verify numerically.
    """
    old = chatterbox_model.s3gen.speaker_encoder.xvector.dense
    new = SafeDenseLayer(old.linear.in_channels, old.linear.out_channels)
    new.linear.weight.data.copy_(old.linear.weight.data)
    chatterbox_model.s3gen.speaker_encoder.xvector.dense = new
    return True


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
                          opset: int, item_patch: bool):
    import torch
    print(f"  exporting speech_encoder.onnx (opset {opset}) ...")
    t0 = time.perf_counter()
    with item_no_op_patch(item_patch):
        torch.onnx.export(
            prepare_conditionals_mod,
            (audio_values,),
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
    print(f"    done ({time.perf_counter() - t0:.1f}s)  {out_path.stat().st_size / 1e6:.2f} MB")


def export_conditional_decoder(cond_decoder_mod, speech_tokens, speaker_embeddings,
                               speaker_features, out_path: Path, opset: int, item_patch: bool):
    """Export the conditional decoder.

    On CUDA, torch.jit.trace (used internally by torch.onnx.export) fails
    inside upstream chatterbox `CausalBlock1D.block(x * mask)` with a
    spurious cuda:0/cpu device-mismatch — confirmed by repro outside ONNX
    via direct torch.jit.trace. Eager mode runs the same code path
    cleanly. Workaround: move the module + inputs to CPU just for the
    export. The resulting ONNX graph is device-independent; ORT will run
    it on CUDA at session-load time.

    `set_default_device("cpu")` is also pinned for the duration of the
    export so any intermediates the tracer materializes land on CPU.
    """
    import torch
    print(f"  exporting conditional_decoder.onnx (opset {opset}) ...")
    t0 = time.perf_counter()
    cond_decoder_mod = cond_decoder_mod.cpu()
    speech_tokens_cpu = speech_tokens.cpu()
    speaker_embeddings_cpu = speaker_embeddings.cpu()
    speaker_features_cpu = speaker_features.cpu()
    # The vendored `istft` singleton is held by reference inside cond_decoder.
    import _chatterbox_internals as ci  # noqa - already imported in main; reimport keeps the function self-contained
    ci.istft.cpu()
    prev_default = torch.get_default_device() if hasattr(torch, "get_default_device") else None
    torch.set_default_device("cpu")
    try:
        with item_no_op_patch(item_patch):
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
            print(f"    falling back to raw externalization")
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

    print("Applying SafeDenseLayer monkeypatch on speaker_encoder.xvector.dense ...")
    patched = apply_safe_dense_patch(chatterbox_model, ci.SafeDenseLayer)

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
        )
        graphs_exported.append("speech_encoder.onnx")

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
        )
        graphs_exported.append("conditional_decoder.onnx")

    if graphs_exported and not args.no_onnxslim:
        print("\nPost-export: onnxslim + external data ...")
        slim_and_externalize(args.output_dir, graphs_exported)

    if args.skip_language_model:
        print("\n[skipped] language_model.onnx (E3 work — not yet implemented)")

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
            "stage": "E2-graphs-only",
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
