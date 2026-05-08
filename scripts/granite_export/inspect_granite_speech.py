#!/usr/bin/env python3
"""
Inspection probe for ibm-granite/granite-speech-4.1-2b (issue #28, Run 2).

Loads the public checkpoint, walks the module tree, and runs a forward
pass on dummy audio. Reports:

  1. Top-level submodule layout (audio tower, projector, language model).
  2. Encoder forward signature and output(s) - confirms whether the
     "dual-head CTC" described in the model card is exposed at inference
     or only used as a training loss.
  3. Projector forward signature - confirms BLIP-2 Q-Former invocation
     pattern and the windowed-vs-monolithic question.
  4. The language model decoder's prefill / step contract - which
     `past_key_values` layout it returns and whether `inputs_embeds` is
     accepted (needed for splicing audio embeddings into the prompt).
  5. The full GraniteSpeechForConditionalGeneration.generate path - how
     audio embeds are fused with text token embeddings.

This is a read-only probe; it writes nothing. Output goes to stdout for
inclusion in docs/dev/granite_speech_investigation.md Run 2.

Usage
-----
    source .venv-granite-export/bin/activate
    python public/scripts/granite_export/inspect_granite_speech.py \\
        --model-repo ibm-granite/granite-speech-4.1-2b
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-repo",
        default="ibm-granite/granite-speech-4.1-2b",
    )
    parser.add_argument(
        "--revision",
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Where to run the forward probe (CPU is fine for shape inspection).",
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=2.0,
        help="Length of dummy audio for the forward probe.",
    )
    return parser.parse_args()


def section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def describe_module(name: str, module) -> None:
    cls = type(module).__name__
    n_params = sum(p.numel() for p in module.parameters())
    print(f"  {name:<32s} {cls:<48s} params={n_params:,}")


def main() -> int:
    args = parse_args()

    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor
    except ImportError as e:
        print(f"Missing dependency: {e}. Install requirements first.", file=sys.stderr)
        return 2

    section("Loading config")
    config = AutoConfig.from_pretrained(args.model_repo, revision=args.revision)
    print(f"  architectures: {config.architectures}")
    print(f"  model_type:    {config.model_type}")
    print(f"  audio_token:   {config.audio_token_index}")

    section("Loading processor")
    processor = AutoProcessor.from_pretrained(args.model_repo, revision=args.revision)
    print(f"  processor: {type(processor).__name__}")
    fe = (
        getattr(processor, "audio_processor", None)
        or getattr(processor, "feature_extractor", None)
    )
    print(f"  feature/audio extractor: {type(fe).__name__}")
    print(f"  tokenizer: {type(processor.tokenizer).__name__}")
    for attr in (
        "sampling_rate",
        "feature_size",
        "num_mel_bins",
        "n_fft",
        "hop_length",
        "win_length",
        "padding_value",
        "return_attention_mask",
    ):
        if hasattr(fe, attr):
            print(f"    fe.{attr} = {getattr(fe, attr)!r}")

    section("Loading model")
    print(f"  loading {args.model_repo} on {args.device}...")
    # AutoModel resolves to GraniteSpeechForConditionalGeneration via the
    # config's `model_type: granite_speech`. AutoModelForSpeechSeq2Seq is
    # for encoder-decoder seq2seq models like Whisper; this model is a
    # decoder-only LLM with audio input projection, not a seq2seq.
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_repo,
            revision=args.revision,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(args.device)
    except Exception:
        # Fall back to the speech-specific class directly.
        from transformers import GraniteSpeechForConditionalGeneration
        model = GraniteSpeechForConditionalGeneration.from_pretrained(
            args.model_repo,
            revision=args.revision,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(args.device)
    model.eval()
    print(f"  top class: {type(model).__name__}")

    section("Top-level submodules")
    for name, child in model.named_children():
        describe_module(name, child)

    inner = getattr(model, "model", model)
    if inner is not model:
        section("model.model submodules")
        for name, child in inner.named_children():
            describe_module(name, child)

    # --- Encoder ---------------------------------------------------------
    section("Encoder probe")
    encoder = None
    for attr in ("encoder", "audio_encoder", "audio_tower", "speech_encoder"):
        if hasattr(inner, attr):
            encoder = getattr(inner, attr)
            print(f"  found encoder at model.{attr}: {type(encoder).__name__}")
            break
    if encoder is not None:
        try:
            sig = inspect.signature(encoder.forward)
            print(f"  encoder.forward{sig}")
        except (TypeError, ValueError):
            print("  (could not introspect encoder.forward signature)")

    # --- Projector -------------------------------------------------------
    section("Projector probe")
    projector = None
    for attr in ("projector", "encoder_projector", "audio_projector", "speech_projector"):
        if hasattr(inner, attr):
            projector = getattr(inner, attr)
            print(f"  found projector at model.{attr}: {type(projector).__name__}")
            break
    if projector is not None:
        try:
            sig = inspect.signature(projector.forward)
            print(f"  projector.forward{sig}")
        except (TypeError, ValueError):
            print("  (could not introspect projector.forward signature)")
        for child_name, _ in projector.named_children():
            print(f"    projector.{child_name}: {type(getattr(projector, child_name)).__name__}")

    # --- Language model -------------------------------------------------
    section("Language model probe")
    lm = None
    for attr in ("language_model", "text_model", "lm", "decoder"):
        if hasattr(model, attr):
            lm = getattr(model, attr)
            print(f"  found lm at model.{attr}: {type(lm).__name__}")
            break
        if hasattr(inner, attr):
            lm = getattr(inner, attr)
            print(f"  found lm at model.model.{attr}: {type(lm).__name__}")
            break
    if lm is not None:
        try:
            sig = inspect.signature(lm.forward)
            print(f"  lm.forward{sig}")
        except (TypeError, ValueError):
            print("  (could not introspect lm.forward signature)")

    # --- Forward probe --------------------------------------------------
    section("Forward probe (dummy audio)")
    sr = getattr(fe, "sampling_rate", 16000)
    n_samples = int(args.audio_seconds * sr)
    dummy_audio = torch.zeros(n_samples)
    print(f"  building dummy audio: {n_samples} samples @ {sr} Hz")

    prompt = processor.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "<|audio|>transcribe the speech with proper punctuation and capitalization.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"  prompt template (first 120 chars): {prompt[:120]!r}")

    try:
        inputs = processor(prompt, dummy_audio, return_tensors="pt").to(args.device)
    except Exception as e:
        print(f"  processor() failed: {e}")
        inputs = None

    if inputs is not None:
        for k, v in inputs.items():
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            dtype = v.dtype if hasattr(v, "dtype") else "n/a"
            print(f"    {k}: shape={shape}, dtype={dtype}")

        section("model.forward(**inputs) probe")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=False, return_dict=True)
        print(f"  output type: {type(out).__name__}")
        for k in dir(out):
            if k.startswith("_"):
                continue
            try:
                v = getattr(out, k)
            except AttributeError:
                continue
            if v is None or callable(v):
                continue
            if hasattr(v, "shape"):
                print(f"    out.{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            elif isinstance(v, (list, tuple)) and v and hasattr(v[0], "shape"):
                print(f"    out.{k}: tuple/list of {len(v)} tensors")
                if v:
                    inner_shape = tuple(v[0].shape) if hasattr(v[0], "shape") else "?"
                    print(f"      [0]: shape={inner_shape}")
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
                print(
                    f"    out.{k}: nested tuple of {len(v)} x {len(v[0]) if v[0] else 0}"
                )
                if v and v[0]:
                    first = v[0][0]
                    if hasattr(first, "shape"):
                        print(f"      [0][0]: shape={tuple(first.shape)}")

    # --- Per-submodule shape probe -------------------------------------
    if inputs is not None and "input_features" in inputs:
        section("Per-submodule shape probe")
        feats = inputs["input_features"]
        feats_mask = inputs.get("input_features_mask")
        print(f"  input_features:        shape={tuple(feats.shape)}, dtype={feats.dtype}")
        if feats_mask is not None:
            print(f"  input_features_mask:   shape={tuple(feats_mask.shape)}, dtype={feats_mask.dtype}")

        with torch.no_grad():
            try:
                enc_out = encoder(feats)
                if hasattr(enc_out, "shape"):
                    print(
                        f"  encoder(input_features): shape={tuple(enc_out.shape)}, dtype={enc_out.dtype}"
                    )
                else:
                    print(f"  encoder(input_features): {type(enc_out).__name__}")
                    for k in dir(enc_out):
                        if k.startswith("_"):
                            continue
                        try:
                            v = getattr(enc_out, k)
                        except AttributeError:
                            continue
                        if hasattr(v, "shape"):
                            print(f"    enc_out.{k}: shape={tuple(v.shape)}")
            except Exception as e:
                print(f"  encoder(input_features) failed: {e}")
                enc_out = None

            if enc_out is not None and hasattr(enc_out, "shape"):
                try:
                    proj_out = projector(enc_out)
                    if hasattr(proj_out, "shape"):
                        print(
                            f"  projector(enc_out):     shape={tuple(proj_out.shape)}, dtype={proj_out.dtype}"
                        )
                    else:
                        print(f"  projector(enc_out): {type(proj_out).__name__}")
                except Exception as e:
                    print(f"  projector(enc_out) failed: {e}")

    # --- Feature extractor parameters (deep) ----------------------------
    section("Feature extractor params (deep)")
    if fe is not None:
        # GraniteSpeechFeatureExtractor stores torchaudio's MelSpectrogram inside.
        for attr in dir(fe):
            if attr.startswith("_"):
                continue
            try:
                v = getattr(fe, attr)
            except Exception:
                continue
            if callable(v):
                continue
            if isinstance(v, (int, float, str, bool, list, tuple)) and not isinstance(v, type):
                print(f"  fe.{attr} = {v!r}")
        # Try the torchaudio MelSpectrogram if present.
        ts_mel = getattr(fe, "melspec", None) or getattr(fe, "mel_spectrogram", None)
        if ts_mel is not None:
            print(f"  torchaudio mel module: {type(ts_mel).__name__}")
            for attr in ("sample_rate", "n_fft", "hop_length", "win_length", "n_mels", "f_min", "f_max", "power"):
                if hasattr(ts_mel, attr):
                    print(f"    ts_mel.{attr} = {getattr(ts_mel, attr)!r}")

    # --- Decoder cache contract ----------------------------------------
    section("Decoder cache contract")
    if lm is not None and inputs is not None:
        # Run a tiny inputs_embeds-only forward with use_cache=True to see
        # what past_key_values come back as.
        with torch.no_grad():
            try:
                # Use the LM's own embeddings on the prompt tokens.
                embeds = lm.get_input_embeddings()(inputs["input_ids"])
                lm_out = lm(
                    inputs_embeds=embeds,
                    use_cache=True,
                    return_dict=True,
                )
                print(f"  lm output type: {type(lm_out).__name__}")
                if hasattr(lm_out, "logits"):
                    print(f"  logits: shape={tuple(lm_out.logits.shape)}")
                pkv = getattr(lm_out, "past_key_values", None)
                print(f"  past_key_values: {type(pkv).__name__}")
                if pkv is None:
                    pass
                elif hasattr(pkv, "layers"):
                    print(f"    Cache.layers: {len(pkv.layers)}")
                    if pkv.layers:
                        layer0 = pkv.layers[0]
                        print(f"    layer 0 type: {type(layer0).__name__}")
                        for attr in ("keys", "values", "key_cache", "value_cache"):
                            if hasattr(layer0, attr):
                                v = getattr(layer0, attr)
                                if hasattr(v, "shape"):
                                    print(f"      layer0.{attr}: shape={tuple(v.shape)}")
                                elif isinstance(v, (list, tuple)) and v and hasattr(v[0], "shape"):
                                    print(
                                        f"      layer0.{attr}: list of {len(v)} tensors, [0]={tuple(v[0].shape)}"
                                    )
                elif isinstance(pkv, (list, tuple)):
                    print(f"    layers: {len(pkv)}")
                    if pkv and isinstance(pkv[0], (list, tuple)) and pkv[0]:
                        print(
                            f"    layer 0: tuple of {len(pkv[0])}, first shape={tuple(pkv[0][0].shape)}"
                        )
            except Exception as e:
                print(f"  lm probe failed: {e}")

    section("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
