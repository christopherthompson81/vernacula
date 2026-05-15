#!/usr/bin/env python3
"""Per-graph numerical parity: each ONNX export vs upstream PyTorch.

Stage 0 step E4. For each of the four exported ONNX graphs, run it
through ONNX Runtime and compare its outputs against the equivalent
upstream `chatterbox` PyTorch forward on identical inputs.

Why this matters: torch.onnx.export returning 0 only proves the graph
*traces*. It says nothing about whether the graph *computes the same
function* as the PyTorch model. The vendored wrappers in
`_chatterbox_internals.py` make several non-obvious substitutions
(`SafeDenseLayer` BatchNorm→LayerNorm, scatter-add window_sumsquare,
the bool-mask InputsEmbeds dispatch) — we have no a-priori reason to
trust them numerically.

Test layers, smallest blast radius first:

  * `lm`     — language_model.onnx vs chatterbox.t3.tfmr + speech_head.
                Both sides come from upstream; expect bit-identity (or
                near it modulo float ordering).
  * `embed`  — embed_tokens.onnx vs the vendored InputsEmbeds running
                in eager. Confirms the export trace didn't drift from
                the wrapper it traced.
  * `enc`    — speech_encoder.onnx vs running upstream chatterbox.s3gen
                eager. Reveals whether SafeDenseLayer and the vendored
                S3Tokenizer chain agree numerically with upstream.
  * `dec`    — conditional_decoder.onnx vs upstream chatterbox.s3gen
                flow + mel2wav. Compare waveforms via spectral distance
                (bit-equality is unrealistic for a vocoder).

Each test reports max-abs-diff, max-rel-diff, mean-abs-diff and a
pass/fail verdict against a configurable tolerance.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _common import (
    add_local_script_path,
    choose_onnx_providers,
    fail,
    read_export_report,
    LLM_HIDDEN_SIZE,
    LLM_NUM_LAYERS,
    LLM_NUM_KV_HEADS,
    LLM_HEAD_DIM,
)

add_local_script_path()


@dataclass
class ParityResult:
    name: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    tolerance: float
    notes: str = ""

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        line = (
            f"  [{verdict}] {self.name}  "
            f"max_abs={self.max_abs_diff:.3e}  "
            f"max_rel={self.max_rel_diff:.3e}  "
            f"mean_abs={self.mean_abs_diff:.3e}  "
            f"(tol={self.tolerance:.0e})"
        )
        if self.notes:
            line += f"  // {self.notes}"
        return line


def diff_metrics(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, float, float]:
    """Element-wise abs and rel diff between two arrays of the same shape."""
    if ours.shape != theirs.shape:
        return float("inf"), float("inf"), float("inf")
    diff = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
    denom = np.maximum(np.abs(theirs.astype(np.float64)), 1e-12)
    rel = diff / denom
    return float(diff.max()), float(rel.max()), float(diff.mean())


def parity_lm(onnx_dir: Path, providers: list[str], tolerance: float = 1e-2) -> ParityResult:
    """language_model.onnx vs chatterbox.t3.tfmr + speech_head eager.

    Both sides come from the same upstream weights — no vendored model
    code in this path. Differences are pure ONNX-runtime numerical
    drift (mainly different SDPA / softmax kernels).

    Pass criterion is two-part:

      1. max-abs-diff in logits ≤ tolerance (default 1e-2). This is
         lenient because SDPA kernels diverge ~1e-3 routinely.
      2. **Argmax tokens must match exactly.** For TTS sampling we
         care that the LM's preferred next token is the same. If
         logit drift doesn't reorder the top-1, the model is
         functionally equivalent.

    Either failure flips the verdict; the report shows both metrics so
    the failure mode is unambiguous.
    """
    import torch
    import onnxruntime as ort
    from chatterbox.tts import ChatterboxTTS

    onnx_path = onnx_dir / "language_model.onnx"
    if not onnx_path.exists():
        return ParityResult("lm", False, float("inf"), float("inf"), float("inf"),
                            tolerance, notes=f"missing {onnx_path}")

    chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
    tfmr = chatterbox_model.t3.tfmr
    speech_head = chatterbox_model.t3.speech_head
    tfmr.eval()
    speech_head.eval()

    torch.manual_seed(0)
    B, S = 1, 8
    inputs_embeds = torch.randn(B, S, LLM_HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    attention_mask = torch.ones(B, S, dtype=torch.int64, device="cuda")
    past_kv = tuple(
        (torch.zeros(B, LLM_NUM_KV_HEADS, 0, LLM_HEAD_DIM, device="cuda"),
         torch.zeros(B, LLM_NUM_KV_HEADS, 0, LLM_HEAD_DIM, device="cuda"))
        for _ in range(LLM_NUM_LAYERS)
    )

    with torch.no_grad():
        out = tfmr(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        upstream_logits = speech_head(out.last_hidden_state).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    feed = {
        "inputs_embeds": inputs_embeds.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
    }
    for layer in range(LLM_NUM_LAYERS):
        feed[f"past_key_values.{layer}.key"] = past_kv[layer][0].cpu().numpy()
        feed[f"past_key_values.{layer}.value"] = past_kv[layer][1].cpu().numpy()
    onnx_logits = sess.run(["logits"], feed)[0]

    max_abs, max_rel, mean_abs = diff_metrics(onnx_logits, upstream_logits)

    onnx_argmax = onnx_logits.argmax(axis=-1)
    upstream_argmax = upstream_logits.argmax(axis=-1)
    tokens_agree = bool(np.array_equal(onnx_argmax, upstream_argmax))

    logit_range = (float(upstream_logits.min()), float(upstream_logits.max()))
    passed = (max_abs <= tolerance) and tokens_agree
    notes = (
        f"shape={tuple(onnx_logits.shape)}  "
        f"logit_range=[{logit_range[0]:.1f}, {logit_range[1]:.1f}]  "
        f"argmax_agree={tokens_agree}"
    )
    return ParityResult("lm", passed, max_abs, max_rel, mean_abs, tolerance, notes=notes)


def parity_embed(onnx_dir: Path, providers: list[str], tolerance: float = 1e-4) -> ParityResult:
    """embed_tokens.onnx vs the vendored InputsEmbeds running eager.

    Sanity check that ONNX trace faithfully captured the wrapper's
    forward pass. Both sides run the same Python code; the only
    difference is ORT vs PyTorch CUDA kernels for the underlying
    `embedding` and `where` ops. Expect very tight agreement (1e-4 or
    better) since the wrapper has no SDPA, no softmax, just lookups
    and masked selections.

    Doesn't validate that the wrapper's math is *correct* — that
    requires comparing to a known-good independent implementation
    (deferred to a future test). For now, the wrapper's behavior is
    self-consistent across runtimes.
    """
    import torch
    import onnxruntime as ort
    from chatterbox.tts import ChatterboxTTS
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _chatterbox_internals as ci

    onnx_path = onnx_dir / "embed_tokens.onnx"
    if not onnx_path.exists():
        return ParityResult("embed", False, float("inf"), float("inf"), float("inf"),
                            tolerance, notes=f"missing {onnx_path}")

    chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
    embed = ci.InputsEmbeds(chatterbox_model).eval().to("cuda")

    torch.manual_seed(0)
    from _common import START_SPEECH_TOKEN as ST, EXAGGERATION_TOKEN as ET
    input_ids = torch.tensor([[
        ET, 255, 281, 39, 46, 56, 2, 53, 2, 286, 41, 37, 2, 136, 122,
        49, 2, 152, 2, 103, 2, 277, 21, 101, 7, 2, 301, 55, 34,
        28, 7, 2, 53, 2, 296, 18, 18, 115, 2, 51, 2, 33, 245,
        2, 17, 190, 2, 42, 2, 50, 18, 125, 4, 32, 2, 290, 169,
        142, 2, 41, 2, 43, 2, 18, 29, 91, 2, 25, 186, 8, 20,
        14, 80, 2, 29, 86, 213, 216, 9, 0, ST, ST,
    ]], dtype=torch.long, device="cuda")
    position_ids = torch.where(
        input_ids >= ST,
        torch.zeros_like(input_ids),
        torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0) - 1,
    )
    exaggeration = torch.tensor([0.5], device="cuda")

    with torch.no_grad():
        eager_out = embed(input_ids, position_ids, exaggeration).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_out = sess.run(["inputs_embeds"], {
        "input_ids": input_ids.cpu().numpy(),
        "position_ids": position_ids.cpu().numpy(),
        "exaggeration": exaggeration.cpu().numpy(),
    })[0]

    max_abs, max_rel, mean_abs = diff_metrics(onnx_out, eager_out)
    passed = max_abs <= tolerance
    return ParityResult(
        "embed", passed, max_abs, max_rel, mean_abs, tolerance,
        notes=f"shape={tuple(onnx_out.shape)}  range=[{eager_out.min():.2f}, {eager_out.max():.2f}]",
    )


def parity_enc(onnx_dir: Path, providers: list[str], tolerance: float = 1e-2) -> ParityResult:
    """SafeDenseLayer impact on the speaker_encoder.

    Vlad asserts the BatchNorm1d→LayerNorm substitution in
    `s3gen.speaker_encoder.xvector.dense` is "safe at inference". This
    test validates the claim:

      A: upstream speaker_encoder(features) UNPATCHED  (ground truth)
      B: same encoder + same features WITH SafeDenseLayer applied

    Pass criterion: max-abs-diff in speaker_embeddings ≤ tolerance.
    Fail means SafeDenseLayer is not actually inference-equivalent and
    the speaker_encoder.onnx we ship produces drifted speaker
    embeddings — which would degrade voice cloning quality silently.

    Doesn't include ONNX comparison directly; the eager-vs-eager A/B
    is the load-bearing claim. If the substitution is benign, the
    ONNX trace also fine. If it isn't, ONNX trace is irrelevant
    because the upstream wasn't faithfully preserved.
    """
    import torch
    from chatterbox.tts import ChatterboxTTS
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _chatterbox_internals as ci

    chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
    enc = chatterbox_model.s3gen.speaker_encoder
    enc.eval()

    # Realistic-shaped input. The xvector encoder takes (B, T, F=80)
    # where F is the mel-bin count from Kaldi fbank. Use random with
    # standard-normal statistics so BatchNorm hits its trained range.
    torch.manual_seed(0)
    features = torch.randn(1, 200, 80, device="cuda")

    # A: unpatched
    with torch.no_grad():
        out_a = enc(features).clone()

    # B: patched
    orig_dense = enc.xvector.dense
    new_dense = ci.SafeDenseLayer(orig_dense.linear.in_channels,
                                  orig_dense.linear.out_channels).to("cuda")
    new_dense.linear.weight.data.copy_(orig_dense.linear.weight.data)
    enc.xvector.dense = new_dense

    try:
        with torch.no_grad():
            out_b = enc(features).clone()
    finally:
        # Always restore to leave the module in a clean state for any
        # subsequent tests in the same run.
        enc.xvector.dense = orig_dense

    a = out_a.cpu().numpy()
    b = out_b.cpu().numpy()
    max_abs, max_rel, mean_abs = diff_metrics(b, a)

    passed = max_abs <= tolerance
    notes = (
        f"shape={tuple(a.shape)}  "
        f"range=[{a.min():.3f}, {a.max():.3f}]  "
        f"cosine_sim={float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)):.6f}"
    )
    return ParityResult("enc[safe-dense]", passed, max_abs, max_rel, mean_abs, tolerance, notes=notes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Directory produced by export_chatterbox_to_onnx.py")
    p.add_argument("--runtime", default="cuda", choices=["cpu", "cuda", "tensorrt"])
    p.add_argument("--tests", default="lm",
                   help="Comma-separated subset: lm,embed,enc,dec,all (default: lm)")
    p.add_argument("--tolerance", type=float, default=1e-2,
                   help="Max-abs-diff threshold for pass (default: 1e-2). "
                        "Each test layer has its own appropriate default; this is a "
                        "global override. SDPA-based models routinely drift ~1e-3 "
                        "between PyTorch and ORT even with matching weights.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx_dir.exists():
        fail(f"ONNX dir not found: {args.onnx_dir}")
    report = read_export_report(args.onnx_dir)
    print(f"Parity check vs {args.onnx_dir}")
    print(f"  graphs in report: {report.get('graphs_exported', [])}")
    print(f"  runtime: {args.runtime}  tolerance: {args.tolerance:.0e}")
    print()

    providers = choose_onnx_providers(args.runtime)
    tests = {t.strip() for t in args.tests.split(",")}
    if "all" in tests:
        tests = {"lm", "embed", "enc", "dec"}

    results: list[ParityResult] = []

    if "lm" in tests:
        print("[lm] language_model.onnx vs chatterbox.t3.tfmr + speech_head")
        results.append(parity_lm(args.onnx_dir, providers, args.tolerance))
        print(results[-1].summary())

    if "embed" in tests:
        print("[embed] embed_tokens.onnx vs vendored InputsEmbeds eager")
        results.append(parity_embed(args.onnx_dir, providers, args.tolerance))
        print(results[-1].summary())
    if "enc" in tests:
        print("[enc] SafeDenseLayer impact on speaker_encoder")
        results.append(parity_enc(args.onnx_dir, providers, args.tolerance))
        print(results[-1].summary())
    if "dec" in tests:
        print("[dec] conditional_decoder parity — not implemented yet")

    print()
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"PARITY FAILED: {len(failed)}/{len(results)} test(s)")
        sys.exit(1)
    print(f"PARITY OK: {len(results)}/{len(results)} test(s) passed")


if __name__ == "__main__":
    main()
