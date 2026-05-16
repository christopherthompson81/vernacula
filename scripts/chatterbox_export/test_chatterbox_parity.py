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
  * `solve_euler` — eager-vs-eager regression guard for the CFM solver
                rewrite that landed in Run 10. Expect bit-identity
                (max_abs = 0); regressions here mean the cat-only
                rewrite has drifted from upstream's broadcast-assign
                semantics. Doesn't touch ONNX, so runs without a
                cond_decoder.onnx artifact.

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
    EXAGGERATION_TOKEN,
    START_SPEECH_TOKEN,
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


def parity_enc_onnx_vs_upstream(onnx_dir: Path, providers: list[str],
                                tolerance: float = 1e-3) -> ParityResult:
    """speech_encoder.onnx speaker_embeddings vs upstream eager.

    The downstream-most parity check: does our exported ONNX
    speech_encoder produce the same speaker embedding (component of
    its 4-tuple output) as running the upstream chatterbox
    speaker_encoder directly on the same input features?

    This is what changed when we dropped SafeDenseLayer: the previous
    `enc[safe-dense]` test showed substituting BatchNorm1d→randomly-
    initialized LayerNorm drifted embeddings by 93%. After dropping
    the substitution and inlining BatchNorm math (so ONNX-export
    works), this test confirms the resulting ONNX produces upstream-
    equivalent embeddings.

    Builds the encoder input the same way the vendored
    PrepareConditionalsModel does (Kaldi fbank from a 16 kHz audio
    clip), runs both paths, compares.
    """
    import torch
    import onnxruntime as ort
    from chatterbox.tts import ChatterboxTTS
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _chatterbox_internals as ci

    onnx_path = onnx_dir / "speech_encoder.onnx"
    if not onnx_path.exists():
        return ParityResult("enc[onnx-vs-upstream]", False,
                            float("inf"), float("inf"), float("inf"),
                            tolerance, notes=f"missing {onnx_path}")

    chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")

    # Run our exported ONNX speech_encoder
    torch.manual_seed(0)
    audio = torch.randn(1, 312_936)  # matches DUMMY_AUDIO_SAMPLES from export
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_out = sess.run(
        ["audio_features", "audio_tokens", "speaker_embeddings", "speaker_features"],
        {"audio_values": audio.numpy()},
    )
    onnx_speaker_embeddings = onnx_out[2]  # the third output

    # Build the equivalent upstream eager output. We use the same
    # vendored PrepareConditionalsModel here because it's the
    # well-tested orchestration of upstream submodules — but with
    # NO SafeDenseLayer patch (our export script also doesn't apply
    # one anymore). So the encoder runs upstream code path.
    prep = ci.PrepareConditionalsModel(chatterbox_model).eval().to("cuda")
    with torch.no_grad():
        _, _, eager_speaker_embeddings, _ = prep(audio.to("cuda"))
    eager_speaker_embeddings_np = eager_speaker_embeddings.cpu().numpy()

    max_abs, max_rel, mean_abs = diff_metrics(onnx_speaker_embeddings, eager_speaker_embeddings_np)
    cos = float(np.dot(onnx_speaker_embeddings.flatten(), eager_speaker_embeddings_np.flatten())
                / (np.linalg.norm(onnx_speaker_embeddings) * np.linalg.norm(eager_speaker_embeddings_np) + 1e-12))
    passed = (max_abs <= tolerance) and (cos > 0.999)
    notes = (
        f"shape={tuple(onnx_speaker_embeddings.shape)}  "
        f"range=[{eager_speaker_embeddings_np.min():.3f}, {eager_speaker_embeddings_np.max():.3f}]  "
        f"cosine_sim={cos:.6f}"
    )
    return ParityResult("enc[onnx-vs-upstream]", passed, max_abs, max_rel, mean_abs, tolerance, notes=notes)


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


def parity_dec(onnx_dir: Path, providers: list[str], tolerance: float = 1e-2) -> ParityResult:
    """conditional_decoder.onnx waveform vs upstream eager.

    Build the same speech_tokens / speaker_emb / speaker_features
    pipeline both ways:
      A: upstream eager (cond decoder runs upstream flow.inference
         + mel2wav.inference in PyTorch)
      C: ONNX via ORT (same wrapper, exported with the four cond-
         decoder patches active).

    Compare with **mel-spectral distance**, NOT time-domain cosine.

    Why: upstream HiFi-GAN-NSF's SourceModule injects fresh
    `torch.randn_like` noise on every call (line in SineGen.forward).
    Time-domain waveforms are inherently non-deterministic — two
    consecutive eager runs already differ by ~4e-2 max-abs and
    cosine ~0.12, before ONNX is involved. The randomness is
    perceptually inaudible (envelope unchanged, noise-floor only),
    but it makes time-domain bit-comparison meaningless.

    Mel-spectral distance compares the spectral envelope, which is
    what the model actually controls. A pass means the ONNX wav is
    perceptually identical to the eager wav up to the inherent
    upstream stochasticity.
    """
    import torch
    import onnxruntime as ort
    from chatterbox.tts import ChatterboxTTS
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _chatterbox_internals as ci
    from _export_patches import patched_cond_decoder_for_export, patched_sinegen_deterministic

    onnx_path = onnx_dir / "conditional_decoder.onnx"
    if not onnx_path.exists():
        return ParityResult("dec", False, float("inf"), float("inf"), float("inf"),
                            tolerance, notes=f"missing {onnx_path}")

    chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
    prep = ci.PrepareConditionalsModel(chatterbox_model).eval().to("cuda")
    et = ci.InputsEmbeds(chatterbox_model).eval().to("cuda")
    cd_eager = ci.ConditionalDecoder(chatterbox_model).eval().to("cuda")
    ci.istft.to("cuda")

    # Same audio prompt as the export uses
    torch.manual_seed(0)
    audio = torch.randn(1, 312_936, device="cuda")
    ids = torch.tensor([[EXAGGERATION_TOKEN] + [255, 281, 39, 46, 56, 2, 53, 2, 286, 41, 37, 2, 136, 122,
        49, 2, 152, 2, 103, 2, 277, 21, 101, 7, 2, 301, 55, 34, 28, 7, 2, 53, 2, 296, 18, 18, 115, 2, 51, 2, 33, 245,
        2, 17, 190, 2, 42, 2, 50, 18, 125, 4, 32, 2, 290, 169, 142, 2, 41, 2, 43, 2, 18, 29, 91, 2, 25, 186, 8, 20,
        14, 80, 2, 29, 86, 213, 216, 9, 0, START_SPEECH_TOKEN, START_SPEECH_TOKEN]], dtype=torch.long, device="cuda")
    pos = torch.where(ids >= START_SPEECH_TOKEN, torch.zeros_like(ids),
                      torch.arange(ids.shape[1], device="cuda").unsqueeze(0) - 1)
    ex = torch.tensor([0.5], device="cuda")

    with torch.no_grad():
        cond_emb, prompt_token, spk_emb, spk_feat = prep(audio_values=audio)
        text_emb = et(ids, pos, ex)
        inputs_embeds = torch.cat((cond_emb, text_emb), dim=1)
        llm = chatterbox_model.t3.tfmr
        sh = chatterbox_model.t3.speech_head
        gt = torch.tensor([[START_SPEECH_TOKEN]], dtype=torch.long, device="cuda")
        pkv = None
        for i in range(256):
            o = llm(inputs_embeds=inputs_embeds, past_key_values=pkv)
            pkv = o.past_key_values
            nl = sh(o.last_hidden_state[:, -1, :])
            nt = torch.argmax(nl, dim=-1).unsqueeze(-1)
            gt = torch.cat((gt, nt), dim=-1)
            if (nt.view(-1) == 6562).all():
                break
            p = torch.full((ids.shape[0], 1), i + 1, dtype=torch.long, device="cuda")
            inputs_embeds = et(nt, p, ex)
        speech_tokens = torch.cat([prompt_token, gt[:, 1:-1]], dim=1)

        # Apply the same patches the ONNX export used, so we're
        # comparing patched-eager vs patched-ONNX (rather than
        # mixing in upstream's torch.istft path here). Includes the
        # deterministic SineGen probe to remove NSF stochasticity as
        # a confound.
        with patched_cond_decoder_for_export(chatterbox_model.s3gen, ci.istft), \
             patched_sinegen_deterministic(chatterbox_model.s3gen.mel2wav):
            wav_eager = cd_eager(speech_tokens, spk_emb, spk_feat).cpu().numpy()

    # Run ONNX at the natural speech_tokens length. The cond decoder is
    # now fully dynamic-shape (see docs/chatterbox_investigation.md Run 10).
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    wav_onnx = sess.run(["waveform"], {
        "speech_tokens": speech_tokens.cpu().numpy(),
        "speaker_embeddings": spk_emb.cpu().numpy(),
        "speaker_features": spk_feat.cpu().numpy(),
    })[0]

    max_abs, max_rel, mean_abs = diff_metrics(wav_onnx, wav_eager)

    # Mel-spectral L1 — robust to NSF noise. Compare 80-bin log-mel
    # spectrograms of the two waveforms; this measures envelope match
    # (what the model actually predicts) and ignores phase + per-call
    # noise-injection randomness. Threshold 0.5 is conservative — for
    # bit-equivalent eager runs this is ~0.05 (signed by NSF noise floor).
    import torchaudio.transforms as T
    mel_t = T.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80)
    log_mel_eager = torch.log(torch.clamp(mel_t(torch.from_numpy(wav_eager[0]).float()), min=1e-5))
    log_mel_onnx = torch.log(torch.clamp(mel_t(torch.from_numpy(wav_onnx[0]).float()), min=1e-5))
    mel_l1 = float((log_mel_eager - log_mel_onnx).abs().mean())

    # Cosine on time-domain reported but informational only — not pass-criterion.
    cos = float(np.dot(wav_eager.flatten(), wav_onnx.flatten())
                / (np.linalg.norm(wav_eager) * np.linalg.norm(wav_onnx) + 1e-12))

    mel_threshold = 0.5
    passed = mel_l1 <= mel_threshold
    notes = (
        f"shape={tuple(wav_onnx.shape)}  "
        f"range=[{wav_eager.min():.3f}, {wav_eager.max():.3f}]  "
        f"mel_log_l1={mel_l1:.4e} (thr={mel_threshold})  "
        f"cosine_sim={cos:.4f} (informational, NSF noise expected)"
    )
    return ParityResult("dec", passed, max_abs, max_rel, mean_abs, mel_threshold, notes=notes)


def parity_solve_euler(tolerance: float = 1e-5) -> ParityResult:
    """Eager-vs-eager: our patched solve_euler + cfm.forward vs upstream.

    The dynamic-shape rewrite (docs/chatterbox_investigation.md Run 10)
    replaced upstream's
        x_in = zeros([2, 80, x.size(2)]); x_in[:] = x
    pattern with
        x_in = torch.cat([x, x], dim=0)
    to avoid baking T into the trace's Reshape+Expand chain. The rewrite
    should be mathematically identical to upstream — cat-of-same-shape
    and broadcast-assign produce the same `[2, 80, T]` tensor.

    This test confirms that equivalence: same mu/mask/spks, same RNG,
    once with upstream's solve_euler and once with our patched version,
    expect bit-identical mel output. No ONNX involved — pure eager.
    """
    import torch
    from chatterbox.tts import ChatterboxTTS
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _chatterbox_internals as ci
    from _export_patches import patched_cond_decoder_for_export, _seeded_rand_noise_like

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    # Pin rand_noise to the SAME deterministic seed on both sides. Without
    # this, upstream uses the model-instance's random init while the
    # patched run uses the context-manager's seeded value — different z
    # entering solve_euler → different mels (it's not a math drift,
    # just the rand_noise confounder from docs/.../Run 11).
    model.s3gen.flow.decoder.rand_noise = _seeded_rand_noise_like(
        model.s3gen.flow.decoder.rand_noise
    )

    # Realistic-shaped dummy inputs. T = 1010 is the natural mel-frame
    # count for a 505-token speech sequence (2x upsample); picking a
    # non-trace-time value would also work since both code paths are
    # eager-symbolic.
    B, T, n_feats = 1, 1010, 80
    torch.manual_seed(0)
    mu = torch.randn(B, n_feats, T, device=device)
    mask = torch.ones(B, 1, T, device=device)
    spk_dim = model.s3gen.flow.spk_embed_affine_layer.out_features
    spks = torch.randn(B, spk_dim, device=device)

    def run(model):
        with torch.no_grad():
            mel, _ = model.s3gen.flow.decoder(
                mu=mu, mask=mask, spks=spks, cond=torch.zeros_like(mu),
                n_timesteps=10,
            )
        return mel.cpu().numpy()

    mel_upstream = run(model)
    with patched_cond_decoder_for_export(model.s3gen, ci.istft):
        mel_patched = run(model)

    max_abs, max_rel, mean_abs = diff_metrics(mel_patched, mel_upstream)
    passed = max_abs <= tolerance
    notes = (
        f"shape={mel_upstream.shape}  "
        f"range=[{mel_upstream.min():.3f}, {mel_upstream.max():.3f}]  "
        f"(eager-vs-eager: bit-identity is the expected outcome)"
    )
    return ParityResult("solve_euler", passed, max_abs, max_rel, mean_abs, tolerance, notes=notes)


def parity_attention(tolerance: float = 1e-5) -> ParityResult:
    """Eager-vs-eager: MinimalAttnProcessor vs diffusers AttnProcessor2_0.

    BasicTransformerBlock.attn1 is the heaviest single source of nodes
    in cond_decoder.onnx (~400 nodes per block × 48 blocks ≈ 57% of the
    graph). MinimalAttnProcessor is a drop-in that does only the work
    our model needs (3D input, self-attn, (B,T,T) bias mask, no
    residual/rescale/norm pre/post-processing). To be safe to swap in
    at export time, it must produce bit-identical output to upstream
    on a realistic block.

    Test: pick a real BasicTransformerBlock from the cond decoder
    estimator, call it once with the original processor, once with the
    minimal one, compare. Same mask, same input.
    """
    import torch
    from chatterbox.tts import ChatterboxTTS
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _export_patches import MinimalAttnProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    block = None
    for n, m in model.s3gen.flow.decoder.estimator.named_modules():
        if type(m).__name__ == "BasicTransformerBlock":
            block = m
            break
    if block is None:
        return ParityResult("attention", False, float("inf"), float("inf"),
                            float("inf"), tolerance, notes="no BasicTransformerBlock found")

    # Synthetic input matching the shapes BasicTransformerBlock sees in
    # the actual CFM solve loop: B = 2 (CFG-doubled), T = 1010 (mel
    # frames), C = inner_dim (whatever the first attn1 expects).
    B, T = 2, 1010
    inner_dim = block.attn1.to_q.in_features
    torch.manual_seed(0)
    hidden = torch.randn(B, T, inner_dim, device=device)
    attn_mask = torch.zeros(B, T, T, device=device)  # all-zero bias = unmasked

    # Baseline: upstream processor (whatever was on the block)
    orig_proc = block.attn1.processor
    with torch.no_grad():
        out_upstream = block.attn1(hidden, attention_mask=attn_mask).cpu().numpy()

    # Swap to minimal; same input
    block.attn1.set_processor(MinimalAttnProcessor())
    with torch.no_grad():
        out_minimal = block.attn1(hidden, attention_mask=attn_mask).cpu().numpy()
    block.attn1.set_processor(orig_proc)  # restore

    max_abs, max_rel, mean_abs = diff_metrics(out_minimal, out_upstream)
    passed = max_abs <= tolerance
    notes = (
        f"shape={out_upstream.shape}  "
        f"upstream_proc={type(orig_proc).__name__}  "
        f"range=[{out_upstream.min():.3f}, {out_upstream.max():.3f}]"
    )
    return ParityResult("attention", passed, max_abs, max_rel, mean_abs, tolerance, notes=notes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Directory produced by export_chatterbox_to_onnx.py")
    p.add_argument("--runtime", default="cuda", choices=["cpu", "cuda", "tensorrt"])
    p.add_argument("--tests", default="lm",
                   help="Comma-separated subset: lm,embed,enc,dec,solve_euler,attention,all (default: lm)")
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
        tests = {"lm", "embed", "enc", "dec", "solve_euler", "attention"}

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
        print("[enc] speech_encoder.onnx speaker_embeddings vs upstream eager")
        results.append(parity_enc_onnx_vs_upstream(args.onnx_dir, providers, args.tolerance))
        print(results[-1].summary())
        # Keep the historical SafeDenseLayer test reachable for diagnostic
        # use; not run by default since SafeDenseLayer was removed.
    if "dec" in tests:
        print("[dec] conditional_decoder.onnx waveform vs upstream eager")
        results.append(parity_dec(args.onnx_dir, providers, args.tolerance))
        print(results[-1].summary())
    if "solve_euler" in tests:
        print("[solve_euler] patched cat-only solve_euler vs upstream (eager)")
        results.append(parity_solve_euler())
        print(results[-1].summary())
    if "attention" in tests:
        print("[attention] MinimalAttnProcessor vs diffusers AttnProcessor (eager)")
        results.append(parity_attention())
        print(results[-1].summary())

    print()
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"PARITY FAILED: {len(failed)}/{len(results)} test(s)")
        sys.exit(1)
    print(f"PARITY OK: {len(results)}/{len(results)} test(s) passed")


if __name__ == "__main__":
    main()
