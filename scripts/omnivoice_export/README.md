# OmniVoice → ONNX export

Exports `k2-fsa/OmniVoice` (Apache-2.0, a diffusion-language-model TTS) into ONNX graphs
for Vernacula. **Phase 1** goal: faithful fp32 graphs + a Python harness validated against
the PyTorch reference. C# integration and quantization are later phases.

Why OmniVoice: it's non-autoregressive (iterative unmasking of multi-codebook audio tokens),
so it can't hallucinate the way an autoregressive LM-backbone TTS (Chatterbox) does — the
"Path B" in `tts.scratch.md`. Investigation log: `docs/omnivoice_onnx_investigation.md`.

## Architecture split

OmniVoice's `generate()` decomposes into **three neural graphs** (exported) and **host
orchestration** (stays in Python now / C# later — tokenizer, `RuleDurationEstimator`, the
diffusion masking schedule, CFG + top-k/gumbel scoring, post-processing).

| Graph | Source | Inputs | Output |
|---|---|---|---|
| `omnivoice_transformer.onnx` | embeds + Qwen3 (hidden 1024) + `audio_heads` | `input_ids[2B,8,S]` i64, `audio_mask[2B,S]` bool, `attention_mask[2B,1,S,S]` bool | `logits[2B,8,S,1025]` f32 |
| `higgs_encoder.onnx` | `HiggsAudioV2TokenizerModel.encode` | `input_values[B,1,T]` f32 @24k | `audio_codes[B,8,Tc]` i64 |
| `higgs_decoder.onnx` | `HiggsAudioV2TokenizerModel.decode` | `audio_codes[B,8,Tc]` i64 | `audio_values[B,1,Tsamp]` f32 @24k |

`2B` = classifier-free-guidance cond+uncond stack. The transformer is invoked `num_step`
(=32 default) times per generation. Dynamic axes: batch and sequence/sample length.

## Environment

Python 3.10–3.12. Dedicated venv (transformers≥5.12 is required for
`HiggsAudioV2TokenizerModel`; the repo-wide kokoro venv pins 5.10.2):

```bash
cd scripts/omnivoice_export
python3 -m venv .venv-omnivoice-export
source .venv-omnivoice-export/bin/activate
pip install torch==2.12.0 torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install "git+https://github.com/k2-fsa/OmniVoice.git"
```

The `nvidia-cudnn-cu13` pin in `requirements.txt` matches the system `libcudnn9` so torch's
bundled cuDNN can't mismatch it (same fix as the kokoro venv). Bump it if the system
`libcudnn9` is upgraded (`dpkg -l | grep libcudnn9`).

Model: `huggingface-cli download k2-fsa/OmniVoice` (or `snapshot_download`), default path
`/mnt/data/models/omnivoice/k2-fsa-OmniVoice`.

## Workflow

```bash
# 0. build a MATCHED voice-clone reference (model-made clip + known transcript).
#    Cloning conditions on (ref_audio, ref_text) and expects them to correspond; a
#    mismatched transcript derails the diffusion into quiet/unintelligible output.
python make_reference.py              # -> capture/ref_voice.wav + capture/ref_voice.txt

# 1. capture in-distribution reference tensors from a real PyTorch run (deterministic)
python capture_reference.py --device cpu

# 2. export the three graphs (fp32), driven by the capture. The transformer's 0.6B
#    weights are consolidated into a single omnivoice_transformer.onnx.data sidecar.
python export_omnivoice.py            # add --no-dynamo to force the legacy exporter

# 3. validate each graph against the capture (full fp32; default provider=cpu)
python parity_check.py                # --provider {cpu,cuda-fp32,cuda-tf32}

# 4. end-to-end: ONNX graphs as drop-in replacements in the real pipeline
python infer_onnx.py                  # compares WAV to the PyTorch reference
```

**Faithful vs fast:** validate parity in full fp32 (`--provider cpu` or `cuda-fp32`).
ORT's CUDA EP defaults to TF32 matmul (`cuda-tf32`), whose ~1e-2 logit error **compounds
through the 32-step diffusion loop** into audibly different speech — a performance-phase
trade-off, not a parity path.

## Validation metrics (and why)

Per `docs/kokoro_onnx_investigation.md` — never use random ids/codes or waveform SNR:

- **transformer**: argmax-token agreement (the diffusion loop only needs the same token
  picks) + logit max-abs. Validated on *captured real* inputs, not random ids.
- **encoder**: exact integer code-match rate (codes are discrete indices).
- **decoder**: **log-spectral L1** (phase-invariant). DAC is GAN-trained → output phase is
  not unique, so waveform SNR/correlation flags inaudible phase diffs as huge errors.

## Files

- `requirements.txt` — deps (+ cuDNN pin rationale)
- `make_reference.py` — synthesise a matched voice-clone reference (clip + transcript)
- `capture_reference.py` — wrap the 3 modules, run `generate()`, dump real I/O → `capture/`
- `export_omnivoice.py` — 3 `nn.Module` wrappers → ONNX (dynamo, legacy fallback) + external-data consolidation
- `parity_check.py` — per-graph parity vs capture
- `infer_onnx.py` — ONNX-backed `generate()`, end-to-end WAV vs reference
- `onnx/` — exported graphs + `export-report.json` (generated)
- `capture/` — reference tensors + WAVs (generated)

## Out of scope (later)

C# runtime port; fp16/int8 quantization (the k2-fsa/OmniVoice#151 mobile ask); long-text
chunking parity.
