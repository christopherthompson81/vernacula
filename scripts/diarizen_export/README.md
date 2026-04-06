# DiariZen ONNX Export

Exports the DiariZen diarization pipeline to ONNX and binary parameter files for use in Vernacula's C# inference code.

The pipeline has two neural components and one set of static transform parameters:

1. **Segmentation model** — WavLM Large + Conformer (EEND), outputs per-frame speaker activity scores
2. **Speaker embedding model** — WeSpeaker ResNet34, outputs 256-dim embeddings from Kaldi Fbank features
3. **LDA/PLDA transform** — static parameters exported as flat binary files for the VBx clustering step

The clustering logic itself is algorithmic (not a neural net) and is reimplemented directly in C#.

## Files

- `export_diarizen_onnx.py` — exports the DiariZen segmentation model to ONNX
- `export_pyannote_wespeaker_onnx.py` — exports the WeSpeaker speaker embedding model to ONNX
- `export_lda_transform.py` — exports the LDA/PLDA transform parameters as flat binary files
- `diarizen_requirements.txt` — export dependencies

## Environment

These scripts require Python 3.10 and specific torch versions. Newer torchaudio releases break the local DiariZen/pyannote fork's import path expectations.

```bash
pip install -r scripts/diarizen_export/diarizen_requirements.txt
```

The segmentation and embedding export scripts also require local installs of the DiariZen repo and its pyannote-audio fork:

```bash
pip install -e /path/to/DiariZen/pyannote-audio
pip install -e /path/to/DiariZen
```

The scripts search for the DiariZen repo at `../../DiariZen` relative to the repo root, then fall back to `/home/chris/Programming/DiariZen`.

## Export

### Segmentation model

Downloads the checkpoint from HuggingFace and exports the WavLM + Conformer segmentation model.

```bash
python scripts/diarizen_export/export_diarizen_onnx.py \
  --model-repo BUT-FIT/diarizen-wavlm-large-s80-md \
  --output-dir ~/models/diarizen_onnx
```

Output: `diarizen_segmentation.onnx`, `metadata.json`

### Speaker embedding model

Exports the WeSpeaker ResNet34 model from `pyannote/wespeaker-voxceleb-resnet34-LM`.

```bash
python scripts/diarizen_export/export_pyannote_wespeaker_onnx.py \
  --output-dir ~/models/diarizen_onnx
```

Output: `wespeaker.onnx`

### LDA/PLDA transform

Exports the LDA projection matrix and PLDA parameters from the HuggingFace snapshot as flat `float32` binary files.

```bash
python scripts/diarizen_export/export_lda_transform.py \
  --repo-id BUT-FIT/diarizen-wavlm-large-s80-md \
  --output-dir ~/models/diarizen_onnx/plda
```

Outputs:

| File | Shape | Description |
|---|---|---|
| `mean1.bin` | `float32[256]` | Subtract from raw WeSpeaker embedding before LDA |
| `lda.bin` | `float32[256×128]` | LDA projection matrix (256→128) |
| `mean2.bin` | `float32[128]` | Subtract after LDA projection |
| `plda_mu.bin` | `float32[128]` | PLDA mean (subtract before PLDA transform) |
| `plda_tr.bin` | `float32[128×128]` | PLDA whitening/eigenspace transform |
| `plda_psi.bin` | `float32[128]` | PLDA eigenvalues (Phi for VBx) |

The full pipeline applied in C#:

```
xvec  = sqrt(128) * l2_norm(lda.T @ (sqrt(256) * l2_norm(raw - mean1)) - mean2)
fea   = (xvec - plda_mu) @ plda_tr.T
```

## ONNX Contract

**diarizen_segmentation.onnx**

| Name | Shape | Description |
|---|---|---|
| `waveform` | `[batch, 1, samples]` | 16 kHz mono audio |
| `scores` | `[batch, frames, speakers]` | Per-frame powerset speaker activity scores |

**wespeaker.onnx**

| Name | Shape | Description |
|---|---|---|
| `fbank` | `[batch, time_frames, 80]` | Kaldi Fbank features (80 mel bins, 10 ms shift) |
| `embedding` | `[batch, 256]` | Raw (un-normalised) speaker embedding |

The embedding model outputs raw embeddings — L2 normalisation is applied after the LDA transform in C#, not inside the ONNX model.

Fbank parameters (must match C# `WeSpeakerEmbedder`):

- `num_mel_bins = 80`
- `frame_length = 25 ms` (400 samples at 16 kHz)
- `frame_shift = 10 ms` (160 samples at 16 kHz)
- `window_type = "hamming"`
- `waveform_scale = 32768`
- `dither = 0.0`, `use_energy = false`
