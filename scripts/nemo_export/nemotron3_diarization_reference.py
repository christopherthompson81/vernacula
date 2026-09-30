#!/usr/bin/env python3
"""Run NeMo's own streaming Nemotron-3-Diarization and save its 10 ms speaker probabilities.

This is the REFERENCE side of the port's fidelity check (see nemotron3_diarization_fidelity.py):
NeMo's `forward_streaming` at the card's offline schedule (cache 264 / fifo 40 / chunk 340 /
right context 40 / period 300 -- the same schedule written into the ONNX metadata), fed NeMo's
own preprocessor with dither off so reruns are bit-stable.

Writes `<out-dir>/<stem>.nemo_preds.npy`, float32 [T_10ms, 8].

`--mel-source vernacula` feeds NeMo the Python transcription of Vernacula's own mel frontend
(benchmark_sortformer_rtf.log_mel_spectrogram) instead of NeMo's preprocessor. That takes the
frontend out of the comparison, so what remains is the streaming loop alone. The frontend is
shared with v2.1 and is not specific to this model; `--mel-source nemo` measures both together.

Needs the NeMo-main venv described in
docs/investigations/nemotron3_diarization_onnx_investigation.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nemo", required=True, type=Path)
    p.add_argument("--audio", nargs="+", required=True, type=Path, help="16 kHz mono WAVs.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--mel-source", choices=("nemo", "vernacula"), default="nemo")
    args = p.parse_args()

    import soundfile as sf
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(str(args.nemo), map_location="cpu")
    model.eval()
    # The card's documented way to select a schedule (Quick Start / "Setting up Streaming
    # Configuration"): set the attributes, then re-validate.
    sm = model.sortformer_modules
    sm.spkcache_len, sm.fifo_len, sm.chunk_len = 264, 40, 340
    sm.chunk_right_context, sm.spkcache_update_period = 40, 300
    model._check_streaming_parameters()
    model.preprocessor.featurizer.dither = 0.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.audio:
        audio, sr = sf.read(str(path), dtype="float32")
        assert sr == 16000 and audio.ndim == 1, (path, sr, audio.shape)
        t0 = time.perf_counter()
        with torch.inference_mode():
            if args.mel_source == "nemo":
                sig = torch.from_numpy(audio)[None]
                feats, feat_len = model.process_signal(
                    audio_signal=sig, audio_signal_length=torch.tensor([len(audio)]))
            else:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import benchmark_sortformer_rtf as B
                mel = np.asarray(B.log_mel_spectrogram(audio))
                mel = mel[0] if mel.ndim == 3 else mel             # [T, 128]
                feats = torch.from_numpy(np.ascontiguousarray(mel.T))[None]
                feat_len = torch.tensor([mel.shape[0]])
            preds = model.forward_streaming(feats, feat_len)
        preds = preds[0, : int(feat_len[0])].numpy().astype(np.float32)
        out = args.out_dir / f"{path.stem}.nemo_preds.npy"
        np.save(out, preds)
        print(f"{path.name}: {len(audio) / sr:.1f}s audio, feats {tuple(feats.shape)}, "
              f"preds {preds.shape}, {time.perf_counter() - t0:.1f}s -> {out.name}")


if __name__ == "__main__":
    main()
