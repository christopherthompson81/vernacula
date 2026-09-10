#!/usr/bin/env python3
"""Fidelity DER: score the ported Sortformer streaming loop against NeMo's own.

This answers "has the port drifted from its reference implementation", which is the
question a port needs answered continuously. It is NOT an accuracy benchmark: it says
nothing about whether NeMo itself is right, and a perfect score here is consistent with
both sides being wrong together. For absolute numbers you need a labelled corpus --
NVIDIA evaluates this checkpoint on AMI, VoxConverse v0.3 and DIHARD III.

Why DER rather than the frame-agreement numbers used during debugging:

  * it applies pyannote's OPTIMAL SPEAKER MAPPING, so a pure relabelling of speakers
    scores 0 instead of looking like total disagreement;
  * it applies a collar, so a boundary that moves by one frame is not counted as error;
  * it scores the SEGMENTS the application actually emits -- median filter and
    binarization included -- rather than raw per-frame posteriors.

⚠ THE REFERENCE MUST BE NeMo ITSELF, NEVER scripts/nemo_export/benchmark_sortformer_rtf.py.
That module is a transcription of the same C# streaming loop, so it shares the port's
bugs and they cancel out of any comparison against it. A wrong-signed score boost in
speaker-cache compression survived three rounds of measurement exactly that way (#165
item 9); it showed up the moment the reference became NeMo's own forward_streaming.

⚠ THE REFERENCE MUST ALSO BE BUILT AT VERNACULA'S STREAMING SCHEDULE. The checkpoint
ships a different one (chunk_len=188, fifo_len=0, lc=rc=1 against Vernacula's
124/124/0/0), and mutating the attributes of a restored model does not fully reconfigure
it -- it makes agreement worse, not better. SortformerModules is rebuilt from the
model's own cfg with the schedule overridden, then the trained weights are loaded in.

Usage:

    python scripts/nemo_export/sortformer_fidelity_der.py \\
        --audio a.wav b.wav \\
        --nemo ~/models/diar_streaming_sortformer_4spk-v2.1.nemo \\
        --onnx ~/models/diar_streaming_sortformer_4spk-v2.1.onnx

Exit code is 1 when DER exceeds --max-der, so this can gate a change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", nargs="+", required=True, type=Path,
                   help="Audio files to score. 16 kHz mono.")
    p.add_argument("--nemo", required=True, type=Path, help="Source .nemo checkpoint.")
    p.add_argument("--onnx", required=True, type=Path, help="Exported ONNX model to test.")
    p.add_argument("--collar", type=float, default=0.25,
                   help="Scoring collar in seconds (default 0.25, the usual convention).")
    p.add_argument("--score-overlap", action="store_true",
                   help="Score overlapping speech too. Off by default, matching convention.")
    p.add_argument("--max-der", type=float, default=None,
                   help="Exit 1 if the aggregate DER exceeds this. Use to gate a change.")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Truncate each file, to keep a check quick.")
    return p.parse_args()


def build_reference_model(nemo_path: Path, chunk_len: int, fifo_len: int,
                          spkcache_len: int, update_period: int):
    """NeMo, rebuilt at Vernacula's streaming schedule with the trained weights kept."""
    import omegaconf
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.modules.sortformer_modules import SortformerModules

    model = SortformerEncLabelModel.restore_from(str(nemo_path), map_location="cpu")
    model.eval()
    model.to("cpu")

    cfg = omegaconf.OmegaConf.to_container(model.cfg.sortformer_modules, resolve=True)
    cfg.pop("_target_", None)
    cfg.update(chunk_len=chunk_len, fifo_len=fifo_len, spkcache_len=spkcache_len,
               spkcache_update_period=update_period,
               chunk_left_context=0, chunk_right_context=0)
    rebuilt = SortformerModules(**cfg)
    rebuilt.load_state_dict(model.sortformer_modules.state_dict(), strict=True)
    rebuilt.eval()
    rebuilt.to("cpu")
    model.sortformer_modules = rebuilt
    return model


def to_annotation(segments, uniq: str):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uniq)
    for start, end, speaker in segments:
        if end > start:
            ann[Segment(start, end)] = speaker
    return ann


def main() -> int:
    args = parse_args()
    import numpy as np
    import soundfile as sf
    import torch

    import benchmark_sortformer_rtf as B
    from nemo.collections.asr.metrics.der import score_labels

    model = build_reference_model(
        args.nemo, B.CHUNK_LENGTH, B.FIFO_LENGTH,
        B.SPEAKER_CACHE_LENGTH, B.SPEAKER_CACHE_UPDATE_PERIOD)
    print(f"reference : NeMo forward_streaming, rebuilt at chunk={B.CHUNK_LENGTH} "
          f"fifo={B.FIFO_LENGTH} spkcache={B.SPEAKER_CACHE_LENGTH} "
          f"period={B.SPEAKER_CACHE_UPDATE_PERIOD}, lc=rc=0")
    print(f"hypothesis: {args.onnx.name} via the ported streaming loop\n")

    audio_map, refs, hyps = {}, [], []
    for path in args.audio:
        uniq = path.stem
        frames = int(args.max_seconds * 16000) if args.max_seconds else -1
        audio, sr = sf.read(str(path), dtype="float32",
                            frames=frames if frames > 0 else -1)
        if sr != 16000:
            sys.exit(f"{path}: expected 16 kHz, got {sr}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        mel = B.log_mel_spectrogram(audio)
        stride = B.CHUNK_LENGTH * B.SUBSAMPLING
        n_chunks = (mel.shape[1] + stride - 1) // stride

        with torch.inference_mode():
            ref_preds = model.forward_streaming(
                torch.from_numpy(mel.transpose(0, 2, 1).copy()),
                torch.tensor([mel.shape[1]], dtype=torch.long),
            )[0].cpu().numpy()

        pipeline = B.OnnxSortformerPipeline(args.onnx, device="cpu", gpu_id=0,
                                            provider_kind="cpu")
        pipeline.reset_state()
        hyp_preds = np.concatenate(
            [pipeline.process_chunk(i, stride, mel.shape[1], mel)[0]
             for i in range(n_chunks)], axis=0)

        # Identical post-processing on both sides, so this isolates the streaming loop.
        n = min(len(ref_preds), len(hyp_preds))
        ref_segs = pipeline.binarize_to_segments(pipeline.filter_preds([ref_preds[:n]]))
        hyp_segs = pipeline.binarize_to_segments(pipeline.filter_preds([hyp_preds[:n]]))

        audio_map[uniq] = {"uem_filepath": None}
        refs.append((uniq, to_annotation(ref_segs, uniq)))
        hyps.append((uniq, to_annotation(hyp_segs, uniq)))
        print(f"  {uniq:24} {len(audio)/sr:6.1f}s  ref {len(ref_segs):3} seg / "
              f"{len(refs[-1][1].labels())} spk   hyp {len(hyp_segs):3} seg / "
              f"{len(hyps[-1][1].labels())} spk")

    result = score_labels(audio_map, refs, hyps, collar=args.collar,
                          ignore_overlap=not args.score_overlap, verbose=False)
    if result is None:
        sys.exit("scoring failed: reference and hypothesis counts differ")
    _metric, _mapping, itemized = result
    der, cer, fa, miss = itemized

    print(f"\nfidelity DER vs NeMo (collar {args.collar}s, "
          f"{'scoring' if args.score_overlap else 'ignoring'} overlap)")
    print(f"  DER          {der * 100:7.3f} %")
    print(f"  confusion    {cer * 100:7.3f} %")
    print(f"  false alarm  {fa * 100:7.3f} %")
    print(f"  missed       {miss * 100:7.3f} %")

    if args.max_der is not None and der * 100 > args.max_der:
        print(f"\nFAIL: DER {der * 100:.3f}% exceeds --max-der {args.max_der}%")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
