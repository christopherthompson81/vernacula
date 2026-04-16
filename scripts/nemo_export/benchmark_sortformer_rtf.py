#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from export_sortformer_nemo_to_onnx import build_wrapper, import_dependencies


SAMPLE_RATE = 16_000
NFft = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
NMELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = np.float32(5.960464478e-8)

CHUNK_LENGTH = 124
FIFO_LENGTH = 124
SUBSAMPLING = 8
EMBEDDING_DIMENSION = 512
NUM_SPEAKERS = 4
SPEAKER_CACHE_LENGTH = 188
SPEAKER_CACHE_UPDATE_PERIOD = 124
FRAME_DURATION = 0.08

SIL_THRESHOLD = 0.2
WINDOW = 11
ONSET_THRESHOLD = 0.641
OFFSET_THRESHOLD = 0.561
PAD_ONSET = 0.229
PAD_OFFSET = 0.079
MIN_DUR_ON = 0.511
MIN_DUR_OFF = 0.296

F_MIN = 0.0
F_SP = 200.0 / 3.0
MIN_LOG_HZ = 1000.0
MIN_LOG_MEL = (1000.0 - 0.0) / (200.0 / 3.0)
LOG_STEP = math.log(6.4) / 27.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Sortformer diarization RTF for the NeMo reference and/or the "
            "exported ONNX graph on CPU or CUDA."
        )
    )
    parser.add_argument("--audio", type=Path, required=True, help="16 kHz mono benchmark audio.")
    parser.add_argument("--runtime", choices=("nemo", "onnx", "both"), default="both")
    parser.add_argument("--nemo", type=Path, default=None, help="Path to diar_streaming_sortformer_4spk-v2.1.nemo")
    parser.add_argument("--onnx", type=Path, default=None, help="Path to diar_streaming_sortformer_4spk-v2.1.onnx")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--max-audio-seconds", type=float, default=None)
    parser.add_argument("--ort-provider", choices=("cuda", "tensorrt"), default="cuda")
    parser.add_argument("--ort-cuda-max-workspace", action="store_true")
    parser.add_argument("--ort-use-tf32", action="store_true")
    parser.add_argument("--ort-profile", type=Path, default=None, help="Write ONNX Runtime JSON profile when benchmarking ONNX.")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def hz_to_mel_slaney(hz: float) -> float:
    if hz >= MIN_LOG_HZ:
        return MIN_LOG_MEL + math.log(hz / MIN_LOG_HZ) / LOG_STEP
    return (hz - F_MIN) / F_SP


def mel_to_hz_slaney(mel: float) -> float:
    if mel >= MIN_LOG_MEL:
        return MIN_LOG_HZ * math.exp(LOG_STEP * (mel - MIN_LOG_MEL))
    return F_MIN + F_SP * mel


def create_mel_filterbank() -> np.ndarray:
    freq_bins = NFft // 2 + 1
    fft_freqs = np.arange(freq_bins, dtype=np.float64) * SAMPLE_RATE / NFft
    fmin_mel = hz_to_mel_slaney(0.0)
    fmax_mel = hz_to_mel_slaney(SAMPLE_RATE / 2.0)
    mel_points = np.linspace(fmin_mel, fmax_mel, NMELS + 2, dtype=np.float64)
    mel_hz = np.asarray([mel_to_hz_slaney(point) for point in mel_points], dtype=np.float64)
    fdiff = np.diff(mel_hz)

    fb = np.zeros((NMELS, freq_bins), dtype=np.float32)
    for i in range(NMELS):
        lower = (fft_freqs - mel_hz[i]) / fdiff[i]
        upper = (mel_hz[i + 2] - fft_freqs) / fdiff[i + 1]
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
        fb[i] *= 2.0 / (mel_hz[i + 2] - mel_hz[i])
    return fb


MEL_FILTERBANK = create_mel_filterbank()
FFT_WINDOW = np.zeros(NFft, dtype=np.float32)
FFT_WINDOW[(NFft - WIN_LENGTH) // 2: (NFft - WIN_LENGTH) // 2 + WIN_LENGTH] = np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float32)


def load_audio(path: Path, max_audio_seconds: float | None) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if int(sample_rate) != SAMPLE_RATE:
        raise SystemExit(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz for {path}.")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if max_audio_seconds is not None:
        audio = audio[: int(max_audio_seconds * SAMPLE_RATE)]
    return np.asarray(audio, dtype=np.float32, order="C")


def log_mel_spectrogram(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32, order="C")
    if signal.size == 0:
        return np.zeros((1, 0, NMELS), dtype=np.float32)

    emphasized = np.empty_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - PREEMPH * signal[:-1]

    padded = np.pad(emphasized, (NFft // 2, NFft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, NFft)[::HOP_LENGTH]
    windowed = frames * FFT_WINDOW
    stft = np.fft.rfft(windowed, n=NFft, axis=-1)
    power = (stft.real * stft.real + stft.imag * stft.imag).astype(np.float32, copy=False)
    mel = power @ MEL_FILTERBANK.T
    return np.log(mel + LOG_ZERO_GUARD).astype(np.float32, copy=False)[None, :, :]


def median_filter(preds: np.ndarray) -> np.ndarray:
    total_frames, num_speakers = preds.shape
    filtered = np.empty_like(preds)
    half = WINDOW // 2
    for spk in range(num_speakers):
        for t in range(total_frames):
            start = max(t - half, 0)
            end = min(t + half + 1, total_frames)
            filtered[t, spk] = np.median(preds[start:end, spk])
    return filtered


@dataclass
class RunMetrics:
    feature_seconds: float
    inference_seconds: float
    postprocess_seconds: float
    total_seconds: float
    num_segments: int


@dataclass
class BenchmarkSummary:
    runtime: str
    device: str
    audio_seconds: float
    warmup_runs: int
    timed_runs: int
    session_load_seconds: float
    avg_feature_seconds: float
    avg_inference_seconds: float
    avg_postprocess_seconds: float
    avg_total_seconds: float
    feature_rtf: float
    inference_rtf: float
    postprocess_rtf: float
    total_rtf: float
    avg_num_segments: float


class SortformerPipelineBase:
    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        self._spkcache = np.zeros((1, 0, EMBEDDING_DIMENSION), dtype=np.float32)
        self._spkcache_preds: np.ndarray | None = None
        self._fifo = np.zeros((1, 0, EMBEDDING_DIMENSION), dtype=np.float32)
        self._fifo_preds = np.zeros((1, 0, NUM_SPEAKERS), dtype=np.float32)
        self._mean_sil_emb = np.zeros((EMBEDDING_DIMENSION,), dtype=np.float32)
        self._n_sil_frames = 0

    def synchronize(self) -> None:
        return None

    def infer_chunk(self, chunk: np.ndarray, current_len: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update_silence_profile(self, embs: np.ndarray, preds: np.ndarray) -> None:
        if embs.shape[1] == 0:
            return
        prob_sum = preds[0].sum(axis=1)
        sil_rows = np.nonzero(prob_sum < SIL_THRESHOLD)[0]
        for idx in sil_rows:
            self._n_sil_frames += 1
            old_sum = self._mean_sil_emb * (self._n_sil_frames - 1)
            self._mean_sil_emb = (old_sum + embs[0, idx]) / self._n_sil_frames

    def speaker_quality_scores(self, preds2d: np.ndarray, min_pos_per_spk: int) -> np.ndarray:
        clipped_p = np.maximum(preds2d, 0.25)
        clipped_q = np.maximum(1.0 - preds2d, 0.25)
        scores = np.log(clipped_p) - np.log(clipped_q)
        scores += np.sum(np.log(clipped_q), axis=1, keepdims=True) - math.log(math.sqrt(2.0))

        pos_count = np.sum(scores > 0.0, axis=0)
        mask_neg = preds2d <= 0.5
        scores[mask_neg] = -np.inf
        for spk in range(NUM_SPEAKERS):
            if pos_count[spk] >= min_pos_per_spk:
                mask = (~mask_neg[:, spk]) & (scores[:, spk] <= 0.0)
                scores[mask, spk] = -np.inf
        return scores

    @staticmethod
    def boost(scores: np.ndarray, n_boost_per_spk: int, scale_factor: float) -> None:
        log_half = 0.5 * math.log(2.0)
        total_frames = scores.shape[0]
        for spk in range(scores.shape[1]):
            order = np.argsort(scores[:, spk])[::-1]
            for t_idx in order[: min(n_boost_per_spk, total_frames)]:
                if not np.isneginf(scores[t_idx, spk]):
                    scores[t_idx, spk] -= scale_factor * log_half

    def compress_cache(self) -> None:
        if self._spkcache_preds is None:
            return

        preds2d = self._spkcache_preds[0]
        total_frames = preds2d.shape[0]
        cache_per_spk = SPEAKER_CACHE_LENGTH // NUM_SPEAKERS - 3
        strong_boost = int(cache_per_spk * 0.75)
        weak_boost = int(cache_per_spk * 1.5)
        min_pos_per_spk = int(cache_per_spk * 0.5)

        scores = self.speaker_quality_scores(preds2d, min_pos_per_spk)
        self.boost(scores, strong_boost, 2.0)
        self.boost(scores, weak_boost, 1.0)

        sil_rows = 3 * NUM_SPEAKERS
        ext_scores = np.full((total_frames + sil_rows, NUM_SPEAKERS), -np.inf, dtype=np.float32)
        ext_scores[:total_frames] = scores

        flat: list[tuple[float, int, int]] = []
        for t_idx in range(ext_scores.shape[0]):
            for s_idx in range(NUM_SPEAKERS):
                flat.append((float(ext_scores[t_idx, s_idx]), t_idx, s_idx))

        flat.sort(key=lambda item: item[0], reverse=True)
        selected = sorted(flat[:SPEAKER_CACHE_LENGTH], key=lambda item: (item[2], item[1]))

        new_embs = np.zeros((1, SPEAKER_CACHE_LENGTH, EMBEDDING_DIMENSION), dtype=np.float32)
        new_preds = np.zeros((1, SPEAKER_CACHE_LENGTH, NUM_SPEAKERS), dtype=np.float32)
        for i, (_, t_idx, _) in enumerate(selected):
            if t_idx >= total_frames:
                new_embs[0, i] = self._mean_sil_emb
            else:
                new_embs[0, i] = self._spkcache[0, t_idx]
                new_preds[0, i] = self._spkcache_preds[0, t_idx]

        self._spkcache = new_embs
        self._spkcache_preds = new_preds

    def process_chunk(self, idx: int, chunk_stride: int, total_frames: int, mel_spec: np.ndarray) -> tuple[np.ndarray, float]:
        start = idx * chunk_stride
        end = min(start + chunk_stride, total_frames)
        current_len = end - start
        n_mel_frames = mel_spec.shape[1]

        chunk = np.zeros((1, chunk_stride, NMELS), dtype=np.float32)
        if current_len > 0:
            chunk[0, :current_len] = mel_spec[0, start:min(end, n_mel_frames)]

        self.synchronize()
        t0 = time.perf_counter()
        preds, embs = self.infer_chunk(chunk, current_len)
        self.synchronize()
        inference_seconds = time.perf_counter() - t0

        cache_t = self._spkcache.shape[1]
        fifo_t = self._fifo.shape[1]
        pred_t_out = preds.shape[1]
        emb_t_out = embs.shape[1]
        valid_frames = (current_len + SUBSAMPLING - 1) // SUBSAMPLING

        fp_start = cache_t
        fp_end = min(fp_start + fifo_t, pred_t_out)
        cp_start = cache_t + fifo_t
        cp_end = min(cp_start + valid_frames, pred_t_out)

        fp = preds[:, fp_start:fp_end]
        chunk_preds = preds[:, cp_start:cp_end]
        chunk_embs = embs[:, : min(valid_frames, emb_t_out)]

        self._fifo = np.concatenate([self._fifo, chunk_embs], axis=1)
        self._fifo_preds = np.concatenate([self._fifo_preds, fp if fp.shape[1] > 0 else chunk_preds], axis=1)

        new_fifo_t = self._fifo.shape[1]
        if new_fifo_t > FIFO_LENGTH:
            pop_len = SPEAKER_CACHE_UPDATE_PERIOD
            pop_len = max(pop_len, (new_fifo_t - FIFO_LENGTH) + new_fifo_t)
            pop_len = min(pop_len, new_fifo_t)

            pop_embs = self._fifo[:, :pop_len]
            pop_preds = self._fifo_preds[:, :pop_len]
            self.update_silence_profile(pop_embs, pop_preds)

            self._fifo = self._fifo[:, pop_len:]
            self._fifo_preds = self._fifo_preds[:, pop_len:]
            self._spkcache = np.concatenate([self._spkcache, pop_embs], axis=1)
            self._spkcache_preds = pop_preds if self._spkcache_preds is None else np.concatenate([self._spkcache_preds, pop_preds], axis=1)

            if self._spkcache.shape[1] > SPEAKER_CACHE_LENGTH:
                self.compress_cache()

        return chunk_preds[0], inference_seconds

    def filter_preds(self, all_preds: list[np.ndarray]) -> np.ndarray:
        if not all_preds:
            return np.zeros((0, NUM_SPEAKERS), dtype=np.float32)
        preds = np.concatenate(all_preds, axis=0)
        return median_filter(preds)

    def binarize_to_segments(self, med_filtered: np.ndarray) -> list[tuple[float, float, str]]:
        num_pred_frames = med_filtered.shape[0]
        all_segments: list[tuple[float, float, str]] = []
        for spk in range(NUM_SPEAKERS):
            in_seg = False
            seg_start = 0
            temp: list[tuple[float, float]] = []
            for t_idx in range(num_pred_frames):
                prob = float(med_filtered[t_idx, spk])
                if prob >= ONSET_THRESHOLD and not in_seg:
                    in_seg = True
                    seg_start = t_idx
                elif prob < OFFSET_THRESHOLD and in_seg:
                    in_seg = False
                    start_s = max(seg_start * FRAME_DURATION - PAD_ONSET, 0.0)
                    end_s = t_idx * FRAME_DURATION + PAD_OFFSET
                    if end_s - start_s >= MIN_DUR_ON:
                        temp.append((start_s, end_s))
            if in_seg:
                start_s = max(seg_start * FRAME_DURATION - PAD_ONSET, 0.0)
                end_s = num_pred_frames * FRAME_DURATION + PAD_OFFSET
                if end_s - start_s >= MIN_DUR_ON:
                    temp.append((start_s, end_s))

            merged: list[tuple[float, float]] = []
            for seg in temp:
                if not merged:
                    merged.append(seg)
                else:
                    prev_start, prev_end = merged[-1]
                    if seg[0] - prev_end < MIN_DUR_OFF:
                        merged[-1] = (prev_start, seg[1])
                    else:
                        merged.append(seg)
            for start_s, end_s in merged:
                all_segments.append((start_s, end_s, f"speaker_{spk}"))

        all_segments.sort(key=lambda item: item[0])
        return all_segments

    def diarize(self, mel_spec: np.ndarray) -> RunMetrics:
        t_feature = 0.0
        total_frames = mel_spec.shape[1]
        chunk_stride = CHUNK_LENGTH * SUBSAMPLING
        num_chunks = (total_frames + chunk_stride - 1) // chunk_stride if total_frames > 0 else 0

        all_preds: list[np.ndarray] = []
        inference_seconds = 0.0
        for idx in range(num_chunks):
            chunk_preds, chunk_time = self.process_chunk(idx, chunk_stride, total_frames, mel_spec)
            all_preds.append(chunk_preds)
            inference_seconds += chunk_time

        t0 = time.perf_counter()
        med_filtered = self.filter_preds(all_preds)
        segments = self.binarize_to_segments(med_filtered)
        postprocess_seconds = time.perf_counter() - t0
        return RunMetrics(
            feature_seconds=t_feature,
            inference_seconds=inference_seconds,
            postprocess_seconds=postprocess_seconds,
            total_seconds=inference_seconds + postprocess_seconds,
            num_segments=len(segments),
        )


class OnnxSortformerPipeline(SortformerPipelineBase):
    def __init__(
        self,
        model_path: Path,
        device: str,
        gpu_id: int,
        *,
        provider_kind: str = "cuda",
        cuda_max_workspace: bool = False,
        use_tf32: bool = False,
        profile_path: Path | None = None,
    ) -> None:
        import onnxruntime as ort

        t0 = time.perf_counter()
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.profile_path = profile_path
        self._profiling_enabled = profile_path is not None
        if self._profiling_enabled:
            session_options.enable_profiling = True
            session_options.profile_file_prefix = profile_path.stem
        available = set(ort.get_available_providers())
        if device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise SystemExit(f"CUDAExecutionProvider is not available. Providers: {sorted(available)}")
            cuda_options: dict[str, Any] = {"device_id": gpu_id}
            if cuda_max_workspace:
                cuda_options["cudnn_conv_use_max_workspace"] = "1"
            if use_tf32:
                cuda_options["use_tf32"] = "1"

            if provider_kind == "tensorrt":
                providers = [
                    ("TensorrtExecutionProvider", {"device_id": gpu_id, "trt_engine_cache_enable": "True", "trt_fp16_enable": "True"}),
                    ("CUDAExecutionProvider", cuda_options),
                    "CPUExecutionProvider",
                ]
            else:
                providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
        self.providers = self._session.get_providers()
        self._input_shapes = {
            item.name: list(getattr(item, "shape", []))
            for item in self._session.get_inputs()
        }
        self.session_load_seconds = time.perf_counter() - t0
        super().__init__()

    def _pad_to_static_shape(self, name: str, value: np.ndarray) -> np.ndarray:
        shape = self._input_shapes.get(name)
        if not shape:
            return value

        target = list(value.shape)
        changed = False
        for idx, dim in enumerate(shape):
            if isinstance(dim, int) and dim >= 0:
                if idx >= len(target):
                    continue
                if target[idx] > dim:
                    slicer = [slice(None)] * value.ndim
                    slicer[idx] = slice(0, dim)
                    value = value[tuple(slicer)]
                    target[idx] = dim
                    changed = True
                elif target[idx] < dim:
                    pad_width = [(0, 0)] * value.ndim
                    pad_width[idx] = (0, dim - target[idx])
                    value = np.pad(value, pad_width, mode="constant")
                    target[idx] = dim
                    changed = True
        return np.ascontiguousarray(value) if changed else value

    def finalize_profile(self) -> Path | None:
        if not self._profiling_enabled:
            return None
        raw_path = Path(self._session.end_profiling())
        if self.profile_path is not None:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            if raw_path.resolve() != self.profile_path.resolve():
                raw_path.replace(self.profile_path)
            return self.profile_path
        return raw_path

    def infer_chunk(self, chunk: np.ndarray, current_len: int) -> tuple[np.ndarray, np.ndarray]:
        chunk_value = self._pad_to_static_shape("chunk", chunk)
        spkcache_value = self._pad_to_static_shape("spkcache", self._spkcache)
        fifo_value = self._pad_to_static_shape("fifo", self._fifo)
        outputs = self._session.run(
            ["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs"],
            {
                "chunk": chunk_value,
                "chunk_lengths": np.asarray([current_len], dtype=np.int64),
                "spkcache": spkcache_value,
                "spkcache_lengths": np.asarray([self._spkcache.shape[1]], dtype=np.int64),
                "fifo": fifo_value,
                "fifo_lengths": np.asarray([self._fifo.shape[1]], dtype=np.int64),
            },
        )
        preds = np.asarray(outputs[0], dtype=np.float32)
        embs = np.asarray(outputs[1], dtype=np.float32)
        return preds, embs


class NemoSortformerPipeline(SortformerPipelineBase):
    def __init__(self, nemo_path: Path, device: str, gpu_id: int) -> None:
        torch, nn, SortformerEncLabelModel = import_dependencies()
        self._torch = torch
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested for NeMo benchmark but torch.cuda.is_available() is false.")

        torch_device = torch.device(f"cuda:{gpu_id}" if resolved_device == "cuda" else "cpu")
        t0 = time.perf_counter()
        model = SortformerEncLabelModel.restore_from(str(nemo_path), map_location=torch_device)
        model.eval()
        model.to(torch_device)
        wrapper = build_wrapper(torch, nn, model)
        wrapper.eval()
        wrapper.to(torch_device)
        self._device = torch_device
        self._wrapper = wrapper
        self.device = resolved_device
        self.session_load_seconds = time.perf_counter() - t0
        super().__init__()

    def synchronize(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.synchronize(self._device)

    def infer_chunk(self, chunk: np.ndarray, current_len: int) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        with torch.inference_mode():
            preds, embs, _ = self._wrapper(
                torch.from_numpy(chunk).to(self._device),
                torch.tensor([current_len], dtype=torch.int64, device=self._device),
                torch.from_numpy(self._spkcache).to(self._device),
                torch.tensor([self._spkcache.shape[1]], dtype=torch.int64, device=self._device),
                torch.from_numpy(self._fifo).to(self._device),
                torch.tensor([self._fifo.shape[1]], dtype=torch.int64, device=self._device),
            )
        return preds.detach().cpu().numpy().astype(np.float32), embs.detach().cpu().numpy().astype(np.float32)


def benchmark_runtime(name: str, pipeline: SortformerPipelineBase, audio: np.ndarray, audio_seconds: float, warmup_runs: int, timed_runs: int) -> BenchmarkSummary:
    def run_once() -> RunMetrics:
        pipeline.reset_state()
        t0 = time.perf_counter()
        mel_spec = log_mel_spectrogram(audio)
        feature_seconds = time.perf_counter() - t0
        metrics = pipeline.diarize(mel_spec)
        metrics.feature_seconds = feature_seconds
        metrics.total_seconds += feature_seconds
        return metrics

    for _ in range(warmup_runs):
        _ = run_once()

    runs = [run_once() for _ in range(timed_runs)]
    avg_feature = statistics.mean(run.feature_seconds for run in runs)
    avg_inference = statistics.mean(run.inference_seconds for run in runs)
    avg_post = statistics.mean(run.postprocess_seconds for run in runs)
    avg_total = statistics.mean(run.total_seconds for run in runs)
    avg_segments = statistics.mean(run.num_segments for run in runs)
    runtime_device = getattr(pipeline, "device", getattr(pipeline, "providers", ["cpu"])[0])
    return BenchmarkSummary(
        runtime=name,
        device=str(runtime_device),
        audio_seconds=audio_seconds,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        session_load_seconds=getattr(pipeline, "session_load_seconds", 0.0),
        avg_feature_seconds=avg_feature,
        avg_inference_seconds=avg_inference,
        avg_postprocess_seconds=avg_post,
        avg_total_seconds=avg_total,
        feature_rtf=avg_feature / audio_seconds if audio_seconds > 0 else float("nan"),
        inference_rtf=avg_inference / audio_seconds if audio_seconds > 0 else float("nan"),
        postprocess_rtf=avg_post / audio_seconds if audio_seconds > 0 else float("nan"),
        total_rtf=avg_total / audio_seconds if audio_seconds > 0 else float("nan"),
        avg_num_segments=avg_segments,
    )


def print_summary(summary: BenchmarkSummary) -> None:
    print(f"\n[{summary.runtime}]")
    print(f"  device            : {summary.device}")
    print(f"  audio_seconds     : {summary.audio_seconds:.2f}")
    print(f"  session_load_s    : {summary.session_load_seconds:.3f}")
    print(f"  avg_feature_s     : {summary.avg_feature_seconds:.3f}  (RTF {summary.feature_rtf:.4f})")
    print(f"  avg_inference_s   : {summary.avg_inference_seconds:.3f}  (RTF {summary.inference_rtf:.4f})")
    print(f"  avg_postprocess_s : {summary.avg_postprocess_seconds:.3f}  (RTF {summary.postprocess_rtf:.4f})")
    print(f"  avg_total_s       : {summary.avg_total_seconds:.3f}  (RTF {summary.total_rtf:.4f})")
    print(f"  avg_segments      : {summary.avg_num_segments:.1f}")


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    runtime_names = ["nemo", "onnx"] if args.runtime == "both" else [args.runtime]
    if "nemo" in runtime_names and args.nemo is None:
        raise SystemExit("--nemo is required when benchmarking the NeMo reference.")
    if "onnx" in runtime_names and args.onnx is None:
        raise SystemExit("--onnx is required when benchmarking the ONNX export.")

    audio = load_audio(audio_path, args.max_audio_seconds)
    audio_seconds = len(audio) / SAMPLE_RATE
    print(f"Audio: {audio_path} ({audio_seconds:.2f}s)")

    if args.device == "auto":
        requested_device = "cuda"
    else:
        requested_device = args.device

    summaries: list[BenchmarkSummary] = []
    for runtime_name in runtime_names:
        if runtime_name == "nemo":
            pipeline = NemoSortformerPipeline(args.nemo.expanduser().resolve(), requested_device, args.gpu_id)
        else:
            pipeline = OnnxSortformerPipeline(
                args.onnx.expanduser().resolve(),
                requested_device,
                args.gpu_id,
                provider_kind=args.ort_provider,
                cuda_max_workspace=args.ort_cuda_max_workspace,
                use_tf32=args.ort_use_tf32,
                profile_path=args.ort_profile.expanduser().resolve() if args.ort_profile is not None else None,
            )
        summary = benchmark_runtime(runtime_name, pipeline, audio, audio_seconds, args.warmup_runs, args.timed_runs)
        summaries.append(summary)
        print_summary(summary)
        if runtime_name == "onnx" and isinstance(pipeline, OnnxSortformerPipeline):
            profile = pipeline.finalize_profile()
            if profile is not None:
                print(f"  ort_profile       : {profile}")

    payload = {
        "audio": str(audio_path),
        "audio_seconds": audio_seconds,
        "device": requested_device,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "results": [asdict(item) for item in summaries],
    }
    if args.json_out is not None:
        json_path = args.json_out.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON written to {json_path}")


if __name__ == "__main__":
    main()
