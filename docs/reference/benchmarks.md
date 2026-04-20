# Benchmarks

Throughput and accuracy numbers for the diarization and ASR backends. Throughput figures come from in-house runs; accuracy is reproduced from the model authors' published benchmarks.

## Throughput

10-minute English audio file, fp32 models. RTF < 1.0 = faster than real-time.

| Backend | Hardware | Diarization | ASR | Total | RTF |
|---|---|---|---|---|---|
| Silero VAD | AMD Ryzen 7 7840U | 2.1s | 50.5s | 52.7s | **0.088** |
| Sortformer | AMD Ryzen 7 7840U | 33.2s | 49.2s | 82.4s | **0.137** |
| DiariZen | AMD Ryzen 7 7840U | 502.0s | 55.8s | 557.8s | 0.930 |
| Silero VAD | NVIDIA RTX 3090 | 2.1s | 5.4s | 7.4s | **0.012** |
| Sortformer | NVIDIA RTX 3090 | 16.0s | 5.5s | 21.4s | **0.036** |
| DiariZen | NVIDIA RTX 3090 | 16.8s | 5.4s | 22.2s | **0.037** |

> DiariZen's segmentation and embedding pipeline is heavily GPU-accelerated — CUDA reduces diarization time from 502s to 16.8s (~30×) and brings total runtime in line with Sortformer.

Parakeet TDT beam search (`--parakeet-beam 4`) adds roughly 3–5× to ASR latency. With KenLM fusion at typical weights the additional lookup cost is a few hundred milliseconds per clip (one-time LM load plus microsecond-per-beam-expansion scoring).

## Accuracy (DER)

Diarization Error Rate from published benchmarks. Lower is better.

| Backend | AMI-SDM | VoxConverse | DIHARD III | Source |
|---|---|---|---|---|
| Sortformer v2-stream | 20.6% | 13.9% | 20.2% | [HuggingFace](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) |
| DiariZen-Large | 13.9% | 9.1% | 14.5% | [BUTSpeechFIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen) |

> Benchmarks use different evaluation conditions (collar, overlap handling) — direct cross-model comparison should be treated as indicative only. The independent survey [Benchmarking Diarization Models (2509.26177)](https://arxiv.org/abs/2509.26177) found Sortformer v2-stream and DiariZen among the top open-source performers overall.
