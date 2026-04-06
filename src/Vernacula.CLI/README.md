# Vernacula Full-Pipeline Benchmark

Compares **CUDA**, **DirectML**, and **CPU** execution providers by running the real production pipeline:
**Sortformer diarization → per-segment Parakeet ASR**.

This mirrors exactly what the main app does, so the RTF result is directly meaningful.
Both the Sortformer and the ASR models (encoder + decoder) use the selected EP.
The preprocessor remains on CPU. The benchmark can optionally batch diarized segments for the
Parakeet encoder pass, which is the main batching lever available with the current exports.

## Why not pass the whole file to the encoder?

The Parakeet encoder's self-attention positional encoding supports a maximum of ~9999 frames
(~100 seconds). Passing a longer recording crashes with a tensor broadcast error. The main app
solves this by diarizing first and feeding short per-speaker segments to the ASR model — so
this benchmark does the same.

## Files

| File | Description |
|---|---|
| `Vernacula.CLI.csproj` | Console project; `EP` build property selects the OnnxRuntime package |
| `Program.cs` | Main benchmark: arg parsing, session loading, pipeline loop, reporting |
| `BenchmarkSortformer.cs` | Adapted `SortformerStreamer`; EP-configurable, `ResetState()` for run isolation |
| `BenchmarkAudioUtils.cs` | Self-contained mel spectrogram pipeline (no WPF/ClosedXML deps) |

## Requirements

- .NET 10 SDK, x64
- The Parakeet model directory containing:
  - `nemo128.onnx`
  - `encoder-model.onnx`
  - `decoder_joint-model.onnx`
  - `diar_streaming_sortformer_4spk-v2.1.onnx`
  - `vocab.txt`
- For CUDA: an NVIDIA GPU with CUDA installed
- For DirectML: any DirectX 12-capable GPU (NVIDIA, AMD, Intel)

## Build and run

From the `Benchmark/` directory:

**CUDA (NVIDIA)**
```
dotnet build -c Release -p:EP=Cuda -p:Platform=x64
dotnet run   -c Release -p:EP=Cuda -p:Platform=x64 -- --audio <file> --model <model-dir>
```

**DirectML (any DX12 GPU)**
```
dotnet build -c Release -p:EP=DirectML -p:Platform=x64
dotnet run   -c Release -p:EP=DirectML -p:Platform=x64 -- --audio <file> --model <model-dir>
```

**CPU (baseline)**
```
dotnet build -c Release -p:EP=Cpu -p:Platform=x64
dotnet run   -c Release -p:EP=Cpu -p:Platform=x64 -- --audio <file> --model <model-dir>
```

### All arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--audio <path>` | Yes | — | Audio file (WAV, MP3, FLAC, etc.) |
| `--model <path>` | Yes | — | Directory containing the ONNX models and vocab |
| `--warmup <n>` | No | 3 | Warmup runs (discarded, used to warm up GPU) |
| `--runs <n>` | No | 10 | Timed runs included in results |
| `--max-duration <s>` | No | (none) | Clip audio to first `s` seconds for faster iteration |
| `--encoder-batch-size <n>` | No | 1 | Batch diarized segments together for the encoder pass |
| `--length-bucket-ms <ms>` | No | 1000 | Group similar-length segments before batching |

### Example

```
dotnet run -c Release -p:EP=Cuda -p:Platform=x64 -- ^
  --audio "C:\audio\sample.wav" --model "C:\models\vernacula" --warmup 3 --runs 10
```

For a quick smoke test on a long file:
```
dotnet run -c Release -p:EP=Cuda -p:Platform=x64 -- ^
  --audio "C:\audio\long.wav" --model "C:\models\vernacula" --max-duration 120 --runs 5
```

To test encoder batching on diarized segments:
```
dotnet run -c Release -p:EP=Cuda -p:Platform=x64 -- ^
  --audio "C:\audio\sample.wav" --model "C:\models\vernacula" ^
  --encoder-batch-size 8 --length-bucket-ms 1000 --runs 10
```

## Output

Per-run progress is printed inline, then a summary table:

```
  Run     1/10  diar=  823ms  asr= 1204ms  total=  2027ms  segs= 18
  Run     2/10  diar=  791ms  asr= 1188ms  total=  1979ms  segs= 18
  ...

┌────────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Phase                  │   Min ms │   Avg ms │   Med ms │   Max ms │
├────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Diarization  (CUDA)    │      ... │      ... │      ... │      ... │
│ ASR          (CUDA)    │      ... │      ... │      ... │      ... │
│ Total pipeline         │      ... │      ... │      ... │      ... │
└────────────────────────┴──────────┴──────────┴──────────┴──────────┘

Audio duration   : 300.00s
Avg total time   : 2010ms
Real-time factor : 0.0067  (lower = faster; <1.0 = faster than real-time)
Session load     : 8423ms  (one-time cost, not in RTF)
```

### Phases explained

| Phase | Notes |
|---|---|
| **Diarization** | Full Sortformer streaming pipeline: chunk processing → median filter → segment binarization |
| **ASR** | Sum of all per-segment: preprocessor (CPU) + encoder (selected EP, optionally batched) + decoder (selected EP) |
| **Total pipeline** | Diarization + ASR end-to-end |
| **RTF** | `total_pipeline_time / audio_duration` — lower is better; < 1.0 = faster than real-time |
| **Session load** | One-time model loading cost; excluded from RTF since it amortises over the full file |

> The Sortformer session is **loaded once** and its streaming state is reset between runs via
> `ResetState()`, so session load time is not inflated by the number of runs.

## Batching Notes

- The current `encoder-model.onnx` export has dynamic batch and sequence dimensions, so the benchmark
  can batch diarized segments together for the encoder pass.
- The current `nemo128.onnx` preprocessor export is still effectively batch-1 on this toolchain, so
  preprocessing remains per-segment even when encoder batching is enabled.
- This means the benchmark is useful for measuring the real benefit of post-diarization bucketing and
  encoder batching without pretending that the whole ASR stack is fully batched yet.
