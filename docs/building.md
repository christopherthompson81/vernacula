# Building from source

All projects are built with `dotnet build`. The `EP` property selects the ONNX Runtime execution provider.

## Execution providers

| `-p:EP=` | Hardware | Notes |
|---|---|---|
| `Cuda` | NVIDIA GPU | Default. Requires the **CUDA 13** runtime (ONNX Runtime 1.29 links CUDA 13; CUDA 12 will not load it). |
| `Cpu` | Any CPU | No GPU required. Slower. |
| `DirectML` | Windows only | Uses DirectX 12; works on AMD/Intel/NVIDIA. Pinned to ONNX Runtime 1.24.4 throughout — the newest DirectML release — because the managed assembly has to match the native runtime it ships with. |

See [Installation](installation.md) for the underlying runtime prerequisites (CUDA runtime, FFmpeg, etc.).

### After changing execution provider or ONNX Runtime version

Optimized graphs are cached next to each model as `<model>.opt.<ep>.<hash>.onnx`, and the hash covers
the ONNX Runtime version, so a runtime upgrade leaves the old files behind unused. They are never
cleaned automatically. To reclaim the space (they can run to several GB across a full model set):

```bash
rm -f /path/to/models/*.opt.*
```

The first run after that is slower while the graphs are rebuilt.

## Vernacula.CLI

```bash
cd src/Vernacula.CLI

# GPU (CUDA)
dotnet build -c Release -p:EP=Cuda -p:Platform=x64

# CPU only
dotnet build -c Release -p:EP=Cpu -p:Platform=x64
```

## Vernacula.Avalonia

```bash
cd src/Vernacula.Avalonia

# Build
dotnet build -c Release -p:EP=Cuda -p:Platform=x64

# Or publish as self-contained (recommended for desktop install)
dotnet publish -c Release -p:EP=Cuda -p:Platform=x64 \
  -r linux-x64 --self-contained true \
  -o ~/apps/vernacula-desktop
```

For a Linux end-user install, the `install.sh` script at the repo root runs a self-contained publish and registers the `.desktop` entry for you — see [Installation](installation.md).
