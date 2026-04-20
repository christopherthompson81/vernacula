# Building from source

All projects are built with `dotnet build`. The `EP` property selects the ONNX Runtime execution provider.

## Execution providers

| `-p:EP=` | Hardware | Notes |
|---|---|---|
| `Cuda` | NVIDIA GPU | Default. Requires CUDA Toolkit. |
| `Cpu` | Any CPU | No GPU required. Slower. |
| `DirectML` | Windows only | Uses DirectX 12; works on AMD/Intel/NVIDIA. |

See [Installation](installation.md) for the underlying runtime prerequisites (CUDA Toolkit, FFmpeg, etc.).

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
