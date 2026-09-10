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

## Execution provider (`-p:EP=`)

`EP` selects which ONNX Runtime package is restored: `Cuda`, `Cpu` or `DirectML`.

**It defaults to whatever the architecture can actually run** — `Cuda` on x64, `Cpu`
everywhere else — so on Apple Silicon (and any other arm64 host) a plain `dotnet build` or
`dotnet run` is correct and needs no flag. There is no CUDA build of ONNX Runtime for
arm64, so `Cuda` was never a possible default there; it only ever produced a build error
telling you to pass the one value the build could work out for itself.

Pass `EP` explicitly to override: a CPU-only build on an x64 machine, or DirectML on
Windows. `-p:EP=Cuda` on arm64 is still rejected, with an error saying why.

⚠ On macOS, `Cpu` is also the build that carries the **CoreML and WebGPU** natives — the
plain `osx-arm64` ONNX Runtime package is the one that ships them. "CPU" names the package,
not the providers you end up with.

## Vernacula.CLI

```bash
cd src/Vernacula.CLI

# GPU (CUDA) — the default on x64
dotnet build -c Release -p:EP=Cuda -p:Platform=x64

# CPU only
dotnet build -c Release -p:EP=Cpu -p:Platform=x64

# Apple Silicon: no flag needed
dotnet build -c Release
```

## Vernacula.Avalonia

This project targets **two** frameworks: `net10.0-windows` and `net10.0`. NAudio 3 hands
the Windows audio backend (`WaveOut`, in `NAudio.WinMM`) only to a Windows target
framework, so `net10.0-windows` is the build with native playback and `net10.0` is the
portable one that plays through `ffplay`. `dotnet build` builds both; `dotnet run` and
`dotnet publish` act on one at a time and so need `-f`.

```bash
cd src/Vernacula.Avalonia

# Build (both frameworks)
dotnet build -c Release -p:EP=Cuda -p:Platform=x64

# Or publish as self-contained (recommended for desktop install)
dotnet publish -c Release -f net10.0 -p:EP=Cuda -p:Platform=x64 \
  -r linux-x64 --self-contained true \
  -o ~/apps/vernacula-desktop

# The same on Windows
dotnet publish -c Release -f net10.0-windows -p:EP=Cuda -p:Platform=x64 \
  -r win-x64 --self-contained true \
  -o %USERPROFILE%/apps/vernacula-desktop
```

For a Linux end-user install, the `install.sh` script at the repo root runs a self-contained publish and registers the `.desktop` entry for you — see [Installation](installation.md).

On macOS, `./package-macos.sh` builds `dist/Vernacula.app`: a real bundle, so it gets a
Dock icon, the right name in the menu bar, and a double-click launcher. It defaults to a
self-contained publish, because a framework-dependent bundle launched from Finder inherits
no `PATH` and fails to find .NET on machines where it came from Homebrew.

```bash
./package-macos.sh                      # dist/Vernacula.app
./package-macos.sh --framework-dependent  # smaller; needs .NET where Finder can see it
cp -R dist/Vernacula.app /Applications/  # -R, not -r: -r mangles bundles
```

Running unbundled (`dotnet run`) still shows the right Dock icon — the app sets it at
startup through AppKit, since a bare executable has no bundle to read one from.
