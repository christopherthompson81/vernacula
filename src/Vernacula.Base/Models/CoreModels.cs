namespace Vernacula.Base.Models;

// CoreML / WebGpu are macOS-only accelerator paths (the osx-arm64 ORT build
// reports CoreMLExecutionProvider + WebGpuExecutionProvider). Appended at the
// end so previously-serialized ordinals keep their meaning.
public enum ExecutionProvider { Auto, Cuda, DirectML, Cpu, CoreML, WebGpu }
public enum ModelPrecision    { Int8, Fp32 }
// Settings persist this as its ordinal too: append new values, never insert.
public enum SegmentationMode  { SileroVad, Sortformer, DiariZen, VibeVoiceBuiltin, Nemotron3Diarization }
