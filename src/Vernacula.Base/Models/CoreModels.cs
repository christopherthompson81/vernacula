namespace Vernacula.Base.Models;

public enum ExecutionProvider { Auto, Cuda, DirectML, Cpu }
public enum ModelPrecision    { Int8, Fp32 }
public enum SegmentationMode  { SileroVad, Sortformer, DiariZen, VibeVoiceBuiltin }
public enum DenoiserMode      { None, DeepFilterNet3 }
