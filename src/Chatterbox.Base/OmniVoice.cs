using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Chatterbox.Base;

/// <summary>
/// Low-level inference over the three Vernacula-exported OmniVoice ONNX graphs
/// (see scripts/omnivoice_export, docs/omnivoice_onnx_investigation.md):
///
///   omnivoice_transformer.onnx  the diffusion denoiser (embeds + Qwen3 + audio_heads),
///                               run once per diffusion step over the CFG cond+uncond batch
///   higgs_encoder.onnx          reference wav @24k -> discrete audio codes (voice cloning)
///   higgs_decoder.onnx          audio codes -> 24 kHz waveform (the vocoder)
///
/// This class only runs the graphs; the diffusion loop, tokenizer, duration estimate,
/// CFG/scoring, and text prep live in the higher-level OmniVoice pipeline. The
/// transformer's embedding lookup is inside the graph, so callers supply only
/// input_ids / audio_mask / attention_mask.
///
/// Parity (CPU / full fp32) is validated against the Phase-1 Python capture by
/// OmniVoiceGraphParityTests. Not thread-safe (matches the underlying sessions).
/// </summary>
public sealed class OmniVoice : IDisposable
{
    public const int SampleRate = 24_000;
    public const int NumCodebooks = 8;
    public const int AudioVocabSize = 1025;
    public const int AudioMaskId = 1024;
    /// <summary>Higgs codec hop length: encoder input length must be a multiple of this.</summary>
    public const int HopLength = 960;

    public const string TransformerFile = "omnivoice_transformer.onnx";
    public const string EncoderFile = "higgs_encoder.onnx";
    public const string DecoderFile = "higgs_decoder.onnx";

    private readonly InferenceSession _transformer;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;

    public OmniVoice(string onnxDir, ExecutionProvider ep, SessionLoadObserver? onLoad = null)
    {
        _transformer = SessionLoader.LoadAndReport(Path.Combine(onnxDir, TransformerFile), ep, onLoad);
        _encoder = SessionLoader.LoadAndReport(Path.Combine(onnxDir, EncoderFile), ep, onLoad);
        _decoder = SessionLoader.LoadAndReport(Path.Combine(onnxDir, DecoderFile), ep, onLoad);
    }

    // ── Transformer ────────────────────────────────────────────────────────────

    /// <summary>
    /// Run the diffusion denoiser. <paramref name="inputIds"/> is flattened
    /// [2B, 8, S] int64; <paramref name="audioMask"/> is [2B, S] bool;
    /// <paramref name="attentionMask"/> is [2B, 1, S, S] bool. Returns logits
    /// flattened row-major as [2B, 8, S, 1025] float32.
    /// </summary>
    public float[] RunTransformer(long[] inputIds, bool[] audioMask, bool[] attentionMask,
                                  int twoB, int seq)
    {
        var idsT = new DenseTensor<long>(inputIds, [twoB, NumCodebooks, seq]);
        var maskT = new DenseTensor<bool>(audioMask, [twoB, seq]);
        var attnT = new DenseTensor<bool>(attentionMask, [twoB, 1, seq, seq]);

        using var outputs = _transformer.Run([
            NamedOnnxValue.CreateFromTensor("input_ids", idsT),
            NamedOnnxValue.CreateFromTensor("audio_mask", maskT),
            NamedOnnxValue.CreateFromTensor("attention_mask", attnT),
        ]);
        return outputs.First(v => v.Name == "logits").AsTensor<float>().ToArray();
    }

    // ── Codec ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Encode a 24 kHz mono waveform (length must be a multiple of <see cref="HopLength"/>)
    /// into discrete audio codes. Returns codes [8, Tc].
    /// </summary>
    public long[,] EncodeAudio(float[] wav24k)
    {
        var inT = new DenseTensor<float>(wav24k, [1, 1, wav24k.Length]);
        using var outputs = _encoder.Run([NamedOnnxValue.CreateFromTensor("input_values", inT)]);
        var codes = outputs.First(v => v.Name == "audio_codes").AsTensor<long>();
        int tc = codes.Dimensions[2];
        var result = new long[NumCodebooks, tc];
        for (int c = 0; c < NumCodebooks; c++)
            for (int t = 0; t < tc; t++)
                result[c, t] = codes[0, c, t];
        return result;
    }

    /// <summary>Decode audio codes [8, Tc] into a 24 kHz mono waveform.</summary>
    public float[] DecodeAudio(long[,] codes)
    {
        int cb = codes.GetLength(0), tc = codes.GetLength(1);
        var flat = new long[cb * tc];
        for (int c = 0; c < cb; c++)
            for (int t = 0; t < tc; t++)
                flat[c * tc + t] = codes[c, t];
        var inT = new DenseTensor<long>(flat, [1, cb, tc]);
        using var outputs = _decoder.Run([NamedOnnxValue.CreateFromTensor("audio_codes", inT)]);
        return outputs.First(v => v.Name == "audio_values").AsTensor<float>().ToArray();
    }

    /// <summary>
    /// Create a reusable IO-bound runner for the diffusion loop. The transformer is invoked
    /// num_step times at a fixed sequence length, so we bind one set of pinned input/output
    /// buffers once and only mutate their contents per step — avoiding a 12.7 MB managed
    /// logits allocation and tensor rebuild on every step.
    /// </summary>
    public TransformerLoop CreateTransformerLoop(int batch, int seq) => new(_transformer, batch, seq);

    public sealed class TransformerLoop : IDisposable
    {
        private readonly InferenceSession _s;
        private readonly OrtIoBinding _io;
        private readonly RunOptions _run = new();
        private readonly OrtValue _idsV, _amV, _atV, _logV;
        public int Batch { get; }
        public int Seq { get; }

        /// <summary>Mutable input buffers: input_ids [B,8,S], audio_mask [B,S], attention_mask
        /// [B,1,S,S] (all flat). Mutate in place between <see cref="Run"/> calls.</summary>
        public long[] Ids { get; }
        public bool[] AudioMask { get; }
        public bool[] Attn { get; }
        /// <summary>Output logits buffer [B,8,S,1025] flat, overwritten by each Run.</summary>
        public float[] Logits { get; }

        public TransformerLoop(InferenceSession s, int batch, int seq)
        {
            _s = s; Batch = batch; Seq = seq;
            Ids = new long[batch * NumCodebooks * seq];
            AudioMask = new bool[batch * seq];
            Attn = new bool[batch * seq * seq];
            Logits = new float[(long)batch * NumCodebooks * seq * AudioVocabSize];

            var cpu = OrtMemoryInfo.DefaultInstance;
            _idsV = OrtValue.CreateTensorValueFromMemory<long>(cpu, Ids,
                new long[] { batch, NumCodebooks, seq });
            _amV = OrtValue.CreateTensorValueFromMemory<bool>(cpu, AudioMask,
                new long[] { batch, seq });
            _atV = OrtValue.CreateTensorValueFromMemory<bool>(cpu, Attn,
                new long[] { batch, 1, seq, seq });
            _logV = OrtValue.CreateTensorValueFromMemory<float>(cpu, Logits,
                new long[] { batch, NumCodebooks, seq, AudioVocabSize });

            _io = _s.CreateIoBinding();
            _io.BindInput("input_ids", _idsV);
            _io.BindInput("audio_mask", _amV);
            _io.BindInput("attention_mask", _atV);
            _io.BindOutput("logits", _logV);
        }

        /// <summary>Run one step; <see cref="Logits"/> holds the result afterwards.</summary>
        public float[] Run()
        {
            _s.RunWithBinding(_run, _io);
            return Logits;
        }

        public void Dispose()
        {
            _io.Dispose(); _run.Dispose();
            _idsV.Dispose(); _amV.Dispose(); _atV.Dispose(); _logV.Dispose();
        }
    }

    /// <summary>Raw codec encode for parity tests: wav [1,1,samples] -> codes flat [1,8,Tc].</summary>
    public long[] EncodeRaw(float[] wav, out int tc)
    {
        var inT = new DenseTensor<float>(wav, [1, 1, wav.Length]);
        using var outputs = _encoder.Run([NamedOnnxValue.CreateFromTensor("input_values", inT)]);
        var codes = outputs.First(v => v.Name == "audio_codes").AsTensor<long>();
        tc = codes.Dimensions[2];
        return codes.ToArray();
    }

    /// <summary>Raw codec decode for parity tests: codes flat [1,8,Tc] -> wav flat.</summary>
    public float[] DecodeRaw(long[] codes, int tc)
    {
        var inT = new DenseTensor<long>(codes, [1, NumCodebooks, tc]);
        using var outputs = _decoder.Run([NamedOnnxValue.CreateFromTensor("audio_codes", inT)]);
        return outputs.First(v => v.Name == "audio_values").AsTensor<float>().ToArray();
    }

    public void Dispose()
    {
        _transformer.Dispose();
        _encoder.Dispose();
        _decoder.Dispose();
    }
}
