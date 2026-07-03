using System.Text.RegularExpressions;
using Google.Protobuf;
using Microsoft.ML.OnnxRuntime;
using Onnx;

namespace Chatterbox.Base;

/// <summary>
/// Folds the IPA fine-tune diff (a small ipa_diff.onnx: per-module LoRA A/B + the sparse
/// changed embed_tokens rows) onto the BASE omnivoice_transformer.onnx AT LOAD TIME, producing
/// in-memory replacement initializers — no 2.45 GB merged file on disk.
///
/// The base transformer stores weights in external data (.onnx.data). We parse the base graph
/// protobuf (only the ~1.5 MB .onnx, not the data) to get each initializer's byte offset/length
/// and the MatMul-node→weight map (the nodes keep their module path,
/// /model/llm/layers.N/self_attn/q_proj/MatMul, while the weight initializer is generically
/// named). For each Linear we read its weight bytes, add ΔWᵀ = ((B@A)·scale)ᵀ (ONNX weight is
/// (in,out) vs PyTorch (out,in)); embed_tokens is read whole and its changed rows overwritten.
/// The folded tensors are handed to the session via SessionOptions.AddInitializer, so ORT uses
/// them for the named initializers and reads the rest from the base .onnx.data unchanged.
/// </summary>
public sealed class OmniVoiceDiff : IDisposable
{
    private static readonly Regex NodeRe =
        new(@"layers\.(\d+)/(self_attn|mlp)/(\w+_proj)/", RegexOptions.Compiled);

    // OrtValues + their backing arrays must outlive the session that references them.
    private readonly List<OrtValue> _values = new();

    /// <summary>Build folded replacement initializers and register them on
    /// <paramref name="opts"/> via AddInitializer. Keep this instance alive for the session's
    /// lifetime (Dispose after the session).</summary>
    public void ApplyTo(SessionOptions opts, string baseOnnxPath, string diffOnnxPath)
    {
        var baseModel = ModelProto.Parser.ParseFrom(File.ReadAllBytes(baseOnnxPath));
        var diffModel = ModelProto.Parser.ParseFrom(File.ReadAllBytes(diffOnnxPath));
        string dataPath = Path.Combine(Path.GetDirectoryName(baseOnnxPath)!,
            // location is identical across initializers; take it from the first external one
            baseModel.Graph.Initializer.First(i => i.ExternalData.Count > 0)
                .ExternalData.First(e => e.Key == "location").Value);

        float scale = float.Parse(diffModel.MetadataProps.First(p => p.Key == "lora_scale").Value);
        var diff = diffModel.Graph.Initializer.ToDictionary(i => i.Name, i => i);
        var inits = baseModel.Graph.Initializer.ToDictionary(i => i.Name, i => i);

        // map (layer, sub, proj) -> weight-initializer name via the MatMul node names
        var node2w = new Dictionary<(int, string, string), string>();
        foreach (var nd in baseModel.Graph.Node)
        {
            if (nd.OpType != "MatMul") continue;
            var mm = NodeRe.Match(nd.Name);
            if (!mm.Success) continue;
            var w = nd.Input.FirstOrDefault(inits.ContainsKey);
            if (w != null)
                node2w[(int.Parse(mm.Groups[1].Value), mm.Groups[2].Value, mm.Groups[3].Value)] = w;
        }

        using var data = File.OpenRead(dataPath);
        foreach (var ((layer, sub, proj), wname) in node2w)
        {
            string key = $"llm.layers.{layer}.{sub}.{proj}";
            if (!diff.TryGetValue($"{key}.lora_A", out var Ai) ||
                !diff.TryGetValue($"{key}.lora_B", out var Bi)) continue;
            var init = inits[wname];
            (long off, long len) = External(init);
            int inN = (int)init.Dims[0], outN = (int)init.Dims[1];
            float[] W = ReadFloat32(data, off, len);         // (in, out) row-major
            float[] A = ToFloat(Ai);                          // (r, in)
            float[] B = ToFloat(Bi);                          // (out, r)
            int r = (int)Ai.Dims[0];
            // W += ΔWᵀ where ΔW=(B@A)*scale is (out,in), so ΔWᵀ[i,o]=scale*Σ_k A[k,i]B[o,k].
            // Transpose A -> At (in,r) so both operands and the W write are contiguous, and
            // parallelise over rows i (cache-friendly; the strided W write was the 25 s killer).
            var At = new float[inN * r];
            for (int k = 0; k < r; k++)
                for (int i = 0; i < inN; i++) At[i * r + k] = A[k * inN + i];
            System.Threading.Tasks.Parallel.For(0, inN, i =>
            {
                int wrow = i * outN, arow = i * r;
                for (int o = 0; o < outN; o++)
                {
                    float acc = 0; int brow = o * r;
                    for (int k = 0; k < r; k++) acc += At[arow + k] * B[brow + k];
                    W[wrow + o] += scale * acc;
                }
            });
            Register(opts, wname, W, init.Dims.ToArray());
        }

        // embed_tokens: read whole table, overwrite changed rows
        var emb = inits["model.llm.embed_tokens.weight"];
        (long eoff, long elen) = External(emb);
        int hidden = (int)emb.Dims[1];
        float[] E = ReadFloat32(data, eoff, elen);
        float[] rows = ToFloat(diff["embed_rows"]);
        int[] idx = ToInt32(diff["embed_idx"]);
        for (int n = 0; n < idx.Length; n++)
            Array.Copy(rows, n * hidden, E, (long)idx[n] * hidden, hidden);
        Register(opts, emb.Name, E, emb.Dims.ToArray());
    }

    private void Register(SessionOptions opts, string name, float[] data, long[] shape)
    {
        var v = OrtValue.CreateTensorValueFromMemory(data, shape);
        _values.Add(v);
        opts.AddInitializer(name, v);
    }

    private static (long, long) External(TensorProto t)
    {
        long off = 0, len = 0;
        foreach (var e in t.ExternalData)
        {
            if (e.Key == "offset") off = long.Parse(e.Value);
            else if (e.Key == "length") len = long.Parse(e.Value);
        }
        return (off, len);
    }

    private static float[] ReadFloat32(FileStream f, long offset, long length)
    {
        f.Seek(offset, SeekOrigin.Begin);
        var bytes = new byte[length];
        int read = 0;
        while (read < length) read += f.Read(bytes, read, (int)(length - read));
        var outv = new float[length / 4];
        Buffer.BlockCopy(bytes, 0, outv, 0, (int)length);
        return outv;
    }

    // TensorProto raw_data -> float[], handling fp16 (data_type 10) and fp32 (1).
    private static float[] ToFloat(TensorProto t)
    {
        var raw = t.RawData.ToByteArray();
        if (t.DataType == (int)TensorProto.Types.DataType.Float16)
        {
            var outv = new float[raw.Length / 2];
            for (int i = 0; i < outv.Length; i++)
                outv[i] = (float)BitConverter.ToHalf(raw, i * 2);
            return outv;
        }
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        return f;
    }

    private static int[] ToInt32(TensorProto t)
    {
        var raw = t.RawData.ToByteArray();
        var outv = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, outv, 0, raw.Length);
        return outv;
    }

    public void Dispose()
    {
        foreach (var v in _values) v.Dispose();
        _values.Clear();
    }
}
