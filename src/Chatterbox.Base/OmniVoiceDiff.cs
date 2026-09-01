using System.Text.RegularExpressions;
using Google.Protobuf;
using Microsoft.ML.OnnxRuntime;
using Onnx;

namespace Chatterbox.Base;

/// <summary>
/// Applies the IPA fine-tune diff (a small ipa_diff_vN.onnx: per-module LoRA A/B at rank 16 plus
/// the sparse changed embed_tokens rows) to the BASE omnivoice_transformer.onnx at load time,
/// without a 2.45 GB merged file on disk.
///
/// ⚠ IT DOES THIS BY REWRITING THE GRAPH, NOT THE WEIGHTS — and the distinction is the whole
/// point. The previous implementation folded ΔW into each weight matrix and handed the results to
/// ORT through <c>SessionOptions.AddInitializer</c>. That is CPU-only by ORT's design: a
/// user-supplied initializer must be a CPU tensor, so when the session is planned on CUDA ORT
/// rejects every one of them —
///     "Cannot use user supplied initializer ... because the ORT planned memory location device
///      is different from what is supplied"
/// — 197 rejections, and then SILENTLY falls back to the base graph's own weights. Measured: the
/// output was bit-identical to running with no diff at all, i.e. stock orthographic OmniVoice fed
/// IPA, which is noise. Nothing in the API reports this as an error.
///
/// So we supply no initializers. Instead:
///
///  · Each Linear's <c>MatMul(x, W) -> Y</c> becomes
///        <c>MatMul(x, W) -> Y__lora_base</c>
///        <c>MatMul(x, Aᵀ) -> t</c>          (Aᵀ is (in, r), r = 16)
///        <c>MatMul(t, scale·Bᵀ) -> d</c>    (Bᵀ is (r, out); the scale is folded into B here so
///                                            no Mul node is needed at runtime)
///        <c>Add(Y__lora_base, d) -> Y</c>
///    Renaming the original output and letting the Add produce the ORIGINAL name means no
///    consumer has to be rewired.
///
///  · embed_tokens' <c>Gather(E, ids) -> G</c> gains a sparse additive correction. The diff's
///    embed_rows are REPLACEMENT rows, so the delta is computed here against the base table —
///    seeking to the 5,572 changed rows in the external data rather than reading all 621 MB.
///    The correction is a compact (changed+1, hidden) table whose row 0 is zeros, plus a
///    (vocab,) int32 map from token id to compact row:
///        <c>Gather(idxMap, ids) -> sel</c>, <c>Gather(delta, sel) -> corr</c>,
///        <c>Add(G__lora_base, corr) -> G</c>
///
/// The base weights are never touched, so ORT loads them from the untouched .onnx.data exactly as
/// it would without a diff — on any execution provider. The patched model is handed to ORT as a
/// byte buffer with <c>session.model_external_initializers_file_folder_path</c> pointing at the
/// directory holding that .onnx.data.
///
/// Cost: ~86 MB added to the 1.5 MB model proto (the LoRA factors widened fp16 → fp32 to match the
/// graph, plus the embed delta), against 2.45 GB for a merged file. Validated by
/// OmniVoiceSmoke --fold-selftest against the Python-merged transformer.
/// </summary>
public sealed class OmniVoiceDiff
{
    private static readonly Regex NodeRe =
        new(@"layers\.(\d+)/(self_attn|mlp)/(\w+_proj)/", RegexOptions.Compiled);

    /// <summary>ORT session-option key that tells a from-memory load where the external data file
    /// lives. Without it, loading a patched model from a byte buffer cannot resolve the base
    /// weights' `location` and the session fails.</summary>
    public const string ExternalDataFolderKey = "session.model_external_initializers_file_folder_path";

    /// <summary>Number of Linear modules patched by the most recent <see cref="BuildPatchedModel"/>,
    /// and the number of embedding rows corrected. For the load report and the self-test.</summary>
    public int PatchedModules { get; private set; }
    public int PatchedEmbedRows { get; private set; }

    /// <summary>Build the patched model as a serialized ModelProto. The caller loads it with
    /// <see cref="CreateSession"/> (or by setting <see cref="ExternalDataFolderKey"/> itself), and
    /// the base .onnx.data is read from disk unchanged.</summary>
    public byte[] BuildPatchedModel(string baseOnnxPath, string diffOnnxPath)
    {
        var baseModel = ModelProto.Parser.ParseFrom(File.ReadAllBytes(baseOnnxPath));
        var diffModel = ModelProto.Parser.ParseFrom(File.ReadAllBytes(diffOnnxPath));
        var graph = baseModel.Graph;

        string dataPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(baseOnnxPath))!,
            // location is identical across initializers; take it from the first external one
            graph.Initializer.First(i => i.ExternalData.Count > 0)
                .ExternalData.First(e => e.Key == "location").Value);

        float scale = float.Parse(diffModel.MetadataProps.First(p => p.Key == "lora_scale").Value,
                                  System.Globalization.CultureInfo.InvariantCulture);
        var diff = diffModel.Graph.Initializer.ToDictionary(i => i.Name, i => i);
        var inits = graph.Initializer.ToDictionary(i => i.Name, i => i);

        var newInits = new List<TensorProto>();
        var patched = new List<NodeProto>(graph.Node.Count + 4 * 400);
        int modules = 0;

        using var data = File.OpenRead(dataPath);

        foreach (var nd in graph.Node)
        {
            // ── Linear: MatMul(x, W) ──────────────────────────────────────────────────────────
            if (nd.OpType == "MatMul" && NodeRe.Match(nd.Name) is { Success: true } mm)
            {
                string wname = nd.Input.FirstOrDefault(inits.ContainsKey) ?? "";
                string key = $"llm.layers.{mm.Groups[1].Value}.{mm.Groups[2].Value}.{mm.Groups[3].Value}";
                if (wname.Length > 0
                    && diff.TryGetValue($"{key}.lora_A", out var Ai)
                    && diff.TryGetValue($"{key}.lora_B", out var Bi))
                {
                    // x is the input that is NOT the weight initializer.
                    string x = nd.Input.First(i => i != wname);
                    string y = nd.Output[0];
                    string baseOut = y + "__lora_base";

                    int r = (int)Ai.Dims[0], inN = (int)Ai.Dims[1], outN = (int)Bi.Dims[0];
                    // A is (r, in) -> Aᵀ (in, r);  B is (out, r) -> scale·Bᵀ (r, out).
                    float[] A = ToFloat(Ai), B = ToFloat(Bi);
                    var At = new float[inN * r];
                    for (int k = 0; k < r; k++)
                        for (int i = 0; i < inN; i++) At[i * r + k] = A[k * inN + i];
                    var Bt = new float[r * outN];
                    for (int o = 0; o < outN; o++)
                        for (int k = 0; k < r; k++) Bt[k * outN + o] = scale * B[o * r + k];

                    newInits.Add(FloatTensor($"{key}.lora_At", At, inN, r));
                    newInits.Add(FloatTensor($"{key}.lora_Bt_scaled", Bt, r, outN));

                    var renamed = nd.Clone();
                    renamed.Output[0] = baseOut;
                    patched.Add(renamed);
                    patched.Add(Node("MatMul", $"{nd.Name}/lora_a",
                        new[] { x, $"{key}.lora_At" }, new[] { y + "__lora_a" }));
                    patched.Add(Node("MatMul", $"{nd.Name}/lora_b",
                        new[] { y + "__lora_a", $"{key}.lora_Bt_scaled" }, new[] { y + "__lora_d" }));
                    patched.Add(Node("Add", $"{nd.Name}/lora_add",
                        new[] { baseOut, y + "__lora_d" }, new[] { y }));
                    modules++;
                    continue;
                }
            }

            // ── embed_tokens: Gather(E, ids) ──────────────────────────────────────────────────
            if (nd.OpType == "Gather" && nd.Input.Count == 2
                && nd.Input[0] == "model.llm.embed_tokens.weight"
                && diff.ContainsKey("embed_rows"))
            {
                var emb = inits[nd.Input[0]];
                (long eoff, _) = External(emb);
                int vocab = (int)emb.Dims[0], hidden = (int)emb.Dims[1];
                float[] rows = ToFloat(diff["embed_rows"]);
                int[] idx = ToInt32(diff["embed_idx"]);

                // Delta against the base rows. Seek to each changed row instead of reading the
                // whole 621 MB table — 5,572 reads of 4 KB.
                var delta = new float[(idx.Length + 1) * hidden];   // row 0 stays zero
                var rowBuf = new byte[hidden * 4];
                var baseRow = new float[hidden];
                for (int n = 0; n < idx.Length; n++)
                {
                    data.Seek(eoff + (long)idx[n] * hidden * 4, SeekOrigin.Begin);
                    ReadExactly(data, rowBuf);
                    Buffer.BlockCopy(rowBuf, 0, baseRow, 0, rowBuf.Length);
                    for (int h = 0; h < hidden; h++)
                        delta[(n + 1) * hidden + h] = rows[n * hidden + h] - baseRow[h];
                }
                var map = new int[vocab];
                for (int n = 0; n < idx.Length; n++) map[idx[n]] = n + 1;

                newInits.Add(FloatTensor("ipa_embed_delta", delta, idx.Length + 1, hidden));
                newInits.Add(Int32Tensor("ipa_embed_map", map, vocab));

                string g = nd.Output[0], gBase = g + "__lora_base", ids = nd.Input[1];
                var renamed = nd.Clone();
                renamed.Output[0] = gBase;
                patched.Add(renamed);
                patched.Add(Node("Gather", $"{nd.Name}/ipa_sel",
                    new[] { "ipa_embed_map", ids }, new[] { g + "__ipa_sel" }));
                patched.Add(Node("Gather", $"{nd.Name}/ipa_corr",
                    new[] { "ipa_embed_delta", g + "__ipa_sel" }, new[] { g + "__ipa_corr" }));
                patched.Add(Node("Add", $"{nd.Name}/ipa_add",
                    new[] { gBase, g + "__ipa_corr" }, new[] { g }));
                PatchedEmbedRows = idx.Length;
                continue;
            }

            patched.Add(nd);
        }

        // ⚠ The embedding half must be as loud about failing as the LoRA half. If the base graph
        // names the table something else, or emits the lookup as anything but a 2-input Gather,
        // PatchedEmbedRows stays 0 and the 5,572 replacement rows are silently never applied — the
        // session still loads and runs, producing a HALF-patched model. That is the silent-partial
        // -application failure this file exists to prevent.
        if (diff.ContainsKey("embed_rows") && PatchedEmbedRows == 0)
            throw new InvalidOperationException(
                $"IPA diff {Path.GetFileName(diffOnnxPath)} carries embed_rows, but no embed_tokens "
                + "Gather was matched in the base graph — the embedding half would be skipped, "
                + "leaving a half-patched model that still loads and runs.");
        if (modules == 0)
            throw new InvalidOperationException(
                $"IPA diff {Path.GetFileName(diffOnnxPath)} patched no modules of "
                + $"{Path.GetFileName(baseOnnxPath)} — the diff and the base graph do not correspond.");
        PatchedModules = modules;

        // Nodes are emitted in the original order with each module's LoRA chain inserted directly
        // after the node it extends, so the graph stays topologically sorted. Appending at the end
        // would place the Add after its own consumers.
        graph.Node.Clear();
        graph.Node.AddRange(patched);
        graph.Initializer.AddRange(newInits);
        return baseModel.ToByteArray();
    }

    /// <summary>Build the patched model and create a session from it on <paramref name="opts"/>.
    /// Works on every execution provider — nothing is supplied as a user initializer.</summary>
    public InferenceSession CreateSession(SessionOptions opts, string baseOnnxPath, string diffOnnxPath)
    {
        byte[] model = BuildPatchedModel(baseOnnxPath, diffOnnxPath);
        opts.AddSessionConfigEntry(ExternalDataFolderKey,
            Path.GetDirectoryName(Path.GetFullPath(baseOnnxPath))!);
        return new InferenceSession(model, opts);
    }

    private static NodeProto Node(string op, string name, string[] inputs, string[] outputs)
    {
        var n = new NodeProto { OpType = op, Name = name };
        n.Input.AddRange(inputs);
        n.Output.AddRange(outputs);
        return n;
    }

    private static TensorProto FloatTensor(string name, float[] v, params long[] dims)
    {
        var bytes = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
        var t = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Float,
            RawData = ByteString.CopyFrom(bytes),
        };
        t.Dims.AddRange(dims);
        return t;
    }

    private static TensorProto Int32Tensor(string name, int[] v, params long[] dims)
    {
        var bytes = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
        var t = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Int32,
            RawData = ByteString.CopyFrom(bytes),
        };
        t.Dims.AddRange(dims);
        return t;
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

    private static void ReadExactly(FileStream f, byte[] buf)
    {
        int read = 0;
        while (read < buf.Length)
        {
            int n = f.Read(buf, read, buf.Length - read);
            if (n <= 0) throw new EndOfStreamException("external data ended early");
            read += n;
        }
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
}
