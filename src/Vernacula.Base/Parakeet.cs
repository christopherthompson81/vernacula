using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// NVIDIA Parakeet TDT ASR — C# port of parakeet.py.
/// Loads three ONNX models (preprocessor, encoder, decoder_joint) and
/// runs end-to-end recognition on pre-segmented audio.
/// </summary>
public sealed class Parakeet : IDisposable
{
    private readonly string _modelPath;

    private readonly InferenceSession _preprocessor;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoderJoint;

    private readonly Dictionary<int, string> _vocab;
    private readonly int _vocabSize;
    private readonly int _blankIdx;

    private readonly int[] _stateShape1;
    private readonly int[] _stateShape2;

    /// <summary>Whether CoreML encoder buckets may be used at all for this instance.</summary>
    private readonly bool _coreMlEligible;

    /// <summary>
    /// Bucket mel-frame size → its session, or null once an open has been tried and failed.
    /// Opened lazily: each bucket costs a ~22 s first compile and ~4.4 GB of CoreML cache,
    /// so a run that never sees a 30 s segment never pays for the 3000 bucket.
    /// </summary>
    private readonly Dictionary<int, InferenceSession?> _coreMlEncoders = [];

    /// <summary>Wall-clock milliseconds spent inside the encoder, whichever graph ran it.</summary>
    public double EncoderMs { get; private set; }

    /// <summary>
    /// Of <see cref="EncoderMs"/>, the part spent opening CoreML bucket sessions rather
    /// than running inference. Charged once per bucket per instance, not per segment.
    /// </summary>
    public double CoreMLLoadMs { get; private set; }

    /// <summary>Segments encoded on a CoreML bucket. Diagnostics; also what the tests assert.</summary>
    public int CoreMLSegmentCount { get; private set; }

    /// <summary>Segments encoded on the stock graph — too long for any bucket, or no buckets present.</summary>
    public int StockSegmentCount { get; private set; }

    /// <summary>
    /// Decoding beam width. <c>1</c> (default) runs the fast greedy-batch
    /// decoder. Values &gt; 1 enable TDT beam search (slower; 3–5× at beam=4).
    /// </summary>
    public int BeamWidth { get; set; } = 1;

    /// <summary>
    /// Optional subword-level language model used for shallow fusion during
    /// beam search. Ignored when <see cref="BeamWidth"/> is 1.
    /// </summary>
    public KenLmScorer? LmScorer { get; set; }

    /// <summary>
    /// Shallow-fusion weight applied to the LM log-prob. Typical values are
    /// 0.1–0.5. Ignored when <see cref="LmScorer"/> is null.
    /// </summary>
    public float LmWeight { get; set; } = 0.3f;

    /// <summary>
    /// Per-emitted-token reward added to each beam's cumulative score. Offsets
    /// the shortening bias that shallow LM fusion otherwise introduces
    /// (every emission pays an LM cost, so beams that emit fewer tokens
    /// would win without this). Typical range 0.0–1.0; 0.6 is a reasonable
    /// default when <see cref="LmWeight"/> is around 0.3.
    /// </summary>
    public float LmLengthPenalty { get; set; } = 0.6f;

    // ── Construction ─────────────────────────────────────────────────────────

    public Parakeet(string modelPath)
        : this(modelPath, Config.EncoderFile, Config.DecoderJointFile) { }

    public Parakeet(string modelPath, string encoderFile, string decoderJointFile,
                       ExecutionProvider ep = ExecutionProvider.Auto,
                       int beamWidth = 1)
    {
        _modelPath = modelPath;
        BeamWidth  = beamWidth;

        var cpuOpts = new SessionOptions();
        _preprocessor = new InferenceSession(
            Path.Combine(modelPath, Config.PreprocessorFile), cpuOpts);

        // ⚠ The CoreML buckets are a SEPARATE artifact with a different contract, so the
        // stock encoder still has to run somewhere: it handles anything longer than the
        // largest bucket, and everything when the buckets are absent. Asking for CoreML
        // here would throw -- the stock graph cannot compile under that EP (see
        // TryOpenCoreMLEncoder) -- so it needs some other provider.
        // ⚠ ONE SessionOptions PER SESSION. Handing the same options object to two sessions
        // segfaults the process on the WebGPU EP: the second Dispose lands in
        // `webgpu::BufferManager::Release` on already-freed buffers. It kills the process at
        // teardown -- AFTER transcription and BEFORE the caller writes anything out -- so
        // the symptom is a crash and a missing transcript rather than a bad one.
        //
        // Reproduced down to two sessions and one shared options object, with nothing else
        // involved: shared -> SIGSEGV, one options object each -> clean. This was previously
        // recorded here as an ORT teardown bug to route around; it is ours.
        ExecutionProvider stockEp = StockEncoderProvider(ep);
        _encoder      = new InferenceSession(Path.Combine(modelPath, encoderFile),
                                             OrtSessionBuilder.Create(stockEp));
        _decoderJoint = new InferenceSession(Path.Combine(modelPath, decoderJointFile),
                                             OrtSessionBuilder.Create(stockEp));

        // Only fp32 has CoreML buckets. An int8 request is an explicit ask for the
        // quantized graph; silently serving fp32 from the ANE would change precision
        // behind the caller's back.
        _coreMlEligible = encoderFile == Config.EncoderFile && CoreMLWanted(ep);

        (_vocab, _vocabSize, _blankIdx) = GetVocab();

        var inputMeta = _decoderJoint.InputMetadata;
        _stateShape1 = inputMeta["input_states_1"].Dimensions;
        _stateShape2 = inputMeta["input_states_2"].Dimensions;
    }

    // ── Vocabulary ───────────────────────────────────────────────────────────

    private (Dictionary<int, string> vocab, int vocabSize, int blankIdx) GetVocab()
    {
        var vocab = new Dictionary<int, string>();
        var lines = File.ReadAllLines(Path.Combine(_modelPath, Config.VocabFile));
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            int lastSpace = line.LastIndexOf(' ');
            string token  = line[..lastSpace].Replace("\u2581", " ");
            int    id     = int.Parse(line[(lastSpace + 1)..]);
            vocab[id]     = token;
        }

        int size     = vocab.Count;
        int blankIdx = vocab.FirstOrDefault(kv => kv.Value == "<blk>").Key;
        return (vocab, size, blankIdx);
    }

    // ── Array helpers ─────────────────────────────────────────────────────────

    public static (float[,] padded, long[] lens) PadList(IReadOnlyList<float[]> arrays)
    {
        if (arrays.Count == 0)
            return (new float[0, 0], Array.Empty<long>());

        var lens   = new long[arrays.Count];
        long maxLen = 0;
        for (int i = 0; i < arrays.Count; i++)
        {
            lens[i] = arrays[i].Length;
            if (lens[i] > maxLen) maxLen = lens[i];
        }

        var padded = new float[arrays.Count, maxLen];
        for (int i = 0; i < arrays.Count; i++)
        {
            int len = (int)Math.Min(lens[i], maxLen);
            for (int j = 0; j < len; j++)
                padded[i, j] = arrays[i][j];
        }

        return (padded, lens);
    }

    // ── Waveform segmentation ─────────────────────────────────────────────────

    public static List<float[]> SegmentWaveform(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio)
    {
        var results = new List<float[]>(segs.Count);
        foreach (var (start, end, _) in segs)
        {
            int startFrame = Math.Max((int)(start * Config.SampleRate), 0);
            int endFrame   = Math.Min((int)(end   * Config.SampleRate), audio.Length);
            int len        = Math.Max(endFrame - startFrame, 0);
            var seg        = new float[len];
            if (len > 0)
                Array.Copy(audio, startFrame, seg, 0, len);
            results.Add(seg);
        }
        return results;
    }

    // ── ONNX inference helpers ────────────────────────────────────────────────

    private (float[,,] features, long[] featLens) PreprocessFeatures(
        float[,] waveforms, long[] waveformsLens)
    {
        int B = waveforms.GetLength(0);
        int T = waveforms.GetLength(1);

        var waveT  = new DenseTensor<float>(Flatten2D(waveforms, B, T), new[] { B, T });
        var waveL  = new DenseTensor<long>(waveformsLens, new[] { B });

        using var results = _preprocessor.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("waveforms",      waveT),
            NamedOnnxValue.CreateFromTensor("waveforms_lens", waveL),
        });

        var featT  = results.First(r => r.Name == "features").AsTensor<float>();
        var featLT = results.First(r => r.Name == "features_lens").AsTensor<long>();

        int D2 = featT.Dimensions[1];
        int T2 = featT.Dimensions[2];
        var features = new float[B, D2, T2];
        for (int b = 0; b < B; b++)
            for (int d = 0; d < D2; d++)
                for (int t = 0; t < T2; t++)
                    features[b, d, t] = featT[b, d, t];

        var featLens = new long[B];
        for (int b = 0; b < B; b++)
            featLens[b] = featLT[b];

        return (features, featLens);
    }

    /// <summary>
    /// Whether a CoreML bucket may be used, given the caller's provider choice.
    /// </summary>
    /// <remarks>
    /// ⚠ <b>Auto does NOT opt in here, unlike the Sortformer variant.</b> The Sortformer
    /// gate reasons that presence of a matching artifact is checkable evidence CoreML suits
    /// the model. For Parakeet that evidence now exists and it says the opposite — measured
    /// end to end on an M5, a 10-minute recording, 132 segments, warm CoreML cache:
    ///
    /// <code>
    ///                       CPU EP      CoreML buckets
    ///   encoder inference   13.6 s       8.7 s     <- the ANE does win, 1.56x
    ///   opening buckets      0.0 s      11.7 s     <- and then loses it all here
    ///   encoder total       13.6 s      20.4 s
    /// </code>
    ///
    /// The inference win is real but small in absolute terms (~5 s on a 10-minute file),
    /// because the CPU EP batches and is already good at this — and a bucket session costs
    /// ~2.9 s to open even warm, charged per bucket per instance. Modelling every ladder
    /// from 1 to 4 buckets against a real segment distribution puts the best at ~16.5 s,
    /// still behind the CPU's 13.6 s: fewer buckets means less loading but more padding
    /// waste and more segments falling through to the stock graph.
    ///
    /// Real segments are also much shorter than the buckets — p50 2.3 s against a 4 s
    /// smallest bucket — so mean fill is only ~47% and half the padded work is wasted.
    ///
    /// So an explicit CoreML request is honoured, and Auto stays on the CPU EP until either
    /// the session-open cost is hidden (opening buckets concurrently with diarization, or
    /// keeping them alive across files) or the ladder is re-cut for the observed
    /// distribution. Diarization is unaffected: the Sortformer variant is a clear win
    /// (13.1 s -> 6.6 s on the same file) and Auto still takes it.
    /// </remarks>
    private static bool CoreMLWanted(ExecutionProvider ep) => ep == ExecutionProvider.CoreML;

    /// <summary>
    /// Which provider the STOCK encoder and decoder-joint sessions run on, given what the
    /// caller asked for.
    /// </summary>
    /// <remarks>
    /// A CoreML request cannot go to these sessions: the stock graph does not compile under
    /// that EP at all (see <see cref="TryOpenCoreMLEncoder"/>), so it resolves to Auto — the
    /// best remaining provider, which is WebGPU on macOS.
    ///
    /// This briefly returned the CPU EP on macOS to dodge a teardown segfault. That was the
    /// wrong fix for a correctly-observed crash: the cause was one `SessionOptions` shared
    /// between two sessions, not the WebGPU EP, and it is fixed at the constructor. Keeping
    /// the dodge would have cost macOS the ~1.5× WebGPU gives the stock encoder over the CPU
    /// EP (7.1 s vs 10.8 s over a 10-minute recording).
    /// </remarks>
    private static ExecutionProvider StockEncoderProvider(ExecutionProvider ep) =>
        ep == ExecutionProvider.CoreML ? ExecutionProvider.Auto : ep;

    /// <summary>The smallest bucket that fits <paramref name="melFrames"/>, or -1.</summary>
    private static int BucketFor(long melFrames)
    {
        foreach (int frames in Config.ParakeetCoreMLEncoderFrames)
            if (melFrames <= frames)
                return frames;
        return -1;      // longer than the largest bucket: the stock graph takes it
    }

    /// <summary>
    /// The bucket session for <paramref name="melFrames"/>, opening it on first use, or
    /// null to fall back to the stock encoder.
    /// </summary>
    private InferenceSession? CoreMLEncoder(int melFrames)
    {
        if (_coreMlEncoders.TryGetValue(melFrames, out var cached))
            return cached;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        InferenceSession? session = TryOpenCoreMLEncoder(_modelPath, melFrames);
        CoreMLLoadMs += sw.Elapsed.TotalMilliseconds;
        _coreMlEncoders[melFrames] = session;
        return session;
    }

    /// <summary>
    /// Opens one CoreML encoder bucket, or returns null to run stock-only.
    /// </summary>
    /// <remarks>
    /// ⚠ THE FILENAME IS NOT THE CONTRACT. A bucket exported for a different mel count, or
    /// an older export without <c>pad_keep</c>, lands at this exact path and loads fine —
    /// then throws on the first segment, turning a graceful fallback into a failed run.
    /// Check the signature.
    /// </remarks>
    private static InferenceSession? TryOpenCoreMLEncoder(string modelPath, int melFrames)
    {
        string path = Path.Combine(modelPath, Config.ParakeetCoreMLEncoderFile(melFrames));
        if (!File.Exists(path))
            return null;

        InferenceSession? sess = null;
        try
        {
            // Always CoreML, never the caller's `ep`: under Auto that would build a WebGPU
            // session for a graph exported specifically for the CoreML EP.
            var opts = OrtSessionBuilder.Create(
                ExecutionProvider.CoreML, GraphOptimizationLevel.ORT_ENABLE_BASIC,
                coreMlModelPath: path);
            sess = new InferenceSession(path, opts);

            if (!SignatureMatchesBucketContract(sess, melFrames))
            {
                sess.Dispose();
                return null;
            }
            return sess;
        }
        catch
        {
            // Missing provider, a graph this ORT declines, anything: the stock encoder
            // handles every segment on its own.
            sess?.Dispose();
            return null;
        }
    }

    /// <summary>
    /// Whether a candidate graph is the bucket <see cref="EncodeOnBucket"/> knows how to
    /// feed: exactly <c>audio_signal [1, 128, F]</c> and <c>pad_keep [1, F]</c>.
    /// </summary>
    private static bool SignatureMatchesBucketContract(InferenceSession sess, int melFrames)
    {
        var expected = new (string Name, int[] Dims)[]
        {
            ("audio_signal", [1, Config.NMels, melFrames]),
            ("pad_keep",     [1, melFrames]),
        };

        var meta = sess.InputMetadata;
        if (meta.Count != expected.Length)
            return false;

        foreach (var (name, dims) in expected)
        {
            if (!meta.TryGetValue(name, out var m) || m.Dimensions.Length != dims.Length)
                return false;
            for (int i = 0; i < dims.Length; i++)
                if (m.Dimensions[i] != dims[i])   // a dynamic axis is negative, so it fails too
                    return false;
        }
        return true;
    }

    /// <summary>
    /// Encodes ONE segment on a CoreML bucket: zero-pad the features to the bucket and
    /// hand the graph an honest <c>pad_keep</c>.
    /// </summary>
    /// <remarks>
    /// ⚠ <c>pad_keep</c> MUST reflect the true frame count. All-ones on a short segment is
    /// exactly the "bake the length" behaviour this export exists to avoid: measured on
    /// real speech it moves the encoder output by 0.17–0.24 and changes the transcript by
    /// 2–4% WER on 8–20 s segments, worse on short ones — and it fails silently, since the
    /// graph is perfectly happy to be lied to.
    /// </remarks>
    private static float[] EncodeOnBucket(
        InferenceSession sess, float[,,] features, int b, int D, int melFrames, long trueFrames)
    {
        var signal = new float[D * melFrames];
        var keep   = new float[melFrames];
        int valid  = (int)Math.Min(trueFrames, melFrames);
        for (int d = 0; d < D; d++)
            for (int t = 0; t < valid; t++)
                signal[d * melFrames + t] = features[b, d, t];
        for (int t = 0; t < valid; t++)
            keep[t] = 1f;

        using var results = sess.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_signal",
                new DenseTensor<float>(signal, [1, D, melFrames])),
            NamedOnnxValue.CreateFromTensor("pad_keep",
                new DenseTensor<float>(keep, [1, melFrames])),
        });

        var outT = results.First(r => r.Name == "outputs").AsTensor<float>();
        var flat = new float[outT.Length];
        for (int i = 0; i < flat.Length; i++) flat[i] = outT.GetValue(i);
        return flat;                                    // [1, D_enc, T_enc] flattened
    }

    /// <summary>
    /// Encodes a whole batch one segment at a time on CoreML buckets, or returns null to
    /// leave the batch to the stock encoder.
    /// </summary>
    /// <remarks>
    /// All-or-nothing per batch, deliberately: <see cref="MakeBatches"/> groups
    /// length-sorted segments, so a batch is nearly homogeneous and a mixed routing would
    /// buy a sub-batch stock call's worth of complexity for almost nothing. In practice the
    /// short batches go to the ANE and a batch holding a >30 s segment goes to the stock
    /// graph whole.
    ///
    /// Dropping batching costs nothing here even though the stock path batches up to 32:
    /// measured on an M5, the CPU EP is FASTER at batch 1 (141.8 ms/segment) than at
    /// batch 8 (162.7 ms), because it already saturates its threads on a single segment.
    /// </remarks>
    private (float[,,] encoderOut, long[] encoderLens)? TryEncodeViaCoreML(
        float[,,] features, long[] lens)
    {
        int B = features.GetLength(0);
        int D = features.GetLength(1);

        var buckets  = new int[B];
        var sessions = new InferenceSession[B];
        for (int b = 0; b < B; b++)
        {
            buckets[b] = BucketFor(lens[b]);
            if (buckets[b] < 0)
                return null;                                  // longer than every bucket
            var sess = CoreMLEncoder(buckets[b]);
            if (sess is null)
                return null;                                  // absent, or a bad signature
            sessions[b] = sess;
        }

        var encLens  = new long[B];
        var outputs  = new float[B][];
        int maxT = 0, dEnc = 0;
        for (int b = 0; b < B; b++)
        {
            int bucketT = Config.ParakeetEncodedFrames(buckets[b]);
            outputs[b]  = EncodeOnBucket(sessions[b], features, b, D, buckets[b], lens[b]);
            dEnc        = outputs[b].Length / bucketT;
            // From the TRUE frame count, not the bucket's: the padded tail is masked out
            // of the encoder, and decoding it would transcribe silence.
            encLens[b]  = Math.Min(Config.ParakeetEncodedFrames((int)lens[b]), bucketT);
            maxT        = Math.Max(maxT, (int)encLens[b]);
        }

        var encoderOut = new float[B, maxT, dEnc];
        for (int b = 0; b < B; b++)
        {
            int bucketT = Config.ParakeetEncodedFrames(buckets[b]);
            var flat    = outputs[b];                          // [1, dEnc, bucketT]
            for (int d = 0; d < dEnc; d++)
                for (int t = 0; t < encLens[b]; t++)
                    encoderOut[b, t, d] = flat[d * bucketT + t];
        }

        CoreMLSegmentCount += B;
        return (encoderOut, encLens);
    }

    private (float[,,] encoderOut, long[] encoderLens) Encode(float[,,] features, long[] lens)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try { return EncodeInner(features, lens); }
        finally { EncoderMs += sw.Elapsed.TotalMilliseconds; }
    }

    private (float[,,] encoderOut, long[] encoderLens) EncodeInner(float[,,] features, long[] lens)
    {
        if (_coreMlEligible)
        {
            var routed = TryEncodeViaCoreML(features, lens);
            if (routed is not null)
                return routed.Value;
        }
        StockSegmentCount += features.GetLength(0);

        int B = features.GetLength(0);
        int D = features.GetLength(1);
        int T = features.GetLength(2);

        var featT = new DenseTensor<float>(Flatten3D(features, B, D, T), new[] { B, D, T });
        var lenT  = new DenseTensor<long>(lens, new[] { B });

        using var results = _encoder.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_signal", featT),
            NamedOnnxValue.CreateFromTensor("length",       lenT),
        });

        var outT  = results.First(r => r.Name == "outputs").AsTensor<float>();
        var outLT = results.First(r => r.Name == "encoded_lengths").AsTensor<long>();

        int D2 = outT.Dimensions[1];
        int T2 = outT.Dimensions[2];
        var encoderOut = new float[B, T2, D2];
        for (int b = 0; b < B; b++)
            for (int d = 0; d < D2; d++)
                for (int t = 0; t < T2; t++)
                    encoderOut[b, t, d] = outT[b, d, t];

        var encoderLens = new long[B];
        for (int b = 0; b < B; b++)
            encoderLens[b] = outLT[b];

        return (encoderOut, encoderLens);
    }

    private (float[,,] s1, float[,,] s2) CreateState()
    {
        int l1 = _stateShape1[0], h1 = _stateShape1[2];
        int l2 = _stateShape2[0], h2 = _stateShape2[2];
        return (new float[l1, 1, h1], new float[l2, 1, h2]);
    }

    /// <summary>
    /// Runs one step of the decoder-joint network. Returns vocab logits,
    /// the full duration-head logits (one score per TDT duration bucket —
    /// typically [0, 1, 2, 3, 4]), and the new RNN-T state. Greedy callers
    /// argmax the durations to get the step value; beam callers need the
    /// full distribution.
    /// </summary>
    private (float[] logits, float[] durLogits, (float[,,] s1, float[,,] s2) state) Decode(
        IReadOnlyList<int> prevTokens,
        (float[,,] s1, float[,,] s2) state,
        float[] encoderFrame)
    {
        int D = encoderFrame.Length;

        var encData = new float[D];
        Array.Copy(encoderFrame, encData, D);
        var encT = new DenseTensor<float>(encData, new[] { 1, D, 1 });

        int lastToken = prevTokens.Count > 0 ? prevTokens[prevTokens.Count - 1] : _blankIdx;
        var targT     = new DenseTensor<int>(new int[] { lastToken }, new[] { 1, 1 });
        var targLenT  = new DenseTensor<int>(new int[] { 1 }, new[] { 1 });

        int l1 = state.s1.GetLength(0), h1 = state.s1.GetLength(2);
        int l2 = state.s2.GetLength(0), h2 = state.s2.GetLength(2);
        var s1T = new DenseTensor<float>(Flatten3D(state.s1, l1, 1, h1), new[] { l1, 1, h1 });
        var s2T = new DenseTensor<float>(Flatten3D(state.s2, l2, 1, h2), new[] { l2, 1, h2 });

        using var results = _decoderJoint.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_outputs", encT),
            NamedOnnxValue.CreateFromTensor("targets",         targT),
            NamedOnnxValue.CreateFromTensor("target_length",   targLenT),
            NamedOnnxValue.CreateFromTensor("input_states_1",  s1T),
            NamedOnnxValue.CreateFromTensor("input_states_2",  s2T),
        });

        var outT = results.First(r => r.Name == "outputs").AsTensor<float>();
        var ns1T = results.First(r => r.Name == "output_states_1").AsTensor<float>();
        var ns2T = results.First(r => r.Name == "output_states_2").AsTensor<float>();

        int outLen = (int)outT.Length;
        var output = new float[outLen];
        for (int i = 0; i < outLen; i++) output[i] = outT.GetValue(i);

        var logits    = output[.._vocabSize];
        var durLogits = output[_vocabSize..];

        var ns1 = new float[ns1T.Dimensions[0], 1, ns1T.Dimensions[2]];
        var ns2 = new float[ns2T.Dimensions[0], 1, ns2T.Dimensions[2]];
        for (int i = 0; i < ns1T.Dimensions[0]; i++)
            for (int j = 0; j < ns1T.Dimensions[2]; j++)
                ns1[i, 0, j] = ns1T[i, 0, j];
        for (int i = 0; i < ns2T.Dimensions[0]; i++)
            for (int j = 0; j < ns2T.Dimensions[2]; j++)
                ns2[i, 0, j] = ns2T[i, 0, j];

        return (logits, durLogits, (ns1, ns2));
    }

    // ── Streaming decoder ─────────────────────────────────────────────────────

    private IEnumerable<(List<int> tokens, List<int> timestamps, List<int> durations, List<float> logprobs)>
        Decoder(float[,,] encoderOut, long[] encoderLens)
        => BeamWidth > 1
            ? DecoderBeam(encoderOut, encoderLens, BeamWidth)
            : DecoderGreedy(encoderOut, encoderLens);

    private IEnumerable<(List<int> tokens, List<int> timestamps, List<int> durations, List<float> logprobs)>
        DecoderGreedy(float[,,] encoderOut, long[] encoderLens)
    {
        int B = encoderOut.GetLength(0);
        int D = encoderOut.GetLength(2);

        for (int b = 0; b < B; b++)
        {
            long encLen = encoderLens[b];

            var prevState     = CreateState();
            var tokens        = new List<int>();
            var timestamps    = new List<int>();
            var durations     = new List<int>();
            var logprobs      = new List<float>();
            var frame         = new float[D];
            int t             = 0;
            int emittedTokens = 0;

            while (t < encLen)
            {
                for (int d = 0; d < D; d++) frame[d] = encoderOut[b, t, d];

                var (logits, durLogits, newState) = Decode(tokens, prevState, frame);
                int token = ArgMax(logits);
                int step  = ArgMax(durLogits);

                if (token != _blankIdx)
                {
                    prevState = newState;
                    tokens.Add(token);
                    timestamps.Add(t);
                    durations.Add(step);
                    emittedTokens++;
                    logprobs.Add(AudioUtils.LogSoftmax(logits)[token]);
                }

                if (step > 0)
                {
                    t += step;
                    emittedTokens = 0;
                }
                else if (token == _blankIdx || emittedTokens == Config.MaxTokensPerStep)
                {
                    t++;
                    emittedTokens = 0;
                }
            }

            yield return (tokens, timestamps, durations, logprobs);
        }
    }

    // ── Beam-search decoder (TDT, alignment-length synchronous) ───────────────
    //
    // Each expansion step runs Decode() on one live hypothesis and branches
    // over the top-K vocab candidates × all duration buckets. Hypotheses
    // advance their frame pointer by the chosen duration independently, so
    // they reach the end of the segment at different alignment lengths. We
    // keep them in one beam pruned by cumulative log-prob (vocab + duration).
    //
    // Simplifications vs. NeMo's reference implementation:
    // - No prefix-merge step: hypotheses that converge on the same (tokens,
    //   state) keep separate score mass instead of being summed.
    // - No language-model fusion yet (planned — LM fusion rides on top of
    //   this loop).

    private sealed class BeamHyp
    {
        public List<int>   Tokens     = new();
        public List<int>   Timestamps = new();
        public List<int>   Durations  = new();
        public List<float> Logprobs   = new();
        public (float[,,] s1, float[,,] s2) State;
        public int    T;
        public double Score;
        public int    EmittedAtT;

        public BeamHyp Clone() => new()
        {
            Tokens     = new List<int>(Tokens),
            Timestamps = new List<int>(Timestamps),
            Durations  = new List<int>(Durations),
            Logprobs   = new List<float>(Logprobs),
            State      = State,   // shared, treated as immutable
            T          = T,
            Score      = Score,
            EmittedAtT = EmittedAtT,
        };
    }

    private IEnumerable<(List<int> tokens, List<int> timestamps, List<int> durations, List<float> logprobs)>
        DecoderBeam(float[,,] encoderOut, long[] encoderLens, int beamWidth)
    {
        int B = encoderOut.GetLength(0);
        int D = encoderOut.GetLength(2);

        for (int b = 0; b < B; b++)
        {
            long encLen = encoderLens[b];
            var frame = new float[D];

            var beam = new List<BeamHyp> {
                new BeamHyp { State = CreateState() }
            };
            var completed = new List<BeamHyp>();

            while (beam.Count > 0)
            {
                var candidates = new List<BeamHyp>(beam.Count * beamWidth * 4);

                foreach (var h in beam)
                {
                    if (h.T >= encLen)
                    {
                        completed.Add(h);
                        continue;
                    }

                    for (int d = 0; d < D; d++) frame[d] = encoderOut[b, h.T, d];
                    var (logits, durLogits, newState) = Decode(h.Tokens, h.State, frame);
                    var vocabLp = AudioUtils.LogSoftmax(logits);
                    var durLp   = AudioUtils.LogSoftmax(durLogits);

                    var topTokens = TopKIndices(vocabLp, beamWidth);

                    foreach (int token in topTokens)
                    {
                        bool isBlank = token == _blankIdx;

                        for (int dIdx = 0; dIdx < durLp.Length; dIdx++)
                        {
                            // Respect MaxTokensPerStep: don't emit a non-blank token
                            // at the same frame past the cap (would stall the decoder).
                            if (!isBlank && dIdx == 0 && h.EmittedAtT >= Config.MaxTokensPerStep)
                                continue;

                            var nh = h.Clone();
                            nh.Score += vocabLp[token] + durLp[dIdx];

                            if (!isBlank)
                            {
                                // Shallow LM fusion: only score non-blank tokens so the
                                // blank head's probability mass isn't double-weighted.
                                // Length penalty counteracts the shortening bias LM
                                // fusion introduces (beams that emit fewer tokens would
                                // otherwise accumulate lower LM cost regardless of fit).
                                if (LmScorer != null)
                                    nh.Score += LmWeight * LmScorer.LogProb(h.Tokens, token)
                                              + LmLengthPenalty;

                                nh.Tokens.Add(token);
                                nh.Timestamps.Add(h.T);
                                nh.Durations.Add(dIdx);
                                nh.Logprobs.Add(vocabLp[token]);
                                nh.State      = newState;
                                nh.EmittedAtT = dIdx == 0 ? h.EmittedAtT + 1 : 0;
                            }
                            else
                            {
                                nh.EmittedAtT = 0;
                            }

                            // Frame advance. Blank at dur==0 would stall; force +1
                            // to match the safety-advance in the greedy path.
                            nh.T = (isBlank && dIdx == 0) ? h.T + 1 : h.T + dIdx;

                            candidates.Add(nh);
                        }
                    }
                }

                if (candidates.Count == 0) break;

                candidates.Sort((a, c) => c.Score.CompareTo(a.Score));
                beam = candidates.Count > beamWidth
                    ? candidates.GetRange(0, beamWidth)
                    : candidates;
            }

            BeamHyp best = completed.Count > 0
                ? completed.OrderByDescending(h => h.Score).First()
                : beam.OrderByDescending(h => h.Score).First();

            yield return (best.Tokens, best.Timestamps, best.Durations, best.Logprobs);
        }
    }

    private static int[] TopKIndices(float[] values, int k)
    {
        int n = values.Length;
        if (k >= n)
        {
            var all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            Array.Sort(all, (a, b) => values[b].CompareTo(values[a]));
            return all;
        }

        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        // Partial sort: the sort is O(n log n) but k is small (≤ beam width)
        // so a full sort is fine compared to a Decode call.
        Array.Sort(idx, (a, b) => values[b].CompareTo(values[a]));
        var result = new int[k];
        Array.Copy(idx, result, k);
        return result;
    }

    // ── VRAM batch sizing ─────────────────────────────────────────────────────

    private static long ComputeMaxFrames()
    {
        var (_, freeMb) = HardwareInfo.GetGpuMemoryMb();
        if (freeMb <= 0)
            return Config.FallbackMaxFrames;

        double available = freeMb - Config.VramSafetyMb;
        if (available <= 0)
            return Config.FallbackMaxFrames;

        long frames = (long)((available - Config.VramInterceptMb) / Config.VramSlopePerSample);
        return frames > 0 ? frames : Config.FallbackMaxFrames;
    }

    // ── Batching ──────────────────────────────────────────────────────────────

    private static IEnumerable<List<float[]>> MakeBatches(
        List<float[]> sorted, int maxBatchSize, long maxFrames)
    {
        var current    = new List<float[]>();
        long currentMax = 0;

        foreach (var wave in sorted)
        {
            long newMax     = Math.Max(currentMax, wave.Length);
            long projected  = (current.Count + 1) * newMax;
            bool sizeLimit  = current.Count >= maxBatchSize;
            bool frameLimit = current.Count > 0 && projected > maxFrames;

            if (sizeLimit || frameLimit)
            {
                yield return current;
                current    = new List<float[]>();
                currentMax = 0;
                newMax     = wave.Length;
            }

            current.Add(wave);
            currentMax = Math.Max(currentMax, wave.Length);
        }

        if (current.Count > 0)
            yield return current;
    }

    // ── Public recognition API ────────────────────────────────────────────────

    /// <summary>
    /// CPU-only phase: extract waveforms, sort, batch, and run the preprocessor.
    /// Returns null if <paramref name="segs"/> is empty.
    /// </summary>
    public PreparedBatch? PrepareBatch(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio)
    {
        var waveforms = SegmentWaveform(segs, audio);
        if (waveforms.Count == 0) return null;

        var sorted = waveforms
            .Select((w, i) => (w, origIdx: i))
            .OrderBy(x => x.w.Length)
            .ToList();

        long maxFrames   = ComputeMaxFrames();
        var sortedWaves  = sorted.Select(x => x.w).ToList();
        var batchedWaves = MakeBatches(sortedWaves, Config.MaxBatchSize, maxFrames).ToList();

        var batches = new List<(float[,,] features, long[] featLens, int segCount)>(batchedWaves.Count);
        foreach (var batch in batchedWaves)
        {
            var (padded, lens)       = PadList(batch);
            var (features, featLens) = PreprocessFeatures(padded, lens);
            batches.Add((features, featLens, batch.Count));
        }

        return new PreparedBatch(sorted.Select(x => x.origIdx).ToList(), batches);
    }

    /// <summary>
    /// GPU phase: encode and decode a batch already preprocessed by
    /// <see cref="PrepareBatch"/>. Results are yielded in original segment order.
    /// </summary>
    public IEnumerable<(int segId, string text, List<int> tokens, List<int> timestamps, List<int> durations, List<float> logprobs)>
        RecognizePrepared(PreparedBatch prepared)
    {
        int sortedPos = 0;
        foreach (var (features, featLens, segCount) in prepared.Batches)
        {
            var (encoderOut, encoderLens) = Encode(features, featLens);

            int j = 0;
            foreach (var (tokens, timestamps, durations, logprobs) in Decoder(encoderOut, encoderLens))
            {
                int origIdx = prepared.OriginalIndices[sortedPos + j];
                string text = string.Concat(tokens.Select(t => _vocab.TryGetValue(t, out var s) ? s : "")).Trim();
                yield return (origIdx, text, tokens, timestamps, durations, logprobs);
                j++;
            }
            sortedPos += segCount;
        }
    }

    /// <summary>
    /// Convenience wrapper: CPU preprocess then GPU encode+decode in one call.
    /// </summary>
    public IEnumerable<(int segId, string text, List<int> tokens, List<int> timestamps, List<int> durations, List<float> logprobs)>
        Recognize(IReadOnlyList<(double start, double end, string spk)> segs, float[] audio)
    {
        var prepared = PrepareBatch(segs, audio);
        if (prepared == null) yield break;
        foreach (var result in RecognizePrepared(prepared))
            yield return result;
    }

    // ── Flat array helpers ────────────────────────────────────────────────────

    private static float[] Flatten2D(float[,] a, int d0, int d1)
    {
        var flat = new float[d0 * d1];
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                flat[i * d1 + j] = a[i, j];
        return flat;
    }

    private static float[] Flatten3D(float[,,] a, int d0, int d1, int d2)
    {
        var flat = new float[d0 * d1 * d2];
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    flat[i * d1 * d2 + j * d2 + k] = a[i, j, k];
        return flat;
    }

    private static int ArgMax(float[] arr)
    {
        int idx = 0;
        float max = float.NegativeInfinity;
        for (int i = 0; i < arr.Length; i++)
            if (arr[i] > max) { max = arr[i]; idx = i; }
        return idx;
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose()
    {
        _preprocessor.Dispose();
        _encoder.Dispose();
        _decoderJoint.Dispose();
        foreach (var sess in _coreMlEncoders.Values)
            sess?.Dispose();
    }
}

/// <summary>
/// CPU-preprocessed features ready for GPU encoding.
/// </summary>
public sealed class PreparedBatch(
    IReadOnlyList<int> originalIndices,
    IReadOnlyList<(float[,,] features, long[] featLens, int segCount)> batches)
{
    public IReadOnlyList<int> OriginalIndices { get; } = originalIndices;
    public IReadOnlyList<(float[,,] features, long[] featLens, int segCount)> Batches { get; } = batches;
}
