using System.Diagnostics;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

string? audioPath = null;
string? modelDir = null;
int warmupRuns = 3;
int timedRuns = 10;
double? maxDuration = null;
int encoderBatchSize = 1;
double lengthBucketMs = 1000.0;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--audio":
            audioPath = args[++i];
            break;
        case "--model":
            modelDir = args[++i];
            break;
        case "--warmup":
            warmupRuns = int.Parse(args[++i]);
            break;
        case "--runs":
            timedRuns = int.Parse(args[++i]);
            break;
        case "--max-duration":
            maxDuration = double.Parse(args[++i]);
            break;
        case "--encoder-batch-size":
            encoderBatchSize = int.Parse(args[++i]);
            break;
        case "--length-bucket-ms":
            lengthBucketMs = double.Parse(args[++i]);
            break;
        case "-h":
        case "--help":
            PrintUsage();
            return 0;
    }
}

if (audioPath is null || modelDir is null)
{
    Console.Error.WriteLine("Error: --audio and --model are required.");
    PrintUsage();
    return 1;
}

if (!File.Exists(audioPath))
{
    Console.Error.WriteLine($"Audio file not found: {audioPath}");
    return 1;
}

if (!Directory.Exists(modelDir))
{
    Console.Error.WriteLine($"Model dir not found: {modelDir}");
    return 1;
}

#if DIRECTML
const string EP_NAME = "DirectML";
#elif CPU
const string EP_NAME = "CPU";
#else
const string EP_NAME = "CUDA";
#endif

Console.WriteLine($"=== Parakeet Full-Pipeline Benchmark [{EP_NAME}] ===");
Console.WriteLine($"Audio  : {audioPath}");
Console.WriteLine($"Models : {modelDir}");
Console.WriteLine($"Warmup : {warmupRuns}  Timed runs: {timedRuns}");
Console.WriteLine($"Encoder batch size : {encoderBatchSize}");
Console.WriteLine($"Length bucket      : {lengthBucketMs:F0}ms");
if (maxDuration.HasValue)
{
    Console.WriteLine($"Clipping audio to first {maxDuration.Value}s");
}
Console.WriteLine();

Console.Write("Loading audio... ");
var sw = Stopwatch.StartNew();
float[] audioFull = BenchmarkAudioUtils.LoadAudio16kMono(audioPath, out double fullDurationSec);
sw.Stop();

float[] audio;
double audioDurationSec;
if (maxDuration.HasValue && audioFull.Length > (int)(maxDuration.Value * BenchmarkAudioUtils.SampleRate))
{
    int clip = (int)(maxDuration.Value * BenchmarkAudioUtils.SampleRate);
    audio = audioFull[..clip];
    audioDurationSec = clip / (double)BenchmarkAudioUtils.SampleRate;
    Console.WriteLine($"{fullDurationSec:F1}s total, clipped to first {audioDurationSec:F1}s ({sw.ElapsedMilliseconds}ms)");
}
else
{
    audio = audioFull;
    audioDurationSec = fullDurationSec;
    Console.WriteLine($"{audioDurationSec:F1}s ({sw.ElapsedMilliseconds}ms)");
}

Console.WriteLine();
Console.Write("Loading ONNX sessions...");
sw.Restart();

var cpuOpts = new SessionOptions();
using var preprocessorSession = new InferenceSession(
    Path.Combine(modelDir, "nemo128.onnx"),
    cpuOpts);

var epOpts = MakeSessionOptions();
using var encoderSession = new InferenceSession(Path.Combine(modelDir, "encoder-model.onnx"), epOpts);
using var decoderJointSession = new InferenceSession(Path.Combine(modelDir, "decoder_joint-model.onnx"), epOpts);
using var sortformer = new BenchmarkSortformer(modelDir);

sw.Stop();
long sessionLoadMs = sw.ElapsedMilliseconds;
Console.WriteLine($" {sessionLoadMs}ms");

var vocab = LoadVocab(Path.Combine(modelDir, "vocab.txt"), out int vocabSize, out int blankIdx);
var inputMeta = decoderJointSession.InputMetadata;
int[] stateShape1 = inputMeta["input_states_1"].Dimensions;
int[] stateShape2 = inputMeta["input_states_2"].Dimensions;

var diarTimes = new List<long>();
var asrTimes = new List<long>();
var totalTimes = new List<long>();

int lastSegCount = 0;
string? lastTranscript = null;

int totalRuns = warmupRuns + timedRuns;
Console.WriteLine($"Running {warmupRuns} warmup + {timedRuns} timed runs...");
Console.WriteLine();

for (int run = 0; run < totalRuns; run++)
{
    bool isTimed = run >= warmupRuns;
    string label = isTimed
        ? $"  Run    {run - warmupRuns + 1,2}/{timedRuns}"
        : $"  Warmup {run + 1,2}/{warmupRuns}";

    Console.Write($"{label}  diar=...");
    var swTotal = Stopwatch.StartNew();

    sortformer.ResetState();
    var swDiar = Stopwatch.StartNew();
    var segs = sortformer.Diarize(audio);
    swDiar.Stop();
    lastSegCount = segs.Count;

    var swAsr = Stopwatch.StartNew();
    var texts = new string[segs.Count];
    var featureJobs = new List<(int Index, float[,] Features, long FeatureLen)>(segs.Count);

    for (int segIndex = 0; segIndex < segs.Count; segIndex++)
    {
        var (segStart, segEnd, _) = segs[segIndex];
        int sampleStart = Math.Max((int)(segStart * BenchmarkAudioUtils.SampleRate), 0);
        int sampleEnd = Math.Min((int)(segEnd * BenchmarkAudioUtils.SampleRate), audio.Length);
        int segLen = Math.Max(sampleEnd - sampleStart, 0);
        if (segLen == 0)
        {
            continue;
        }

        var waveform2d = new float[1, segLen];
        for (int i = 0; i < segLen; i++)
        {
            waveform2d[0, i] = audio[sampleStart + i];
        }

        var waveLens = new long[] { segLen };
        var (features, featLens) = Preprocess(preprocessorSession, waveform2d, waveLens);
        featureJobs.Add((segIndex, Slice2D(features, 0), featLens[0]));
    }

    foreach (var batch in BuildFeatureBatches(featureJobs, encoderBatchSize, lengthBucketMs))
    {
        var (batchFeatures, batchLens, batchIndices) = PadFeatureBatch(batch);
        var (encoderOut, encoderLens) = Encode(encoderSession, batchFeatures, batchLens);
        var decoded = DecodeBatch(
            decoderJointSession,
            encoderOut,
            encoderLens,
            stateShape1,
            stateShape2,
            blankIdx,
            vocabSize);

        for (int i = 0; i < batchIndices.Count; i++)
        {
            string text = string.Concat(decoded[i].tokens.Select(t => vocab.TryGetValue(t, out var s) ? s : "")).Trim();
            texts[batchIndices[i]] = text;
        }
    }

    swAsr.Stop();
    swTotal.Stop();

    var sb = new StringBuilder();
    foreach (string text in texts)
    {
        if (!string.IsNullOrWhiteSpace(text))
        {
            sb.Append(text).Append(' ');
        }
    }

    lastTranscript = sb.ToString().TrimEnd();

    Console.Write(
        $"\r{label}  diar={swDiar.ElapsedMilliseconds,5}ms  " +
        $"asr={swAsr.ElapsedMilliseconds,6}ms  " +
        $"total={swTotal.ElapsedMilliseconds,6}ms  " +
        $"segs={segs.Count,3}");
    Console.WriteLine();

    if (isTimed)
    {
        diarTimes.Add(swDiar.ElapsedMilliseconds);
        asrTimes.Add(swAsr.ElapsedMilliseconds);
        totalTimes.Add(swTotal.ElapsedMilliseconds);
    }
}

Console.WriteLine();
Console.WriteLine($"Transcript ({lastSegCount} segments):");
Console.WriteLine($"  \"{lastTranscript}\"");
Console.WriteLine();
Console.WriteLine("+------------------------+----------+----------+----------+----------+");
Console.WriteLine("| Phase                  |   Min ms |   Avg ms |   Med ms |   Max ms |");
Console.WriteLine("+------------------------+----------+----------+----------+----------+");
PrintRow($"Diarization  ({EP_NAME})", diarTimes);
PrintRow($"ASR          ({EP_NAME})", asrTimes);
PrintRow("Total pipeline", totalTimes);
Console.WriteLine("+------------------------+----------+----------+----------+----------+");
Console.WriteLine();

double avgTotalSec = totalTimes.Average() / 1000.0;
double rtf = avgTotalSec / audioDurationSec;
Console.WriteLine($"Audio duration   : {audioDurationSec:F2}s");
Console.WriteLine($"Avg total time   : {avgTotalSec * 1000:F0}ms");
Console.WriteLine($"Real-time factor : {rtf:F4}  (lower = faster; <1.0 = faster than real-time)");
Console.WriteLine($"Session load     : {sessionLoadMs}ms  (one-time cost, not in RTF)");
Console.WriteLine();
Console.WriteLine($"[{EP_NAME}] benchmark complete.");

return 0;

static void PrintUsage()
{
    Console.WriteLine("Usage: ParakeetBenchmark --audio <file> --model <dir> [--warmup N] [--runs N] [--max-duration S] [--encoder-batch-size N] [--length-bucket-ms MS]");
    Console.WriteLine("  --max-duration S       Clip audio to first S seconds.");
    Console.WriteLine("  --encoder-batch-size N Batch diarized segments together for the encoder pass.");
    Console.WriteLine("  --length-bucket-ms MS  Group similar-length segments before batching (default 1000ms).");
    Console.WriteLine("Build: dotnet build -c Release -p:EP=Cuda|DirectML|Cpu -p:Platform=x64");
}

static void PrintRow(string label, List<long> times)
{
    if (times.Count == 0)
    {
        return;
    }

    var sorted = times.OrderBy(x => x).ToList();
    double min = sorted.First();
    double avg = times.Average();
    double med = sorted.Count % 2 == 1
        ? sorted[sorted.Count / 2]
        : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2.0;
    double max = sorted.Last();
    Console.WriteLine($"| {label,-22} | {min,8:F0} | {avg,8:F1} | {med,8:F1} | {max,8:F0} |");
}

static SessionOptions MakeSessionOptions()
{
    var opts = new SessionOptions();
#if DIRECTML
    try
    {
        opts.AppendExecutionProvider_DML(0);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Warning: DirectML EP unavailable ({ex.Message}), falling back to CPU.");
    }
#elif CPU
    // CPU only.
#else
    try
    {
        opts.AppendExecutionProvider_CUDA(0);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Warning: CUDA EP unavailable ({ex.Message}), falling back to CPU.");
    }
#endif
    return opts;
}

static (float[,,] Features, long[] FeatureLens) Preprocess(
    InferenceSession session,
    float[,] waveforms,
    long[] waveLens)
{
    int batch = waveforms.GetLength(0);
    int samples = waveforms.GetLength(1);
    var waveTensor = new DenseTensor<float>(Flatten2D(waveforms, batch, samples), [batch, samples]);
    var lensTensor = new DenseTensor<long>(waveLens, [batch]);

    using var res = session.Run(
    [
        NamedOnnxValue.CreateFromTensor("waveforms", waveTensor),
        NamedOnnxValue.CreateFromTensor("waveforms_lens", lensTensor),
    ]);

    var featTensor = res.First(r => r.Name == "features").AsTensor<float>();
    var featLensTensor = res.First(r => r.Name == "features_lens").AsTensor<long>();

    int featureDim = featTensor.Dimensions[1];
    int frames = featTensor.Dimensions[2];
    var features = new float[batch, featureDim, frames];
    for (int b = 0; b < batch; b++)
        for (int d = 0; d < featureDim; d++)
            for (int t = 0; t < frames; t++)
                features[b, d, t] = featTensor[b, d, t];

    var lens = new long[batch];
    for (int b = 0; b < batch; b++)
    {
        lens[b] = featLensTensor[b];
    }

    return (features, lens);
}

static (float[,,] EncoderOut, long[] EncoderLens) Encode(
    InferenceSession session,
    float[,,] features,
    long[] lens)
{
    int batch = features.GetLength(0);
    int featureDim = features.GetLength(1);
    int frames = features.GetLength(2);
    var featTensor = new DenseTensor<float>(Flatten3D(features, batch, featureDim, frames), [batch, featureDim, frames]);
    var lensTensor = new DenseTensor<long>(lens, [batch]);

    using var res = session.Run(
    [
        NamedOnnxValue.CreateFromTensor("audio_signal", featTensor),
        NamedOnnxValue.CreateFromTensor("length", lensTensor),
    ]);

    var outTensor = res.First(r => r.Name == "outputs").AsTensor<float>();
    var outLensTensor = res.First(r => r.Name == "encoded_lengths").AsTensor<long>();

    int encoderDim = outTensor.Dimensions[1];
    int encoderFrames = outTensor.Dimensions[2];
    var encoderOut = new float[batch, encoderFrames, encoderDim];
    for (int b = 0; b < batch; b++)
        for (int d = 0; d < encoderDim; d++)
            for (int t = 0; t < encoderFrames; t++)
                encoderOut[b, t, d] = outTensor[b, d, t];

    var encoderLens = new long[batch];
    for (int b = 0; b < batch; b++)
    {
        encoderLens[b] = outLensTensor[b];
    }

    return (encoderOut, encoderLens);
}

static List<(List<int> tokens, List<int> timestamps, List<float> logprobs)> DecodeBatch(
    InferenceSession session,
    float[,,] encoderOut,
    long[] encoderLens,
    int[] stateShape1,
    int[] stateShape2,
    int blankIdx,
    int vocabSize)
{
    int batch = encoderOut.GetLength(0);
    var results = new List<(List<int> tokens, List<int> timestamps, List<float> logprobs)>(batch);
    for (int b = 0; b < batch; b++)
    {
        results.Add(DecodeOne(session, encoderOut, encoderLens, b, stateShape1, stateShape2, blankIdx, vocabSize));
    }

    return results;
}

static (List<int> tokens, List<int> timestamps, List<float> logprobs) DecodeOne(
    InferenceSession session,
    float[,,] encoderOut,
    long[] encoderLens,
    int batchIndex,
    int[] stateShape1,
    int[] stateShape2,
    int blankIdx,
    int vocabSize)
{
    if (encoderOut.GetLength(0) == 0)
    {
        return ([], [], []);
    }

    int encoderDim = encoderOut.GetLength(2);
    long encLen = encoderLens[batchIndex];

    int l1 = stateShape1[0];
    int h1 = stateShape1[2];
    int l2 = stateShape2[0];
    int h2 = stateShape2[2];

    var s1 = new float[l1, 1, h1];
    var s2 = new float[l2, 1, h2];
    var tokens = new List<int>();
    var timestamps = new List<int>();
    var logprobs = new List<float>();
    var frame = new float[encoderDim];

    int t = 0;
    int emitted = 0;
    while (t < encLen)
    {
        for (int d = 0; d < encoderDim; d++)
        {
            frame[d] = encoderOut[batchIndex, t, d];
        }

        var encTensor = new DenseTensor<float>(frame, [1, encoderDim, 1]);
        int lastTok = tokens.Count > 0 ? tokens[^1] : blankIdx;
        var targTensor = new DenseTensor<int>(new int[] { lastTok }, [1, 1]);
        var targLenTensor = new DenseTensor<int>(new int[] { 1 }, [1]);
        var s1Tensor = new DenseTensor<float>(Flatten3D(s1, l1, 1, h1), [l1, 1, h1]);
        var s2Tensor = new DenseTensor<float>(Flatten3D(s2, l2, 1, h2), [l2, 1, h2]);

        using var res = session.Run(
        [
            NamedOnnxValue.CreateFromTensor("encoder_outputs", encTensor),
            NamedOnnxValue.CreateFromTensor("targets", targTensor),
            NamedOnnxValue.CreateFromTensor("target_length", targLenTensor),
            NamedOnnxValue.CreateFromTensor("input_states_1", s1Tensor),
            NamedOnnxValue.CreateFromTensor("input_states_2", s2Tensor),
        ]);

        var outTensor = res.First(r => r.Name == "outputs").AsTensor<float>();
        var ns1Tensor = res.First(r => r.Name == "output_states_1").AsTensor<float>();
        var ns2Tensor = res.First(r => r.Name == "output_states_2").AsTensor<float>();

        int outLen = (int)outTensor.Length;
        var output = new float[outLen];
        for (int i = 0; i < outLen; i++)
        {
            output[i] = outTensor.GetValue(i);
        }

        var logits = output[..vocabSize];
        int token = ArgMax(logits);

        int stepIdx = 0;
        float stepMax = float.NegativeInfinity;
        for (int i = vocabSize; i < outLen; i++)
        {
            if (output[i] > stepMax)
            {
                stepMax = output[i];
                stepIdx = i - vocabSize;
            }
        }

        for (int i = 0; i < ns1Tensor.Dimensions[0]; i++)
            for (int j = 0; j < ns1Tensor.Dimensions[2]; j++)
                s1[i, 0, j] = ns1Tensor[i, 0, j];

        for (int i = 0; i < ns2Tensor.Dimensions[0]; i++)
            for (int j = 0; j < ns2Tensor.Dimensions[2]; j++)
                s2[i, 0, j] = ns2Tensor[i, 0, j];

        if (token != blankIdx)
        {
            tokens.Add(token);
            timestamps.Add(t);
            logprobs.Add(LogSoftmax(logits)[token]);
            emitted++;
        }

        if (stepIdx > 0)
        {
            t += stepIdx;
            emitted = 0;
        }
        else if (token == blankIdx || emitted == 10)
        {
            t++;
            emitted = 0;
        }
    }

    return (tokens, timestamps, logprobs);
}

static float[,] Slice2D(float[,,] input, int batchIndex)
{
    int d0 = input.GetLength(1);
    int d1 = input.GetLength(2);
    var slice = new float[d0, d1];
    for (int i = 0; i < d0; i++)
        for (int j = 0; j < d1; j++)
            slice[i, j] = input[batchIndex, i, j];

    return slice;
}

static List<List<(int Index, float[,] Features, long FeatureLen)>> BuildFeatureBatches(
    List<(int Index, float[,] Features, long FeatureLen)> featureJobs,
    int batchSize,
    double lengthBucketMs)
{
    if (batchSize <= 1)
    {
        return featureJobs
            .Select(job => new List<(int Index, float[,] Features, long FeatureLen)> { job })
            .ToList();
    }

    const double featureFrameMs = 10.0;
    int bucketFrames = Math.Max(1, (int)Math.Ceiling(lengthBucketMs / featureFrameMs));

    var ordered = featureJobs
        .OrderBy(job => (job.FeatureLen + bucketFrames - 1) / bucketFrames)
        .ThenBy(job => job.FeatureLen)
        .ToList();

    var batches = new List<List<(int Index, float[,] Features, long FeatureLen)>>();
    foreach (var group in ordered.GroupBy(job => (job.FeatureLen + bucketFrames - 1) / bucketFrames))
    {
        var jobs = group.ToList();
        for (int i = 0; i < jobs.Count; i += batchSize)
        {
            batches.Add(jobs.Skip(i).Take(batchSize).ToList());
        }
    }

    return batches;
}

static (float[,,] Features, long[] FeatureLens, List<int> BatchIndices) PadFeatureBatch(
    List<(int Index, float[,] Features, long FeatureLen)> batch)
{
    int batchSize = batch.Count;
    int featureDim = batch[0].Features.GetLength(0);
    int maxFrames = batch.Max(job => (int)job.FeatureLen);

    var features = new float[batchSize, featureDim, maxFrames];
    var lens = new long[batchSize];
    var indices = new List<int>(batchSize);

    for (int b = 0; b < batchSize; b++)
    {
        var job = batch[b];
        lens[b] = job.FeatureLen;
        indices.Add(job.Index);

        for (int d = 0; d < featureDim; d++)
            for (int t = 0; t < job.FeatureLen; t++)
                features[b, d, t] = job.Features[d, t];
    }

    return (features, lens, indices);
}

static Dictionary<int, string> LoadVocab(string path, out int vocabSize, out int blankIdx)
{
    var vocab = new Dictionary<int, string>();
    foreach (var line in File.ReadAllLines(path))
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            continue;
        }

        int sp = line.LastIndexOf(' ');
        string token = line[..sp].Replace("\u2581", " ");
        int id = int.Parse(line[(sp + 1)..]);
        vocab[id] = token;
    }

    vocabSize = vocab.Count;
    blankIdx = vocab.FirstOrDefault(kv => kv.Value == "<blk>").Key;
    return vocab;
}

static int ArgMax(float[] arr)
{
    int idx = 0;
    float max = float.NegativeInfinity;
    for (int i = 0; i < arr.Length; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
            idx = i;
        }
    }

    return idx;
}

static float[] LogSoftmax(float[] x)
{
    float max = float.NegativeInfinity;
    foreach (float v in x)
    {
        if (v > max)
        {
            max = v;
        }
    }

    double sum = 0;
    foreach (float v in x)
    {
        sum += Math.Exp(v - max);
    }

    float lse = (float)(Math.Log(sum) + max);
    var result = new float[x.Length];
    for (int i = 0; i < x.Length; i++)
    {
        result[i] = x[i] - lse;
    }

    return result;
}

static float[] Flatten2D(float[,] a, int d0, int d1)
{
    var flat = new float[d0 * d1];
    for (int i = 0; i < d0; i++)
        for (int j = 0; j < d1; j++)
            flat[i * d1 + j] = a[i, j];

    return flat;
}

static float[] Flatten3D(float[,,] a, int d0, int d1, int d2)
{
    var flat = new float[d0 * d1 * d2];
    for (int i = 0; i < d0; i++)
        for (int j = 0; j < d1; j++)
            for (int k = 0; k < d2; k++)
                flat[i * d1 * d2 + j * d2 + k] = a[i, j, k];

    return flat;
}
