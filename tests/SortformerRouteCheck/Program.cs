// Sortformer CoreML steady-state routing check (#162).
//
// Proves two things about SortformerStreamer's dual-graph routing, end to end over a
// full streaming run rather than a single chunk:
//
//   1. the routing engages at all (steady-state chunk count > 0), and
//   2. routing through the CoreML variant gives the same answer as the stock graph
//      alone, after 30-odd chunks of accumulating spkcache/FIFO state.
//
// A single-chunk parity check cannot show (2): the risk in this design is that a
// mis-routed chunk corrupts the streaming state and the error compounds silently.
//
// MANUAL TOOL -- not in Vernacula.slnx and not run by CI, like the other tests/*Smoke
// harnesses. It needs both model files present in ~/models:
//
//   diar_streaming_sortformer_4spk-v2.1.onnx          (stock, required)
//   diar_streaming_sortformer_4spk-v2.1.coreml.onnx   (variant, else the CoreML run is a no-op)
//
// Run:  dotnet run --project tests/SortformerRouteCheck -p:EP=Cpu

using Vernacula.Base;
using Vernacula.Base.Models;

// Optional arg: models root (default ~/models). Used to point at a deliberately
// mismatched variant when checking that the signature guard rejects it.
string modelsRoot = args.Length > 0 ? args[0] : Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

// ~5 minutes of structured synthetic audio: alternating tone pairs over noise, so the
// diarizer sees changing content rather than stationary hiss. Identical for both runs.
const int sr = 16000, seconds = 300;
var audio = new float[sr * seconds];
var rng = new Random(7);
for (int i = 0; i < audio.Length; i++)
{
    double t = (double)i / sr;
    int spk = ((int)(t / 3.7)) % 2;               // switch "speaker" every 3.7 s
    double f = spk == 0 ? 180.0 : 320.0;
    audio[i] = (float)(0.35 * Math.Sin(2 * Math.PI * f * t)
                     + 0.15 * Math.Sin(2 * Math.PI * f * 2.5 * t)
                     + 0.02 * (rng.NextDouble() - 0.5));
}

var mel = AudioUtils.LogMelSpectrogram(audio);

(List<float[,]> preds, int steady, int stock, bool loaded, double ms) Run(ExecutionProvider ep)
{
    var sload = System.Diagnostics.Stopwatch.StartNew();
    using var s = new SortformerStreamer(modelsRoot, ep);
    sload.Stop();
    var (totalFrames, chunkStride, numChunks) = s.GetPredParams(mel);
    var all = new List<float[,]>();
    var pattern  = new System.Text.StringBuilder();
    var steadyMs = new List<double>();
    var stockMs  = new List<double>();
    int prevSteady = 0;
    var sw = System.Diagnostics.Stopwatch.StartNew();
    foreach (var (_, idx, p) in s.GetPreds(mel, totalFrames, chunkStride, numChunks))
    {
        double el = sw.Elapsed.TotalMilliseconds;
        bool wasSteady = s.SteadyStateChunkCount > prevSteady;
        (wasSteady ? steadyMs : stockMs).Add(el);
        pattern.Append(wasSteady ? 'S' : '.');
        prevSteady = s.SteadyStateChunkCount;
        sw.Restart();
        all.Add(p);
    }
    static double Med(List<double> v)
    { if (v.Count == 0) return 0; var c = new List<double>(v); c.Sort(); return c[c.Count / 2]; }
    Console.WriteLine($"   load {sload.Elapsed.TotalSeconds:F2} s | per-chunk median: steady {Med(steadyMs):F1} ms, stock {Med(stockMs):F1} ms");
    Console.WriteLine($"   routing (S=steady . =stock): {pattern}");
    return (all, s.SteadyStateChunkCount, s.StockChunkCount, s.UsesSteadyStateGraph,
            steadyMs.Sum() + stockMs.Sum());
}

var base_ = Run(ExecutionProvider.Cpu);
Console.WriteLine($"stock-only (Cpu)   : variantLoaded={base_.loaded,-5} steady={base_.steady,-3} stock={base_.stock,-3} chunks={base_.preds.Count} {base_.ms:F0} ms");

var auto = Run(ExecutionProvider.Auto);
Console.WriteLine($"auto-detect        : variantLoaded={auto.loaded,-5} steady={auto.steady,-3} stock={auto.stock,-3} chunks={auto.preds.Count} {auto.ms:F0} ms");

var routed = Run(ExecutionProvider.CoreML);
Console.WriteLine($"routed (CoreML)    : variantLoaded={routed.loaded,-5} steady={routed.steady,-3} stock={routed.stock,-3} chunks={routed.preds.Count} {routed.ms:F0} ms");

if (!routed.loaded) { Console.WriteLine("\n!! variant not loaded -- routing never exercised"); return 1; }
if (routed.steady == 0) { Console.WriteLine("\n!! steady-state graph never used"); return 1; }

// Auto must reach the variant on its own -- that is the whole claim of the detection
// path, and without asserting it the harness passes even if Auto stops detecting.
if (!auto.loaded) { Console.WriteLine("\n!! Auto did not detect the variant"); return 1; }
if (auto.steady != routed.steady)
{
    Console.WriteLine($"\n!! Auto routed {auto.steady} chunks, explicit CoreML routed {routed.steady}");
    return 1;
}

if (base_.preds.Count != routed.preds.Count)
{ Console.WriteLine($"\n!! chunk count differs: {base_.preds.Count} vs {routed.preds.Count}"); return 1; }

double worst = 0; int worstIdx = -1; double sum = 0; long n = 0;
for (int c = 0; c < base_.preds.Count; c++)
{
    var a = base_.preds[c]; var b = routed.preds[c];
    if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
    { Console.WriteLine($"\n!! chunk {c} shape {a.GetLength(0)}x{a.GetLength(1)} vs {b.GetLength(0)}x{b.GetLength(1)}"); return 1; }
    for (int i = 0; i < a.GetLength(0); i++)
        for (int j = 0; j < a.GetLength(1); j++)
        {
            double d = Math.Abs(a[i, j] - b[i, j]);
            sum += d * d; n++;
            if (d > worst) { worst = d; worstIdx = c; }
        }
}
Console.WriteLine($"\npreds vs stock-only over {base_.preds.Count} chunks ({n} values)");
Console.WriteLine($"  maxAbsDiff = {worst:E3}  (chunk {worstIdx})");
Console.WriteLine($"  rms        = {Math.Sqrt(sum / n):E3}");
return 0;
