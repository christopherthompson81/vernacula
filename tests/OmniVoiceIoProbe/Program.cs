// Diagnose the C# CUDA IO-binding corruption (ORT 1.24.4): does mutating a bound input
// OrtValue's host buffer IN PLACE between runs get re-uploaded to the device, or does ORT
// reuse a stale device copy? The OmniVoice diffusion loop mutates input_ids every step, so a
// stale upload would explain the noise. Compares, on CUDA (use_tf32=0):
//
//   plain(A), plain(B)                         -- reference outputs for two input sets
//   bound, run with A, then mutate buffer->B, run   -- does the 2nd run match plain(B)?
//   bound, rebind a fresh B OrtValue, run           -- mitigation candidate
//
// Usage: dotnet run --project tests/OmniVoiceIoProbe -- <onnx-dir>

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

string onnxDir = args.Length > 0 ? args[0] : "scripts/omnivoice_export/onnx";
string path = Path.Combine(onnxDir, "omnivoice_transformer.onnx");

const int B = 2, C = 8, S = 160, V = 1025, Mask = 1024;

// Two distinct, valid input sets (style/text region = some ids, audio region = mask).
long[] MakeIds(int seed)
{
    var r = new Random(seed);
    var a = new long[B * C * S];
    for (int i = 0; i < a.Length; i++) a[i] = r.Next(0, 1000);
    return a;
}
var idsA = MakeIds(1);
var idsB = MakeIds(2);
var amask = new bool[B * S];
for (int i = 0; i < amask.Length; i++) amask[i] = true;
var attn = new bool[B * S * S];
Array.Fill(attn, true);

var so = OrtSessionBuilder.Create(ExecutionProvider.Cuda, GraphOptimizationLevel.ORT_ENABLE_ALL,
                                  enableProfiling: false, out _, disableTf32: true);
using var sess = new InferenceSession(path, so);

float[] Plain(long[] ids)
{
    var idsT = new DenseTensor<long>(ids, [B, C, S]);
    var amT = new DenseTensor<bool>(amask, [B, S]);
    var atT = new DenseTensor<bool>(attn, [B, 1, S, S]);
    using var o = sess.Run([
        NamedOnnxValue.CreateFromTensor("input_ids", idsT),
        NamedOnnxValue.CreateFromTensor("audio_mask", amT),
        NamedOnnxValue.CreateFromTensor("attention_mask", atT)]);
    return o.First(v => v.Name == "logits").AsTensor<float>().ToArray();
}

double MaxAbs(float[] x, float[] y)
{
    double m = 0;
    for (int i = 0; i < x.Length; i++) m = Math.Max(m, Math.Abs(x[i] - y[i]));
    return m;
}

var plainA = Plain(idsA);
var plainB = Plain(idsB);
Console.WriteLine($"sanity: plain(A) vs plain(B) max-abs = {MaxAbs(plainA, plainB):e3}  (should be large)");

// Bound path: persistent OrtValues over managed buffers, mutate input in place between runs.
var idsBuf = (long[])idsA.Clone();
var logits = new float[B * C * S * V];
var cpu = OrtMemoryInfo.DefaultInstance;
using (var idsV = OrtValue.CreateTensorValueFromMemory<long>(cpu, idsBuf, [B, C, S]))
using (var amV = OrtValue.CreateTensorValueFromMemory<bool>(cpu, amask, [B, S]))
using (var atV = OrtValue.CreateTensorValueFromMemory<bool>(cpu, attn, [B, 1, S, S]))
using (var logV = OrtValue.CreateTensorValueFromMemory<float>(cpu, logits, [B, C, S, V]))
using (var run = new RunOptions())
using (var io = sess.CreateIoBinding())
{
    io.BindInput("input_ids", idsV);
    io.BindInput("audio_mask", amV);
    io.BindInput("attention_mask", atV);
    io.BindOutput("logits", logV);

    sess.RunWithBinding(run, io);
    var boundRun1 = (float[])logits.Clone();
    Console.WriteLine($"bound run1 (buffer=A) vs plain(A) max-abs = {MaxAbs(boundRun1, plainA):e3}  (want ~0)");

    Array.Copy(idsB, idsBuf, idsBuf.Length);   // mutate the bound input buffer IN PLACE -> B
    sess.RunWithBinding(run, io);
    var boundRun2 = (float[])logits.Clone();
    Console.WriteLine($"bound run2 (buffer mutated A to B) vs plainB max-abs = {MaxAbs(boundRun2, plainB):e3}  (want ~0; large means stale)");
    Console.WriteLine($"bound run2 vs plainA              max-abs = {MaxAbs(boundRun2, plainA):e3}  (~0 means ORT reused stale A)");

    // MITIGATION (no ORT upgrade): re-create + re-bind a FRESH input OrtValue for B, keeping
    // the output buffer bound. Does a fresh OrtValue force a fresh device upload?
    using var idsV2 = OrtValue.CreateTensorValueFromMemory<long>(cpu, (long[])idsB.Clone(), [B, C, S]);
    io.BindInput("input_ids", idsV2);
    sess.RunWithBinding(run, io);
    var boundRun3 = (float[])logits.Clone();
    Console.WriteLine($"bound run3 (rebound fresh B OrtValue) vs plainB max-abs = {MaxAbs(boundRun3, plainB):e3}  (want ~0 = mitigation works)");
}

return 0;
