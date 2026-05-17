// C#-side smoke test for the AcousticLM attention-capture path.
//
// Runs ChatterboxPipeline.Lm.Generate with captureAlignment: true on
// a single short input, dumps the resulting (NumSpeechRows, NumTextCols)
// alignment matrix as a flat float32 file alongside a JSON shape descriptor.
// The Python spike (scripts/chatterbox_attention_spike) produces a
// comparable matrix via PyTorch — if the C# probe's output is the same
// shape and roughly the same diagonal pattern, the ONNX → C# pipeline
// is wired correctly.
//
// Run:
//   dotnet run --project scripts/ChatterboxAttentionProbe -c Release -- \
//     --text "Hello world." \
//     --voice /path/to/voice.wav \
//     --onnx-dir /path/to/chatterbox_export \
//     --out /tmp/cs_attn_probe

using Chatterbox.Base;
using Chatterbox.Base.Alignment;
using Vernacula.Base.Models;

string? text = null, voicePath = null, onnxDir = null, outDir = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--text":      text       = args[++i]; break;
        case "--voice":     voicePath  = args[++i]; break;
        case "--onnx-dir":  onnxDir    = args[++i]; break;
        case "--out":       outDir     = args[++i]; break;
        default: Console.Error.WriteLine($"Unknown arg: {args[i]}"); return 2;
    }
}
if (text is null || voicePath is null || onnxDir is null || outDir is null)
{
    Console.Error.WriteLine("Required: --text --voice --onnx-dir --out");
    return 2;
}
Directory.CreateDirectory(outDir);

Console.WriteLine("[load] ChatterboxPipeline...");
using var pipeline = new ChatterboxPipeline(onnxDir, ExecutionProvider.Auto);
if (pipeline.Tokenizer is null)
{
    Console.Error.WriteLine("No tokenizer.json found in the bundle's HF cache.");
    return 1;
}

Console.WriteLine($"[input] text='{text}', voice={voicePath}");
var spk = pipeline.Embedder.Embed(voicePath);
var tokenIds = pipeline.Tokenizer.WrapForLm(text);
Console.WriteLine($"[token] {tokenIds.Length} wrapped LM tokens");

Console.WriteLine("[lm   ] Generate with captureAlignment: true...");
var sw = System.Diagnostics.Stopwatch.StartNew();
var result = pipeline.Lm.Generate(spk.CondEmb, tokenIds, maxSteps: 1024, captureAlignment: true);
sw.Stop();
Console.WriteLine($"[lm   ] {result.Steps} steps, {result.RawGeneratedTokens.Count} tokens in {sw.ElapsedMilliseconds} ms");

if (result.Alignment is null)
{
    Console.Error.WriteLine("[ERROR] alignment capture returned null");
    return 1;
}

int rows = result.Alignment.NumSpeechRows;
int cols = result.Alignment.NumTextCols;
Console.WriteLine($"[align] matrix shape: {rows} speech rows × {cols} text cols");

// Argmax per row — should advance monotonically through text positions
// if the alignment is diagonal. Print first 30 and last 5 as a sanity strip.
var argmaxes = new int[rows];
for (int r = 0; r < rows; r++)
{
    var row = result.Alignment.Row(r);
    int best = 0;
    float bv = row[0];
    for (int c = 1; c < cols; c++) if (row[c] > bv) { bv = row[c]; best = c; }
    argmaxes[r] = best;
}
Console.Write("[align] argmax per row (first 30): ");
for (int i = 0; i < Math.Min(30, rows); i++) Console.Write($"{argmaxes[i]} ");
Console.WriteLine();
if (rows > 30)
{
    Console.Write("[align] argmax per row (last 5):   ");
    for (int i = Math.Max(0, rows - 5); i < rows; i++) Console.Write($"{argmaxes[i]} ");
    Console.WriteLine();
}

// Save the matrix as flat fp32 + a JSON meta. The Python spike's plotter
// can reuse the same heatmap script to compare visually.
var flat = result.Alignment.ToFlatRowMajor();
var binBytes = new byte[flat.Length * sizeof(float)];
Buffer.BlockCopy(flat, 0, binBytes, 0, binBytes.Length);
File.WriteAllBytes(Path.Combine(outDir, "alignment.f32"), binBytes);
File.WriteAllText(Path.Combine(outDir, "alignment.meta.json"),
    $"{{\n  \"rows\": {rows},\n  \"cols\": {cols},\n  \"steps\": {result.Steps},\n  \"text\": \"{text.Replace("\\", "\\\\").Replace("\"", "\\\"")}\"\n}}\n");
Console.WriteLine($"[save ] {outDir}/alignment.f32 + alignment.meta.json");

// ── Exercise the aligner end-to-end ────────────────────────────────
// Synthesize audio so we have a real total-sample count for the
// per-step-seconds conversion.
Console.WriteLine("[voc  ] vocoder synthesis (for audio length)...");
var speechTokens = result.BuildSpeechTokens(spk.AudioTokens);
var audio = pipeline.Vocoder.Synthesize(speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
Console.WriteLine($"[voc  ] {audio.Length} samples @ {ChatterboxConstants.S3GenSr} Hz = {audio.Length / (double)ChatterboxConstants.S3GenSr:F2} s");

var wordTimings = ChatterboxAttentionAligner.Align(
    result.Alignment, text, pipeline.Tokenizer, audio.Length, ChatterboxConstants.S3GenSr);
Console.WriteLine($"[align] aligner produced {wordTimings.Count} word timings:");
foreach (var w in wordTimings)
    Console.WriteLine($"  [{w.StartSeconds:F2}-{w.EndSeconds:F2}s]  {w.Text}");

return 0;
