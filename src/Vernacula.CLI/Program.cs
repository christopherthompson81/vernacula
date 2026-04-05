using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Vernacula.Base;
using Vernacula.Base.Models;
using ParakeetAsr = Vernacula.Base.Parakeet;

// ── Argument parsing ──────────────────────────────────────────────────────────

string? audioPath       = null;
string? modelDir        = null;
string? outputPath      = null;
string? segmentsPath    = null;
string  exportFormat    = "md";
string  diarization     = "sortformer"; // sortformer, diarizen, or vad
float   ahcThreshold    = Config.DiariZenAhcThreshold;
bool    showBenchmark   = false;
bool    skipAsr         = false;
ModelPrecision precision = ModelPrecision.Fp32;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--audio":         audioPath    = args[++i]; break;
        case "--model":         modelDir     = args[++i]; break;
        case "--output":        outputPath   = args[++i]; break;
        case "--segments":      segmentsPath = args[++i]; break;
        case "--export-format": exportFormat = args[++i].ToLowerInvariant(); break;
        case "--diarization":
            diarization = args[++i].ToLowerInvariant();
            if (diarization is not ("sortformer" or "diarizen" or "vad"))
            {
                Console.Error.WriteLine($"Unknown diarization backend: {diarization}. Choose: sortformer, diarizen, vad.");
                return 1;
            }
            break;
        case "--vad":           diarization = "vad"; break; // deprecated but supported
        case "--ahc-threshold": ahcThreshold = float.Parse(args[++i]); break;
        case "--benchmark":     showBenchmark = true; break;
        case "--skip-asr":      skipAsr = true; break;
        case "--precision":
            precision = args[++i].ToLowerInvariant() switch {
                "int8" => ModelPrecision.Int8,
                _      => ModelPrecision.Fp32,
            };
            break;
        case "-h":
        case "--help":
            PrintUsage();
            return 0;
        default:
            Console.Error.WriteLine($"Unknown argument: {args[i]}");
            return 1;
    }
}

if (audioPath is null || modelDir is null)
{
    Console.Error.WriteLine("Error: --audio and --model are required.");
    PrintUsage();
    return 1;
}

if (!File.Exists(audioPath))     { Console.Error.WriteLine($"Audio file not found: {audioPath}");  return 1; }
if (!Directory.Exists(modelDir)) { Console.Error.WriteLine($"Model dir not found: {modelDir}");    return 1; }

if (exportFormat is not ("md" or "txt" or "json" or "srt"))
{
    Console.Error.WriteLine($"Unknown export format: {exportFormat}. Choose: md, txt, json, srt.");
    return 1;
}

// ── Output path with autoincrement ────────────────────────────────────────────

string ext = exportFormat switch { "md" => ".md", "txt" => ".txt", "json" => ".json", "srt" => ".srt", _ => ".md" };

if (outputPath is null)
{
    string baseName = Path.GetFileNameWithoutExtension(audioPath);
    string dir      = Path.GetDirectoryName(Path.GetFullPath(audioPath)) ?? ".";
    string candidate = Path.Combine(dir, baseName + ext);
    if (!File.Exists(candidate))
    {
        outputPath = candidate;
    }
    else
    {
        int n = 1;
        while (File.Exists(Path.Combine(dir, $"{baseName}_{n}{ext}")))
            n++;
        outputPath = Path.Combine(dir, $"{baseName}_{n}{ext}");
    }
}

// ── Ctrl+C cancellation ───────────────────────────────────────────────────────

using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

try
{
    // ── Phase 1: Load audio ───────────────────────────────────────────────────

    Console.Write("Loading audio... ");
    var swLoad = Stopwatch.StartNew();
    var (rawSamples, sampleRate, channels) = AudioUtils.ReadAudio(audioPath);
    float[] audio = AudioUtils.AudioTo16000Mono(rawSamples, sampleRate, channels);
    swLoad.Stop();
    double audioDurationSec = audio.Length / (double)Config.SampleRate;
    Console.WriteLine($"{audioDurationSec:F1}s ({swLoad.ElapsedMilliseconds}ms)");

    cts.Token.ThrowIfCancellationRequested();

    // ── Phase 2: Segmentation (Diarization or VAD) ────────────────────────────

    List<(double start, double end, string spkId)> segs;
    var swDiar = Stopwatch.StartNew();

    if (segmentsPath is not null)
    {
        // Load pre-computed segments from JSON — skips diarization entirely.
        // Expected format: [{start, end, speaker}, ...]
        Console.Write($"Loading segments from {segmentsPath}... ");
        var json = File.ReadAllText(segmentsPath);
        using var doc = JsonDocument.Parse(json);
        segs = doc.RootElement.EnumerateArray()
            .Select(el => (
                el.GetProperty("start").GetDouble(),
                el.GetProperty("end").GetDouble(),
                el.GetProperty("speaker").GetString() ?? "speaker_0"
            ))
            .OrderBy(s => s.Item1)
            .ToList();
        swDiar.Stop();
        Console.WriteLine($"{segs.Count} segment(s) ({swDiar.ElapsedMilliseconds}ms)");
    }
    else if (diarization == "vad")
    {
        Console.Write("Detecting speech (VAD)... ");
        using var vad = new VadSegmenter(modelDir);
        var vadSegs = vad.GetSegments(audio);
        segs = vadSegs.Select(s => (s.start, s.end, "speaker_1")).ToList();
        swDiar.Stop();
        Console.WriteLine($"{segs.Count} segment(s) ({swDiar.ElapsedMilliseconds}ms)");
    }
    else if (diarization == "diarizen")
    {
        Console.Write("Diarizing (DiariZen)... ");
        // Check <modelDir>/diarizen/ subdirectory first (matches Avalonia app layout),
        // then fall back to modelDir root for backward compatibility.
        string diarizenSubDir = Path.Combine(modelDir, "diarizen");
        string diarizenBase   = Directory.Exists(diarizenSubDir) ? diarizenSubDir : modelDir;
        string diarizenModel  = Path.Combine(diarizenBase, Config.DiariZenFile);
        if (!File.Exists(diarizenModel))
        {
            Console.Error.WriteLine($"\nError: DiariZen model not found: {diarizenModel}");
            Console.Error.WriteLine("Expected at <model-dir>/diarizen/diarizen_segmentation.onnx");
            return 1;
        }

        string? embedderModel = Path.Combine(diarizenBase, Config.DiariZenEmbedderFile);
        if (!File.Exists(embedderModel)) embedderModel = null;
        using var diarizer = new DiariZenDiarizer(diarizenModel, embedderModel);
        var diarSegments = diarizer.Diarize(
            audio,
            minSpeakers: 1,
            maxSpeakers: 8,
            ahcThreshold: ahcThreshold,
            progress: message => Console.Write($"\r  {message,-72}"));
        segs = diarSegments.Select(s => (s.Start, s.End, s.Speaker)).ToList();
        swDiar.Stop();
        Console.WriteLine($"\rDiarizing (DiariZen)... {segs.Count} segment(s) ({swDiar.ElapsedMilliseconds}ms)");
    }
    else // sortformer (default)
    {
        Console.Write("Diarizing (Sortformer)... ");
        using var sortformer = new SortformerStreamer(modelDir);
        segs = sortformer.Diarize(audio,
            (idx, total) => Console.Write($"\r  Diarizing chunk {idx}/{total}..."));
        swDiar.Stop();
        Console.WriteLine($"\rDiarizing (Sortformer)... {segs.Count} segment(s) ({swDiar.ElapsedMilliseconds}ms)");
    }

    cts.Token.ThrowIfCancellationRequested();

    // ── Phase 3: ASR ─────────────────────────────────────────────────────────

    var results = new List<(double start, double end, string spkId, string text)>(segs.Count);
    var swAsr = Stopwatch.StartNew();

    if (skipAsr)
    {
        results.AddRange(segs.Select(s => (s.start, s.end, s.spkId, string.Empty)));
        swAsr.Stop();
    }
    else
    {
        var (encoderFile, decoderJointFile) = Config.GetAsrFiles(precision);
        using var parakeet = new ParakeetAsr(modelDir, encoderFile, decoderJointFile);

        int totalSegs = segs.Count;
        int completed = 0;

        Console.WriteLine($"Transcribing {totalSegs} segment(s)...");
        foreach (var (segId, text, _, _, _) in parakeet.Recognize(segs, audio))
        {
            cts.Token.ThrowIfCancellationRequested();
            completed++;
            var (start, end, spkId) = segs[segId];
            results.Add((start, end, spkId, text));
            Console.Write($"\r  {completed}/{totalSegs}");
        }

        Console.WriteLine();
        swAsr.Stop();
    }

    // Sort by start time (Recognize may return results in batch order)
    results.Sort((a, b) => a.start.CompareTo(b.start));

    // ── Phase 4: Export ───────────────────────────────────────────────────────

    string content = exportFormat switch
    {
        "txt"  => BuildTxt(results),
        "json" => BuildJson(results),
        "srt"  => BuildSrt(results),
        _      => BuildMd(results),
    };

    File.WriteAllText(outputPath, content, Encoding.UTF8);
    Console.WriteLine($"Saved → {outputPath}");

    // ── Benchmark summary ─────────────────────────────────────────────────────

    if (showBenchmark)
    {
        long   totalMs = swDiar.ElapsedMilliseconds + swAsr.ElapsedMilliseconds;
        double rtf     = (totalMs / 1000.0) / audioDurationSec;
        Console.WriteLine();
        Console.WriteLine($"Audio duration  : {audioDurationSec:F2}s");
        Console.WriteLine($"Diarization/VAD : {swDiar.ElapsedMilliseconds}ms");
        Console.WriteLine($"ASR             : {swAsr.ElapsedMilliseconds}ms");
        Console.WriteLine($"Total           : {totalMs}ms");
        Console.WriteLine($"Real-time factor: {rtf:F4}  (< 1.0 = faster than real-time)");
    }
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine("\nCancelled.");
    return 130;
}

return 0;

// ── Format builders ───────────────────────────────────────────────────────────

static string BuildMd(List<(double start, double end, string spkId, string text)> results)
{
    var sb = new StringBuilder();
    foreach (var (start, end, spkId, text) in results)
    {
        sb.AppendLine($"## [{spkId}] [{FormatTime(start)} - {FormatTime(end)}]");
        sb.AppendLine();
        sb.AppendLine(text);
        sb.AppendLine();
    }
    return sb.ToString();
}

static string BuildTxt(List<(double start, double end, string spkId, string text)> results)
{
    var sb = new StringBuilder();
    foreach (var (start, end, spkId, text) in results)
    {
        sb.AppendLine($"[{spkId}] {FormatTime(start)} - {FormatTime(end)}");
        sb.AppendLine(text);
        sb.AppendLine();
    }
    return sb.ToString();
}

static string BuildJson(List<(double start, double end, string spkId, string text)> results)
{
    var items = results.Select(r => new
    {
        speaker = r.spkId,
        start   = Math.Round(r.start, 3),
        end     = Math.Round(r.end, 3),
        text    = r.text,
    });
    return JsonSerializer.Serialize(items, new JsonSerializerOptions { WriteIndented = true });
}

static string BuildSrt(List<(double start, double end, string spkId, string text)> results)
{
    var sb  = new StringBuilder();
    int idx = 1;
    foreach (var (start, end, spkId, text) in results)
    {
        sb.AppendLine(idx.ToString());
        sb.AppendLine($"{SrtTime(start)} --> {SrtTime(end)}");
        sb.AppendLine($"[{spkId}] {text}");
        sb.AppendLine();
        idx++;
    }
    return sb.ToString();
}

static string FormatTime(double seconds)
    => TimeSpan.FromSeconds(seconds).ToString(@"hh\:mm\:ss");

static string SrtTime(double seconds)
{
    var ts = TimeSpan.FromSeconds(seconds);
    return $"{ts.Hours:D2}:{ts.Minutes:D2}:{ts.Seconds:D2},{ts.Milliseconds:D3}";
}

static void PrintUsage()
{
    Console.WriteLine("Usage: parakeet-cli --audio <file> --model <dir> [options]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --segments <path>                  Load pre-computed segments JSON, skip diarization");
    Console.WriteLine("  --export-format <md|txt|json|srt>  Output format (default: md)");
    Console.WriteLine("  --output <path>                    Override output file path");
    Console.WriteLine("  --diarization <backend>            Diarization backend: sortformer, diarizen, vad");
    Console.WriteLine("                                     (default: sortformer)");
    Console.WriteLine("  --vad                              Use VAD instead of diarization (deprecated)");
    Console.WriteLine("  --precision <fp32|int8>            Model precision (default: fp32)");
    Console.WriteLine("  --skip-asr                         Export diarization/VAD segments without transcription");
    Console.WriteLine("  --benchmark                        Print timing / RTF after transcription");
    Console.WriteLine("  -h, --help                         Show this help");
    Console.WriteLine();
    Console.WriteLine("Build: dotnet build -c Release -p:EP=Cuda|DirectML|Cpu -p:Platform=x64");
}
