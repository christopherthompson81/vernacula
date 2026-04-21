using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Vernacula.Base;
using Vernacula.Base.Models;
using ParakeetAsr = Vernacula.Base.Parakeet;

// ── Argument parsing ──────────────────────────────────────────────────────────

string? audioPath       = null;
string? modelDir        = null;
string? outputPath      = null;
string? segmentsPath    = null;
string  exportFormat    = "md";
string? diarization     = null;         // null = pick default for ASR backend
float   ahcThreshold    = Config.DiariZenAhcThreshold;
bool    showBenchmark   = false;
bool    skipAsr         = false;
string  asrBackend      = "parakeet";   // parakeet, cohere, qwen3asr, or vibevoice
string? cohereModelDir  = null;         // defaults to <modelDir>/cohere_transcribe
string? cohereLanguage  = null;         // ISO 639-1 forced language (e.g. "en")
string? qwen3AsrModelDir = null;        // defaults to <modelDir>/qwen3asr
bool    forceQwen3AsrSerial = false;    // --qwen3asr-serial disables experimental batching
GraphOptimizationLevel qwen3AsrOrtOptLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
string? vibevoiceModelDir = null;       // defaults to <modelDir>/vibevoice_asr
string? profileOutputDir  = null;       // ORT profiling output dir (vibevoice only)
int     profileMaxTokens  = 200;        // cap maxNewTokens during profiling to stay under ORT 1M event limit
double  minAsrSeconds     = 5.0;        // minimum group span (seconds) when using segmented VibeVoice ASR
double  asrBufferSeconds  = 0.0;        // audio padding on each side of a group (seconds)
bool    profileSortformer = false;      // --profile-sortformer: print fine-grained timing for Sortformer
bool    downloadVoxLingua = false;      // --download-voxlingua: fetch the LID model, then exit
bool    runLid             = false;      // --lid: run LID on --audio and print result, then exit
bool    runWhisperCheck    = false;      // --whisper-check: Phase 2a sanity test (mel + encoder only)
int     whisperBatchSize   = 1;          // --whisper-batch N: N>=2 uses RecognizeBatched (see investigation doc — batching is currently slower than sequential on short-variable-segment workloads; wire up for testing)
ModelPrecision precision = ModelPrecision.Fp32;
int     parakeetBeam      = 1;          // --parakeet-beam N: 1 = greedy (default), >1 = TDT beam search
string? parakeetLmPath    = null;       // --lm <path> to ARPA(.gz) subword n-gram model; implies beam ≥ 4
float   parakeetLmWeight  = 0.3f;       // --lm-weight <w> shallow fusion scalar
float   parakeetLmLen     = 0.6f;       // --lm-length-penalty <p> per-emitted-token reward to offset LM shortening bias

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
            if (diarization is not ("sortformer" or "diarizen" or "vad" or "vibevoice-asr-builtin"))
            {
                Console.Error.WriteLine($"Unknown diarization backend: {diarization}. Choose: sortformer, diarizen, vad, vibevoice-asr-builtin.");
                return 1;
            }
            break;
        case "--vad":           diarization = "vad"; break; // deprecated but supported
        case "--ahc-threshold": ahcThreshold = float.Parse(args[++i]); break;
        case "--benchmark":     showBenchmark = true; break;
        case "--skip-asr":      skipAsr = true; break;
        case "--asr":
            asrBackend = args[++i].ToLowerInvariant();
            if (asrBackend is not ("parakeet" or "cohere" or "qwen3asr" or "vibevoice" or "whisper"))
            {
                Console.Error.WriteLine($"Unknown ASR backend: {asrBackend}. Choose: parakeet, cohere, qwen3asr, vibevoice, whisper.");
                return 1;
            }
            break;
        case "--cohere-model":     cohereModelDir    = args[++i]; break;
        case "--qwen3asr-model":   qwen3AsrModelDir  = args[++i]; break;
        case "--qwen3asr-serial":  forceQwen3AsrSerial = true; break;
        case "--qwen3asr-ort-opt":
            string qwenOptLevel = args[++i].ToLowerInvariant();
            qwen3AsrOrtOptLevel = qwenOptLevel switch
            {
                "extended" => GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "basic" => GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "disabled" => GraphOptimizationLevel.ORT_DISABLE_ALL,
                _ => (GraphOptimizationLevel)(-1),
            };
            if ((int)qwen3AsrOrtOptLevel == -1)
            {
                Console.Error.WriteLine($"Unknown Qwen ORT optimization level: {qwenOptLevel}. Choose: extended, basic, disabled.");
                return 1;
            }
            break;
        case "--vibevoice-model":  vibevoiceModelDir = args[++i]; break;
        case "--min-asr-seconds":  minAsrSeconds     = double.Parse(args[++i]); break;
        case "--asr-buffer":       asrBufferSeconds  = double.Parse(args[++i]); break;
        case "--profile":          profileOutputDir  = args[++i]; break;
        case "--profile-steps":    profileMaxTokens  = int.Parse(args[++i]); break;
        case "--profile-sortformer": profileSortformer = true; break;
        case "--download-voxlingua": downloadVoxLingua = true; break;
        case "--lid":                runLid = true; break;
        case "--whisper-check":      runWhisperCheck = true; break;
        case "--whisper-batch":
            if (!int.TryParse(args[++i], out whisperBatchSize) || whisperBatchSize < 1)
            {
                Console.Error.WriteLine("--whisper-batch expects a positive integer (1 = sequential, 2+ = batched).");
                return 1;
            }
            break;
        case "--language":        cohereLanguage = args[++i]; break;
        case "--precision":
            precision = args[++i].ToLowerInvariant() switch {
                "int8" => ModelPrecision.Int8,
                _      => ModelPrecision.Fp32,
            };
            break;
        case "--parakeet-beam":
            if (!int.TryParse(args[++i], out parakeetBeam) || parakeetBeam < 1)
            {
                Console.Error.WriteLine("--parakeet-beam expects a positive integer (1 = greedy).");
                return 1;
            }
            break;
        case "--lm":
            parakeetLmPath = args[++i];
            break;
        case "--lm-weight":
            if (!float.TryParse(args[++i], System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture, out parakeetLmWeight))
            {
                Console.Error.WriteLine("--lm-weight expects a float (typical range 0.1–0.5).");
                return 1;
            }
            break;
        case "--lm-length-penalty":
            if (!float.TryParse(args[++i], System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture, out parakeetLmLen))
            {
                Console.Error.WriteLine("--lm-length-penalty expects a float (typical 0.0–1.0).");
                return 1;
            }
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

// ── Model-management actions (don't require --audio) ─────────────────────────

if (downloadVoxLingua)
{
    // The Avalonia app's SettingsService.DefaultModelsDir resolves to
    // <LocalApplicationData>/Vernacula/models; match that when the caller
    // didn't override via --model so the downloaded assets land where the
    // GUI would expect them.
    string modelsRoot = modelDir ?? DefaultModelsDir();
    string destDir = Path.Combine(modelsRoot, Config.VoxLinguaSubDir);
    return await DownloadVoxLinguaAsync(destDir);
}

if (runLid)
{
    if (audioPath is null)
    {
        Console.Error.WriteLine("Error: --lid requires --audio <file>.");
        return 1;
    }
    string modelsRoot = modelDir ?? DefaultModelsDir();
    return RunLidAction(audioPath, modelsRoot);
}

if (runWhisperCheck)
{
    if (audioPath is null)
    {
        Console.Error.WriteLine("Error: --whisper-check requires --audio <file>.");
        return 1;
    }
    string modelsRoot = modelDir ?? DefaultModelsDir();
    return RunWhisperCheckAction(audioPath, modelsRoot);
}

if (audioPath is null || modelDir is null)
{
    Console.Error.WriteLine("Error: --audio and --model are required.");
    PrintUsage();
    return 1;
}

// Resolve diarization default based on ASR backend
diarization ??= asrBackend == "vibevoice" ? "vibevoice-asr-builtin" : "sortformer";

// Validate that vibevoice-asr-builtin is only used with vibevoice ASR
if (diarization == "vibevoice-asr-builtin" && asrBackend != "vibevoice")
{
    Console.Error.WriteLine("Error: --diarization vibevoice-asr-builtin requires --asr vibevoice.");
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
int cancelPressCount = 0;
Console.CancelKeyPress += (_, e) =>
{
    int press = Interlocked.Increment(ref cancelPressCount);
    if (press == 1)
    {
        e.Cancel = true;
        Console.Error.WriteLine("\nCancellation requested. Press Ctrl+C again to force exit.");
        cts.Cancel();
        return;
    }

    Console.Error.WriteLine("\nForce exiting...");
    Environment.Exit(130);
};

try
{
    // ── Phase 1: Load audio ───────────────────────────────────────────────────

    Console.Write("Loading audio... ");
    var swLoad = Stopwatch.StartNew();
    var (rawSamples, sampleRate, channels) = AudioUtils.ReadAudio(audioPath);
    swLoad.Stop();

    float[] audio = AudioUtils.AudioTo16000Mono(rawSamples, sampleRate, channels);

    double audioDurationSec = audio.Length / (double)Config.SampleRate;
    Console.WriteLine($"Audio: {audioDurationSec:F1}s (load: {swLoad.ElapsedMilliseconds}ms)");

    cts.Token.ThrowIfCancellationRequested();

    // ── Phase 2: Segmentation (Diarization or VAD) ────────────────────────────

    // When using VibeVoice built-in diarization, Phase 2 is skipped entirely —
    // the model handles segmentation and speaker assignment internally.
    List<(double start, double end, string spkId)> segs;
    var swDiar = Stopwatch.StartNew();

    if (diarization == "vibevoice-asr-builtin" && segmentsPath is null)
    {
        segs = []; // populated in Phase 3
        swDiar.Stop();
    }
    else if (segmentsPath is not null)
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
        if (profileSortformer)
        {
            // ── Profiled Sortformer path ──────────────────────────────────────
            var sw = Stopwatch.StartNew();

            var sw1 = Stopwatch.StartNew();
            Console.Write("Loading Sortformer model... ");
            using var sortformer = new SortformerStreamer(modelDir);
            Console.WriteLine($"DONE ({sw1.ElapsedMilliseconds,6} ms)");

            var sw2 = Stopwatch.StartNew();
            Console.Write("Computing mel spectrogram... ");
            float[,,] melSpec = AudioUtils.LogMelSpectrogram(audio);
            Console.WriteLine($"DONE ({sw2.ElapsedMilliseconds,6} ms)");

            var sw3 = Stopwatch.StartNew();
            Console.Write("Computing pred params... ");
            var (totalFrames, chunkStride, numChunks) = sortformer.GetPredParams(melSpec);
            Console.WriteLine($"DONE ({sw3.ElapsedMilliseconds,6} ms)  (frames={totalFrames}, chunks={numChunks})");

            var sw4 = Stopwatch.StartNew();
            Console.Write("Processing chunks (ONNX inference)... ");
            var allPreds = new List<float[,]>(numChunks);
            int chunkIdx = 0;
            foreach (var (nc, idx, chunkPreds) in sortformer.GetPreds(melSpec, totalFrames, chunkStride, numChunks))
            {
                allPreds.Add(chunkPreds);
                chunkIdx = idx + 1;
                Console.Write($"\r  chunk {chunkIdx}/{numChunks}...");
            }
            Console.WriteLine($"\rDONE ({sw4.ElapsedMilliseconds,6} ms)  ({numChunks} chunks)");

            var sw5 = Stopwatch.StartNew();
            Console.Write("Filtering predictions (median filter)... ");
            var (numPredFrames, medFiltered) = sortformer.FilterPreds(allPreds, totalFrames);
            Console.WriteLine($"DONE ({sw5.ElapsedMilliseconds,6} ms)");

            var sw6 = Stopwatch.StartNew();
            Console.Write("Binarizing segments... ");
            segs = sortformer.BinarizePredToSegments(numPredFrames, medFiltered);
            Console.WriteLine($"DONE ({sw6.ElapsedMilliseconds,6} ms)  ({segs.Count} segments)");

            sw.Stop();
            Console.WriteLine();
            Console.WriteLine(new string('=', 64));
            Console.WriteLine("Sortformer profile breakdown:");
            Console.WriteLine(new string('=', 64));
            Console.WriteLine($"  1. ONNX Runtime model load   : {sw1.ElapsedMilliseconds,6} ms");
            Console.WriteLine($"  2. Mel spectrogram           : {sw2.ElapsedMilliseconds,6} ms");
            Console.WriteLine($"  3. Pred params               : {sw3.ElapsedMilliseconds,6} ms");
            Console.WriteLine($"  4. Chunk inference (all)     : {sw4.ElapsedMilliseconds,6} ms  ({sw4.ElapsedMilliseconds / numChunks,6} ms/chunk)");
            Console.WriteLine($"  5. Median filter             : {sw5.ElapsedMilliseconds,6} ms");
            Console.WriteLine($"  6. Binarization              : {sw6.ElapsedMilliseconds,6} ms");
            Console.WriteLine(new string('-', 64));
            Console.WriteLine($"  TOTAL Sortformer             : {sw.ElapsedMilliseconds,6} ms");
            Console.WriteLine(new string('=', 64));
        }
        else
        {
            Console.Write("Diarizing (Sortformer)... ");
            using var sortformer = new SortformerStreamer(modelDir);
            segs = sortformer.Diarize(audio,
                (idx, total) => Console.Write($"\r  Diarizing chunk {idx}/{total}..."));
            swDiar.Stop();
            Console.WriteLine($"\rDiarizing (Sortformer)... {segs.Count} segment(s) ({swDiar.ElapsedMilliseconds}ms)");
        }
    }

    cts.Token.ThrowIfCancellationRequested();

    // ── Phase 3: ASR ─────────────────────────────────────────────────────────

    var results = new List<(double start, double end, string spkId, string text)>(segs.Count);
    var swAsr = Stopwatch.StartNew();
    QwenExperimentalBatchBenchmark? qwenBatchBenchmark = null;

    if (skipAsr)
    {
        results.AddRange(segs.Select(s => (s.start, s.end, s.spkId, string.Empty)));
        swAsr.Stop();
    }
    else if (asrBackend == "vibevoice")
    {
        string vibevoiceDir = vibevoiceModelDir
            ?? (Directory.Exists(Path.Combine(modelDir, Config.VibeVoiceSubDir))
                ? Path.Combine(modelDir, Config.VibeVoiceSubDir)
                : modelDir);

        if (!File.Exists(Path.Combine(vibevoiceDir, VibeVoiceAsr.AudioEncoderFile)))
        {
            Console.Error.WriteLine($"\nError: VibeVoice-ASR model not found in: {vibevoiceDir}");
            Console.Error.WriteLine($"Expected {VibeVoiceAsr.AudioEncoderFile} and related files there.");
            Console.Error.WriteLine("Use --vibevoice-model <dir> to specify the directory explicitly.");
            return 1;
        }

        // Persist the encoder across calls only in segmented mode (vad / diarizen).
        // In built-in whole-recording mode the encoder is created and disposed
        // inside each Transcribe() call so its VRAM is freed before the long decode.
        bool persistEncoder = diarization != "vibevoice-asr-builtin";
        using var vibevoice = new VibeVoiceAsr(vibevoiceDir, persistEncoder: persistEncoder,
            profileOutputDir: profileOutputDir);

        if (diarization == "vibevoice-asr-builtin")
        {
            // Whole-recording path: VibeVoice handles segmentation internally.
            Console.Write("Transcribing (VibeVoice-ASR built-in diarization)... ");
            var vibevoiceSegs = vibevoice.Transcribe(rawSamples, sampleRate, channels,
                maxNewTokens: profileOutputDir is not null ? profileMaxTokens : 8_192,
                ct: cts.Token);
            swAsr.Stop();

            foreach (var seg in vibevoiceSegs)
                results.Add((seg.Start, seg.End, $"speaker_{seg.Speaker}", seg.Content));

            Console.WriteLine($"{results.Count} segment(s) ({swAsr.ElapsedMilliseconds}ms)");
        }
        else
        {
            // Segmented path: use VAD/diarizer segments from Phase 2, merge short
            // groups, and run VibeVoice on each group independently.
            var rawGroups = segs.Select(s => (s.start, s.end)).ToList();
            var groups    = VadSegmenter.MergeShortGroups(rawGroups, minAsrSeconds);

            Console.WriteLine($"Transcribing {groups.Count} group(s) from {segs.Count} segment(s) " +
                              $"(VibeVoice-ASR, min {minAsrSeconds:F1}s)...");

            // Collect VibeVoice sub-segments with absolute timestamps.
            var vibeSegs = new List<(double start, double end, string spkId, string text)>();
            int completed = 0;
            foreach (var (grpStart, grpEnd) in groups)
            {
                cts.Token.ThrowIfCancellationRequested();

                // Slice raw audio at original sample rate (channel-frame-aligned).
                // Pad each side by asrBufferSeconds to give VibeVoice context at
                // group boundaries; the midpoint filter below discards any output
                // segment whose midpoint falls outside [grpStart, grpEnd], so the
                // same buffer audio in adjacent groups never produces duplicate text.
                double sliceStart = Math.Max(0, grpStart - asrBufferSeconds);
                double sliceEnd   = Math.Min(rawSamples.Length / (double)(sampleRate * channels),
                                             grpEnd + asrBufferSeconds);
                int startFrame = (int)(sliceStart * sampleRate);
                int endFrame   = Math.Min((int)Math.Ceiling(sliceEnd * sampleRate),
                                          rawSamples.Length / channels);
                float[] slice  = rawSamples[(startFrame * channels)..(endFrame * channels)];
                double sliceStartSec = (double)startFrame / sampleRate;

                // Token cap: proportional to the full slice duration (including
                // buffer) so hallucination loops are still bounded.
                double sliceDuration = (double)(endFrame - startFrame) / sampleRate;
                int groupMaxTokens = Math.Min(8_192, (int)(sliceDuration * 40) + 150);

                var groupSegs = vibevoice.Transcribe(slice, sampleRate, channels,
                    maxNewTokens: groupMaxTokens,
                    ct: cts.Token);

                // Convert to absolute timestamps and apply midpoint filter.
                // Only keep segments whose midpoint lies within [grpStart, grpEnd]
                // — this deduplicates the buffer overlap region between groups.
                foreach (var seg in groupSegs)
                {
                    double absStart = sliceStartSec + seg.Start;
                    double absEnd   = sliceStartSec + seg.End;
                    double mid      = (absStart + absEnd) / 2.0;
                    if (mid >= grpStart && mid <= grpEnd)
                        vibeSegs.Add((absStart, absEnd, $"speaker_{seg.Speaker}", seg.Content));
                }

                completed++;
                Console.Write($"\r  Group {completed}/{groups.Count} → {vibeSegs.Count} sub-segment(s)...");
            }

            swAsr.Stop();

            // Use VibeVoice's segments directly — they carry accurate timestamps
            // and meaningful speech boundaries.  Assign each VibeVoice segment the
            // speaker ID from whichever diarizer segment overlaps it most.
            //
            // Attempting to redistribute text back onto the original diarizer
            // segment boundaries causes blank segments and repeated text: when a
            // VibeVoice sub-segment spans multiple diarizer segments, all of its
            // text falls on one bucket and the others stay empty.
            foreach (var vibe in vibeSegs.OrderBy(v => v.start))
            {
                if (string.IsNullOrWhiteSpace(vibe.text))
                    continue;

                // Find diarizer segment with maximum time overlap.
                int    bestIdx     = -1;
                double bestOverlap = 0;
                for (int i = 0; i < segs.Count; i++)
                {
                    double ov = Math.Max(0,
                        Math.Min(vibe.end, segs[i].end) - Math.Max(vibe.start, segs[i].start));
                    if (ov > bestOverlap) { bestOverlap = ov; bestIdx = i; }
                }
                if (bestIdx < 0) // no overlap — fall back to nearest midpoint
                {
                    double vibeMid = (vibe.start + vibe.end) / 2;
                    bestIdx = 0; double bestDist = double.MaxValue;
                    for (int i = 0; i < segs.Count; i++)
                    {
                        double dist = Math.Abs((segs[i].start + segs[i].end) / 2 - vibeMid);
                        if (dist < bestDist) { bestDist = dist; bestIdx = i; }
                    }
                }
                results.Add((vibe.start, vibe.end, segs[bestIdx].spkId, vibe.text));
            }

            Console.WriteLine($"\r  {groups.Count} group(s) → {vibeSegs.Count} VibeVoice sub-segment(s) " +
                              $"→ {results.Count} output segment(s) ({swAsr.ElapsedMilliseconds}ms)");
        }
    }
    else if (asrBackend == "cohere")
    {
        string cohereDir = cohereModelDir
            ?? (Directory.Exists(Path.Combine(modelDir, "cohere_transcribe"))
                ? Path.Combine(modelDir, "cohere_transcribe")
                : modelDir);

        if (!File.Exists(Path.Combine(cohereDir, CohereTranscribe.MelFile)))
        {
            Console.Error.WriteLine($"\nError: Cohere model not found in: {cohereDir}");
            Console.Error.WriteLine($"Expected {CohereTranscribe.MelFile} and related files there.");
            Console.Error.WriteLine("Use --cohere-model <dir> to specify the directory explicitly.");
            return 1;
        }

        using var cohere = new CohereTranscribe(cohereDir);
        int totalSegs = segs.Count;
        int completed = 0;

        Console.WriteLine($"Transcribing {totalSegs} segment(s) (Cohere)...");
        foreach (var (segId, text, meta) in cohere.Recognize(segs, audio,
                     forceLanguage: cohereLanguage))
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
    else if (asrBackend == "qwen3asr")
    {
        string qwen3AsrDir = qwen3AsrModelDir
            ?? (Directory.Exists(Path.Combine(modelDir, Config.Qwen3AsrSubDir))
                ? Path.Combine(modelDir, Config.Qwen3AsrSubDir)
                : modelDir);

        if (!File.Exists(Path.Combine(qwen3AsrDir, Qwen3Asr.EncoderFile)))
        {
            Console.Error.WriteLine($"\nError: Qwen3-ASR model not found in: {qwen3AsrDir}");
            Console.Error.WriteLine($"Expected {Qwen3Asr.EncoderFile} and related files there.");
            Console.Error.WriteLine("Use --qwen3asr-model <dir> to specify the directory explicitly.");
            return 1;
        }

        bool hasExperimentalQwenBatchingArtifacts;
        {
            bool hasBatchedFiles = !forceQwen3AsrSerial &&
                                   File.Exists(Path.Combine(qwen3AsrDir, Qwen3Asr.EncoderBatchedFile)) &&
                                   (File.Exists(Path.Combine(qwen3AsrDir, Qwen3Asr.DecoderFile)) ||
                                    File.Exists(Path.Combine(qwen3AsrDir, Qwen3Asr.DecoderInitBatchedFile)));
            using var qwen3Asr = new Qwen3Asr(
                qwen3AsrDir,
                preferBatched: hasBatchedFiles,
                optimizationLevel: qwen3AsrOrtOptLevel);
            int totalSegs = segs.Count;
            int completed = 0;
            hasExperimentalQwenBatchingArtifacts = qwen3Asr.HasExperimentalBatchingArtifacts();
            string batchNote = forceQwen3AsrSerial
                ? " (serial forced)"
                : hasExperimentalQwenBatchingArtifacts ? " (batched encoder+prefill)" : "";

            Console.WriteLine($"Transcribing {totalSegs} segment(s) (Qwen3-ASR{batchNote})...");
            var recognitionResults = hasExperimentalQwenBatchingArtifacts
                ? qwen3Asr.RecognizeBatchedDetailed(segs, audio)
                : qwen3Asr.RecognizeDetailed(segs, audio);

            foreach (var result in recognitionResults)
            {
                cts.Token.ThrowIfCancellationRequested();
                completed++;
                var (start, end, spkId) = segs[result.SegmentId];
                results.Add((start, end, spkId, result.Text));
                Console.Write($"\r  {completed}/{totalSegs}");
            }
        }

        Console.WriteLine();
        swAsr.Stop();

        if (showBenchmark)
        {
            if (hasExperimentalQwenBatchingArtifacts)
            {
                Console.Write("Benchmarking experimental Qwen batching... ");
                qwenBatchBenchmark = Qwen3Asr.BenchmarkExperimentalBatching(
                    qwen3AsrDir,
                    ExecutionProvider.Auto,
                    segs,
                    audio,
                    qwen3AsrOrtOptLevel);
                Console.WriteLine("done");
            }
            else
            {
                Console.WriteLine("Experimental Qwen batching artifacts not found; skipping batching benchmark.");
            }
        }
    }
    else if (asrBackend == "whisper")
    {
        string whisperDir = Path.Combine(modelDir, Config.WhisperTurboSubDir);
        string[] required = [
            WhisperTurbo.EncoderFile,
            WhisperTurbo.DecoderInitFile,
            WhisperTurbo.DecoderStepFile,
            WhisperTurbo.TokenizerFile,
            WhisperTurbo.GenerationConfigFile,
        ];
        foreach (string f in required)
        {
            if (!File.Exists(Path.Combine(whisperDir, f)))
            {
                Console.Error.WriteLine($"\nError: Whisper-turbo model missing: {whisperDir}/{f}");
                Console.Error.WriteLine(
                    "Select the WhisperTurbo ASR backend in the app and download the model files, " +
                    "or copy them into that directory.");
                return 1;
            }
        }

        using var whisper = new WhisperTurbo(whisperDir);
        int totalSegs = segs.Count;
        int completed = 0;

        var recognitionResults = whisperBatchSize > 1
            ? whisper.RecognizeBatched(segs, audio, forceLanguage: cohereLanguage,
                                       initialBatchSize: whisperBatchSize)
            : whisper.Recognize(segs, audio, forceLanguage: cohereLanguage);
        string batchNote = whisperBatchSize > 1 ? $", batch={whisperBatchSize}" : "";

        Console.WriteLine($"Transcribing {totalSegs} segment(s) (Whisper-turbo{batchNote})...");
        foreach (var result in recognitionResults)
        {
            cts.Token.ThrowIfCancellationRequested();
            completed++;
            var (start, end, spkId) = segs[result.SegmentId];
            results.Add((start, end, spkId, result.Text));
            Console.Write($"\r  {completed}/{totalSegs}");
        }
        Console.WriteLine();
        swAsr.Stop();
    }
    else
    {
        var (encoderFile, decoderJointFile) = Config.GetAsrFiles(precision);

        // LM fusion only takes effect during beam search. Auto-bump to a
        // minimal beam when --lm is passed without an explicit --parakeet-beam
        // so `--lm foo.arpa` Just Works.
        int effectiveBeam = parakeetLmPath != null && parakeetBeam < 2 ? 4 : parakeetBeam;

        using var parakeet = new ParakeetAsr(modelDir, encoderFile, decoderJointFile,
            beamWidth: effectiveBeam);

        if (parakeetLmPath != null)
        {
            Console.Write($"Loading language model {Path.GetFileName(parakeetLmPath)}... ");
            var lmSw = Stopwatch.StartNew();
            parakeet.LmScorer       = KenLmScorer.LoadArpa(parakeetLmPath);
            parakeet.LmWeight       = parakeetLmWeight;
            parakeet.LmLengthPenalty = parakeetLmLen;
            Console.WriteLine($"order={parakeet.LmScorer.Order} weight={parakeetLmWeight:F2} length-penalty={parakeetLmLen:F2} ({lmSw.ElapsedMilliseconds}ms)");
        }

        int totalSegs = segs.Count;
        int completed = 0;

        Console.WriteLine($"Transcribing {totalSegs} segment(s)...");
        foreach (var (segId, text, _, _, _, _) in parakeet.Recognize(segs, audio))
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

        if (qwenBatchBenchmark is not null)
        {
            double serialStageMs = qwenBatchBenchmark.SerialEncoderMilliseconds + qwenBatchBenchmark.SerialPrefillMilliseconds;
            double batchedStageMs = qwenBatchBenchmark.BatchedEncoderMilliseconds + qwenBatchBenchmark.BatchedPrefillMilliseconds;
            double speedup = batchedStageMs > 0 ? serialStageMs / batchedStageMs : 0;

            Console.WriteLine();
            Console.WriteLine("Qwen Experimental Batching (encoder + prefill only):");
            Console.WriteLine($"  Free VRAM      : {qwenBatchBenchmark.FreeGpuMemoryMb} MB");
            Console.WriteLine($"  Segments       : {qwenBatchBenchmark.SegmentCount}");
            Console.WriteLine($"  Batch runs     : {qwenBatchBenchmark.BatchRuns}");
            Console.WriteLine($"  Seconds ceiling: {qwenBatchBenchmark.TotalSecondsCeiling:F0}s");
            Console.WriteLine($"  Serial encoder : {qwenBatchBenchmark.SerialEncoderMilliseconds:F1}ms");
            Console.WriteLine($"  Serial prefill : {qwenBatchBenchmark.SerialPrefillMilliseconds:F1}ms");
            Console.WriteLine($"  Batched encoder: {qwenBatchBenchmark.BatchedEncoderMilliseconds:F1}ms");
            Console.WriteLine($"  Batched prefill: {qwenBatchBenchmark.BatchedPrefillMilliseconds:F1}ms");
            Console.WriteLine($"  Stage speedup  : {speedup:F2}x");
            foreach (var batch in qwenBatchBenchmark.Batches)
            {
                Console.WriteLine(
                    $"    batch segs={batch.SegmentCount}, total={batch.TotalSeconds:F1}s, max={batch.MaxSegmentSeconds:F1}s, " +
                    $"encoder={batch.EncoderMilliseconds:F1}ms, prefill={batch.PrefillMilliseconds:F1}ms");
            }
        }
    }
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine("\nCancelled.");
    return 130;
}
catch (OnnxRuntimeException ex) when (IsLikelyOutOfMemory(ex))
{
    Console.Error.WriteLine("\nONNX Runtime ran out of memory while loading or running the model.");
    Console.Error.WriteLine("Free GPU memory and try again, or rerun with a smaller workload / CPU build.");
    Console.Error.WriteLine(ex.Message);
    return 1;
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
        sb.AppendLine($"[{spkId}] {text}".Trim());
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

static bool IsLikelyOutOfMemory(OnnxRuntimeException ex)
{
    string message = ex.Message;
    return message.Contains("out of memory", StringComparison.OrdinalIgnoreCase)
        || message.Contains("failed to allocate memory", StringComparison.OrdinalIgnoreCase)
        || message.Contains("cuda out of memory", StringComparison.OrdinalIgnoreCase)
        || message.Contains("bfcarena", StringComparison.OrdinalIgnoreCase);
}

/// <summary>
/// Resolves the same default models directory the Avalonia SettingsService
/// uses (<c>LocalApplicationData/Vernacula/models</c>), so CLI downloads
/// land where the GUI would expect them.
/// </summary>
static string DefaultModelsDir() => Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
    "Vernacula", "models");

/// <summary>
/// Headless LID validation: load audio, run VAD, pick the longest segment,
/// classify with VoxLinguaLid (escalating to a longer window on ambiguity),
/// and print a structured result. No diarization, no ASR.
///
/// Mirrors what <c>LangIdService.DetectLanguage</c> does inside the Avalonia
/// pipeline — both call <c>VoxLinguaLid.ClassifyLongestSegment</c>, so this
/// exercise is a real integration test of the LID code path.
/// </summary>
static int RunLidAction(string audioPath, string modelsRoot)
{
    string voxDir = Path.Combine(modelsRoot, Config.VoxLinguaSubDir);
    string vadDir = Path.Combine(modelsRoot, Config.VadSubDir);

    if (!File.Exists(audioPath))
    {
        Console.Error.WriteLine($"Audio file not found: {audioPath}");
        return 1;
    }
    if (!File.Exists(Path.Combine(voxDir, Config.VoxLinguaModelFile)))
    {
        Console.Error.WriteLine(
            $"VoxLingua107 model not found at {voxDir}. " +
            $"Run --download-voxlingua first.");
        return 1;
    }
    if (!File.Exists(Path.Combine(vadDir, Config.VadFile)))
    {
        Console.Error.WriteLine(
            $"Silero VAD model not found at {vadDir}/{Config.VadFile}. " +
            $"The --lid action uses VAD to pick the best speech segment; " +
            $"install the core diarization model pack and retry.");
        return 1;
    }

    Console.WriteLine($"[lid] loading audio: {audioPath}");
    var (raw, sr, channels) = AudioUtils.ReadAudio(audioPath);
    float[] audio = AudioUtils.AudioTo16000Mono(raw, sr, channels);
    Console.WriteLine($"[lid] decoded: {audio.Length} samples @ 16 kHz ({audio.Length / 16000.0:F1} s)");

    Console.WriteLine("[lid] running Silero VAD");
    List<(double start, double end)> vadSegs;
    using (var vad = new VadSegmenter(vadDir))
        vadSegs = vad.GetSegments(audio);

    if (vadSegs.Count == 0)
    {
        Console.Error.WriteLine("[lid] VAD found no speech segments — nothing to classify.");
        return 1;
    }
    var longest = vadSegs.OrderByDescending(s => s.end - s.start).First();
    Console.WriteLine(
        $"[lid] VAD: {vadSegs.Count} segments; longest is " +
        $"{longest.start:F2}s–{longest.end:F2}s ({longest.end - longest.start:F2}s)");

    Console.WriteLine($"[lid] loading VoxLinguaLid from {voxDir}");
    var sw = System.Diagnostics.Stopwatch.StartNew();
    using var lid = new VoxLinguaLid(voxDir);
    Console.WriteLine($"[lid] loaded in {sw.ElapsedMilliseconds} ms");

    sw.Restart();
    var result = lid.ClassifyLongestSegment(audio, vadSegs);
    sw.Stop();
    if (result is null)
    {
        Console.Error.WriteLine("[lid] classification returned null (longest VAD segment too short).");
        return 1;
    }

    Console.WriteLine();
    Console.WriteLine($"Detected language : {result.Top.Name} ({result.Iso})  p={result.TopProbability:P1}");
    Console.WriteLine($"Ambiguous         : {result.IsAmbiguous}");
    Console.WriteLine($"Clip duration     : {result.ClipDurationSeconds:F1} s");
    Console.WriteLine($"Classify latency  : {sw.ElapsedMilliseconds} ms");
    Console.WriteLine("Top-5 candidates  :");
    foreach (var c in result.TopK)
        Console.WriteLine($"  {c.Iso,-4} {c.Name,-20} {c.Probability:P2}");
    Console.WriteLine();
    Console.WriteLine("Summary           : " + result.FormatSummary());
    return 0;
}

/// <summary>
/// Phase-2a sanity check: load audio → 16 kHz mono → log-mel (128 × 3000) →
/// encoder → print shape and stats.  No decoder, no tokenizer.  Exercises
/// the mel frontend and encoder session in isolation so we can confirm
/// the ONNX export loads and produces sensible activations before building
/// out the decode loop.
/// </summary>
static int RunWhisperCheckAction(string audioPath, string modelsRoot)
{
    string whisperDir = Path.Combine(modelsRoot, Config.WhisperTurboSubDir);

    if (!File.Exists(audioPath))
    {
        Console.Error.WriteLine($"Audio file not found: {audioPath}");
        return 1;
    }
    foreach (string f in new[] { WhisperTurbo.EncoderFile, WhisperTurbo.DecoderInitFile,
                                 WhisperTurbo.DecoderStepFile, WhisperTurbo.TokenizerFile,
                                 WhisperTurbo.GenerationConfigFile })
    {
        if (!File.Exists(Path.Combine(whisperDir, f)))
        {
            Console.Error.WriteLine(
                $"Missing {whisperDir}/{f}.\n" +
                $"Select the WhisperTurbo ASR backend in the app and download the model files first.");
            return 1;
        }
    }

    Console.WriteLine($"[whisper-check] loading audio: {audioPath}");
    var (raw, sr, channels) = AudioUtils.ReadAudio(audioPath);
    float[] audio = AudioUtils.AudioTo16000Mono(raw, sr, channels);
    Console.WriteLine($"[whisper-check] decoded: {audio.Length} samples @ 16 kHz ({audio.Length / 16000.0:F1} s)");

    Console.WriteLine($"[whisper-check] loading whisper-turbo from {whisperDir}");
    var swLoad = Stopwatch.StartNew();
    using var whisper = new WhisperTurbo(whisperDir);
    swLoad.Stop();
    Console.WriteLine($"[whisper-check] loaded in {swLoad.ElapsedMilliseconds} ms");

    var swTrans = Stopwatch.StartNew();
    var result = whisper.Transcribe(audio, languageIso: "en");
    swTrans.Stop();

    int chunkSec = Math.Min(30, (int)(audio.Length / 16000.0));
    double rtf = swTrans.Elapsed.TotalSeconds / Math.Max(chunkSec, 1);
    Console.WriteLine();
    Console.WriteLine($"[whisper-check] transcribed {result.Tokens.Count} tokens in {swTrans.ElapsedMilliseconds} ms (RTF {rtf:F3} on first 30 s)");
    Console.WriteLine("[whisper-check] transcript:");
    Console.WriteLine();
    Console.WriteLine(result.Text);
    Console.WriteLine();
    Console.WriteLine("[whisper-check] OK");
    return 0;
}

/// <summary>
/// Fetch the VoxLingua107 LID assets (<c>voxlingua107.onnx</c> +
/// <c>lang_map.json</c>) from the HuggingFace repo. Skips any file that
/// already exists on disk with a non-zero size. Streams with progress
/// so large downloads are visible over SSH.
/// </summary>
static async Task<int> DownloadVoxLinguaAsync(string destDir)
{
    const string repoBase =
        "https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx/resolve/main";
    string[] files = [Config.VoxLinguaModelFile, Config.VoxLinguaLangMapFile];

    Directory.CreateDirectory(destDir);
    Console.WriteLine($"[download] destination: {destDir}");

    using var http = new HttpClient(new HttpClientHandler { AllowAutoRedirect = true });
    http.Timeout = TimeSpan.FromMinutes(15);

    foreach (string name in files)
    {
        string destPath = Path.Combine(destDir, name);
        if (File.Exists(destPath) && new FileInfo(destPath).Length > 0)
        {
            Console.WriteLine($"[download] {name}: already present ({new FileInfo(destPath).Length / 1024 / 1024} MiB), skipping");
            continue;
        }

        string url = $"{repoBase}/{name}";
        Console.Write($"[download] {name} ← {url} ");
        try
        {
            using var resp = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            resp.EnsureSuccessStatusCode();
            long? total = resp.Content.Headers.ContentLength;
            string tmpPath = destPath + ".download";

            long bytes = 0;
            var buffer = new byte[1 << 20];
            int lastPct = -1;

            await using (var src = await resp.Content.ReadAsStreamAsync())
            await using (var dst = File.Create(tmpPath))
            {
                while (true)
                {
                    int read = await src.ReadAsync(buffer);
                    if (read == 0) break;
                    await dst.WriteAsync(buffer.AsMemory(0, read));
                    bytes += read;

                    if (total is > 0)
                    {
                        int pct = (int)(100 * bytes / total.Value);
                        if (pct != lastPct && pct % 10 == 0)
                        {
                            Console.Write($"{pct}% ");
                            lastPct = pct;
                        }
                    }
                }
            }

            File.Move(tmpPath, destPath, overwrite: true);
            Console.WriteLine($"done ({bytes / 1024 / 1024} MiB)");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"\n[download] FAILED ({name}): {ex.Message}");
            return 1;
        }
    }

    Console.WriteLine("[download] VoxLingua107 ready.");
    return 0;
}

static void PrintUsage()
{
    Console.WriteLine("Usage: vernacula-cli --audio <file> --model <dir> [options]");
    Console.WriteLine("       vernacula-cli --download-voxlingua [--model <dir>]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --segments <path>                  Load pre-computed segments JSON, skip diarization");
    Console.WriteLine("  --export-format <md|txt|json|srt>  Output format (default: md)");
    Console.WriteLine("  --output <path>                    Override output file path");
    Console.WriteLine("  --diarization <backend>            Diarization backend: sortformer, diarizen, vad, vibevoice-asr-builtin");
    Console.WriteLine("                                     (default: sortformer, or vibevoice-asr-builtin when --asr vibevoice)");
    Console.WriteLine("  --vad                              Use VAD instead of diarization (deprecated)");
    Console.WriteLine("  --asr <parakeet|cohere|qwen3asr|vibevoice>  ASR backend (default: parakeet)");
    Console.WriteLine("  --cohere-model <dir>               Path to Cohere Transcribe model dir (default: <model>/cohere_transcribe)");
    Console.WriteLine("  --qwen3asr-model <dir>             Path to Qwen3-ASR model dir (default: <model>/qwen3asr)");
    Console.WriteLine("  --qwen3asr-serial                  Force serial Qwen3-ASR, disabling experimental batching");
    Console.WriteLine("  --qwen3asr-ort-opt <extended|basic|disabled>");
    Console.WriteLine("                                     ONNX Runtime graph optimization level for Qwen3-ASR");
    Console.WriteLine("  --vibevoice-model <dir>            Path to VibeVoice-ASR model dir (default: <model>/vibevoice_asr)");
    Console.WriteLine("  --min-asr-seconds <n>              Minimum audio span (s) per ASR group when using segmented VibeVoice (default: 5.0)");
    Console.WriteLine("  --asr-buffer <n>                   Seconds of audio padding on each side of a group (default: 0.0); helps boundary transitions");
    Console.WriteLine("  --profile <dir>                    Write ORT Chrome-trace JSON to <dir>/ (vibevoice only; for perf analysis)");
    Console.WriteLine("  --profile-steps <n>                Cap decode tokens during --profile run (default: 200; ORT limit: ~1M events)");
    Console.WriteLine("  --language <code>                  Force language for Cohere ASR (ISO 639-1, e.g. en, fr, de)");
    Console.WriteLine("  --precision <fp32|int8>            Model precision (default: fp32, parakeet only)");
    Console.WriteLine("  --parakeet-beam <N>                Parakeet TDT beam width (default: 1 = greedy; 4–8 = beam search, 3–5x slower)");
    Console.WriteLine("  --lm <path>                        Parakeet shallow LM fusion — ARPA(.gz) subword n-gram. Auto-bumps beam to 4.");
    Console.WriteLine("  --lm-weight <w>                    Shallow-fusion weight (default: 0.3; typical 0.1–0.5)");
    Console.WriteLine("  --lm-length-penalty <p>            Per-token reward offsetting LM shortening bias (default: 0.6; typical 0.0–1.0)");
    Console.WriteLine("  --skip-asr                         Export diarization/VAD segments without transcription");
    Console.WriteLine("  --benchmark                        Print timing / RTF after transcription");
    Console.WriteLine("  --profile-sortformer               Print fine-grained timing breakdown for Sortformer");
    Console.WriteLine("  --download-voxlingua               Download the VoxLingua107 LID model and exit");
    Console.WriteLine("                                     (defaults to ~/.local/share/Vernacula/models)");
    Console.WriteLine("  --lid                              Run VAD + LID on --audio and print the detected");
    Console.WriteLine("                                     language, then exit. Uses --model as the models");
    Console.WriteLine("                                     root, or the default dir if omitted.");
    Console.WriteLine("  -h, --help                         Show this help");
    Console.WriteLine();
    Console.WriteLine("Build: dotnet build -c Release -p:EP=Cuda|DirectML|Cpu -p:Platform=x64");
}
