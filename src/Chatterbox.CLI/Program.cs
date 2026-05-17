// Chatterbox TTS CLI.
//
// Thin command-line front end over Chatterbox.Base. Loads the ONNX
// bundle once per invocation, synthesizes one utterance, writes a
// 24 kHz mono float32 WAV. Designed for the one-shot case; chunked
// synthesis and continuous-reader modes will land in a follow-up.
//
// Usage:
//   chatterbox --onnx-dir <dir> --voice <wav> --text "..." --out <wav>
//   chatterbox --onnx-dir <dir> --voice <wav> --text-file in.md --out <wav>
//
// See README.md for full flag reference.

using System.Diagnostics;
using System.Globalization;
using Chatterbox.Base;
using Chatterbox.Base.Markdown;
using NAudio.Wave;
using Vernacula.Base.Alignment;
using Vernacula.Base.Models;

// ── Argument parsing ──────────────────────────────────────────────────────────

string? onnxDir       = null;
string? voicePath     = null;
string? outPath       = null;
string? text          = null;
string? textFile      = null;
string? tokenizerJson = null;
string? alignmentOut  = null;  // JSON sidecar of per-word audio timings
string? nfaBundle     = null;  // path to scripts/nemo_export/export_nfa_ctc_to_onnx.py output dir
string  ep            = "auto";
bool    verbose       = false;
bool?   useIoBinding  = null;
float   exaggeration  = ChatterboxConstants.DefaultExaggeration;
// CLI default is intentionally higher than ChatterboxConstants.DefaultMaxLmSteps
// (256). The constant is the smoke/bench reference; the CLI sees long-form
// markdown where typical paragraph chunks run 300-500 LM steps and would
// silently truncate at 256. 1024 is a comfortable cap that natural STOP_SPEECH
// reaches first on real prose; users can lower it if they want hard limits.
int     maxSteps      = 1024;

// Bounds-checked value reader. Caller passes the flag name so the error
// mentions which arg was missing its value. Args are parsed via a manual
// loop because the codebase convention (see Vernacula.CLI) is to avoid
// the System.CommandLine dependency.
string Next(string flag, int i)
{
    if (i + 1 >= args.Length)
        throw new ArgumentException($"{flag} requires a value");
    return args[i + 1];
}

try
{
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--onnx-dir":       onnxDir       = Next("--onnx-dir", i);       i++; break;
            case "--voice":          voicePath     = Next("--voice", i);          i++; break;
            case "--out":            outPath       = Next("--out", i);            i++; break;
            case "--text":           text          = Next("--text", i);           i++; break;
            case "--text-file":      textFile      = Next("--text-file", i);      i++; break;
            case "--tokenizer-json": tokenizerJson = Next("--tokenizer-json", i); i++; break;
            case "--alignment-out":  alignmentOut  = Next("--alignment-out", i);  i++; break;
            case "--nfa-bundle":     nfaBundle     = Next("--nfa-bundle", i);     i++; break;
            case "--ep":             ep            = Next("--ep", i).ToLowerInvariant(); i++; break;
            case "--verbose" or "-v": verbose      = true; break;
            case "--io-binding":     useIoBinding  = true; break;
            case "--no-io-binding":  useIoBinding  = false; break;
            case "--exaggeration":
                // InvariantCulture: a German-locale machine would otherwise
                // reject "0.5" and require "0,5" — opposite of what CLI
                // users will type. Same applies to --max-steps below.
                if (!float.TryParse(Next("--exaggeration", i),
                        NumberStyles.Float, CultureInfo.InvariantCulture, out exaggeration))
                {
                    Console.Error.WriteLine("--exaggeration expects a float (default 0.5).");
                    return 2;
                }
                i++;
                break;
            case "--max-steps":
                if (!int.TryParse(Next("--max-steps", i),
                        NumberStyles.Integer, CultureInfo.InvariantCulture, out maxSteps)
                    || maxSteps < 1)
                {
                    Console.Error.WriteLine("--max-steps expects a positive integer (default 1024).");
                    return 2;
                }
                i++;
                break;
            case "--help" or "-h":
                PrintUsage();
                return 0;
            default:
                Console.Error.WriteLine($"Unknown arg: {args[i]}");
                PrintUsage();
                return 2;
        }
    }
}
catch (ArgumentException ex)
{
    Console.Error.WriteLine(ex.Message);
    return 2;
}

// ── Validate ───────────────────────────────────────────────────────────────────

if (onnxDir is null || voicePath is null)
{
    Console.Error.WriteLine("Missing required: --onnx-dir <dir> --voice <wav>");
    PrintUsage();
    return 2;
}
if (text is null && textFile is null)
{
    Console.Error.WriteLine("Provide one of: --text \"...\"  or  --text-file <path>");
    PrintUsage();
    return 2;
}
if (text is not null && textFile is not null)
{
    Console.Error.WriteLine("--text and --text-file are mutually exclusive.");
    return 2;
}
// Map --ep to ExecutionProvider. Distinct from --ep cuda is --ep auto:
// `cuda` requires the CUDA EP to be available (throws if not); `auto`
// silently falls back from CUDA to DirectML when CUDA isn't available
// — the right call on heterogeneous Windows boxes. The csproj also
// supports DirectML builds (`-p:EP=DirectML`); `--ep directml` makes
// that wiring reachable at runtime.
ExecutionProvider epEnum;
switch (ep)
{
    case "auto":     epEnum = ExecutionProvider.Auto;     break;
    case "cuda":     epEnum = ExecutionProvider.Cuda;     break;
    case "cpu":      epEnum = ExecutionProvider.Cpu;      break;
    case "directml": epEnum = ExecutionProvider.DirectML; break;
    default:
        Console.Error.WriteLine($"Unknown EP: {ep}. Choose cpu, cuda, directml, or auto.");
        return 2;
}

onnxDir   = ExpandHome(onnxDir);
voicePath = ExpandHome(voicePath);
outPath   = ExpandHome(outPath ?? "chatterbox_out.wav");
if (textFile is not null) textFile = ExpandHome(textFile);
if (tokenizerJson is not null) tokenizerJson = ExpandHome(tokenizerJson);
if (alignmentOut is not null) alignmentOut = ExpandHome(alignmentOut);
if (nfaBundle is not null) nfaBundle = ExpandHome(nfaBundle);

// --alignment-out requires --nfa-bundle. The aligner needs the exported
// CTC ASR bundle; we don't auto-discover it because there's no canonical
// location and we don't want to silently align with the wrong model.
if (alignmentOut is not null && nfaBundle is null)
{
    Console.Error.WriteLine(
        "--alignment-out requires --nfa-bundle <dir>. Export via "
        + "scripts/nemo_export/export_nfa_ctc_to_onnx.py and pass the output dir here.");
    return 2;
}
if (nfaBundle is not null && !Directory.Exists(nfaBundle))
{
    Console.Error.WriteLine($"--nfa-bundle not found: {nfaBundle}");
    return 1;
}

if (!Directory.Exists(onnxDir))
{
    Console.Error.WriteLine($"--onnx-dir not found: {onnxDir}");
    return 1;
}
if (!File.Exists(voicePath))
{
    Console.Error.WriteLine($"--voice not found: {voicePath}");
    return 1;
}
if (textFile is not null && !File.Exists(textFile))
{
    Console.Error.WriteLine($"--text-file not found: {textFile}");
    return 1;
}

// ── Read text source ──────────────────────────────────────────────────────────
// --text-file routed through MarkdownTextExtractor when the file extension
// is .md or .markdown (case-insensitive) so headings/lists/code/emphasis
// markup gets stripped before tokenization. --text always treated as plain
// text — callers passing a string usually want it spoken verbatim. The
// extractor's behavior matrix and the source-range index (for downstream
// forced alignment) are documented on the class itself.

string textToSpeak;
if (text is not null)
{
    textToSpeak = text;
}
else
{
    var raw = File.ReadAllText(textFile!);
    var ext = Path.GetExtension(textFile!).ToLowerInvariant();
    if (ext is ".md" or ".markdown")
    {
        var extracted = MarkdownTextExtractor.Extract(raw);
        if (verbose)
            Console.WriteLine($"Markdown extracted: {raw.Length} chars → {extracted.Text.Length} chars, {extracted.Ranges.Count} source-range entries");
        textToSpeak = extracted.Text;
    }
    else
    {
        textToSpeak = raw;
    }
}
if (string.IsNullOrWhiteSpace(textToSpeak))
{
    Console.Error.WriteLine("Text is empty (after reading --text-file).");
    return 1;
}

// ── Resolve tokenizer ─────────────────────────────────────────────────────────

var tokenizerPath = tokenizerJson ?? ChatterboxPipeline.LocateCachedTokenizerJson();
if (tokenizerPath is null)
{
    Console.Error.WriteLine(
        "No tokenizer.json found. Pass --tokenizer-json <path>, or download via:");
    Console.Error.WriteLine(
        "  huggingface-cli download ResembleAI/chatterbox tokenizer.json");
    return 1;
}

// ── Load + synthesize ─────────────────────────────────────────────────────────

var totalSw = Stopwatch.StartNew();

SessionLoadObserver? onLoad = verbose
    ? e => Console.WriteLine(
        $"  {e.FileName}: {e.ElapsedMs} ms  cache={(e.CacheHit ? "HIT" : "miss")}  "
        + $"ep={(e.UsedCuda ? "cuda" : "cpu/dml")}  src={e.SourceSizeBytes / 1e6:F0} MB")
    : null;

if (verbose) Console.WriteLine($"Loading ONNX bundle from {onnxDir} (ep={ep}) ...");
var loadSw = Stopwatch.StartNew();
using var pipeline = new ChatterboxPipeline(onnxDir, epEnum, tokenizerPath, onLoad);
loadSw.Stop();
if (verbose)
{
    Console.WriteLine($"  vocoder mode: {pipeline.Vocoder.Mode}");
    Console.WriteLine($"Loaded sessions in {loadSw.ElapsedMilliseconds} ms total  (requested-ep={ep})");
}

// pipeline.Tokenizer is non-null here: we resolved `tokenizerPath` above
// and would have exited 1 if no tokenizer.json was findable, so the
// pipeline's auto-locate fallback never fires.

var synthSw = Stopwatch.StartNew();
var spk = pipeline.Embedder.Embed(voicePath);
if (verbose) Console.WriteLine(
    $"speech_encoder: cond_emb=({string.Join(",", spk.CondEmb.Dimensions.ToArray())})  "
    + $"audio_tokens=(1,{spk.AudioTokens.Length})");

// ── Chunking decision ────────────────────────────────────────────────────────
// ParagraphChunker returns >1 chunks only when input is long enough AND has
// paragraph breaks (\n\n). Short or single-paragraph inputs collapse to a
// single chunk; we use the existing one-shot path to avoid orchestration
// overhead for those. Tokenization happens per-path (whole text once vs per
// chunk) so we don't waste a tokenize-everything pass on the chunked path.
var chunks = ParagraphChunker.Chunk(textToSpeak);
// Per-chunk audio is preserved past the synthesis step so the alignment
// pass below can resample + align each chunk independently. Both branches
// populate chunkAudios[] in input order; the final WAV is concat-of-chunks.
float[][] chunkAudios;
if (chunks.Count <= 1)
{
    // One-shot path (existing behavior).
    var tokenIds = pipeline.Tokenizer!.WrapForLm(textToSpeak);
    if (verbose)
    {
        var preview = textToSpeak.Length > 60 ? textToSpeak[..60] + "..." : textToSpeak;
        Console.WriteLine($"Tokenized \"{preview.Replace("\n", " ")}\" → {tokenIds.Length} tokens");
    }
    var lmResult = pipeline.Lm.Generate(spk.CondEmb, tokenIds,
        useIoBinding: useIoBinding,
        exaggeration: exaggeration,
        maxSteps: maxSteps);
    if (verbose) Console.WriteLine(
        $"LM: {lmResult.Steps} steps, generated {lmResult.RawGeneratedTokens.Count - 1} tokens");

    var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
    var oneShotAudio = pipeline.Vocoder.Synthesize(speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
    // Normalize the one-shot path to the same shape as the chunked path
    // (single-element chunkAudios + a one-element chunks list). Keeps the
    // alignment + WAV-write code below path-agnostic. The textToSpeak
    // whitespace check at the top of this section guarantees the chunker
    // returns ≥1 entries, so we don't guard chunks.Count == 0 here.
    chunkAudios = new[] { oneShotAudio };
}
else
{
    // Long-form path: ChunkedSynthesizer pipelines LM(N+1) with voc(N) across
    // chunk boundaries via a Channel<T> producer/consumer. groupSize=1 (no
    // LM batching) because real paragraphs have different lengths; the
    // batched mode requires same-length prompts. Pipelining alone still
    // saves ~7% wall by overlapping vocoder with the next chunk's LM —
    // see Run 2 in docs/chatterbox_perf_investigation.md.
    if (verbose)
        Console.WriteLine($"Chunked into {chunks.Count} paragraphs "
            + $"(min/max char count: {chunks.Min(c => c.Length)}/{chunks.Max(c => c.Length)})");

    var tokensPerChunk = chunks.Select(c => pipeline.Tokenizer!.WrapForLm(c)).ToArray();
    var synth = new ChunkedSynthesizer(pipeline);
    var result = synth.Synthesize(spk, tokensPerChunk,
        useIoBinding: useIoBinding,
        exaggeration: exaggeration,
        maxSteps: maxSteps);

    if (verbose)
    {
        for (int i = 0; i < result.ChunkTimings.Count; i++)
        {
            var t = result.ChunkTimings[i];
            Console.WriteLine($"  chunk {i + 1}/{chunks.Count}: "
                + $"LM {t.LmMs} ms ({t.LmSteps} steps), "
                + $"voc {t.VocoderMs} ms, "
                + $"audio {t.AudioSamples / (float)ChatterboxConstants.S3GenSr:F2}s");
        }
    }

    chunkAudios = result.Waveforms.ToArray();
}

// Concatenate per-chunk waveforms for the WAV write. No fades / silence
// injection between chunks — paragraph boundaries already cue the TTS
// to taper naturally, and the LM's STOP_SPEECH emits a brief trailing
// silence per chunk.
int totalSamples = chunkAudios.Sum(w => w.Length);
var samples = new float[totalSamples];
{
    int off = 0;
    foreach (var w in chunkAudios)
    {
        Array.Copy(w, 0, samples, off, w.Length);
        off += w.Length;
    }
}
synthSw.Stop();

// ── Forced alignment (optional, --alignment-out) ──────────────────────────────
// Per-chunk: resample 24 kHz → 16 kHz, run NemoNfaAligner against the
// chunk's reference text, translate chunk-relative timings to absolute
// playback time via the cumulative chunk-sample offset. JSON sidecar
// holds a flat words list (sorted by start_seconds) + a chunks summary;
// flat is what the Avalonia app's binary-search-by-playback-time lookup
// will need.
if (alignmentOut is not null)
{
    var alignSw = Stopwatch.StartNew();
    using var aligner = new NemoNfaAligner(nfaBundle!, epEnum);
    if (verbose)
        Console.WriteLine($"Aligning {chunks.Count} chunk(s) against {nfaBundle} ...");

    var allWords = new List<object>();
    var chunkRecords = new List<object>();
    int sampleOffset = 0;
    for (int i = 0; i < chunkAudios.Length; i++)
    {
        var chunkAudio24k = chunkAudios[i];
        double chunkStartSec = sampleOffset / (double)ChatterboxConstants.S3GenSr;
        double chunkEndSec = (sampleOffset + chunkAudio24k.Length) / (double)ChatterboxConstants.S3GenSr;
        sampleOffset += chunkAudio24k.Length;

        // 24 kHz mono float → 16 kHz mono float via AudioUtils. The cleanup
        // pass (75 Hz HPF + mains-hum notches) is harmless for TTS audio
        // and matches what other ASR consumers in Vernacula get.
        var chunkAudio16k = Vernacula.Base.AudioUtils.AudioTo16000Mono(
            chunkAudio24k, ChatterboxConstants.S3GenSr, channels: 1);
        var words = aligner.Align(chunkAudio16k, chunks[i], "en");

        foreach (var w in words)
        {
            allWords.Add(new
            {
                text = w.Text,
                start_seconds = chunkStartSec + w.StartSeconds,
                end_seconds = chunkStartSec + w.EndSeconds,
                chunk_index = i,
            });
        }
        chunkRecords.Add(new
        {
            index = i,
            audio_start_seconds = chunkStartSec,
            audio_end_seconds = chunkEndSec,
            text = chunks[i],
            word_count = words.Count,
        });
        if (verbose)
            Console.WriteLine($"  chunk {i + 1}/{chunkAudios.Length}: "
                + $"{chunkAudio24k.Length / (double)ChatterboxConstants.S3GenSr:F2}s audio → "
                + $"{words.Count} aligned words");
    }
    alignSw.Stop();

    var payload = new
    {
        audio_path = outPath,
        sample_rate = ChatterboxConstants.S3GenSr,
        audio_duration_seconds = totalSamples / (double)ChatterboxConstants.S3GenSr,
        aligner = "nemo_nfa",
        nfa_bundle = nfaBundle,
        chunks = chunkRecords,
        words = allWords,
    };
    // Snake-case schema (start_seconds, chunk_index, ...) is part of the
    // consumer contract — don't add PropertyNamingPolicy.CamelCase here
    // unless you also update every downstream parser. Default options
    // preserve declared anonymous-type member names verbatim.
    var json = System.Text.Json.JsonSerializer.Serialize(payload,
        new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
    // Atomic write: serialize to a sibling .tmp, then rename. A SIGINT
    // mid-File.WriteAllText would otherwise leave a truncated JSON that
    // downstream consumers fail to parse.
    var tmpPath = alignmentOut + ".tmp";
    File.WriteAllText(tmpPath, json);
    File.Move(tmpPath, alignmentOut, overwrite: true);

    if (verbose)
        Console.WriteLine($"Alignment: {alignSw.ElapsedMilliseconds} ms, "
            + $"{allWords.Count} words → {alignmentOut}");
}

// ── Write WAV ─────────────────────────────────────────────────────────────────

var fmt = WaveFormat.CreateIeeeFloatWaveFormat(ChatterboxConstants.S3GenSr, 1);
using (var writer = new WaveFileWriter(outPath, fmt))
{
    writer.WriteSamples(samples, 0, samples.Length);
}
totalSw.Stop();

float audioSeconds = samples.Length / (float)ChatterboxConstants.S3GenSr;
string chunkInfo = chunks.Count > 1 ? $", {chunks.Count} chunks" : "";
Console.WriteLine(
    $"Synthesized {audioSeconds:F2}s of audio → {outPath} "
    + $"({totalSw.ElapsedMilliseconds / 1000.0:F1}s total, "
    + $"{synthSw.ElapsedMilliseconds / 1000.0:F1}s synth{chunkInfo})");
return 0;


static string ExpandHome(string path)
    => path.StartsWith("~/")
        ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), path[2..])
        : path;

static void PrintUsage()
{
    Console.Error.WriteLine("""
        Usage: chatterbox --onnx-dir <dir> --voice <wav> (--text "..." | --text-file <path>) [options]

        Required:
          --onnx-dir <dir>         Directory containing the Chatterbox ONNX bundle
                                   (speech_encoder, embed_tokens, language_model,
                                    + one of the cond-decoder layouts).
          --voice <wav>            Reference voice clip. Any sample rate / channels.
          --text "..."   OR        Text to synthesize. Mutually exclusive with --text-file.
          --text-file <path>       Read text from a file. Files with .md or .markdown
                                   extensions are routed through the markdown extractor
                                   (headings/lists/emphasis/links stripped; code blocks,
                                   tables, images, HTML dropped entirely). Other extensions
                                   are passed through as plain text.

        Optional:
          --out <wav>              Output WAV path. Default: chatterbox_out.wav
          --ep <name>              Execution provider, one of:
                                     auto      — CUDA, fall back to DirectML (default)
                                     cuda      — CUDA only, fail if unavailable
                                     directml  — DirectML only, fail if unavailable
                                     cpu       — CPU only
                                   The csproj's `-p:EP=...` build flag must include
                                   the runtime you ask for here (cuda needs OnnxRuntime.Gpu,
                                   directml needs OnnxRuntime.DirectML).
          --tokenizer-json <path>  Path to chatterbox tokenizer.json. Default:
                                   auto-locate from the HF hub cache.
          --alignment-out <path>   Write a JSON sidecar with per-word audio timings,
                                   one entry per spoken word with absolute playback
                                   start/end seconds. Requires --nfa-bundle.
          --nfa-bundle <dir>       NFA CTC ASR bundle directory exported by
                                   scripts/nemo_export/export_nfa_ctc_to_onnx.py.
                                   Required when --alignment-out is set.
          --io-binding /           Force GPU-resident KV-cache chaining for the LM.
          --no-io-binding          Default: auto-detect from the effective EP.
          --exaggeration <float>   Conditioning scalar passed to embed_tokens. Default: 0.5.
                                   Typical range 0.0 – 1.0; out-of-range values are
                                   accepted but produce increasingly unusual audio.
          --max-steps <int>        Cap on LM rollout length. Default: 1024.
                                   The LM naturally emits STOP_SPEECH at a paragraph's
                                   end (usually 200-500 steps); this cap is a safety
                                   net. Long-form markdown chunks need more headroom
                                   than the smoke-bench default of 256.
          --verbose / -v           Print per-stage timing and cache info.
          --help / -h              Show this message.
        """);
}
