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
using Chatterbox.Base;
using NAudio.Wave;
using Vernacula.Base.Models;

// ── Argument parsing ──────────────────────────────────────────────────────────

string? onnxDir       = null;
string? voicePath     = null;
string? outPath       = null;
string? text          = null;
string? textFile      = null;
string? tokenizerJson = null;
string  ep            = "cuda";
bool    verbose       = false;
bool?   useIoBinding  = null;
float   exaggeration  = ChatterboxConstants.DefaultExaggeration;
int     maxSteps      = ChatterboxConstants.DefaultMaxLmSteps;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--onnx-dir":       onnxDir       = args[++i]; break;
        case "--voice":          voicePath     = args[++i]; break;
        case "--out":            outPath       = args[++i]; break;
        case "--text":           text          = args[++i]; break;
        case "--text-file":      textFile      = args[++i]; break;
        case "--tokenizer-json": tokenizerJson = args[++i]; break;
        case "--ep":             ep            = args[++i].ToLowerInvariant(); break;
        case "--verbose" or "-v": verbose      = true; break;
        case "--io-binding":     useIoBinding  = true; break;
        case "--no-io-binding":  useIoBinding  = false; break;
        case "--exaggeration":
            if (!float.TryParse(args[++i], out exaggeration))
            {
                Console.Error.WriteLine("--exaggeration expects a float (default 0.5).");
                return 2;
            }
            break;
        case "--max-steps":
            if (!int.TryParse(args[++i], out maxSteps) || maxSteps < 1)
            {
                Console.Error.WriteLine("--max-steps expects a positive integer (default 256).");
                return 2;
            }
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
if (ep is not ("cpu" or "cuda"))
{
    Console.Error.WriteLine($"Unknown EP: {ep}. Choose cpu or cuda.");
    return 2;
}

onnxDir   = ExpandHome(onnxDir);
voicePath = ExpandHome(voicePath);
outPath   = ExpandHome(outPath ?? "chatterbox_out.wav");
if (textFile is not null) textFile = ExpandHome(textFile);
if (tokenizerJson is not null) tokenizerJson = ExpandHome(tokenizerJson);

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
// --text-file is currently passed through as plain text. Markdown
// parsing (Stage 1 step 4) will land in a follow-up so a `.md` source
// has heading/list/emphasis markers stripped before synthesis. Today,
// any markdown punctuation appears verbatim in the output.

string textToSpeak = text ?? File.ReadAllText(textFile!);
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
var epEnum = ep == "cuda" ? ExecutionProvider.Auto : ExecutionProvider.Cpu;

SessionLoadObserver? onLoad = verbose
    ? e => Console.WriteLine(
        $"  {e.FileName}: {e.ElapsedMs} ms  cache={(e.CacheHit ? "HIT" : "miss")}  "
        + $"ep={(e.UsedCuda ? "cuda" : "cpu/dml")}  src={e.SourceSizeBytes / 1e6:F0} MB")
    : null;

if (verbose) Console.WriteLine($"Loading ONNX bundle from {onnxDir} (ep={ep}) ...");
using var pipeline = new ChatterboxPipeline(onnxDir, epEnum, tokenizerPath, onLoad);
if (verbose) Console.WriteLine($"  vocoder mode: {pipeline.Vocoder.Mode}");

var tokenIds = pipeline.Tokenizer!.WrapForLm(textToSpeak);
if (verbose)
{
    var preview = textToSpeak.Length > 60 ? textToSpeak[..60] + "..." : textToSpeak;
    Console.WriteLine($"Tokenized \"{preview.Replace("\n", " ")}\" → {tokenIds.Length} tokens");
}

var synthSw = Stopwatch.StartNew();
var spk = pipeline.Embedder.Embed(voicePath);
if (verbose) Console.WriteLine(
    $"speech_encoder: cond_emb=({string.Join(",", spk.CondEmb.Dimensions.ToArray())})  "
    + $"audio_tokens=(1,{spk.AudioTokens.Length})");

var lmResult = pipeline.Lm.Generate(spk.CondEmb, tokenIds,
    useIoBinding: useIoBinding,
    exaggeration: exaggeration,
    maxSteps: maxSteps);
if (verbose) Console.WriteLine(
    $"LM: {lmResult.Steps} steps, generated {lmResult.RawGeneratedTokens.Count - 1} tokens");

var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
var samples = pipeline.Vocoder.Synthesize(speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
synthSw.Stop();

// ── Write WAV ─────────────────────────────────────────────────────────────────

var fmt = WaveFormat.CreateIeeeFloatWaveFormat(ChatterboxConstants.S3GenSr, 1);
using (var writer = new WaveFileWriter(outPath, fmt))
{
    writer.WriteSamples(samples, 0, samples.Length);
}
totalSw.Stop();

float audioSeconds = samples.Length / (float)ChatterboxConstants.S3GenSr;
Console.WriteLine(
    $"Synthesized {audioSeconds:F2}s of audio → {outPath} "
    + $"({totalSw.ElapsedMilliseconds / 1000.0:F1}s total, "
    + $"{synthSw.ElapsedMilliseconds / 1000.0:F1}s synth)");
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
          --text-file <path>       Read text from a file. Currently passed through as
                                   plain text (markdown stripping is a follow-up).

        Optional:
          --out <wav>              Output WAV path. Default: chatterbox_out.wav
          --ep cpu | cuda          Execution provider. Default: cuda.
          --tokenizer-json <path>  Path to chatterbox tokenizer.json. Default:
                                   auto-locate from the HF hub cache.
          --io-binding /           Force GPU-resident KV-cache chaining for the LM.
          --no-io-binding          Default: auto-detect based on the effective EP.
          --exaggeration <float>   Conditioning scalar passed to embed_tokens. Default: 0.5.
          --max-steps <int>        Cap on LM rollout length. Default: 256.
          --verbose / -v           Print per-stage timing and cache info.
          --help / -h              Show this message.
        """);
}
