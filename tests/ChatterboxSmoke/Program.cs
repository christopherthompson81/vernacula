// Chatterbox TTS — C# smoke test.
//
// Originally a monolithic port of scripts/chatterbox_export/.../listen_test.py;
// now a thin CLI on top of Chatterbox.Base (SpeakerEmbedder + AcousticLM +
// Vocoder + ChatterboxPipeline). Keeps the per-stage timing prints that make
// the smoke test useful for perf+parity checks.
//
// Usage:
//   dotnet run --project tests/ChatterboxSmoke -- \
//       --onnx-dir /tmp/cb_dyn5 \
//       --voice ~/Downloads/voice.wav \
//       --text "Hello world." \
//       --out   /tmp/chatterbox_out_cs.wav \
//       --ep    cuda

using System.Diagnostics;
using Chatterbox.Base;
using Chatterbox.Base.AudioIo;
using Chatterbox.Base.Tokenization;
using NAudio.Wave;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.ChatterboxSmoke;

internal static class Program
{
    // Backward-compat: the Ezreal-and-Jinx sentence, pre-tokenized via
    // chatterbox's EnTokenizer. Used when --text is not given so the smoke
    // test still passes without HF cache / tokenizer.json being present.
    // Wrapping: [EXAGGERATION_TOKEN, ...text..., START_SPEECH, START_SPEECH].
    private static readonly long[] FallbackInputIds =
    [
        ChatterboxConstants.ExaggerationToken,
        255, 281, 39, 46, 56, 2, 53, 2, 286, 41, 37, 2, 136, 122,
        49, 2, 152, 2, 103, 2, 277, 21, 101, 7, 2, 301, 55, 34, 28, 7,
        2, 53, 2, 296, 18, 18, 115, 2, 51, 2, 33, 245, 2, 17, 190, 2,
        42, 2, 50, 18, 125, 4, 32, 2, 290, 169, 142, 2, 41, 2, 43, 2,
        18, 29, 91, 2, 25, 186, 8, 20, 14, 80, 2, 29, 86, 213, 216, 9,
        0, ChatterboxConstants.StartSpeechToken, ChatterboxConstants.StartSpeechToken,
    ];

    private static int Main(string[] args)
    {
        string? onnxDir = null;
        string? voicePath = null;
        string? outPath = "/tmp/chatterbox_out_cs.wav";
        string ep = "cuda";
        string? diagDir = null;
        bool useIoBinding = true;
        string? text = null;
        string? tokenizerJson = null;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--onnx-dir":       onnxDir = args[++i]; break;
                case "--voice":          voicePath = args[++i]; break;
                case "--out":            outPath = args[++i]; break;
                case "--ep":             ep = args[++i].ToLowerInvariant(); break;
                case "--diag":           diagDir = args[++i]; break;
                case "--io-binding":     useIoBinding = true; break;
                case "--no-io-binding":  useIoBinding = false; break;
                case "--text":           text = args[++i]; break;
                case "--tokenizer-json": tokenizerJson = args[++i]; break;
                default:
                    Console.Error.WriteLine($"Unknown arg: {args[i]}");
                    return 2;
            }
        }
        if (diagDir is not null)
        {
            diagDir = ExpandHome(diagDir);
            Directory.CreateDirectory(diagDir);
            Console.WriteLine($"[diag] dumping LM step-0/step-1 + token sequence to {diagDir}");
            if (useIoBinding)
            {
                Console.WriteLine("[diag] --io-binding is on: past_kv at step 1 is GPU-resident "
                    + "and will NOT be dumped (extracting would defeat IoBinding). "
                    + "Use --no-io-binding for full past_kv diag artifacts.");
            }
        }
        if (onnxDir is null || voicePath is null || outPath is null)
        {
            Console.Error.WriteLine(
                "Usage: --onnx-dir <dir> --voice <wav> [--out <wav>] [--ep cpu|cuda]");
            return 2;
        }
        if (ep is not ("cpu" or "cuda"))
        {
            Console.Error.WriteLine($"Unknown EP: {ep}. Choose cpu or cuda.");
            return 2;
        }

        voicePath = ExpandHome(voicePath);
        outPath = ExpandHome(outPath);

        var totalSw = Stopwatch.StartNew();
        var sw = new Stopwatch();
        var epEnum = ep == "cuda" ? ExecutionProvider.Auto : ExecutionProvider.Cpu;

        // ── Resolve tokenizer (needed only if --text was given) ───────────
        EnTokenizer? tokenizer = null;
        if (text is not null)
        {
            var tokenizerPath = tokenizerJson ?? ChatterboxPipeline.LocateCachedTokenizerJson();
            if (tokenizerPath is null)
            {
                Console.Error.WriteLine(
                    "--text given but no tokenizer.json found. Pass --tokenizer-json <path>, "
                    + "or download via `huggingface-cli download ResembleAI/chatterbox tokenizer.json`.");
                return 2;
            }
            tokenizer = new EnTokenizer(tokenizerPath);
        }

        // ── Load all sessions via the Base pipeline; print per-graph timing ──
        // Observer prints one line per session loaded (preserves the format
        // the pre-factor monolith used so perf-tuning logs stay comparable).
        // `ep=` reports the EFFECTIVE EP per the OrtSessionBuilder usedCuda
        // probe — distinct from the requested EP, so `Auto` falling back
        // from CUDA to DirectML is visible here instead of silently breaking
        // downstream IoBinding. We also accumulate the per-session EP flags
        // so the summary line below reports the actual back-end mix rather
        // than echoing the CLI string.
        var perSessionUsedCuda = new List<bool>();
        void OnSessionLoad(SessionLoadEvent e)
        {
            perSessionUsedCuda.Add(e.UsedCuda);
            Console.WriteLine($"  {e.FileName}: {e.ElapsedMs} ms  cache={(e.CacheHit ? "HIT" : "miss")}  "
                + $"ep={(e.UsedCuda ? "cuda" : "cpu/dml")}  src={e.SourceSizeBytes / 1e6:F0} MB");
        }

        var totalLoadSw = Stopwatch.StartNew();
        using var embedder = new SpeakerEmbedder(
            Path.Combine(onnxDir, "speech_encoder.onnx"), epEnum, OnSessionLoad);
        using var lm = new AcousticLM(
            Path.Combine(onnxDir, "embed_tokens.onnx"),
            Path.Combine(onnxDir, "language_model.onnx"),
            epEnum, OnSessionLoad);
        using var vocoder = new Vocoder(onnxDir, epEnum, OnSessionLoad);
        totalLoadSw.Stop();
        // Effective EP for the summary: "cuda" if every session got CUDA,
        // "cpu/dml" if none did, "mixed" if the EPs disagreed (shouldn't
        // happen unless the cache for one graph was built under a
        // different EP and the user didn't clear it).
        string effectiveEp = perSessionUsedCuda.All(x => x) ? "cuda"
            : perSessionUsedCuda.All(x => !x) ? "cpu/dml"
            : "mixed";
        Console.WriteLine($"  vocoder mode: {vocoder.Mode}");
        Console.WriteLine($"Loaded sessions in {totalLoadSw.ElapsedMilliseconds} ms total  (requested={ep}, effective={effectiveEp})");

        // ── Speaker embedding ──────────────────────────────────────────────
        sw.Restart();
        var audio = VoicePromptLoader.Load(voicePath);
        sw.Stop();
        Console.WriteLine($"Loaded voice {voicePath}: {audio.Length} samples "
            + $"({audio.Length / (float)ChatterboxConstants.S3GenSr:F2}s)  [{sw.ElapsedMilliseconds} ms]");

        sw.Restart();
        var spk = embedder.Embed(audio);
        sw.Stop();
        Console.WriteLine($"speech_encoder: cond_emb=({string.Join(",", spk.CondEmb.Dimensions.ToArray())})  "
            + $"audio_tokens=(1,{spk.AudioTokens.Length})  [{sw.ElapsedMilliseconds} ms]");

        // ── Tokens (from --text or the hardcoded fallback) ─────────────────
        long[] inputIds;
        if (tokenizer is not null)
        {
            inputIds = tokenizer.WrapForLm(text!);
            var preview = text!.Length > 50 ? text[..50] + "..." : text;
            Console.WriteLine($"Tokenized \"{preview}\" → {inputIds.Length} tokens");
        }
        else
        {
            inputIds = FallbackInputIds;
            Console.WriteLine($"Using hardcoded Ezreal sentence ({inputIds.Length} tokens). "
                + "Pass --text \"...\" for arbitrary input.");
        }

        // ── LM rollout ─────────────────────────────────────────────────────
        var lmSw = Stopwatch.StartNew();
        var lmResult = lm.Generate(spk.CondEmb, inputIds,
            useIoBinding: useIoBinding,
            diagDir: diagDir);
        lmSw.Stop();
        Console.WriteLine($"LM ({(useIoBinding ? "io-binding" : "basic")}): {lmResult.Steps} steps, "
            + $"generated {lmResult.RawGeneratedTokens.Count - 1} tokens "
            + $"[{lmSw.ElapsedMilliseconds} ms, {lmSw.ElapsedMilliseconds / (double)lmResult.Steps:F1} ms/step]");

        if (diagDir is not null)
        {
            File.WriteAllBytes(Path.Combine(diagDir, "cs_tokens.bin"),
                System.Runtime.InteropServices.MemoryMarshal.AsBytes<long>(
                    lmResult.RawGeneratedTokens.ToArray()).ToArray());
            Console.WriteLine($"[diag] wrote {diagDir}/cs_tokens.bin ({lmResult.RawGeneratedTokens.Count} tokens)");
        }

        // ── Build speech_tokens + vocoder ──────────────────────────────────
        var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
        Console.WriteLine($"speech_tokens: shape=(1, {speechTokens.Length})  "
            + $"({spk.AudioTokens.Length} from voice + {speechTokens.Length - spk.AudioTokens.Length} from LM)");

        sw.Restart();
        var samples = vocoder.Synthesize(speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
        sw.Stop();
        Console.WriteLine($"cond_decoder ({vocoder.Mode}): waveform=(1, {samples.Length}) "
            + $"→ {samples.Length / (float)ChatterboxConstants.S3GenSr:F2}s  [{sw.ElapsedMilliseconds} ms]");

        // ── Write WAV ──────────────────────────────────────────────────────
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(ChatterboxConstants.S3GenSr, 1);
        using (var writer = new WaveFileWriter(outPath, fmt))
        {
            writer.WriteSamples(samples, 0, samples.Length);
        }
        totalSw.Stop();
        Console.WriteLine($"Wrote {outPath}  [total {totalSw.ElapsedMilliseconds / 1000.0:F1}s]");
        return 0;
    }

    private static string ExpandHome(string path)
        => path.StartsWith("~/")
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), path[2..])
            : path;
}
