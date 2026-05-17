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
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
        string? lmPath = null;  // --lm-path: override path to language_model*.onnx (Run 8 quantization sweep)
        int benchChunks = 1;
        bool pipelined = false;
        int benchBatchedLm = 0;  // 0 = disabled; >0 = run probe with B=this
        int benchBatchedVoc = 0; // 0 = disabled; >0 = run vocoder-only probe at B=this
        int vocBatch = 1;        // --voc-batch B: group bench-chunks vocoder calls similarly to --lm-batch
        int lmBatch = 1;  // --lm-batch B: group benchChunks into batched LM calls of size B
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
                case "--lm-path":        lmPath = args[++i]; break;
                case "--bench-chunks":
                    // Loop the post-load synthesis N times. The pipeline
                    // (sessions, voice embedding, tokenization) is set up
                    // once; each iteration reruns LM + vocoder on the same
                    // text. Use this to measure per-chunk steady-state cost
                    // amortized over warmup — drives the long-form perf work.
                    if (!int.TryParse(args[++i], out benchChunks) || benchChunks < 1)
                    {
                        Console.Error.WriteLine("--bench-chunks expects a positive integer.");
                        return 2;
                    }
                    break;
                case "--pipelined":
                    // Use ChunkedSynthesizer (LM and Vocoder concurrent on
                    // separate threads, one chunk apart). Only meaningful
                    // with --bench-chunks N for N>1; for N=1 collapses to
                    // a serial single-chunk call. When --lm-batch B is
                    // also set, the pipelining happens at the group level
                    // (batched LM(N+1) overlapped with batched voc(N)).
                    pipelined = true;
                    break;
                case "--lm-batch":
                    // Group bench-chunks N into batched LM calls of size B.
                    // Each group runs one GenerateBatch with B same-text prompts,
                    // then B vocoder calls serially. With --bench-chunks 8
                    // --lm-batch 4, that's 2 batched LM calls × 8 vocoder calls.
                    if (!int.TryParse(args[++i], out lmBatch) || lmBatch < 1)
                    {
                        Console.Error.WriteLine("--lm-batch expects a positive integer.");
                        return 2;
                    }
                    break;
                case "--bench-batched-lm":
                    // Probe whether the LM ONNX accepts B>1 inputs at all
                    // (its dynamic_axes say yes; this confirms ORT actually
                    // honors it). Bypasses voice/tokenizer/vocoder — loads
                    // only embed_tokens + language_model, runs a B-batch
                    // prefill + a handful of step iterations, reports
                    // per-step ms. Use to compute the amortization factor
                    // vs serial B=1: lower than 1.0 means batching wins.
                    if (!int.TryParse(args[++i], out benchBatchedLm) || benchBatchedLm < 1)
                    {
                        Console.Error.WriteLine("--bench-batched-lm expects a positive integer batch size.");
                        return 2;
                    }
                    break;
                case "--bench-batched-voc":
                    // Sister probe to --bench-batched-lm: does the merged
                    // conditional_decoder_loop.onnx (or split / monolithic
                    // fallback) accept B>1? Replicates one speaker+tokens
                    // input across the B dimension, runs one Synthesize at
                    // batch B, reports wall + per-batch-elem ms.
                    if (!int.TryParse(args[++i], out benchBatchedVoc) || benchBatchedVoc < 1)
                    {
                        Console.Error.WriteLine("--bench-batched-voc expects a positive integer batch size.");
                        return 2;
                    }
                    break;
                case "--voc-batch":
                    if (!int.TryParse(args[++i], out vocBatch) || vocBatch < 1)
                    {
                        Console.Error.WriteLine("--voc-batch expects a positive integer.");
                        return 2;
                    }
                    break;
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
        // --voc-batch is meaningful only under non-pipelined --lm-batch>1.
        // Under --pipelined the group size is what drives voc batching (the
        // group is the batching unit); --voc-batch alone with no --lm-batch
        // falls through to the serial branch and is silently dropped.
        if (vocBatch > 1 && (pipelined || lmBatch <= 1))
        {
            string why = pipelined
                ? "under --pipelined the group size (--lm-batch) drives voc batching"
                : "--voc-batch requires --lm-batch>1 — without batched LM there are no groups to batch the voc over";
            Console.Error.WriteLine($"warning: --voc-batch {vocBatch} ignored: {why}.");
            vocBatch = 1;
        }

        // --bench-batched-{lm,voc} both skip voice loading; only need --onnx-dir.
        // Other modes need --voice. --out always has a default.
        bool isProbe = benchBatchedLm > 0 || benchBatchedVoc > 0;
        if (onnxDir is null || (voicePath is null && !isProbe))
        {
            Console.Error.WriteLine(
                "Usage: --onnx-dir <dir> --voice <wav> [--out <wav>] [--ep cpu|cuda]\n"
                + "   or: --onnx-dir <dir> --bench-batched-lm <B>\n"
                + "   or: --onnx-dir <dir> --bench-batched-voc <B>");
            return 2;
        }
        if (ep is not ("cpu" or "cuda"))
        {
            Console.Error.WriteLine($"Unknown EP: {ep}. Choose cpu or cuda.");
            return 2;
        }

        if (voicePath is not null) voicePath = ExpandHome(voicePath);
        outPath = ExpandHome(outPath ?? "/tmp/chatterbox_out_cs.wav");

        var totalSw = Stopwatch.StartNew();
        var sw = new Stopwatch();
        var epEnum = ep == "cuda" ? ExecutionProvider.Auto : ExecutionProvider.Cpu;

        // ── Probe mode: --bench-batched-lm / --bench-batched-voc ────────────
        if (benchBatchedLm > 0)
        {
            return RunBatchedLmProbe(onnxDir, epEnum, benchBatchedLm);
        }
        if (benchBatchedVoc > 0)
        {
            return RunBatchedVocProbe(onnxDir, epEnum, benchBatchedVoc);
        }

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
        // --lm-path: when set, load the LM graph from an arbitrary path
        // (e.g. /tmp/cb_dyn5/language_model.fp16.onnx) instead of the
        // bundle default. Lets the Run 8 quantization sweep swap LMs
        // without rebuilding the whole bundle dir.
        var lmGraphPath = lmPath is not null ? ExpandHome(lmPath) : Path.Combine(onnxDir, "language_model.onnx");
        if (lmPath is not null)
            Console.WriteLine($"  [lm-path override] {lmGraphPath}");
        using var lm = new AcousticLM(
            Path.Combine(onnxDir, "embed_tokens.onnx"),
            lmGraphPath,
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
        // voicePath is non-null here: the only path that allows it to be
        // null is benchBatchedLm > 0, which returned above.
        sw.Restart();
        var audio = VoicePromptLoader.Load(voicePath!);
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

        // ── Synthesis loop ─────────────────────────────────────────────────
        // benchChunks=1 (default): single-chunk run, writes WAV and prints
        // per-stage timing as before. benchChunks>1: same chunk N times.
        // Two execution modes:
        //   serial (default): LM and vocoder run sequentially per chunk
        //   --pipelined: ChunkedSynthesizer overlaps LM(N+1) with voc(N)
        // Drives the long-form perf measurements in
        // docs/chatterbox_perf_investigation.md.
        var chunkLmMs = new long[benchChunks];
        var chunkVocMs = new long[benchChunks];
        var chunkSteps = new int[benchChunks];
        float[]? lastSamples = null;
        long pipelinedChunkWallMs = 0;  // populated when --pipelined; serial branch uses sums below

        if (pipelined && benchChunks > 1)
        {
            // ChunkedSynthesizer mode. groupSize=lmBatch:
            //   lmBatch=1: chunk-at-a-time LM/voc pipelining (Run 2)
            //   lmBatch>1: per-group LM/voc batched + pipelined across
            //              groups (Run 7). Voc-batch implicitly == lmBatch
            //              in this mode (group is the batching unit).
            var synth = new ChunkedSynthesizer(lm, vocoder);
            var tokensPerChunk = Enumerable.Repeat(inputIds, benchChunks).ToArray();
            var result = synth.Synthesize(
                spk, tokensPerChunk,
                useIoBinding: useIoBinding,
                groupSize: lmBatch);
            for (int chunk = 0; chunk < benchChunks; chunk++)
            {
                var t = result.ChunkTimings[chunk];
                chunkLmMs[chunk] = t.LmMs;
                chunkVocMs[chunk] = t.VocoderMs;
                chunkSteps[chunk] = t.LmSteps;
                lastSamples = result.Waveforms[chunk];
                Console.WriteLine($"  chunk {chunk + 1}/{benchChunks}: "
                    + $"LM-share {t.LmMs} ms ({t.LmSteps} steps), "
                    + $"voc-share {t.VocoderMs} ms, "
                    + $"audio {t.AudioSamples / (float)ChatterboxConstants.S3GenSr:F2}s");
            }
            pipelinedChunkWallMs = result.TotalWallMs;
        }
        else if (lmBatch > 1 && benchChunks > 1)
        {
            // Batched-LM mode (no pipelining). Group benchChunks into
            // lmBatch-sized groups; each group: 1 batched GenerateBatch
            // + B serial vocoder calls. Measures the per-chunk amortization
            // established in Run 3 but end-to-end (real STOP_SPEECH-driven
            // rollouts, full vocoder).
            var batchWallSw = Stopwatch.StartNew();
            int produced = 0;
            while (produced < benchChunks)
            {
                int B = Math.Min(lmBatch, benchChunks - produced);
                var condEmbs = Enumerable.Repeat(spk.CondEmb, B).ToArray();
                var tokensPerB = Enumerable.Repeat(inputIds, B).ToArray();

                var batchLmSw = Stopwatch.StartNew();
                var batched = lm.GenerateBatch(condEmbs, tokensPerB);
                batchLmSw.Stop();

                // Build speech_tokens per element ahead of vocoder dispatch.
                var speechTokensPerB = new long[B][];
                for (int b = 0; b < B; b++)
                    speechTokensPerB[b] = batched.PerChunkResults[b].BuildSpeechTokens(spk.AudioTokens);

                // Vocoder dispatch: batched if vocBatch > 1, serial otherwise.
                long batchVocMs = 0;
                float[][] batchSamples;
                if (vocBatch > 1)
                {
                    // MVP requires same-length speech_tokens; the test driver
                    // replicates the same chunk so this holds. Real long-form
                    // would need ragged-input support (Phase 2).
                    var vocSw = Stopwatch.StartNew();
                    batchSamples = vocoder.SynthesizeBatch(
                        speechTokensPerB, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
                    vocSw.Stop();
                    batchVocMs = vocSw.ElapsedMilliseconds;
                }
                else
                {
                    batchSamples = new float[B][];
                    for (int b = 0; b < B; b++)
                    {
                        sw.Restart();
                        batchSamples[b] = vocoder.Synthesize(
                            speechTokensPerB[b], spk.SpeakerEmbeddings, spk.SpeakerFeatures);
                        sw.Stop();
                        chunkVocMs[produced + b] = sw.ElapsedMilliseconds;
                    }
                }

                for (int b = 0; b < B; b++)
                {
                    var lmResult = batched.PerChunkResults[b];
                    int globalIdx = produced + b;
                    chunkLmMs[globalIdx] = batchLmSw.ElapsedMilliseconds / B;  // amortized share
                    if (vocBatch > 1) chunkVocMs[globalIdx] = batchVocMs / B;  // amortized share
                    chunkSteps[globalIdx] = lmResult.Steps;
                    lastSamples = batchSamples[b];
                    Console.WriteLine($"  chunk {globalIdx + 1}/{benchChunks} [batch {(produced / lmBatch) + 1}, elem {b + 1}/{B}]: "
                        + $"LM-share {chunkLmMs[globalIdx]} ms, "
                        + $"voc{(vocBatch > 1 ? "-share" : "")} {chunkVocMs[globalIdx]} ms, "
                        + $"audio {batchSamples[b].Length / (float)ChatterboxConstants.S3GenSr:F2}s");
                }
                Console.WriteLine($"    batch LM call: {batchLmSw.ElapsedMilliseconds} ms total for B={B}"
                    + (vocBatch > 1 ? $"; batch voc call: {batchVocMs} ms total for B={B}" : ""));
                produced += B;
            }
            batchWallSw.Stop();
            pipelinedChunkWallMs = batchWallSw.ElapsedMilliseconds;  // reuse for the summary
        }
        else
        {
            for (int chunk = 0; chunk < benchChunks; chunk++)
            {
                var lmSw = Stopwatch.StartNew();
                var lmResult = lm.Generate(spk.CondEmb, inputIds,
                    useIoBinding: useIoBinding,
                    diagDir: chunk == 0 ? diagDir : null);  // only diag the first chunk
                lmSw.Stop();
                chunkLmMs[chunk] = lmSw.ElapsedMilliseconds;
                chunkSteps[chunk] = lmResult.Steps;

                if (chunk == 0 && diagDir is not null)
                {
                    File.WriteAllBytes(Path.Combine(diagDir, "cs_tokens.bin"),
                        System.Runtime.InteropServices.MemoryMarshal.AsBytes<long>(
                            lmResult.RawGeneratedTokens.ToArray()).ToArray());
                    Console.WriteLine($"[diag] wrote {diagDir}/cs_tokens.bin ({lmResult.RawGeneratedTokens.Count} tokens)");
                }

                var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
                sw.Restart();
                var samples = vocoder.Synthesize(speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
                sw.Stop();
                chunkVocMs[chunk] = sw.ElapsedMilliseconds;
                lastSamples = samples;

                if (benchChunks == 1)
                {
                    // Existing single-chunk verbose output.
                    Console.WriteLine($"LM ({(useIoBinding ? "io-binding" : "basic")}): {lmResult.Steps} steps, "
                        + $"generated {lmResult.RawGeneratedTokens.Count - 1} tokens "
                        + $"[{chunkLmMs[chunk]} ms, {chunkLmMs[chunk] / (double)lmResult.Steps:F1} ms/step]");
                    Console.WriteLine($"speech_tokens: shape=(1, {speechTokens.Length})  "
                        + $"({spk.AudioTokens.Length} from voice + {speechTokens.Length - spk.AudioTokens.Length} from LM)");
                    Console.WriteLine($"cond_decoder ({vocoder.Mode}): waveform=(1, {samples.Length}) "
                        + $"→ {samples.Length / (float)ChatterboxConstants.S3GenSr:F2}s  [{chunkVocMs[chunk]} ms]");
                }
                else
                {
                    Console.WriteLine($"  chunk {chunk + 1}/{benchChunks}: "
                        + $"LM {chunkLmMs[chunk]} ms ({chunkSteps[chunk]} steps), "
                        + $"voc {chunkVocMs[chunk]} ms, "
                        + $"audio {samples.Length / (float)ChatterboxConstants.S3GenSr:F2}s");
                }
            }
        }

        // ── Write last chunk's WAV (or only chunk in non-bench mode) ──────
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(ChatterboxConstants.S3GenSr, 1);
        using (var writer = new WaveFileWriter(outPath, fmt))
        {
            writer.WriteSamples(lastSamples!, 0, lastSamples!.Length);
        }
        totalSw.Stop();

        if (benchChunks > 1)
        {
            long lmSum = 0, vocSum = 0;
            foreach (var v in chunkLmMs) lmSum += v;
            foreach (var v in chunkVocMs) vocSum += v;
            // For pipelined mode, "chunks-total" is the actual wall measured
            // by ChunkedSynthesizer (not the sum, which over-counts because
            // LM and vocoder overlap). For serial mode, sum and chunks-total
            // are the same thing.
            // For batched / pipelined modes, "chunks-wall" is the actual
            // wall measured for the orchestrator (not the sum, which
            // over-counts because of batching amortization or LM/voc
            // overlap). For serial mode, sum and chunks-wall coincide.
            double chunksTotalSec = (pipelined || lmBatch > 1)
                ? pipelinedChunkWallMs / 1000.0
                : (lmSum + vocSum) / 1000.0;
            string mode;
            if (pipelined && lmBatch > 1) mode = $"pipelined+group-batch={lmBatch}";
            else if (pipelined) mode = "pipelined";
            else if (lmBatch > 1) mode = vocBatch > 1
                ? $"lm-batch={lmBatch}+voc-batch={vocBatch}"
                : $"lm-batch={lmBatch}";
            else mode = "serial";
            Console.WriteLine($"Bench: {benchChunks} chunks ({mode}), "
                + $"LM avg {lmSum / (double)benchChunks:F0} ms, "
                + $"voc avg {vocSum / (double)benchChunks:F0} ms, "
                + $"per-chunk-sum-avg {(lmSum + vocSum) / (double)benchChunks:F0} ms, "
                + $"chunks-wall {chunksTotalSec:F1}s "
                + $"[total wall {totalSw.ElapsedMilliseconds / 1000.0:F1}s]");
        }
        else
        {
            Console.WriteLine($"Wrote {outPath}  [total {totalSw.ElapsedMilliseconds / 1000.0:F1}s]");
        }
        return 0;
    }

    private static string ExpandHome(string path)
        => path.StartsWith("~/")
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), path[2..])
            : path;

    /// <summary>
    /// Bypasses voice/tokenizer/vocoder; loads only embed_tokens + language_model,
    /// runs a single prefill at batch=B plus a handful of autoregressive step
    /// iterations, reports per-step ms. Used to answer:
    ///   1. Does ORT actually honor the LM ONNX's B>1 dynamic_axes? (yes/no)
    ///   2. What is the amortization factor — wall(B=N) / wall(B=1)?
    ///      Lower than 1.0 means batching wins; close to N means we're
    ///      memory-bandwidth-bound and batching adds little.
    ///
    /// Same prompt is replicated across the B dimension to keep the prefill
    /// length identical per batch element. Real per-element variable-length
    /// prefill is a Stage-2-implementation concern, not a probe concern.
    /// </summary>
    private static int RunBatchedLmProbe(string onnxDir, ExecutionProvider ep, int batchSize)
    {
        const int StepIterations = 10;  // average across this many step calls

        Console.WriteLine($"Batched-LM probe: B={batchSize}, ep={ep}");
        var loadSw = Stopwatch.StartNew();
        using var embed = OrtSessionBuilder.CreateCachedSession(
            Path.Combine(onnxDir, "embed_tokens.onnx"), ep);
        using var lm = OrtSessionBuilder.CreateCachedSession(
            Path.Combine(onnxDir, "language_model.onnx"), ep);
        loadSw.Stop();
        Console.WriteLine($"  loaded embed_tokens + language_model in {loadSw.ElapsedMilliseconds} ms");

        // Probe is fp32-hardcoded (DenseTensor<float> for inputs_embeds + past_kv);
        // an fp16 LM would throw ORT type-mismatch at the first BindInput with
        // no useful message. Fail clearly instead.
        if (lm.InputMetadata.TryGetValue("inputs_embeds", out var iemd)
            && iemd.ElementDataType == TensorElementType.Float16)
        {
            Console.Error.WriteLine(
                "--bench-batched-lm is fp32-hardcoded and the loaded LM expects fp16 inputs. "
                + "Re-point the bundle's language_model.onnx at an fp32 export, or use the full "
                + "synthesis path (--bench-chunks) which routes through AcousticLM's dtype-aware binding.");
            return 2;
        }

        // Build B-batched input. Replicate FallbackInputIds (the Ezreal sentence
        // wrapped with [EXAGGERATION, ..., START_SPEECH, START_SPEECH]) across B.
        int sText = FallbackInputIds.Length;
        var idsB = new long[batchSize * sText];
        var posB = new long[batchSize * sText];
        for (int b = 0; b < batchSize; b++)
        {
            Array.Copy(FallbackInputIds, 0, idsB, b * sText, sText);
            for (int i = 0; i < sText; i++)
                posB[b * sText + i] = FallbackInputIds[i] >= ChatterboxConstants.StartSpeechToken ? 0 : i - 1;
        }
        var exagB = new float[batchSize];
        Array.Fill(exagB, ChatterboxConstants.DefaultExaggeration);

        // embed_tokens at B, S (dynamic axes support it).
        var embSw = Stopwatch.StartNew();
        using var embOut = embed.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(idsB, new[] { batchSize, sText })),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(posB, new[] { batchSize, sText })),
            NamedOnnxValue.CreateFromTensor("exaggeration", new DenseTensor<float>(exagB, new[] { batchSize })),
        });
        var textEmb = embOut.First().AsTensor<float>();
        embSw.Stop();
        Console.WriteLine($"  embed_tokens B={batchSize}: {embSw.ElapsedMilliseconds} ms  text_emb shape={DimsOf(textEmb.Dimensions)}");

        // inputs_embeds = text_emb directly (no cond_emb prepend; this probe
        // doesn't care about parity, only kernel timing). Shape: [B, S, 1024].
        var inputsEmbeds = textEmb.ToArray();
        int sTotal = sText;
        int hidden = ChatterboxConstants.LlmHidden;

        // ── Prefill ─────────────────────────────────────────────────────
        var attentionMask = new long[batchSize * sTotal];
        Array.Fill(attentionMask, 1);
        var pastKv = new float[ChatterboxConstants.LlmLayers * 2][];
        var pastKvShape = new int[] { batchSize, ChatterboxConstants.LlmKvHeads, 0, ChatterboxConstants.LlmHeadDim };
        for (int k = 0; k < pastKv.Length; k++) pastKv[k] = [];

        var prefillSw = Stopwatch.StartNew();
        var prefillOut = RunLmOnce(lm, inputsEmbeds, batchSize, sTotal, attentionMask, pastKv, pastKvShape);
        prefillSw.Stop();
        var prefillList = prefillOut.ToList();
        int vocab = prefillList[0].AsTensor<float>().Dimensions[2];
        for (int layer = 0; layer < ChatterboxConstants.LlmLayers; layer++)
        {
            pastKv[2 * layer] = prefillList[1 + 2 * layer].AsTensor<float>().ToArray();
            pastKv[2 * layer + 1] = prefillList[1 + 2 * layer + 1].AsTensor<float>().ToArray();
        }
        pastKvShape = prefillList[1].AsTensor<float>().Dimensions.ToArray();
        prefillOut.Dispose();
        Console.WriteLine($"  prefill B={batchSize} S={sTotal}: {prefillSw.ElapsedMilliseconds} ms  vocab={vocab}  past_kv per-layer shape={DimsOf(pastKvShape)}");

        // ── StepIterations autoregressive steps (random input embeds, B,1,1024) ──
        var stepMs = new long[StepIterations];
        for (int step = 0; step < StepIterations; step++)
        {
            var stepEmbeds = new float[batchSize * 1 * hidden];
            // Use deterministic dummy data; this probe measures cost, not output correctness.
            for (int j = 0; j < stepEmbeds.Length; j++) stepEmbeds[j] = (j % 17) / 17.0f;

            attentionMask = GrowBatchedMask(attentionMask, batchSize, 1);
            var stepSw = Stopwatch.StartNew();
            var stepOut = RunLmOnce(lm, stepEmbeds, batchSize, 1, attentionMask, pastKv, pastKvShape);
            stepSw.Stop();
            stepMs[step] = stepSw.ElapsedMilliseconds;

            var stepList = stepOut.ToList();
            for (int layer = 0; layer < ChatterboxConstants.LlmLayers; layer++)
            {
                pastKv[2 * layer] = stepList[1 + 2 * layer].AsTensor<float>().ToArray();
                pastKv[2 * layer + 1] = stepList[1 + 2 * layer + 1].AsTensor<float>().ToArray();
            }
            pastKvShape = stepList[1].AsTensor<float>().Dimensions.ToArray();
            stepOut.Dispose();
        }

        long stepSum = 0;
        foreach (var ms in stepMs) stepSum += ms;
        double stepAvgPerCall = stepSum / (double)StepIterations;
        double stepAvgPerBatchElem = stepAvgPerCall / batchSize;
        Console.WriteLine($"  step iter avg: {stepAvgPerCall:F1} ms/call  ({stepAvgPerBatchElem:F1} ms/batch-elem  → B={batchSize} amortization)");
        return 0;
    }

    private static IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunLmOnce(
        InferenceSession lm, float[] inputsEmbeds, int batch, int seqLen,
        long[] attentionMask, float[][] pastKv, int[] pastKvShape)
    {
        const int LlmHidden = ChatterboxConstants.LlmHidden;
        const int LlmLayers = ChatterboxConstants.LlmLayers;
        var embedT = new DenseTensor<float>(inputsEmbeds, new[] { batch, seqLen, LlmHidden });
        var maskT = new DenseTensor<long>(attentionMask, new[] { batch, attentionMask.Length / batch });
        var inputs = new List<NamedOnnxValue>(2 + 2 * LlmLayers)
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", embedT),
            NamedOnnxValue.CreateFromTensor("attention_mask", maskT),
        };
        for (int layer = 0; layer < LlmLayers; layer++)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                $"past_key_values.{layer}.key", new DenseTensor<float>(pastKv[2 * layer], pastKvShape)));
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                $"past_key_values.{layer}.value", new DenseTensor<float>(pastKv[2 * layer + 1], pastKvShape)));
        }
        return lm.Run(inputs);
    }

    private static long[] GrowBatchedMask(long[] mask, int batch, int growBy)
    {
        int oldPerB = mask.Length / batch;
        int newPerB = oldPerB + growBy;
        var grown = new long[batch * newPerB];
        for (int b = 0; b < batch; b++)
        {
            Array.Copy(mask, b * oldPerB, grown, b * newPerB, oldPerB);
            for (int i = oldPerB; i < newPerB; i++) grown[b * newPerB + i] = 1;
        }
        return grown;
    }

    private static string DimsOf(IReadOnlyList<int> dims)
        => "(" + string.Join(", ", dims) + ")";
    private static string DimsOf(System.ReadOnlySpan<int> dims)
        => "(" + string.Join(", ", dims.ToArray()) + ")";

    /// <summary>
    /// Sister to <see cref="RunBatchedLmProbe"/> — confirms whether the
    /// merged conditional_decoder_loop.onnx (or split / monolithic
    /// fallback) honors B>1 inputs. Uses random data at the right
    /// shapes (we don't care about output quality here, only whether
    /// the graph accepts the batched call and how the wall time scales).
    ///
    /// ALSO probes the split flow_encoder + cfm_estimator + mel2wav
    /// graphs at the same B, since the merged graph has CFG-doubling
    /// baked in (axis-0 trace at 2) which blocks B>1 there.
    /// </summary>
    private static int RunBatchedVocProbe(string onnxDir, ExecutionProvider ep, int batchSize)
    {
        // Reuse Vocoder's auto-detect to pick the layout we'd actually
        // ship, then call its underlying merged session directly with a
        // B-shaped input. The probe needs direct ORT access because
        // Vocoder.Synthesize hardcodes batch=1 today (the SynthesizeBatch
        // method is the thing we're motivating with this probe).
        string mergedPath = Path.Combine(onnxDir, "conditional_decoder_loop.onnx");
        if (!File.Exists(mergedPath))
        {
            Console.Error.WriteLine($"--bench-batched-voc requires the merged conditional_decoder_loop.onnx in --onnx-dir; not found at {mergedPath}");
            return 1;
        }

        Console.WriteLine($"Batched-voc probe: B={batchSize}, ep={ep}");
        var loadSw = Stopwatch.StartNew();
        using var session = OrtSessionBuilder.CreateCachedSession(mergedPath, ep);
        loadSw.Stop();
        Console.WriteLine($"  loaded conditional_decoder_loop.onnx in {loadSw.ElapsedMilliseconds} ms");

        // Synthetic inputs at the canonical shapes Vocoder.Synthesize uses.
        // speech_tokens: [B, T_tok] of int64 in [0, SPEECH_VOCAB_SIZE);
        //                we pick T_tok = 423 to match the Ezreal-sentence
        //                speech_tokens length the bench produces (250
        //                from voice + 173 from LM).
        // speaker_embeddings: [B, 192] of float
        // speaker_features:   [B, 500, 80] of float
        const int TTok = 423;
        const int SpeakerDim = 192;
        const int PromptLen = ChatterboxConstants.PromptLen;
        const int MelBins = ChatterboxConstants.MelBins;

        var rng = new Random(0);
        var speechTokensB = new long[batchSize * TTok];
        for (int i = 0; i < speechTokensB.Length; i++)
            speechTokensB[i] = rng.Next(0, 6561);  // SPEECH_BASE_VOCAB_SIZE
        var spkEmbB = new float[batchSize * SpeakerDim];
        for (int i = 0; i < spkEmbB.Length; i++) spkEmbB[i] = (float)(rng.NextDouble() * 2 - 1);
        var spkFeatB = new float[batchSize * PromptLen * MelBins];
        for (int i = 0; i < spkFeatB.Length; i++) spkFeatB[i] = (float)(rng.NextDouble() * 2 - 1);

        var runSw = Stopwatch.StartNew();
        try
        {
            using var output = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("speech_tokens",
                    new DenseTensor<long>(speechTokensB, new[] { batchSize, TTok })),
                NamedOnnxValue.CreateFromTensor("speaker_embeddings",
                    new DenseTensor<float>(spkEmbB, new[] { batchSize, SpeakerDim })),
                NamedOnnxValue.CreateFromTensor("speaker_features",
                    new DenseTensor<float>(spkFeatB, new[] { batchSize, PromptLen, MelBins })),
            });
            runSw.Stop();
            var wavTensor = output.First().AsTensor<float>();
            int outBatch = wavTensor.Dimensions[0];
            int outSamples = wavTensor.Dimensions[1];
            double msPerElem = runSw.ElapsedMilliseconds / (double)batchSize;
            Console.WriteLine($"  vocoder B={batchSize}: {runSw.ElapsedMilliseconds} ms wall  ({msPerElem:F1} ms/batch-elem  → output shape ({outBatch},{outSamples}))");
            return 0;
        }
        catch (Exception ex)
        {
            runSw.Stop();
            Console.Error.WriteLine($"  vocoder B={batchSize} FAILED after {runSw.ElapsedMilliseconds} ms: {ex.GetType().Name}: {ex.Message.Split('\n')[0]}");
            // Fall through to the split-graph probe — that's where the
            // hope is, since the merged graph baked the CFG-doubling
            // hardcoded at trace time.
        }

        // ── Split-graph probe (flow_encoder + cfm_estimator + mel2wav) ──
        return ProbeSplitVocoderAtBatch(onnxDir, ep, batchSize, speechTokensB, spkEmbB, spkFeatB,
            TTok, SpeakerDim, PromptLen, MelBins);
    }

    private static int ProbeSplitVocoderAtBatch(string onnxDir, ExecutionProvider ep, int batchSize,
        long[] speechTokensB, float[] spkEmbB, float[] spkFeatB,
        int TTok, int SpeakerDim, int PromptLen, int MelBins)
    {
        string flowPath = Path.Combine(onnxDir, "flow_encoder.onnx");
        string cfmPath = Path.Combine(onnxDir, "cfm_estimator.onnx");
        string m2wPath = Path.Combine(onnxDir, "mel2wav.onnx");
        if (!File.Exists(flowPath) || !File.Exists(cfmPath) || !File.Exists(m2wPath))
        {
            Console.Error.WriteLine($"  split vocoder graphs not in {onnxDir}; can't fall back to split probe.");
            return 1;
        }
        Console.WriteLine($"  Probing split graphs at B={batchSize} ...");
        using var flow = OrtSessionBuilder.CreateCachedSession(flowPath, ep);
        using var cfm = OrtSessionBuilder.CreateCachedSession(cfmPath, ep);
        using var m2w = OrtSessionBuilder.CreateCachedSession(m2wPath, ep);

        // flow_encoder at B>1
        var flowSw = Stopwatch.StartNew();
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue>? flowOut = null;
        try
        {
            flowOut = flow.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("speech_tokens",
                    new DenseTensor<long>(speechTokensB, new[] { batchSize, TTok })),
                NamedOnnxValue.CreateFromTensor("speaker_embeddings",
                    new DenseTensor<float>(spkEmbB, new[] { batchSize, SpeakerDim })),
                NamedOnnxValue.CreateFromTensor("speaker_features",
                    new DenseTensor<float>(spkFeatB, new[] { batchSize, PromptLen, MelBins })),
            });
            flowSw.Stop();
            var muT = flowOut.ToList()[0].AsTensor<float>();
            Console.WriteLine($"  flow_encoder B={batchSize}: {flowSw.ElapsedMilliseconds} ms  mu={DimsOf(muT.Dimensions)}");
        }
        catch (Exception ex)
        {
            flowSw.Stop();
            Console.Error.WriteLine($"  flow_encoder B={batchSize} FAILED after {flowSw.ElapsedMilliseconds} ms: {ex.Message.Split('\n')[0]}");
            return 1;
        }

        // cfm_estimator at B>2 (CFG-doubled): replicate flow outputs ×2 along batch
        var encList = flowOut!.ToList();
        var muArr = encList[0].AsTensor<float>().ToArray();
        var maskArr = encList[1].AsTensor<float>().ToArray();
        var embedArr = encList[2].AsTensor<float>().ToArray();
        var condArr = encList[3].AsTensor<float>().ToArray();
        var zArr = encList[4].AsTensor<float>().ToArray();
        int tMel = encList[0].AsTensor<float>().Dimensions[2];

        // CFG-doubled to [2*B, ...] per the C# split-vocoder convention.
        int cfgB = 2 * batchSize;
        var muIn = new float[cfgB * MelBins * tMel];
        Array.Copy(muArr, 0, muIn, 0, muArr.Length);
        // second half stays zeros (uncond)
        var maskIn = new float[cfgB * 1 * tMel];
        Array.Copy(maskArr, 0, maskIn, 0, maskArr.Length);
        Array.Copy(maskArr, 0, maskIn, maskArr.Length, maskArr.Length);
        var xIn = new float[cfgB * MelBins * tMel];
        Array.Copy(zArr, 0, xIn, 0, zArr.Length);
        Array.Copy(zArr, 0, xIn, zArr.Length, zArr.Length);
        var spksIn = new float[cfgB * MelBins];
        Array.Copy(embedArr, 0, spksIn, 0, embedArr.Length);
        var condIn = new float[cfgB * MelBins * tMel];
        Array.Copy(condArr, 0, condIn, 0, condArr.Length);
        var tIn = new float[cfgB];
        Array.Fill(tIn, 0.0f);

        var cfmSw = Stopwatch.StartNew();
        try
        {
            using var cfmOut = cfm.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("x_in",    new DenseTensor<float>(xIn,    new[] { cfgB, MelBins, tMel })),
                NamedOnnxValue.CreateFromTensor("mask_in", new DenseTensor<float>(maskIn, new[] { cfgB, 1, tMel })),
                NamedOnnxValue.CreateFromTensor("mu_in",   new DenseTensor<float>(muIn,   new[] { cfgB, MelBins, tMel })),
                NamedOnnxValue.CreateFromTensor("t_in",    new DenseTensor<float>(tIn,    new[] { cfgB })),
                NamedOnnxValue.CreateFromTensor("spks_in", new DenseTensor<float>(spksIn, new[] { cfgB, MelBins })),
                NamedOnnxValue.CreateFromTensor("cond_in", new DenseTensor<float>(condIn, new[] { cfgB, MelBins, tMel })),
            });
            cfmSw.Stop();
            var dphi = cfmOut.First().AsTensor<float>();
            Console.WriteLine($"  cfm_estimator B={cfgB} (CFG-doubled): {cfmSw.ElapsedMilliseconds} ms  dphi={DimsOf(dphi.Dimensions)}");
        }
        catch (Exception ex)
        {
            cfmSw.Stop();
            Console.Error.WriteLine($"  cfm_estimator B={cfgB} FAILED after {cfmSw.ElapsedMilliseconds} ms: {ex.Message.Split('\n')[0]}");
            flowOut.Dispose();
            return 1;
        }

        // mel2wav at B>1 — needs a mel of shape [B, 80, T_mel_trimmed]
        int tTrim = tMel - PromptLen;
        var melB = new float[batchSize * MelBins * tTrim];
        // (random values; we only care about timing + crash-free)
        var rng = new Random(0);
        for (int i = 0; i < melB.Length; i++) melB[i] = (float)(rng.NextDouble() * 2 - 1);
        var m2wSw = Stopwatch.StartNew();
        try
        {
            using var m2wOut = m2w.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("mel",
                    new DenseTensor<float>(melB, new[] { batchSize, MelBins, tTrim })),
            });
            m2wSw.Stop();
            var wav = m2wOut.First().AsTensor<float>();
            Console.WriteLine($"  mel2wav B={batchSize}: {m2wSw.ElapsedMilliseconds} ms  wav={DimsOf(wav.Dimensions)}");
        }
        catch (Exception ex)
        {
            m2wSw.Stop();
            Console.Error.WriteLine($"  mel2wav B={batchSize} FAILED after {m2wSw.ElapsedMilliseconds} ms: {ex.Message.Split('\n')[0]}");
            flowOut.Dispose();
            return 1;
        }

        flowOut.Dispose();
        return 0;
    }
}
