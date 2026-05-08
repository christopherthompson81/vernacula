// Granite Speech 4.1 — C# CLI parity smoke + perf tracer.
//
// Drives the exported ONNX bundle end-to-end from a real audio file:
//   audio.wav -> mel.onnx -> encoder.onnx -> projector.onnx ->
//                decoder.onnx (unified) OR decoder_init + decoder_step (split)
//                -> tokens -> ByteLevel BPE decode -> text
//
// Compares against the golden output saved to Fixtures/expected_text.txt
// AND prints a structured timing report so we can find the C#-side
// hotspots (tensor build, output extract, argmax, ORT.Run) on top of the
// ORT runtime cost the Python profiler already characterises.
//
// Usage:
//   dotnet run --project tests/GraniteSpeechSmoke -- \
//       --onnx-dir ./models/granite_speech_4_1_2b \
//       --audio /path/to/clip.wav \
//       --fixtures ./tests/GraniteSpeechSmoke/Fixtures \
//       --max-new-tokens 64 \
//       --ep cuda \
//       --warmup 1

using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.GraniteSpeechSmoke;

internal static class Program
{
    private const int NumDecoderLayers = 40;
    private const int AudioTokenId = 100352;
    private const int EosTokenId = 100257;
    private const int NumKvHeads = 4;
    private const int HeadDim = 128;

    private static int Main(string[] args)
    {
        // ── Argument parsing ──────────────────────────────────────────────
        string? onnxDir = null;
        string? audioPath = null;
        string? fixturesDir = null;
        int maxNewTokens = 64;
        string ep = "cpu";          // cpu | cuda
        int warmup = 0;
        bool ioBinding = false;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--onnx-dir":          onnxDir = args[++i]; break;
                case "--audio":             audioPath = args[++i]; break;
                case "--fixtures":          fixturesDir = args[++i]; break;
                case "--max-new-tokens":    maxNewTokens = int.Parse(args[++i]); break;
                case "--ep":                ep = args[++i].ToLowerInvariant(); break;
                case "--warmup":            warmup = int.Parse(args[++i]); break;
                case "--io-binding":        ioBinding = true; break;
                default:
                    Console.Error.WriteLine($"Unknown arg: {args[i]}");
                    return 2;
            }
        }
        if (onnxDir is null || audioPath is null || fixturesDir is null)
        {
            Console.Error.WriteLine("Usage: --onnx-dir <dir> --audio <wav> --fixtures <dir> "
                                  + "[--max-new-tokens N] [--ep cpu|cuda] [--warmup N]");
            return 2;
        }
        if (ep is not ("cpu" or "cuda"))
        {
            Console.Error.WriteLine($"Unknown EP: {ep}. Choose cpu or cuda.");
            return 2;
        }

        // ── Load audio ────────────────────────────────────────────────────
        var (raw, sr, ch) = AudioUtils.ReadAudio(audioPath);
        float[] audio = AudioUtils.DownmixToMono(raw, ch);
        if (sr != 16000)
        {
            Console.Error.WriteLine($"Audio must be 16 kHz; got {sr}. Resample first.");
            return 2;
        }
        Console.WriteLine($"audio: {audio.Length} samples @ {sr} Hz "
                        + $"({audio.Length / (double)sr:F2}s)");

        // ── Load fixtures ─────────────────────────────────────────────────
        long[] inputIds = ReadInt64Bin(Path.Combine(fixturesDir, "input_ids.bin"));
        string expectedText = File.ReadAllText(Path.Combine(fixturesDir, "expected_text.txt")).Trim();
        Console.WriteLine($"prompt: {inputIds.Length} tokens (from fixtures)");

        // ── Load tokenizer for decode-side ────────────────────────────────
        var (idToToken, addedTokens) = LoadTokenizer(Path.Combine(onnxDir, "tokenizer.json"));
        var byteLevelDecode = BuildByteLevelDecodeTable();

        // ── Build ORT sessions ────────────────────────────────────────────
        // Use Vernacula.Base's OrtSessionBuilder so we share the same EP
        // wiring as the production backends.
        var execProvider = ep == "cuda" ? ExecutionProvider.Cuda : ExecutionProvider.Cpu;
        var so = OrtSessionBuilder.Create(execProvider, GraphOptimizationLevel.ORT_ENABLE_ALL);

        bool unified = File.Exists(Path.Combine(onnxDir, "decoder.onnx"));
        Console.WriteLine($"decoder mode: {(unified ? "unified" : "split init/step")}");
        Console.WriteLine($"execution provider: {ep}");
        Console.WriteLine($"io-binding: {ioBinding}");
        if (ioBinding && !unified)
        {
            Console.Error.WriteLine("--io-binding currently requires the unified decoder bundle.");
            return 2;
        }
        if (ioBinding && ep != "cuda")
        {
            Console.Error.WriteLine("--io-binding only makes sense with --ep cuda.");
            return 2;
        }
        if (ioBinding)
        {
            Console.Error.WriteLine("WARNING: --io-binding is experimental — it produces incorrect "
                                  + "transcripts on the unified Granite decoder. Both RunWithBinding "
                                  + "and the OrtValue-based Run() overload produce garbage when fed "
                                  + "OrtValue inputs created via CreateTensorValueFromMemory; only "
                                  + "the DenseTensor + NamedOnnxValue path produces correct output. "
                                  + "Use this flag for perf measurement only. See "
                                  + "docs/dev/granite_speech_perf_investigation.md Run 5.");
        }

        var loadStopwatch = Stopwatch.StartNew();
        using var melSess = new InferenceSession(Path.Combine(onnxDir, "mel.onnx"), so);
        using var encSess = new InferenceSession(Path.Combine(onnxDir, "encoder.onnx"), so);
        using var projSess = new InferenceSession(Path.Combine(onnxDir, "projector.onnx"), so);
        using var initSess = new InferenceSession(
            Path.Combine(onnxDir, unified ? "decoder.onnx" : "decoder_init.onnx"), so);
        using var stepSess = unified
            ? initSess
            : new InferenceSession(Path.Combine(onnxDir, "decoder_step.onnx"), so);
        loadStopwatch.Stop();
        Console.WriteLine($"sessions loaded in {loadStopwatch.ElapsedMilliseconds} ms");

        // ── Warm-up runs ──────────────────────────────────────────────────
        for (int w = 0; w < warmup; w++)
        {
            Console.WriteLine($"\n--- Warm-up {w + 1}/{warmup} ---");
            RunPipeline(melSess, encSess, projSess, initSess, stepSess, unified,
                audio, inputIds, maxNewTokens, byteLevelDecode, idToToken, addedTokens,
                ioBinding: ioBinding, trace: null);
        }

        // ── Timed run ─────────────────────────────────────────────────────
        var trace = new TraceReport();
        Console.WriteLine($"\n--- Timed run ---");
        var result = RunPipeline(melSess, encSess, projSess, initSess, stepSess, unified,
            audio, inputIds, maxNewTokens, byteLevelDecode, idToToken, addedTokens,
            ioBinding: ioBinding, trace: trace);

        Console.WriteLine();
        Console.WriteLine($"  ORT  transcript: {Quote(result.text)}");
        Console.WriteLine($"  Ref  transcript: {Quote(expectedText)}");
        bool match = result.text.Trim() == expectedText.Trim();
        Console.WriteLine($"  exact match: {match}");

        Console.WriteLine();
        trace.Print(audio.Length / (double)sr);
        return match ? 0 : 1;
    }

    // ── Pipeline ────────────────────────────────────────────────────────

    private sealed class TraceReport
    {
        // Stage timings (one per stage; ms).
        public long MelMs;
        public long EncoderMs;
        public long ProjectorMs;
        public long DecoderInitMs;

        // Step loop aggregate breakdown (ms).
        public long StepRunTotalMs;        // ORT.Run() time across all steps
        public long StepInputBuildMs;      // building DenseTensors + NamedOnnxValue list
        public long StepOutputExtractMs;   // ToArray() copies of K/V + logits
        public long StepArgmaxMs;          // argmax over the vocab
        public long StepBookkeepingMs;     // attention_mask alloc, cache_position, etc.
        public int  StepCount;

        public void Print(double audioSeconds)
        {
            long stepTotal = StepRunTotalMs + StepInputBuildMs + StepOutputExtractMs
                           + StepArgmaxMs + StepBookkeepingMs;
            long total = MelMs + EncoderMs + ProjectorMs + DecoderInitMs + stepTotal;

            Console.WriteLine("=== Stage timings ===");
            Console.WriteLine($"  mel               {MelMs,8} ms");
            Console.WriteLine($"  encoder           {EncoderMs,8} ms");
            Console.WriteLine($"  projector         {ProjectorMs,8} ms");
            Console.WriteLine($"  decoder_init      {DecoderInitMs,8} ms");
            Console.WriteLine($"  decoder_step x{StepCount,-3} {stepTotal,8} ms total"
                            + $"  ({(stepTotal / (double)Math.Max(StepCount, 1)):F2} ms/token,"
                            + $" {1000.0 / (stepTotal / (double)Math.Max(StepCount, 1)):F1} tok/s)");
            Console.WriteLine($"  ----");
            Console.WriteLine($"  TOTAL             {total,8} ms"
                            + $"   (RTF {audioSeconds * 1000 / total:F2}x realtime)");

            Console.WriteLine();
            Console.WriteLine("=== decoder_step breakdown ===");
            void Row(string name, long ms) =>
                Console.WriteLine($"  {name,-22} {ms,7} ms"
                                + $"  ({100.0 * ms / Math.Max(stepTotal, 1):F1}%)");
            Row("ORT.Run()",          StepRunTotalMs);
            Row("Output extract",     StepOutputExtractMs);
            Row("Input build",        StepInputBuildMs);
            Row("Argmax",             StepArgmaxMs);
            Row("Bookkeeping",        StepBookkeepingMs);
        }
    }

    private sealed record PipelineResult(string text, List<long> tokenIds);

    private static PipelineResult RunPipeline(
        InferenceSession melSess,
        InferenceSession encSess,
        InferenceSession projSess,
        InferenceSession initSess,
        InferenceSession stepSess,
        bool unified,
        float[] audio,
        long[] inputIds,
        int maxNewTokens,
        Dictionary<char, byte> byteLevelDecode,
        string?[] idToToken,
        Dictionary<int, string> addedTokens,
        bool ioBinding,
        TraceReport? trace)
    {
        var sw = Stopwatch.StartNew();
        // ── Mel ───────────────────────────────────────────────────────────
        var audioTensor = new DenseTensor<float>(audio, [1, audio.Length]);
        sw.Restart();
        using var melResults = melSess.Run([NamedOnnxValue.CreateFromTensor("audio", audioTensor)]);
        var inputFeatures = (DenseTensor<float>)melResults[0].AsTensor<float>();
        var ifShape = inputFeatures.Dimensions.ToArray();
        var ifData = inputFeatures.ToArray();
        sw.Stop();
        if (trace != null) trace.MelMs = sw.ElapsedMilliseconds;
        Console.WriteLine($"  mel: input_features={Shape(ifShape)} ({sw.ElapsedMilliseconds} ms)");

        // ── Encoder ───────────────────────────────────────────────────────
        sw.Restart();
        var encInputT = new DenseTensor<float>(ifData, ifShape);
        using var encResults = encSess.Run([NamedOnnxValue.CreateFromTensor("input_features", encInputT)]);
        var encoderHidden = encResults[0].AsTensor<float>();
        var encShape = encoderHidden.Dimensions.ToArray();
        var encData = encoderHidden.ToArray();
        sw.Stop();
        if (trace != null) trace.EncoderMs = sw.ElapsedMilliseconds;
        Console.WriteLine($"  encoder: encoder_hidden={Shape(encShape)} ({sw.ElapsedMilliseconds} ms)");

        // ── Projector ─────────────────────────────────────────────────────
        sw.Restart();
        var projInputT = new DenseTensor<float>(encData, encShape);
        using var projResults = projSess.Run([NamedOnnxValue.CreateFromTensor("encoder_hidden", projInputT)]);
        var audioEmbeds = projResults[0].AsTensor<float>();
        var projShape = audioEmbeds.Dimensions.ToArray();
        var projData = audioEmbeds.ToArray();
        sw.Stop();
        if (trace != null) trace.ProjectorMs = sw.ElapsedMilliseconds;
        Console.WriteLine($"  projector: audio_embeds={Shape(projShape)} ({sw.ElapsedMilliseconds} ms)");

        if (ioBinding)
        {
            // IOBinding path keeps KV cache GPU-resident across steps.
            // Prefill + step both go through the unified decoder graph.
            return RunPipelineIoBinding(
                initSess, unified, projData, projShape,
                inputIds, maxNewTokens, idToToken, addedTokens, byteLevelDecode,
                ifShape, encShape, trace);
        }

        // ── Decoder init / prefill ────────────────────────────────────────
        int promptLen = inputIds.Length;
        var inputIdsT = new DenseTensor<long>(inputIds, [1, promptLen]);
        var attentionMask = new long[promptLen];
        for (int i = 0; i < promptLen; i++) attentionMask[i] = 1L;
        var attentionMaskT = new DenseTensor<long>(attentionMask, [1, promptLen]);
        var audioEmbedsT = new DenseTensor<float>(projData, projShape);

        sw.Restart();
        var initInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsT),
            NamedOnnxValue.CreateFromTensor(unified ? "audio_embeds" : "audio_embeds", audioEmbedsT),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskT),
        };
        if (unified)
        {
            // cache_position [S]
            var cachePosArr = new long[promptLen];
            for (int i = 0; i < promptLen; i++) cachePosArr[i] = i;
            initInputs.Add(NamedOnnxValue.CreateFromTensor(
                "cache_position", new DenseTensor<long>(cachePosArr, [promptLen])));
            // Zero-length past_kv per layer (tells the unified graph this is prefill).
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                initInputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_{L}",
                    new DenseTensor<float>(Array.Empty<float>(), [1, NumKvHeads, 0, HeadDim])));
                initInputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_value_{L}",
                    new DenseTensor<float>(Array.Empty<float>(), [1, NumKvHeads, 0, HeadDim])));
            }
        }
        var initResults = initSess.Run(initInputs);
        var initLogits = initResults[0].AsTensor<float>();
        var initShape = initLogits.Dimensions.ToArray();
        int vocabSize = (int)initShape[2];

        // Pre-allocate per-layer K/V buffers sized for the MAX cache length
        // (promptLen + maxNewTokens). Each step copies the new present_kv
        // into the *prefix* of these buffers via Buffer.BlockCopy and binds
        // the next step's past_kv as a DenseTensor view of [..currentLen].
        // This avoids 20 000 .ToArray() allocations across a 90 s run.
        int maxCacheLen = promptLen + maxNewTokens;
        int kvElemsPerLayer = 1 * NumKvHeads * maxCacheLen * HeadDim;
        var pastKeys = new float[NumDecoderLayers][];
        var pastValues = new float[NumDecoderLayers][];
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            pastKeys[L] = new float[kvElemsPerLayer];
            pastValues[L] = new float[kvElemsPerLayer];
            // Copy the prefill outputs into the prefix of the pre-allocated
            // buffers. We use a temporary array because AsTensor<float>().ToArray()
            // is the cleanest way to materialise the ORT tensor; one extra
            // copy here is amortised across the run.
            int prefillElems = 1 * NumKvHeads * promptLen * HeadDim;
            var kSrc = initResults[1 + L].AsTensor<float>().ToArray();
            var vSrc = initResults[1 + NumDecoderLayers + L].AsTensor<float>().ToArray();
            Buffer.BlockCopy(kSrc, 0, pastKeys[L], 0, prefillElems * sizeof(float));
            Buffer.BlockCopy(vSrc, 0, pastValues[L], 0, prefillElems * sizeof(float));
        }
        long nextToken = ArgmaxLastPosition(initLogits, promptLen, vocabSize);
        initResults.Dispose();
        sw.Stop();
        if (trace != null) trace.DecoderInitMs = sw.ElapsedMilliseconds;
        Console.WriteLine($"  decoder_init: logits={Shape(initShape)} ({sw.ElapsedMilliseconds} ms)");

        // ── Decoder step loop ─────────────────────────────────────────────
        var generated = new List<long>(maxNewTokens);
        int pastLen = promptLen;

        // Step-time dummy audio_embeds (unified graph only). The cumsum
        // merge collapses to a no-op because next_token != audio_token_id.
        // We allocate ONCE and reuse across all steps.
        var stepAudioEmbeds = unified
            ? new DenseTensor<float>(new float[1 * 1 * (int)projShape[2]], [1, 1, (int)projShape[2]])
            : null;

        var stepLoopStart = Stopwatch.StartNew();
        var subsw = new Stopwatch();
        for (int step = 0; step < maxNewTokens; step++)
        {
            if (nextToken == EosTokenId) break;
            generated.Add(nextToken);
            int totalLen = pastLen + 1;

            // --- bookkeeping
            subsw.Restart();
            var stepInputId = new DenseTensor<long>(new[] { nextToken }, [1, 1]);
            var stepAttn = new long[totalLen];
            for (int i = 0; i < totalLen; i++) stepAttn[i] = 1L;
            var stepAttnT = new DenseTensor<long>(stepAttn, [1, totalLen]);
            var cachePos = new DenseTensor<long>(new[] { (long)pastLen }, [1]);
            subsw.Stop();
            if (trace != null) trace.StepBookkeepingMs += subsw.ElapsedMilliseconds;

            // --- input build
            subsw.Restart();
            var feed = new List<NamedOnnxValue>(4 + 2 * NumDecoderLayers);
            if (unified)
            {
                feed.Add(NamedOnnxValue.CreateFromTensor("input_ids", stepInputId));
                feed.Add(NamedOnnxValue.CreateFromTensor("audio_embeds", stepAudioEmbeds!));
                feed.Add(NamedOnnxValue.CreateFromTensor("attention_mask", stepAttnT));
                feed.Add(NamedOnnxValue.CreateFromTensor("cache_position", cachePos));
            }
            else
            {
                feed.Add(NamedOnnxValue.CreateFromTensor("input_id", stepInputId));
                feed.Add(NamedOnnxValue.CreateFromTensor("attention_mask", stepAttnT));
                feed.Add(NamedOnnxValue.CreateFromTensor("cache_position", cachePos));
            }
            // past_kv shapes use the actual current cache length, but the
            // backing array is the over-allocated pastKeys/Values[L] (sized
            // for maxCacheLen). DenseTensor reads only `1 * NumKvHeads *
            // pastLen * HeadDim` elements from the start of the array, so
            // the unused tail is harmless.
            int pastElems = 1 * NumKvHeads * pastLen * HeadDim;
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                feed.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_{L}",
                    new DenseTensor<float>(
                        new Memory<float>(pastKeys[L], 0, pastElems),
                        [1, NumKvHeads, pastLen, HeadDim])));
                feed.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_value_{L}",
                    new DenseTensor<float>(
                        new Memory<float>(pastValues[L], 0, pastElems),
                        [1, NumKvHeads, pastLen, HeadDim])));
            }
            subsw.Stop();
            if (trace != null) trace.StepInputBuildMs += subsw.ElapsedMilliseconds;

            // --- ORT.Run()
            subsw.Restart();
            using var stepResults = stepSess.Run(feed);
            subsw.Stop();
            if (trace != null) trace.StepRunTotalMs += subsw.ElapsedMilliseconds;

            // --- output extract: KV roll-forward via Buffer.BlockCopy into
            // the pre-allocated buffers. Avoids 80 fresh array allocations
            // per step (and the GC pressure that came with them).
            subsw.Restart();
            int totalElems = 1 * NumKvHeads * totalLen * HeadDim;
            int copyBytes = totalElems * sizeof(float);
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                var kTensor = (DenseTensor<float>)stepResults[1 + L].AsTensor<float>();
                var vTensor = (DenseTensor<float>)stepResults[1 + NumDecoderLayers + L].AsTensor<float>();
                // Use the underlying buffer directly via Span<T>.CopyTo for
                // managed-managed copies (Buffer.BlockCopy can't take a
                // Span source). For ORT outputs, the DenseTensor wraps a
                // contiguous buffer so the span is the right size.
                kTensor.Buffer.Span.CopyTo(new Span<float>(pastKeys[L], 0, totalElems));
                vTensor.Buffer.Span.CopyTo(new Span<float>(pastValues[L], 0, totalElems));
            }
            var stepLogits = stepResults[0].AsTensor<float>();
            subsw.Stop();
            if (trace != null) trace.StepOutputExtractMs += subsw.ElapsedMilliseconds;

            // --- argmax
            subsw.Restart();
            nextToken = ArgmaxLastPosition(stepLogits, 1, vocabSize);
            subsw.Stop();
            if (trace != null) trace.StepArgmaxMs += subsw.ElapsedMilliseconds;

            pastLen = totalLen;
        }
        stepLoopStart.Stop();
        if (trace != null) trace.StepCount = generated.Count;
        Console.WriteLine($"  decoder_step: {generated.Count} tokens ({stepLoopStart.ElapsedMilliseconds} ms total)");

        string text = DecodeTokens(generated, idToToken, addedTokens, byteLevelDecode);
        return new PipelineResult(text, generated);
    }

    // ── IOBinding decode path ────────────────────────────────────────────
    //
    // Goals:
    //   * Eliminate the per-step .ToArray() that copies 80 GPU KV tensors
    //     back to managed heap (29.6% of step time at 90 s in the baseline).
    //   * Let ORT keep the KV outputs of step N as direct GPU inputs to
    //     step N+1, with no host roundtrip.
    //
    // The trick is that `binding.GetOutputValues()` returns the ORT-allocated
    // OrtValues for the bound outputs, with their backing storage on whatever
    // device we asked for. We take ownership of those, then bind them as
    // inputs to the next step. ORT sees them already on CUDA and skips the
    // input-side memcpy entirely.
    //
    // The logits stay bound to CPU memory because we need them for argmax.
    private static PipelineResult RunPipelineIoBinding(
        InferenceSession decSess,
        bool unified,                      // always true here (gated upstream)
        float[] projData,
        int[] projShape,
        long[] inputIds,
        int maxNewTokens,
        string?[] idToToken,
        Dictionary<int, string> addedTokens,
        Dictionary<char, byte> byteLevelDecode,
        int[] _ifShape,                    // unused, kept for signature parity
        int[] _encShape,                   // unused
        TraceReport? trace)
    {
        var sw = Stopwatch.StartNew();
        int promptLen = inputIds.Length;
        long audioDim = projShape[2];

        var cudaMem = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        var cpuMem = OrtMemoryInfo.DefaultInstance;

        using var runOpts = new RunOptions();

        // Reusable input buffers that change shape every step. We allocate
        // them on the host; ORT auto-copies to device per call. They are
        // small (4-8 bytes for cache_position, ~10 KB for attention_mask at
        // long context) so this is negligible.
        OrtValue MakeInputIds(long[] ids, long[] shape) =>
            OrtValue.CreateTensorValueFromMemory(ids, shape);

        // Step-time dummy audio_embeds: 1 row, value 0. Cumsum-merge no-op.
        float[] dummyAudio = new float[1 * 1 * (int)audioDim];
        // Build per-step OrtValues for inputs that change shape. We dispose
        // them after RunWithBinding completes.

        // ── Prefill ──────────────────────────────────────────────────────
        sw.Restart();
        var prefillBinding = decSess.CreateIoBinding();
        var binding = prefillBinding;

        long[] inputIdsShape = { 1, promptLen };
        using var inputIdsVal = MakeInputIds(inputIds, inputIdsShape);
        binding.BindInput("input_ids", inputIdsVal);

        long[] audioShape = { 1, projShape[1], projShape[2] };
        using var audioVal = OrtValue.CreateTensorValueFromMemory(projData, audioShape);
        binding.BindInput("audio_embeds", audioVal);

        long[] attnShape = { 1, promptLen };
        long[] attnMaskArr = new long[promptLen];
        for (int i = 0; i < promptLen; i++) attnMaskArr[i] = 1L;
        using var attnVal = OrtValue.CreateTensorValueFromMemory(attnMaskArr, attnShape);
        binding.BindInput("attention_mask", attnVal);

        long[] cachePosArr = new long[promptLen];
        for (int i = 0; i < promptLen; i++) cachePosArr[i] = i;
        using var cpVal = OrtValue.CreateTensorValueFromMemory(cachePosArr, new long[] { promptLen });
        binding.BindInput("cache_position", cpVal);

        // Zero-length past_kv inputs at prefill.
        var emptyPasts = new List<OrtValue>(2 * NumDecoderLayers);
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            var k = OrtValue.CreateTensorValueFromMemory(
                Array.Empty<float>(), new long[] { 1, NumKvHeads, 0, HeadDim });
            var v = OrtValue.CreateTensorValueFromMemory(
                Array.Empty<float>(), new long[] { 1, NumKvHeads, 0, HeadDim });
            emptyPasts.Add(k);
            emptyPasts.Add(v);
            binding.BindInput($"past_key_{L}", k);
            binding.BindInput($"past_value_{L}", v);
        }

        // Outputs: logits to CPU (we need argmax). present_kv also to CPU
        // for now (diagnostic) — copy into fresh OrtValues we own so the
        // step loop reads from buffers we control.
        binding.BindOutputToDevice("logits", cpuMem);
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            binding.BindOutputToDevice($"present_key_{L}", cpuMem);
            binding.BindOutputToDevice($"present_value_{L}", cpuMem);
        }

        decSess.RunWithBinding(runOpts, binding);

        // Copy prefill outputs into buffers we own — do not retain references
        // to binding-allocated OrtValues across the step loop. CRITICAL: we
        // also keep the BACKING ARRAYS alive in `pastKvArrays`. OrtValue.
        // CreateTensorValueFromMemory does not keep its source array rooted;
        // if the array is GC'd, the OrtValue's pointer becomes invalid and
        // ORT reads garbage. Holding a parallel array of references prevents
        // this.
        var prefillOutputs = binding.GetOutputValues();
        var prefillLogits = prefillOutputs[0];
        var pastKvs = new OrtValue[2 * NumDecoderLayers];
        var pastKvArrays = new float[2 * NumDecoderLayers][];
        long[] prefillKvShape = { 1, NumKvHeads, promptLen, HeadDim };
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            pastKvArrays[2 * L]     = prefillOutputs[1 + L].GetTensorDataAsSpan<float>().ToArray();
            pastKvArrays[2 * L + 1] = prefillOutputs[1 + NumDecoderLayers + L].GetTensorDataAsSpan<float>().ToArray();
            pastKvs[2 * L]     = OrtValue.CreateTensorValueFromMemory(pastKvArrays[2 * L], prefillKvShape);
            pastKvs[2 * L + 1] = OrtValue.CreateTensorValueFromMemory(pastKvArrays[2 * L + 1], prefillKvShape);
            prefillOutputs[1 + L].Dispose();
            prefillOutputs[1 + NumDecoderLayers + L].Dispose();
        }
        sw.Stop();
        if (trace != null) trace.DecoderInitMs = sw.ElapsedMilliseconds;

        // Argmax on prefill last position.
        var prefillLogitsTensor = prefillLogits.GetTensorDataAsSpan<float>();
        long[] prefillShape = prefillLogits.GetTensorTypeAndShape().Shape;
        int vocab = (int)prefillShape[2];
        long nextToken = ArgmaxLastPositionSpan(prefillLogitsTensor, promptLen, vocab);
        prefillLogits.Dispose();
        Console.WriteLine($"  decoder_init (IOBinding): logits={Shape(prefillShape.Select(s => (int)s).ToArray())} ({sw.ElapsedMilliseconds} ms)");

        foreach (var p in emptyPasts) p.Dispose();

        // ── Step loop ────────────────────────────────────────────────────
        var generated = new List<long>(maxNewTokens);
        int pastLen = promptLen;
        var subsw = new Stopwatch();
        var stepLoopStart = Stopwatch.StartNew();

        // Step-time small inputs. We allocate scratch arrays sized for the
        // longest possible attention_mask once; cache_position is always [1].
        long[] stepInputIdArr = new long[1];
        long[] cachePosStepArr = new long[1];
        // attention_mask grows; allocate up to promptLen + maxNewTokens.
        long[] attnScratch = new long[promptLen + maxNewTokens + 1];
        for (int i = 0; i < attnScratch.Length; i++) attnScratch[i] = 1L;

        OrtValue MakeStepInputId(long tok)
        {
            stepInputIdArr[0] = tok;
            return OrtValue.CreateTensorValueFromMemory(stepInputIdArr, new long[] { 1, 1 });
        }
        OrtValue MakeCachePos(int len)
        {
            cachePosStepArr[0] = len;
            return OrtValue.CreateTensorValueFromMemory(cachePosStepArr, new long[] { 1 });
        }

        // We keep every binding alive for the whole run. Disposing a binding
        // also frees the OrtValues it allocated for its outputs, and we
        // retain those outputs as past_kv for the next step. The bindings
        // themselves are cheap (just a handle to a native binding object);
        // the heavy data is the GPU KV buffers, which we'd be holding either
        // way.
        var allBindings = new List<OrtIoBinding> { prefillBinding };
        OrtValue[]? prevStepPastKvs = null;

        // The names lists are constant across steps; reuse them.
        var stepInputNames = new List<string>(4 + 2 * NumDecoderLayers)
        {
            "input_ids", "audio_embeds", "attention_mask", "cache_position",
        };
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            stepInputNames.Add($"past_key_{L}");
            stepInputNames.Add($"past_value_{L}");
        }
        var stepOutputNames = new List<string>(1 + 2 * NumDecoderLayers) { "logits" };
        for (int L = 0; L < NumDecoderLayers; L++) stepOutputNames.Add($"present_key_{L}");
        for (int L = 0; L < NumDecoderLayers; L++) stepOutputNames.Add($"present_value_{L}");
        for (int step = 0; step < maxNewTokens; step++)
        {
            if (nextToken == EosTokenId) break;
            generated.Add(nextToken);
            int totalLen = pastLen + 1;

            // --- bookkeeping
            subsw.Restart();
            using var stepInputId = MakeStepInputId(nextToken);
            using var cachePos = MakeCachePos(pastLen);
            using var stepAttn = OrtValue.CreateTensorValueFromMemory(
                attnScratch, new long[] { 1, totalLen });
            using var stepAudio = OrtValue.CreateTensorValueFromMemory(
                dummyAudio, new long[] { 1, 1, audioDim });
            subsw.Stop();
            if (trace != null) trace.StepBookkeepingMs += subsw.ElapsedMilliseconds;

            // --- input build
            subsw.Restart();
            // After a long detour through OrtIoBinding (which produced
            // constant per-step latency + garbage transcript on this graph,
            // see Run 4 of the perf investigation doc), we use the OrtValue-
            // based `Run(inputNames, inputValues, outputNames)` overload
            // instead. This keeps all the IOBinding-style benefits — pass
            // OrtValues directly with no DenseTensor wrapping, return
            // OrtValues that we can re-feed as inputs without going through
            // managed arrays — but goes through ORT's regular Run execution
            // path, which actually consumes the past_kv inputs correctly on
            // the unified decoder.
            // Dispose the prior step's pastKvs now that they're not needed.
            if (prevStepPastKvs != null)
            {
                foreach (var ov in prevStepPastKvs) ov.Dispose();
                prevStepPastKvs = null;
            }
            var stepInputValues = new List<OrtValue>(4 + 2 * NumDecoderLayers)
            {
                stepInputId, stepAudio, stepAttn, cachePos,
            };
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                stepInputValues.Add(pastKvs[2 * L]);
                stepInputValues.Add(pastKvs[2 * L + 1]);
            }
            subsw.Stop();
            if (trace != null) trace.StepInputBuildMs += subsw.ElapsedMilliseconds;

            // --- ORT.Run() with OrtValues
            // Important: DO NOT `using` the result — Dispose on the
            // IDisposableReadOnlyCollection<OrtValue> may dispose the
            // contained OrtValues, invalidating the references we capture
            // below as pastKvs. The list shell GC-collects fine without
            // disposal because it has no finalizer (see the VibeVoiceAsr
            // comment in src/Vernacula.Base for the same pattern).
            subsw.Restart();
            var stepRunResult = decSess.Run(runOpts, stepInputNames, stepInputValues, stepOutputNames);
            subsw.Stop();
            if (trace != null) trace.StepRunTotalMs += subsw.ElapsedMilliseconds;

            // --- output extract: take ownership of fresh OrtValues from
            // ORT's Run, defer disposing prior step's pastKvs until next
            // iteration to avoid any chance of input-still-in-use issues.
            subsw.Restart();
            var stepLogits = stepRunResult[0];
            prevStepPastKvs = (OrtValue[])pastKvs.Clone();
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                pastKvs[2 * L]     = stepRunResult[1 + L];
                pastKvs[2 * L + 1] = stepRunResult[1 + NumDecoderLayers + L];
            }
            subsw.Stop();
            if (trace != null) trace.StepOutputExtractMs += subsw.ElapsedMilliseconds;

            // --- argmax on logits
            subsw.Restart();
            var stepLogitsSpan = stepLogits.GetTensorDataAsSpan<float>();
            nextToken = ArgmaxLastPositionSpan(stepLogitsSpan, 1, vocab);
            stepLogits.Dispose();
            subsw.Stop();
            if (trace != null) trace.StepArgmaxMs += subsw.ElapsedMilliseconds;

            pastLen = totalLen;
        }
        stepLoopStart.Stop();
        if (trace != null) trace.StepCount = generated.Count;
        Console.WriteLine($"  decoder_step (IOBinding): {generated.Count} tokens "
                        + $"({stepLoopStart.ElapsedMilliseconds} ms total)");

        // Cleanup: dispose remaining past_kv handles + every binding.
        if (prevStepPastKvs != null)
            foreach (var ov in prevStepPastKvs) ov.Dispose();
        foreach (var ov in pastKvs) ov.Dispose();
        foreach (var b in allBindings) b.Dispose();

        string text = DecodeTokens(generated, idToToken, addedTokens, byteLevelDecode);
        return new PipelineResult(text, generated);
    }

    private static long ArgmaxLastPositionSpan(ReadOnlySpan<float> span, int seqLen, int vocab)
    {
        int offset = (seqLen - 1) * vocab;
        long best = 0;
        float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = span[offset + v];
            if (val > bestVal) { bestVal = val; best = v; }
        }
        return best;
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static string Shape(int[] dims) => "(" + string.Join(", ", dims) + ")";

    private static string Quote(string s) => "\"" + s + "\"";

    private static long ArgmaxLastPosition(Tensor<float> logits, int seqLen, int vocab)
    {
        var span = logits.ToArray();
        int offset = (seqLen - 1) * vocab;
        long best = 0;
        float bestVal = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = span[offset + v];
            if (val > bestVal) { bestVal = val; best = v; }
        }
        return best;
    }

    private static long[] ReadInt64Bin(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length % 8 != 0)
            throw new InvalidDataException($"{path}: size {bytes.Length} not a multiple of 8.");
        long[] result = new long[bytes.Length / 8];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }

    private static (string?[] idToToken, Dictionary<int, string> addedContent)
        LoadTokenizer(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var vocab = root.GetProperty("model").GetProperty("vocab");

        int maxId = 0;
        foreach (var kv in vocab.EnumerateObject())
            if (kv.Value.GetInt32() > maxId) maxId = kv.Value.GetInt32();

        var idToToken = new string?[maxId + 1];
        foreach (var kv in vocab.EnumerateObject())
            idToToken[kv.Value.GetInt32()] = kv.Name;

        var addedContent = new Dictionary<int, string>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var at in addedTokens.EnumerateArray())
            {
                int atId = at.GetProperty("id").GetInt32();
                string atContent = at.GetProperty("content").GetString() ?? "";
                addedContent[atId] = atContent;
            }
        }

        return (idToToken, addedContent);
    }

    private static Dictionary<char, byte> BuildByteLevelDecodeTable()
    {
        var printable = new HashSet<int>();
        for (int b = 33; b <= 126; b++) printable.Add(b);
        for (int b = 161; b <= 172; b++) printable.Add(b);
        for (int b = 174; b <= 255; b++) printable.Add(b);

        var dict = new Dictionary<char, byte>(280);
        foreach (int b in printable) dict[(char)b] = (byte)b;
        int extra = 0;
        for (int b = 0; b < 256; b++)
            if (!printable.Contains(b))
                dict[(char)(0x100 + extra++)] = (byte)b;
        return dict;
    }

    private static string DecodeTokens(
        IReadOnlyList<long> tokenIds,
        string?[] idToToken,
        Dictionary<int, string> addedTokens,
        Dictionary<char, byte> byteLevelDecode)
    {
        var bytes = new List<byte>(tokenIds.Count * 4);
        foreach (long id in tokenIds)
        {
            int iid = (int)id;
            if (addedTokens.TryGetValue(iid, out string? special))
            {
                bytes.AddRange(Encoding.UTF8.GetBytes(special));
                continue;
            }
            string? raw = iid >= 0 && iid < idToToken.Length ? idToToken[iid] : null;
            if (raw is null) continue;
            foreach (char ch in raw)
                if (byteLevelDecode.TryGetValue(ch, out byte b))
                    bytes.Add(b);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }
}
