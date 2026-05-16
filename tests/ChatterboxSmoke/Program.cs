// Chatterbox TTS — C# smoke test (port of scripts/chatterbox_export/.../listen_test.py).
//
// Loads the four ONNX graphs produced by export_chatterbox_to_onnx.py
// and runs them end-to-end to produce a WAV. Hardcodes the same text
// token sequence the Python listen test uses (a tokenizer port is a
// separate concern; see chatterbox.scratch.md Stage 1 step 1).
//
// Usage:
//   dotnet run --project tests/ChatterboxSmoke -- \
//       --onnx-dir /tmp/cb_dyn5 \
//       --voice ~/Downloads/VCTK_p303.wav \
//       --out   /tmp/chatterbox_out_cs.wav \
//       --ep    cuda
//
// Output should be acoustically close to the Python /tmp/chatterbox_out.wav
// (modulo ORT-vs-PyTorch CUDA kernel drift, ~1e-5).

using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using Vernacula.Base;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.ChatterboxSmoke;

internal static class Program
{
    // Constants — must match scripts/chatterbox_export/_common.py and the
    // chatterbox upstream values it pulls from. Each entry below cites the
    // Python source-of-truth. If those change, this list silently drifts;
    // a future pass should read the dynamic ones from export-report.json.
    private const int StartSpeechToken = 6561;   // _common.py::START_SPEECH_TOKEN
    private const int StopSpeechToken = 6562;    // _common.py::STOP_SPEECH_TOKEN
    private const int ExaggerationToken = 6563;  // _common.py::EXAGGERATION_TOKEN
    private const int LlmLayers = 30;            // _common.py::LLM_NUM_LAYERS
    private const int LlmKvHeads = 16;           // _common.py::LLM_NUM_KV_HEADS
    private const int LlmHeadDim = 64;           // _common.py::LLM_HEAD_DIM
    private const int LlmHidden = 1024;          // _common.py::LLM_HIDDEN_SIZE
    private const int S3GenSr = 24_000;          // _common.py::S3GEN_SR
    private const int DummyAudioSamples = 312_936;  // _common.py::DUMMY_AUDIO_SAMPLES (13.04 s @ 24 kHz)
    private const int MelBins = 80;              // chatterbox.s3gen.flow.output_size
    private const int PromptLen = 500;           // speaker_features.shape[1], fixed by the speech_encoder export contract
    private const int CfmSteps = 10;             // flow.decoder n_timesteps; chatterbox tunes for this value
    private const float CfgRate = 0.7f;          // chatterbox.s3gen.flow.decoder.inference_cfg_rate

    // The Ezreal-and-Jinx sentence, pre-tokenized via chatterbox's EnTokenizer.
    // Wrapping: [EXAGGERATION_TOKEN, ...text..., START_SPEECH_TOKEN, START_SPEECH_TOKEN].
    private static readonly long[] InputIds =
    [
        ExaggerationToken,
        255, 281, 39, 46, 56, 2, 53, 2, 286, 41, 37, 2, 136, 122,
        49, 2, 152, 2, 103, 2, 277, 21, 101, 7, 2, 301, 55, 34, 28, 7,
        2, 53, 2, 296, 18, 18, 115, 2, 51, 2, 33, 245, 2, 17, 190, 2,
        42, 2, 50, 18, 125, 4, 32, 2, 290, 169, 142, 2, 41, 2, 43, 2,
        18, 29, 91, 2, 25, 186, 8, 20, 14, 80, 2, 29, 86, 213, 216, 9,
        0, StartSpeechToken, StartSpeechToken,
    ];

    private const float Exaggeration = 0.5f;     // listen_test.py::EXAGGERATION (default-ish)
    private const float RepetitionPenalty = 1.2f;  // listen_test.py LM-loop rep-penalty divisor
    private const int MaxLmSteps = 256;          // listen_test.py LM-loop max_new_tokens

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

        // Expand ~ to $HOME.
        voicePath = ExpandHome(voicePath);
        outPath = ExpandHome(outPath);

        var totalSw = Stopwatch.StartNew();
        var sw = new Stopwatch();
        var epEnum = ep == "cuda" ? ExecutionProvider.Auto : ExecutionProvider.Cpu;

        // ── Load the four sessions ────────────────────────────────────────
        InferenceSession LoadOne(string name)
        {
            sw.Restart();
            var s = OrtSessionBuilder.CreateCachedSession(
                Path.Combine(onnxDir, name), epEnum, out var hit);
            sw.Stop();
            var sz = new FileInfo(Path.Combine(onnxDir, name)).Length;
            Console.WriteLine($"  {name}: {sw.ElapsedMilliseconds} ms  cache={Hit(hit)}  src={sz / 1e6:F0} MB");
            return s;
        }
        // Detect cond decoder layout. Preference order:
        //   1. MERGED  — conditional_decoder_loop.onnx (Phase 2: single graph
        //                with embedded ONNX Loop for the CFM solve)
        //   2. SPLIT   — flow_encoder + cfm_estimator + mel2wav (Phase 1)
        //   3. MONOLITHIC — conditional_decoder.onnx (pre-perf-work)
        bool mergedAvail = File.Exists(Path.Combine(onnxDir, "conditional_decoder_loop.onnx"));
        bool splitAvail = File.Exists(Path.Combine(onnxDir, "flow_encoder.onnx"))
                       && File.Exists(Path.Combine(onnxDir, "cfm_estimator.onnx"))
                       && File.Exists(Path.Combine(onnxDir, "mel2wav.onnx"));
        string mode = mergedAvail ? "MERGED" : (splitAvail ? "SPLIT" : "MONOLITHIC");
        Console.WriteLine($"Cond decoder mode: {mode}");

        var totalLoadSw = Stopwatch.StartNew();
        var enc = LoadOne("speech_encoder.onnx");
        var emb = LoadOne("embed_tokens.onnx");
        var lm  = LoadOne("language_model.onnx");
        InferenceSession? dec = null, flowEnc = null, cfmEst = null, m2w = null, merged = null;
        if (mergedAvail)
        {
            merged = LoadOne("conditional_decoder_loop.onnx");
        }
        else if (splitAvail)
        {
            flowEnc = LoadOne("flow_encoder.onnx");
            cfmEst  = LoadOne("cfm_estimator.onnx");
            m2w     = LoadOne("mel2wav.onnx");
        }
        else
        {
            dec = LoadOne("conditional_decoder.onnx");
        }
        totalLoadSw.Stop();
        int sessionCount = 3 + (mergedAvail ? 1 : (splitAvail ? 3 : 1));
        Console.WriteLine($"Loaded {sessionCount} sessions in {totalLoadSw.ElapsedMilliseconds} ms total  (ep={ep})");
        using var _enc = enc; using var _emb = emb; using var _lm = lm;
        using var _dec = dec; using var _flowEnc = flowEnc; using var _cfmEst = cfmEst;
        using var _m2w = m2w; using var _merged = merged;

        // ── Voice prompt → 24 kHz mono, padded/cropped to 312_936 ─────────
        sw.Restart();
        var audio = LoadVoicePrompt(voicePath);
        sw.Stop();
        Console.WriteLine($"Loaded voice {voicePath}: {audio.Length} samples ({audio.Length / (float)S3GenSr:F2}s)  [{sw.ElapsedMilliseconds} ms]");

        // ── speech_encoder.onnx ──────────────────────────────────────────
        sw.Restart();
        var audioT = new DenseTensor<float>(audio, [1, audio.Length]);
        using var encOut = enc.Run([NamedOnnxValue.CreateFromTensor("audio_values", audioT)]);
        var encList = encOut.ToList();
        var condEmb = encList[0].AsTensor<float>();          // [1, S_cond, 1024]
        var audioTokens = encList[1].AsTensor<long>();       // [1, T_audio]
        var spkEmb = encList[2].AsTensor<float>();           // [1, 192]
        var spkFeat = encList[3].AsTensor<float>();          // [1, 500, 80]
        sw.Stop();
        Console.WriteLine($"speech_encoder: cond_emb={ShapeStr(condEmb.Dimensions)}  audio_tokens={ShapeStr(audioTokens.Dimensions)}  [{sw.ElapsedMilliseconds} ms]");

        // ── Compute LM input_ids — from --text via the tokenizer if given,
        //    else fall back to the hardcoded Ezreal sentence (backward-compat). ──
        long[] inputIds;
        if (text is not null)
        {
            var tokenizerPath = tokenizerJson ?? LocateCachedTokenizerJson();
            if (tokenizerPath is null)
            {
                Console.Error.WriteLine(
                    "--text given but no tokenizer.json found. Pass --tokenizer-json <path>, "
                    + "or download via `huggingface-cli download ResembleAI/chatterbox tokenizer.json`.");
                return 2;
            }
            var tokenizer = new EnTokenizer(tokenizerPath);
            inputIds = tokenizer.WrapForLm(text);
            Console.WriteLine($"Tokenized \"{text[..Math.Min(text.Length, 50)]}{(text.Length > 50 ? "..." : "")}\" "
                + $"→ {inputIds.Length} tokens");
        }
        else
        {
            inputIds = InputIds;
            Console.WriteLine($"Using hardcoded Ezreal sentence ({inputIds.Length} tokens). Pass --text \"...\" for arbitrary input.");
        }

        // ── embed_tokens.onnx — text prompt to embeddings ─────────────────
        sw.Restart();
        int sText = inputIds.Length;
        var positionIds = new long[sText];
        for (int i = 0; i < sText; i++)
            positionIds[i] = inputIds[i] >= StartSpeechToken ? 0 : i - 1;
        var inputIdsT = new DenseTensor<long>(inputIds.ToArray(), [1, sText]);
        var posIdsT = new DenseTensor<long>(positionIds, [1, sText]);
        var exagT = new DenseTensor<float>(new float[] { Exaggeration }, [1]);
        using var embOut = emb.Run([
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsT),
            NamedOnnxValue.CreateFromTensor("position_ids", posIdsT),
            NamedOnnxValue.CreateFromTensor("exaggeration", exagT),
        ]);
        var textEmbTensor = embOut.First().AsTensor<float>();  // [1, sText, 1024]
        sw.Stop();
        Console.WriteLine($"embed_tokens: text_emb={ShapeStr(textEmbTensor.Dimensions)}  [{sw.ElapsedMilliseconds} ms]");

        // Concat cond_emb + text_emb along sequence dim → inputs_embeds.
        int sCond = condEmb.Dimensions[1];
        int sTotal = sCond + sText;
        var inputsEmbeds = ConcatSeq(condEmb, textEmbTensor, LlmHidden);  // [1, sTotal, 1024]

        // ── LM autoregressive loop ────────────────────────────────────────
        var lmSw = Stopwatch.StartNew();
        var (generateTokens, actualSteps) = useIoBinding
            ? RunLmLoopIoBinding(lm, emb, inputsEmbeds, sTotal, diagDir)
            : RunLmLoopBasic(lm, emb, inputsEmbeds, sTotal, diagDir);
        lmSw.Stop();
        Console.WriteLine($"LM ({(useIoBinding ? "io-binding" : "basic")}): {actualSteps} steps, generated {generateTokens.Count - 1} tokens "
            + $"[{lmSw.ElapsedMilliseconds} ms, {lmSw.ElapsedMilliseconds / (double)actualSteps:F1} ms/step]");

        if (diagDir is not null)
        {
            // Full token sequence for divergence-hunting (line up against
            // Python's listen_test sequence to find the first differing step).
            File.WriteAllBytes(Path.Combine(diagDir, "cs_tokens.bin"),
                MemoryMarshal.AsBytes<long>(generateTokens.ToArray()).ToArray());
            Console.WriteLine($"[diag] wrote {diagDir}/cs_tokens.bin ({generateTokens.Count} tokens)");
        }

        // ── Build speech_tokens: [audio_tokens, generated[1:-1]] ──────────
        // Strip START_SPEECH at front; strip STOP at back if present.
        int genStart = 1;
        int genEnd = generateTokens[^1] == StopSpeechToken ? generateTokens.Count - 1 : generateTokens.Count;
        int genCount = genEnd - genStart;
        var audioTokArr = audioTokens.ToArray();
        var speechTokens = new long[audioTokArr.Length + genCount];
        Array.Copy(audioTokArr, speechTokens, audioTokArr.Length);
        for (int i = 0; i < genCount; i++)
            speechTokens[audioTokArr.Length + i] = generateTokens[genStart + i];
        Console.WriteLine($"speech_tokens: shape=(1, {speechTokens.Length})  ({audioTokArr.Length} from voice + {genCount} from LM)");

        // ── Cond decoder: merged Loop graph (Phase 2), 3-graph split (Phase 1),
        //    or monolithic (pre-perf-work). Picked by the detection above.
        sw.Restart();
        var speechTokT = new DenseTensor<long>(speechTokens, [1, speechTokens.Length]);
        var spkEmbT = new DenseTensor<float>(spkEmb.ToArray(), spkEmb.Dimensions.ToArray());
        var spkFeatT = new DenseTensor<float>(spkFeat.ToArray(), spkFeat.Dimensions.ToArray());

        float[] samples;
        int nSamples;
        if (mergedAvail)
        {
            // Single Run on conditional_decoder_loop.onnx — same I/O contract
            // as the monolithic conditional_decoder.onnx, but internally drives
            // the CFM solve via an ONNX Loop body. No C# loop orchestration.
            using var loopOut = merged!.Run([
                NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokT),
                NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkEmbT),
                NamedOnnxValue.CreateFromTensor("speaker_features", spkFeatT),
            ]);
            var wavTensor = loopOut.First().AsTensor<float>();
            sw.Stop();
            nSamples = wavTensor.Dimensions[1];
            Console.WriteLine($"cond_decoder (merged-Loop): waveform=(1, {nSamples}) → {nSamples / (float)S3GenSr:F2}s  [{sw.ElapsedMilliseconds} ms]");
            samples = wavTensor.ToArray();
        }
        else if (splitAvail)
        {
            samples = RunSplitCondDecoder(flowEnc!, cfmEst!, m2w!, speechTokT, spkEmbT, spkFeatT, sw);
            nSamples = samples.Length;
        }
        else
        {
            using var decOut = dec!.Run([
                NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokT),
                NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkEmbT),
                NamedOnnxValue.CreateFromTensor("speaker_features", spkFeatT),
            ]);
            var wavTensor = decOut.First().AsTensor<float>();
            sw.Stop();
            nSamples = wavTensor.Dimensions[1];
            Console.WriteLine($"cond_decoder (monolithic): waveform=(1, {nSamples}) → {nSamples / (float)S3GenSr:F2}s  [{sw.ElapsedMilliseconds} ms]");
            samples = wavTensor.ToArray();
        }

        // ── Write WAV ─────────────────────────────────────────────────────
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(S3GenSr, 1);
        using (var writer = new WaveFileWriter(outPath, fmt))
        {
            writer.WriteSamples(samples, 0, samples.Length);
        }
        totalSw.Stop();
        Console.WriteLine($"Wrote {outPath}  [total {totalSw.ElapsedMilliseconds / 1000.0:F1}s]");
        return 0;
    }

    /// <summary>
    /// LM autoregressive loop, basic Run path. Each step builds 60
    /// NamedOnnxValue inputs (30 layers × {key,value}) from CPU-resident
    /// past_kv arrays, runs the session, copies all 60 KV outputs back
    /// to CPU via .AsTensor&lt;float&gt;().ToArray(). On a 3090, ~32 ms/step
    /// dominated by the host roundtrip. Kept for A/B comparison vs the
    /// IoBinding version below.
    /// </summary>
    private static (List<long> tokens, int steps) RunLmLoopBasic(
        InferenceSession lm, InferenceSession emb,
        float[] inputsEmbeds, int sTotal, string? diagDir)
    {
        var attentionMask = new long[sTotal];
        Array.Fill(attentionMask, 1);
        var pastKv = new float[LlmLayers * 2][];
        var pastKvShape = new int[] { 1, LlmKvHeads, 0, LlmHeadDim };
        for (int k = 0; k < pastKv.Length; k++) pastKv[k] = [];

        var generateTokens = new List<long> { StartSpeechToken };
        int actualSteps = 0;
        for (int step = 0; step < MaxLmSteps; step++)
        {
            var inputs = new List<NamedOnnxValue>(2 + 2 * LlmLayers);
            int seqIn = step == 0 ? sTotal : 1;
            inputs.Add(NamedOnnxValue.CreateFromTensor("inputs_embeds",
                new DenseTensor<float>(inputsEmbeds, [1, seqIn, LlmHidden])));
            inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask",
                new DenseTensor<long>(attentionMask, [1, attentionMask.Length])));
            int tPast = pastKvShape[2];
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.key",
                    new DenseTensor<float>(pastKv[2 * layer], pastKvShape)));
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.value",
                    new DenseTensor<float>(pastKv[2 * layer + 1], pastKvShape)));
            }

            using var lmOut = lm.Run(inputs);
            var outList = lmOut.ToList();
            var logits = outList[0].AsTensor<float>();
            int vocab = logits.Dimensions[2];
            var lastLogits = new float[vocab];
            int logitsOffset = (seqIn - 1) * vocab;
            var logitsBuf = logits.ToArray();
            Array.Copy(logitsBuf, logitsOffset, lastLogits, 0, vocab);

            MaybeDumpStep(diagDir, step, lastLogits, inputsEmbeds, attentionMask,
                          pastKv, pastKvShape);

            foreach (var t in generateTokens)
                if (lastLogits[t] > 0) lastLogits[t] /= RepetitionPenalty;
            long nextToken = Argmax(lastLogits);
            generateTokens.Add(nextToken);
            if (nextToken == StopSpeechToken) { actualSteps = step + 1; break; }

            int tPresent = tPast + seqIn;
            pastKvShape = [1, LlmKvHeads, tPresent, LlmHeadDim];
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                pastKv[2 * layer]     = outList[1 + 2 * layer].AsTensor<float>().ToArray();
                pastKv[2 * layer + 1] = outList[2 + 2 * layer].AsTensor<float>().ToArray();
            }

            var nextIdT = new DenseTensor<long>(new long[] { nextToken }, [1, 1]);
            var nextPosT = new DenseTensor<long>(new long[] { (long)(step + 1) }, [1, 1]);
            var nextExagT = new DenseTensor<float>(new float[] { Exaggeration }, [1]);
            using var nextEmbOut = emb.Run([
                NamedOnnxValue.CreateFromTensor("input_ids", nextIdT),
                NamedOnnxValue.CreateFromTensor("position_ids", nextPosT),
                NamedOnnxValue.CreateFromTensor("exaggeration", nextExagT),
            ]);
            inputsEmbeds = nextEmbOut.First().AsTensor<float>().ToArray();

            var grown = new long[attentionMask.Length + 1];
            Array.Copy(attentionMask, grown, attentionMask.Length);
            grown[^1] = 1;
            attentionMask = grown;
            actualSteps = step + 1;
        }
        return (generateTokens, actualSteps);
    }

    /// <summary>
    /// LM autoregressive loop, IoBinding path. KV cache stays GPU-resident
    /// across steps — each step's `present.{layer}.{key,value}` outputs are
    /// bound to CUDA memory, then the NEXT step's `past_key_values.{layer}.*`
    /// inputs reference those same OrtValues directly. Logits are bound to
    /// CPU memory so we can do argmax + rep-penalty on host. Embeds stay on
    /// host (small, cheap to copy per step). Pattern is adapted from
    /// WhisperTurbo.cs::TranscribeBatch's step loop.
    /// </summary>
    private static (List<long> tokens, int steps) RunLmLoopIoBinding(
        InferenceSession lm, InferenceSession emb,
        float[] inputsEmbeds, int sTotal, string? diagDir)
    {
        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cpuMemInfo  = new OrtMemoryInfo("Cpu",  OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts     = new RunOptions();

        var attentionMask = new long[sTotal];
        Array.Fill(attentionMask, 1);
        var generateTokens = new List<long> { StartSpeechToken };

        // Prefill (step 0): empty past_kv. The empty OrtValues live for the
        // lifetime of the prefill binding only (we replace them with prefill
        // outputs on step 1).
        var emptyPastValues = new List<OrtValue>(LlmLayers * 2);
        for (int i = 0; i < LlmLayers * 2; i++)
            emptyPastValues.Add(OrtValue.CreateTensorValueFromMemory(
                Array.Empty<float>(), [1L, LlmKvHeads, 0L, LlmHeadDim]));

        IDisposableReadOnlyCollection<OrtValue> prefillOutputs;
        {
            using var prefillEmbeds = OrtValue.CreateTensorValueFromMemory(
                inputsEmbeds, [1L, sTotal, LlmHidden]);
            using var prefillMask = OrtValue.CreateTensorValueFromMemory(
                attentionMask, [1L, sTotal]);
            using var binding = lm.CreateIoBinding();
            binding.BindInput("inputs_embeds", prefillEmbeds);
            binding.BindInput("attention_mask", prefillMask);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindInput($"past_key_values.{layer}.key",   emptyPastValues[2 * layer]);
                binding.BindInput($"past_key_values.{layer}.value", emptyPastValues[2 * layer + 1]);
            }
            binding.BindOutputToDevice("logits", cpuMemInfo);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindOutputToDevice($"present.{layer}.key",   cudaMemInfo);
                binding.BindOutputToDevice($"present.{layer}.value", cudaMemInfo);
            }
            lm.RunWithBinding(runOpts, binding);
            prefillOutputs = binding.GetOutputValues();
        }
        foreach (var v in emptyPastValues) v.Dispose();

        // First argmax: prefill logits has shape [1, sTotal, vocab]; we want the [-1, :] row.
        // Read vocab from tensor metadata directly (robust against future batch>1
        // changes that would silently miscompute as `batch_size * vocab` if derived
        // from span length).
        int vocab = (int)prefillOutputs[0].GetTensorTypeAndShape().Shape[2];
        var prefillLogitsSpan = prefillOutputs[0].GetTensorDataAsSpan<float>();
        var lastLogits = prefillLogitsSpan.Slice((sTotal - 1) * vocab, vocab).ToArray();
        MaybeDumpStep(diagDir, 0, lastLogits, inputsEmbeds, attentionMask, null, null);
        foreach (var t in generateTokens)
            if (lastLogits[t] > 0) lastLogits[t] /= RepetitionPenalty;
        long nextToken = Argmax(lastLogits);
        generateTokens.Add(nextToken);
        if (nextToken == StopSpeechToken)
        {
            prefillOutputs.Dispose();
            return (generateTokens, 1);
        }

        // Embed the new token (CPU output).
        inputsEmbeds = EmbedOne(emb, nextToken, 1);
        attentionMask = Grow(attentionMask, 1);

        // Step loop. prevStep[1..60] = present_kv (CUDA), prevStep[0] = logits (CPU).
        IDisposableReadOnlyCollection<OrtValue> prevStep = prefillOutputs;
        int actualSteps = 1;
        for (int step = 1; step < MaxLmSteps; step++)
        {
            using var stepEmbeds = OrtValue.CreateTensorValueFromMemory(
                inputsEmbeds, [1L, 1L, LlmHidden]);
            using var stepMask = OrtValue.CreateTensorValueFromMemory(
                attentionMask, [1L, attentionMask.Length]);
            using var binding = lm.CreateIoBinding();
            binding.BindInput("inputs_embeds", stepEmbeds);
            binding.BindInput("attention_mask", stepMask);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                // Chain prev step's CUDA-resident KV directly — no host roundtrip.
                binding.BindInput($"past_key_values.{layer}.key",   prevStep[1 + 2 * layer]);
                binding.BindInput($"past_key_values.{layer}.value", prevStep[1 + 2 * layer + 1]);
            }
            binding.BindOutputToDevice("logits", cpuMemInfo);
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                binding.BindOutputToDevice($"present.{layer}.key",   cudaMemInfo);
                binding.BindOutputToDevice($"present.{layer}.value", cudaMemInfo);
            }
            lm.RunWithBinding(runOpts, binding);
            var curStep = binding.GetOutputValues();

            // Safe to dispose prevStep here even though `binding` (which holds
            // BindInput references to prevStep's OrtValues) is still alive:
            // RunWithBinding has completed and ORT only reads the input
            // OrtValues during the Run call, not afterward. Same pattern as
            // WhisperTurbo.cs::TranscribeBatch — established in this codebase.
            prevStep.Dispose();
            prevStep = curStep;

            // Argmax on step logits ([1, 1, vocab]).
            var stepLogitsSpan = curStep[0].GetTensorDataAsSpan<float>();
            lastLogits = stepLogitsSpan.Slice(0, vocab).ToArray();
            MaybeDumpStep(diagDir, step, lastLogits, inputsEmbeds, attentionMask, null, null);
            foreach (var t in generateTokens)
                if (lastLogits[t] > 0) lastLogits[t] /= RepetitionPenalty;
            nextToken = Argmax(lastLogits);
            generateTokens.Add(nextToken);
            if (nextToken == StopSpeechToken) { actualSteps = step + 1; break; }

            inputsEmbeds = EmbedOne(emb, nextToken, step + 1);
            attentionMask = Grow(attentionMask, 1);
            actualSteps = step + 1;
        }
        prevStep.Dispose();
        return (generateTokens, actualSteps);
    }

    /// <summary>Run embed_tokens.onnx for a single token at the given position. CPU output.</summary>
    private static float[] EmbedOne(InferenceSession emb, long token, int position)
    {
        var idT  = new DenseTensor<long>(new long[] { token }, [1, 1]);
        var posT = new DenseTensor<long>(new long[] { (long)position }, [1, 1]);
        var exT  = new DenseTensor<float>(new float[] { Exaggeration }, [1]);
        using var o = emb.Run([
            NamedOnnxValue.CreateFromTensor("input_ids", idT),
            NamedOnnxValue.CreateFromTensor("position_ids", posT),
            NamedOnnxValue.CreateFromTensor("exaggeration", exT),
        ]);
        return o.First().AsTensor<float>().ToArray();
    }

    /// <summary>Grow attention_mask by `n` trailing 1s.</summary>
    private static long[] Grow(long[] mask, int n)
    {
        var grown = new long[mask.Length + n];
        Array.Copy(mask, grown, mask.Length);
        for (int i = mask.Length; i < grown.Length; i++) grown[i] = 1;
        return grown;
    }

    /// <summary>
    /// Diagnostic dump for LM step 0/1 — shared by both basic and IoBinding
    /// LM-loop paths. Only fires when diagDir is non-null. The basic path passes
    /// past_kv (CPU arrays); the IoBinding path passes null since KV lives on
    /// CUDA and host extraction would defeat the purpose.
    /// </summary>
    private static void MaybeDumpStep(string? diagDir, int step,
        float[] lastLogits, float[] inputsEmbeds, long[] attentionMask,
        float[][]? pastKv, int[]? pastKvShape)
    {
        if (diagDir is null || step > 1) return;
        Console.WriteLine($"[step{step}] logits[-1, :10] (pre-penalty): "
            + string.Join(", ", lastLogits.Take(10).Select(x => x.ToString("F6"))));
        Console.WriteLine($"[step{step}] logits sum: {lastLogits.Sum():F4}, argmax(pre-penalty): {Argmax(lastLogits)}");
        File.WriteAllBytes(Path.Combine(diagDir, $"cs_step{step}_logits.bin"),
            MemoryMarshal.AsBytes<float>(lastLogits).ToArray());
        File.WriteAllBytes(Path.Combine(diagDir, $"cs_step{step}_inputs_embeds.bin"),
            MemoryMarshal.AsBytes<float>(inputsEmbeds).ToArray());
        File.WriteAllBytes(Path.Combine(diagDir, $"cs_step{step}_attention_mask.bin"),
            MemoryMarshal.AsBytes<long>(attentionMask).ToArray());
        if (step == 1 && pastKv is not null && pastKvShape is not null)
        {
            File.WriteAllBytes(Path.Combine(diagDir, "cs_step1_past_kv_l0_key.bin"),
                MemoryMarshal.AsBytes<float>(pastKv[0]).ToArray());
            File.WriteAllBytes(Path.Combine(diagDir, "cs_step1_past_kv_l0_value.bin"),
                MemoryMarshal.AsBytes<float>(pastKv[1]).ToArray());
            Console.WriteLine($"[step1] past_kv shape={ShapeStr(pastKvShape)}  "
                + $"l0_key sum={pastKv[0].Sum():F4}  l0_value sum={pastKv[1].Sum():F4}");
        }
    }

    /// <summary>
    /// Orchestrate the 3-graph split cond decoder: run flow_encoder, then
    /// the CFM solve loop (10 cosine-scheduled Euler steps with CFG, one
    /// cfm_estimator.onnx call per step), trim the prompt prefix from the
    /// final mel, then run mel2wav. Mirrors parity_split_pipeline in
    /// scripts/chatterbox_export/test_chatterbox_parity.py.
    /// </summary>
    private static float[] RunSplitCondDecoder(
        InferenceSession flowEnc, InferenceSession cfmEst, InferenceSession m2w,
        DenseTensor<long> speechTokT, DenseTensor<float> spkEmbT, DenseTensor<float> spkFeatT,
        Stopwatch sw)
    {
        // 1) flow_encoder: speech_tokens → mu, mel_mask, embedding, cond, z
        sw.Restart();
        using var encOut = flowEnc.Run([
            NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokT),
            NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkEmbT),
            NamedOnnxValue.CreateFromTensor("speaker_features", spkFeatT),
        ]);
        var encList = encOut.ToList();
        var muT       = encList[0].AsTensor<float>();   // [1, 80, T_mel]
        var maskT     = encList[1].AsTensor<float>();   // [1, 1,  T_mel]
        var embedT    = encList[2].AsTensor<float>();   // [1, 80]
        var condT     = encList[3].AsTensor<float>();   // [1, 80, T_mel]
        var zT        = encList[4].AsTensor<float>();   // [1, 80, T_mel]
        int tMel = muT.Dimensions[2];
        sw.Stop();
        Console.WriteLine($"flow_encoder: mu={ShapeStr(muT.Dimensions)}  T_mel={tMel}  [{sw.ElapsedMilliseconds} ms]");

        // Materialize to flat arrays once; the CFM loop builds CFG-doubled
        // copies per step (cat-along-batch) without re-reading the originals.
        float[] mu      = muT.ToArray();
        float[] mask    = maskT.ToArray();
        float[] embed   = embedT.ToArray();
        float[] cond    = condT.ToArray();
        float[] x       = zT.ToArray();    // current CFM state, evolves each step

        // 2) CFM solve loop. Cosine-scheduled t_span (matches upstream's
        //    t_scheduler='cosine'). 10 Euler steps. Per step: CFG-double
        //    the inputs along batch, run cfm_estimator, split + combine
        //    with CFG rate, take an Euler step.
        sw.Restart();
        var tSpan = new float[CfmSteps + 1];
        for (int i = 0; i <= CfmSteps; i++)
        {
            float linear = i / (float)CfmSteps;
            tSpan[i] = 1.0f - MathF.Cos(linear * 0.5f * MathF.PI);
        }
        float t = tSpan[0];
        float dt = tSpan[1] - tSpan[0];
        int muLen = mu.Length;       // = T_mel * 80
        int maskLen = mask.Length;   // = T_mel
        int embedLen = embed.Length; // = 80
        for (int step = 1; step <= CfmSteps; step++)
        {
            // CFG-double along batch dim 0. Each "_in" is shape [2, ...].
            var xIn = CatBatch(x, x);
            var maskIn = CatBatch(mask, mask);
            var muIn = CatBatch(mu, new float[muLen]);             // row 0 = mu, row 1 = zeros
            var condIn = CatBatch(cond, new float[cond.Length]);
            var spksIn = CatBatch(embed, new float[embedLen]);
            var tIn = new float[] { t, t };

            using var estOut = cfmEst.Run([
                NamedOnnxValue.CreateFromTensor("x_in",     new DenseTensor<float>(xIn, [2, MelBins, tMel])),
                NamedOnnxValue.CreateFromTensor("mask_in",  new DenseTensor<float>(maskIn, [2, 1, tMel])),
                NamedOnnxValue.CreateFromTensor("mu_in",    new DenseTensor<float>(muIn, [2, MelBins, tMel])),
                NamedOnnxValue.CreateFromTensor("t_in",     new DenseTensor<float>(tIn, [2])),
                NamedOnnxValue.CreateFromTensor("spks_in",  new DenseTensor<float>(spksIn, [2, MelBins])),
                NamedOnnxValue.CreateFromTensor("cond_in",  new DenseTensor<float>(condIn, [2, MelBins, tMel])),
            ]);
            var dphi = estOut.First().AsTensor<float>().ToArray();  // [2, 80, T_mel]
            int half = dphi.Length / 2;
            // CFG combine: (1+cfg) * cond_part - cfg * uncond_part; Euler step.
            float a = 1.0f + CfgRate;
            for (int i = 0; i < half; i++)
            {
                float c = dphi[i];
                float u = dphi[half + i];
                x[i] = x[i] + dt * (a * c - CfgRate * u);
            }
            t = tSpan[step];
            if (step < CfmSteps) dt = tSpan[step + 1] - t;
        }
        sw.Stop();
        Console.WriteLine($"cfm_solve_loop: {CfmSteps} steps  [{sw.ElapsedMilliseconds} ms, {sw.ElapsedMilliseconds / (double)CfmSteps:F1} ms/step]");

        // 3) Trim mel: drop the prompt prefix (the first PromptLen mel frames).
        //    x is row-major [1, 80, T_mel]; we want [1, 80, T_mel - PromptLen]
        //    keeping the tail along the last dim.
        int tTrim = tMel - PromptLen;
        var feat = new float[MelBins * tTrim];
        for (int c = 0; c < MelBins; c++)
        {
            Array.Copy(x, c * tMel + PromptLen, feat, c * tTrim, tTrim);
        }

        // 4) mel2wav: trimmed mel → waveform (trim_fade is baked into the graph).
        sw.Restart();
        using var m2wOut = m2w.Run([
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(feat, [1, MelBins, tTrim])),
        ]);
        var wavTensor = m2wOut.First().AsTensor<float>();
        sw.Stop();
        int n = wavTensor.Dimensions[1];
        Console.WriteLine($"mel2wav: waveform=(1, {n}) → {n / (float)S3GenSr:F2}s  [{sw.ElapsedMilliseconds} ms]");
        return wavTensor.ToArray();
    }

    /// <summary>
    /// Concatenate two flat tensors of identical size along (implicit) dim 0.
    /// Used to build the CFG-pair tensors for cfm_estimator: row 0 = real,
    /// row 1 = the other half (zeros for mu/cond/spks; same as row 0 for x/mask).
    /// </summary>
    private static float[] CatBatch(float[] a, float[] b)
    {
        var r = new float[a.Length + b.Length];
        Array.Copy(a, 0, r, 0, a.Length);
        Array.Copy(b, 0, r, a.Length, b.Length);
        return r;
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// <summary>
    /// Load a WAV at any sample rate / channel count and return a 24 kHz mono
    /// float32 array, padded with zeros or cropped to DummyAudioSamples (the
    /// trace-time canonical length the speech_encoder was exported with).
    /// </summary>
    private static float[] LoadVoicePrompt(string path)
    {
        var (raw, sr, channels) = AudioUtils.ReadAudio(path);
        float[] mono = AudioUtils.DownmixToMono(raw, channels);

        float[] at24k;
        if (sr == S3GenSr)
        {
            at24k = ReferenceEquals(mono, raw) ? (float[])mono.Clone() : mono;
        }
        else
        {
            var srcFmt = WaveFormat.CreateIeeeFloatWaveFormat(sr, 1);
            var provider = new FloatArraySampleProvider(mono, srcFmt);
            var resampler = new WdlResamplingSampleProvider(provider, S3GenSr);
            var outList = new List<float>((int)((long)mono.Length * S3GenSr / sr + 1024));
            var buf = new float[8192];
            int n;
            while ((n = resampler.Read(buf, 0, buf.Length)) > 0)
                for (int i = 0; i < n; i++) outList.Add(buf[i]);
            at24k = outList.ToArray();
        }

        if (at24k.Length >= DummyAudioSamples)
            return at24k.AsSpan(0, DummyAudioSamples).ToArray();

        var padded = new float[DummyAudioSamples];
        Array.Copy(at24k, padded, at24k.Length);
        return padded;
    }

    /// <summary>
    /// NAudio float-array source mirroring Vernacula.Base's internal helper.
    /// Lives here too because that class is internal to Vernacula.Base.
    /// </summary>
    private sealed class FloatArraySampleProvider : ISampleProvider
    {
        private readonly float[] _data;
        private int _pos;
        public FloatArraySampleProvider(float[] data, WaveFormat fmt) { _data = data; WaveFormat = fmt; }
        public WaveFormat WaveFormat { get; }
        public int Read(float[] buffer, int offset, int count)
        {
            int remain = _data.Length - _pos;
            int take = Math.Min(remain, count);
            if (take <= 0) return 0;
            Array.Copy(_data, _pos, buffer, offset, take);
            _pos += take;
            return take;
        }
    }

    /// <summary>
    /// Concatenate two [1, S_i, D] tensors along the sequence dim, returning
    /// a flat float[] of length (S_a + S_b) * D, row-major.
    /// </summary>
    private static float[] ConcatSeq(Tensor<float> a, Tensor<float> b, int hiddenDim)
    {
        int sa = a.Dimensions[1];
        int sb = b.Dimensions[1];
        var aArr = a.ToArray();
        var bArr = b.ToArray();
        var outArr = new float[(sa + sb) * hiddenDim];
        Array.Copy(aArr, 0, outArr, 0, sa * hiddenDim);
        Array.Copy(bArr, 0, outArr, sa * hiddenDim, sb * hiddenDim);
        return outArr;
    }

    private static long Argmax(float[] arr)
    {
        int best = 0;
        float bestVal = arr[0];
        for (int i = 1; i < arr.Length; i++)
        {
            if (arr[i] > bestVal) { bestVal = arr[i]; best = i; }
        }
        return best;
    }

    private static string ShapeStr(ReadOnlySpan<int> dims)
        => "(" + string.Join(", ", dims.ToArray()) + ")";

    private static string ExpandHome(string path)
        => path.StartsWith("~/") ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), path[2..]) : path;

    private static string Hit(bool b) => b ? "HIT" : "miss";

    /// <summary>
    /// Best-effort lookup of the chatterbox tokenizer.json in the standard HF
    /// hub cache. Returns null if not present; the user can override with
    /// --tokenizer-json. Matches the layout
    /// `~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/tokenizer.json`.
    /// </summary>
    private static string? LocateCachedTokenizerJson()
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var snapshotsDir = Path.Combine(home,
            ".cache", "huggingface", "hub", "models--ResembleAI--chatterbox", "snapshots");
        if (!Directory.Exists(snapshotsDir)) return null;
        foreach (var snap in Directory.EnumerateDirectories(snapshotsDir))
        {
            var p = Path.Combine(snap, "tokenizer.json");
            if (File.Exists(p)) return p;
        }
        return null;
    }
}
