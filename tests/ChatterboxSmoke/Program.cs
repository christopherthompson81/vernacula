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
    // Constants — must match scripts/chatterbox_export/_common.py.
    private const int StartSpeechToken = 6561;
    private const int StopSpeechToken = 6562;
    private const int ExaggerationToken = 6563;
    private const int LlmLayers = 30;
    private const int LlmKvHeads = 16;
    private const int LlmHeadDim = 64;
    private const int LlmHidden = 1024;
    private const int S3GenSr = 24_000;
    private const int DummyAudioSamples = 312_936;  // 13.04 s @ 24 kHz

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

    private const float Exaggeration = 0.5f;
    private const float RepetitionPenalty = 1.2f;
    private const int MaxLmSteps = 256;

    private static int Main(string[] args)
    {
        string? onnxDir = null;
        string? voicePath = null;
        string? outPath = "/tmp/chatterbox_out_cs.wav";
        string ep = "cuda";
        bool skipDec = false;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--onnx-dir": onnxDir = args[++i]; break;
                case "--voice":    voicePath = args[++i]; break;
                case "--out":      outPath = args[++i]; break;
                case "--ep":       ep = args[++i].ToLowerInvariant(); break;
                case "--skip-cond-decoder": skipDec = true; break;
                default:
                    Console.Error.WriteLine($"Unknown arg: {args[i]}");
                    return 2;
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
        var totalLoadSw = Stopwatch.StartNew();
        var enc = LoadOne("speech_encoder.onnx");
        var emb = LoadOne("embed_tokens.onnx");
        var lm  = LoadOne("language_model.onnx");
        var dec = LoadOne("conditional_decoder.onnx");
        totalLoadSw.Stop();
        Console.WriteLine($"Loaded 4 sessions in {totalLoadSw.ElapsedMilliseconds} ms total  (ep={ep})");
        using var _enc = enc; using var _emb = emb; using var _lm = lm; using var _dec = dec;

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

        // ── embed_tokens.onnx — text prompt to embeddings ─────────────────
        sw.Restart();
        int sText = InputIds.Length;
        var positionIds = new long[sText];
        for (int i = 0; i < sText; i++)
            positionIds[i] = InputIds[i] >= StartSpeechToken ? 0 : i - 1;
        var inputIdsT = new DenseTensor<long>(InputIds.ToArray(), [1, sText]);
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
        // attention_mask starts at sTotal, grows by 1 per step.
        // past_kv starts empty (T_past=0), grows by sTotal at step 0 then by 1 per step.
        // generate_tokens accumulates the sampled speech tokens.
        var attentionMask = new long[sTotal];
        Array.Fill(attentionMask, 1);
        var pastKv = new float[LlmLayers * 2][];          // pairs of (key, value) per layer
        var pastKvShape = new int[] { 1, LlmKvHeads, 0, LlmHeadDim };
        for (int k = 0; k < pastKv.Length; k++) pastKv[k] = [];

        var generateTokens = new List<long> { StartSpeechToken };
        var lmSw = new Stopwatch();
        lmSw.Start();
        int actualSteps = 0;
        for (int step = 0; step < MaxLmSteps; step++)
        {
            // Build LM input dict for this step.
            var inputs = new List<NamedOnnxValue>(2 + 2 * LlmLayers);
            int seqIn = step == 0 ? sTotal : 1;
            inputs.Add(NamedOnnxValue.CreateFromTensor("inputs_embeds",
                new DenseTensor<float>(inputsEmbeds, [1, seqIn, LlmHidden])));
            inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask",
                new DenseTensor<long>(attentionMask, [1, attentionMask.Length])));
            int tPast = pastKvShape[2];
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{layer}.key",
                    new DenseTensor<float>(pastKv[2 * layer], pastKvShape)));
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{layer}.value",
                    new DenseTensor<float>(pastKv[2 * layer + 1], pastKvShape)));
            }

            using var lmOut = lm.Run(inputs);
            var outList = lmOut.ToList();
            var logits = outList[0].AsTensor<float>();  // [1, seqIn, vocab]

            // Slice logits of the LAST position; apply repetition penalty in-place.
            int vocab = logits.Dimensions[2];
            var lastLogits = new float[vocab];
            int logitsOffset = (seqIn - 1) * vocab;
            // logits is row-major: [1, seqIn, vocab]; copy out the [-1, :] row.
            var logitsBuf = logits.ToArray();
            Array.Copy(logitsBuf, logitsOffset, lastLogits, 0, vocab);

            if (step <= 1)
            {
                // DIAGNOSTIC: dump per-step logits + LM-input snapshot for cross-check
                // vs Python listen_test. Step 0 isolates the prefill; step 1 isolates
                // the KV-refeed cycle. Remove once parity is established.
                Console.WriteLine($"[step{step}] logits[-1, :10] (pre-penalty): "
                    + string.Join(", ", lastLogits.Take(10).Select(x => x.ToString("F6"))));
                Console.WriteLine($"[step{step}] logits sum: {lastLogits.Sum():F4}, argmax(pre-penalty): {Argmax(lastLogits)}");
                File.WriteAllBytes($"/tmp/cs_step{step}_logits.bin", MemoryMarshal.AsBytes<float>(lastLogits).ToArray());
                File.WriteAllBytes($"/tmp/cs_step{step}_inputs_embeds.bin", MemoryMarshal.AsBytes<float>(inputsEmbeds).ToArray());
                File.WriteAllBytes($"/tmp/cs_step{step}_attention_mask.bin", MemoryMarshal.AsBytes<long>(attentionMask).ToArray());
                if (step == 1)
                {
                    // Dump the first KV (key+value) layer so we can verify the
                    // KV-refeed roundtrip didn't corrupt anything.
                    File.WriteAllBytes("/tmp/cs_step1_past_kv_l0_key.bin",
                        MemoryMarshal.AsBytes<float>(pastKv[0]).ToArray());
                    File.WriteAllBytes("/tmp/cs_step1_past_kv_l0_value.bin",
                        MemoryMarshal.AsBytes<float>(pastKv[1]).ToArray());
                    Console.WriteLine($"[step1] past_kv shape={ShapeStr(pastKvShape)}  "
                        + $"l0_key sum={pastKv[0].Sum():F4}  l0_value sum={pastKv[1].Sum():F4}");
                }
            }

            foreach (var t in generateTokens)
            {
                if (lastLogits[t] > 0) lastLogits[t] /= RepetitionPenalty;
            }
            long nextToken = Argmax(lastLogits);
            generateTokens.Add(nextToken);
            if (nextToken == StopSpeechToken) { actualSteps = step + 1; break; }

            // Stash present_kv into past_kv for next iter. The output shape's
            // T grows by seqIn each step (sTotal on step 0, then 1).
            int tPresent = tPast + seqIn;
            pastKvShape = [1, LlmKvHeads, tPresent, LlmHeadDim];
            for (int layer = 0; layer < LlmLayers; layer++)
            {
                pastKv[2 * layer]     = outList[1 + 2 * layer].AsTensor<float>().ToArray();
                pastKv[2 * layer + 1] = outList[2 + 2 * layer].AsTensor<float>().ToArray();
            }

            // Re-embed the new token at position (step + 1).
            var nextIdT = new DenseTensor<long>(new long[] { nextToken }, [1, 1]);
            var nextPosT = new DenseTensor<long>(new long[] { (long)(step + 1) }, [1, 1]);
            var nextExagT = new DenseTensor<float>(new float[] { Exaggeration }, [1]);
            using var nextEmbOut = emb.Run([
                NamedOnnxValue.CreateFromTensor("input_ids", nextIdT),
                NamedOnnxValue.CreateFromTensor("position_ids", nextPosT),
                NamedOnnxValue.CreateFromTensor("exaggeration", nextExagT),
            ]);
            inputsEmbeds = nextEmbOut.First().AsTensor<float>().ToArray();

            // Grow attention_mask by 1.
            var grown = new long[attentionMask.Length + 1];
            Array.Copy(attentionMask, grown, attentionMask.Length);
            grown[^1] = 1;
            attentionMask = grown;
            actualSteps = step + 1;
        }
        lmSw.Stop();
        Console.WriteLine($"LM: {actualSteps} steps, generated {generateTokens.Count - 1} tokens (incl. STOP/no-STOP)  [{lmSw.ElapsedMilliseconds} ms, {lmSw.ElapsedMilliseconds / (double)actualSteps:F1} ms/step]");

        // DIAGNOSTIC: dump full token sequence for divergence hunting.
        File.WriteAllBytes("/tmp/cs_tokens.bin",
            MemoryMarshal.AsBytes<long>(generateTokens.ToArray()).ToArray());
        Console.WriteLine($"[diag] wrote /tmp/cs_tokens.bin ({generateTokens.Count} tokens)");

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

        // ── conditional_decoder.onnx → waveform ───────────────────────────
        sw.Restart();
        var speechTokT = new DenseTensor<long>(speechTokens, [1, speechTokens.Length]);
        var spkEmbT = new DenseTensor<float>(spkEmb.ToArray(), spkEmb.Dimensions.ToArray());
        var spkFeatT = new DenseTensor<float>(spkFeat.ToArray(), spkFeat.Dimensions.ToArray());
        using var decOut = dec.Run([
            NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokT),
            NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkEmbT),
            NamedOnnxValue.CreateFromTensor("speaker_features", spkFeatT),
        ]);
        var wavTensor = decOut.First().AsTensor<float>();    // [1, N]
        sw.Stop();
        int nSamples = wavTensor.Dimensions[1];
        Console.WriteLine($"cond_decoder: waveform=(1, {nSamples}) → {nSamples / (float)S3GenSr:F2}s  [{sw.ElapsedMilliseconds} ms]");

        // ── Write WAV ─────────────────────────────────────────────────────
        var samples = wavTensor.ToArray();
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(S3GenSr, 1);
        using (var writer = new WaveFileWriter(outPath, fmt))
        {
            writer.WriteSamples(samples, 0, samples.Length);
        }
        totalSw.Stop();
        Console.WriteLine($"Wrote {outPath}  [total {totalSw.ElapsedMilliseconds / 1000.0:F1}s]");
        return 0;
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
}
