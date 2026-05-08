// Granite Speech 4.1 — C# CLI parity smoke.
//
// Drives the exported ONNX bundle end-to-end from a real audio file:
//   audio.wav -> mel.onnx -> encoder.onnx -> projector.onnx -> decoder_init.onnx -> decoder_step (loop) -> text
//
// Compares the resulting transcript against the golden output produced by
// `model.generate(...)` (saved to Fixtures/expected_text.txt by
// scripts/granite_export/dump_inputs_for_csharp_smoke.py).
//
// What this smoke validates that the per-stage Python parity test cannot:
//   * mel.onnx works against an actual WAV file decoded by NAudio
//   * The 4-graph pipeline composes correctly across language boundaries
//     (numpy -> ORT tensors -> next stage's ORT tensors)
//   * The KV cache handoff between decoder_init and decoder_step works under
//     the C# ORT runtime
//   * GPT-2 ByteLevel BPE decode produces the same UTF-8 text as Python
//
// What it does NOT yet validate:
//   * Encoding arbitrary prompts (we cheat by loading pre-tokenised input_ids
//     from Fixtures/input_ids.bin — full BPE encoder is a follow-up).
//   * Batching, IOBinding, GPU execution, dtype paths.
//
// Usage:
//   dotnet run --project tests/GraniteSpeechSmoke -- \
//       --onnx-dir ./models/granite_speech_4_1_2b \
//       --audio /path/to/clip.wav \
//       --fixtures ./tests/GraniteSpeechSmoke/Fixtures \
//       --max-new-tokens 64

using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base;

namespace Vernacula.GraniteSpeechSmoke;

internal static class Program
{
    private const int NumDecoderLayers = 40;
    private const int AudioTokenId = 100352;
    private const int EosTokenId = 100257;

    private static int Main(string[] args)
    {
        // ── Argument parsing ──────────────────────────────────────────────
        string? onnxDir = null;
        string? audioPath = null;
        string? fixturesDir = null;
        int maxNewTokens = 64;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--onnx-dir":          onnxDir = args[++i]; break;
                case "--audio":             audioPath = args[++i]; break;
                case "--fixtures":          fixturesDir = args[++i]; break;
                case "--max-new-tokens":    maxNewTokens = int.Parse(args[++i]); break;
                default:
                    Console.Error.WriteLine($"Unknown arg: {args[i]}");
                    return 2;
            }
        }
        if (onnxDir is null || audioPath is null || fixturesDir is null)
        {
            Console.Error.WriteLine("Usage: --onnx-dir <dir> --audio <wav> --fixtures <dir> [--max-new-tokens N]");
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
        Console.WriteLine($"audio: {audio.Length} samples @ {sr} Hz ({audio.Length / (double)sr:F2}s)");

        // ── Load fixtures ─────────────────────────────────────────────────
        // input_ids.bin is the prompt token IDs for a fixed audio length;
        // expected_text.txt is the golden transcript from `model.generate(...)`.
        long[] inputIds = ReadInt64Bin(Path.Combine(fixturesDir, "input_ids.bin"));
        string expectedText = File.ReadAllText(Path.Combine(fixturesDir, "expected_text.txt")).Trim();
        Console.WriteLine($"prompt: {inputIds.Length} tokens (from fixtures)");

        // ── Load tokenizer for decode-side ────────────────────────────────
        var (idToToken, addedTokens) = LoadTokenizer(Path.Combine(onnxDir, "tokenizer.json"));
        var byteLevelDecode = BuildByteLevelDecodeTable();

        // ── Build ORT sessions ────────────────────────────────────────────
        var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        using var melSess = new InferenceSession(Path.Combine(onnxDir, "mel.onnx"), so);
        using var encSess = new InferenceSession(Path.Combine(onnxDir, "encoder.onnx"), so);
        using var projSess = new InferenceSession(Path.Combine(onnxDir, "projector.onnx"), so);
        using var initSess = new InferenceSession(Path.Combine(onnxDir, "decoder_init.onnx"), so);
        using var stepSess = new InferenceSession(Path.Combine(onnxDir, "decoder_step.onnx"), so);
        Console.WriteLine("ONNX sessions loaded.");

        // ── Mel ───────────────────────────────────────────────────────────
        var audioTensor = new DenseTensor<float>(audio, [1, audio.Length]);
        using var melResults = melSess.Run([NamedOnnxValue.CreateFromTensor("audio", audioTensor)]);
        DenseTensor<float> inputFeatures = (DenseTensor<float>)melResults[0].AsTensor<float>();
        var ifShape = inputFeatures.Dimensions.ToArray();  // [B, T_stacked, 160]
        Console.WriteLine($"mel: input_features={Shape(ifShape)}");

        // ── Encoder ───────────────────────────────────────────────────────
        // Re-wrap in a fresh DenseTensor since the IDisposableReadOnlyCollection
        // becomes invalid once `melResults` is disposed.
        var encInputData = inputFeatures.ToArray();
        var encInputT = new DenseTensor<float>(encInputData, ifShape);
        using var encResults = encSess.Run([NamedOnnxValue.CreateFromTensor("input_features", encInputT)]);
        var encoderHidden = encResults[0].AsTensor<float>();
        var encShape = encoderHidden.Dimensions.ToArray();  // [B, T_stacked, 1024]
        var encData = encoderHidden.ToArray();
        Console.WriteLine($"encoder: encoder_hidden={Shape(encShape)}");

        // ── Projector ─────────────────────────────────────────────────────
        var projInputT = new DenseTensor<float>(encData, encShape);
        using var projResults = projSess.Run([NamedOnnxValue.CreateFromTensor("encoder_hidden", projInputT)]);
        var audioEmbeds = projResults[0].AsTensor<float>();
        var projShape = audioEmbeds.Dimensions.ToArray();  // [B, audio_len, 2048]
        var projData = audioEmbeds.ToArray();
        Console.WriteLine($"projector: audio_embeds={Shape(projShape)}");

        // ── Decoder init (prefill) ────────────────────────────────────────
        int promptLen = inputIds.Length;
        var inputIdsT = new DenseTensor<long>(inputIds, [1, promptLen]);
        var attentionMask = new long[promptLen];
        for (int i = 0; i < promptLen; i++) attentionMask[i] = 1L;
        var attentionMaskT = new DenseTensor<long>(attentionMask, [1, promptLen]);
        var audioEmbedsT = new DenseTensor<float>(projData, projShape);

        var initInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsT),
            NamedOnnxValue.CreateFromTensor("audio_embeds", audioEmbedsT),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskT),
        };
        var initResults = initSess.Run(initInputs);  // keep alive while we extract
        var initLogits = initResults[0].AsTensor<float>();
        var initShape = initLogits.Dimensions.ToArray();  // [1, prompt_len, vocab]
        int vocabSize = (int)initShape[2];
        Console.WriteLine($"decoder_init: logits={Shape(initShape)}");

        // Extract per-layer K/V into managed arrays so we can pass them to step.
        var pastKeys = new float[NumDecoderLayers][];
        var pastValues = new float[NumDecoderLayers][];
        var kvShape = new[] { 1, 4, promptLen, 128 };  // [B, num_kv_heads, seq, head_dim]
        for (int L = 0; L < NumDecoderLayers; L++)
        {
            pastKeys[L] = initResults[1 + L].AsTensor<float>().ToArray();
            pastValues[L] = initResults[1 + NumDecoderLayers + L].AsTensor<float>().ToArray();
        }

        // Greedy next-token from the last-position logits of the prefill.
        long nextToken = ArgmaxLastPosition(initLogits, promptLen, vocabSize);
        initResults.Dispose();

        // ── Decoder step loop ─────────────────────────────────────────────
        var generated = new List<long>(maxNewTokens);
        int pastLen = promptLen;
        for (int step = 0; step < maxNewTokens; step++)
        {
            if (nextToken == EosTokenId) break;
            generated.Add(nextToken);

            int totalLen = pastLen + 1;
            var stepInputId = new DenseTensor<long>(new[] { nextToken }, [1, 1]);
            var stepAttn = new long[totalLen];
            for (int i = 0; i < totalLen; i++) stepAttn[i] = 1L;
            var stepAttnT = new DenseTensor<long>(stepAttn, [1, totalLen]);
            var cachePos = new DenseTensor<long>(new[] { (long)pastLen }, [1]);

            var feed = new List<NamedOnnxValue>(3 + 2 * NumDecoderLayers)
            {
                NamedOnnxValue.CreateFromTensor("input_id", stepInputId),
                NamedOnnxValue.CreateFromTensor("attention_mask", stepAttnT),
                NamedOnnxValue.CreateFromTensor("cache_position", cachePos),
            };
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                feed.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_{L}",
                    new DenseTensor<float>(pastKeys[L], [1, 4, pastLen, 128])));
                feed.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_value_{L}",
                    new DenseTensor<float>(pastValues[L], [1, 4, pastLen, 128])));
            }

            using var stepResults = stepSess.Run(feed);
            var stepLogits = stepResults[0].AsTensor<float>();
            nextToken = ArgmaxLastPosition(stepLogits, 1, vocabSize);

            // Roll forward KV.
            for (int L = 0; L < NumDecoderLayers; L++)
            {
                pastKeys[L] = stepResults[1 + L].AsTensor<float>().ToArray();
                pastValues[L] = stepResults[1 + NumDecoderLayers + L].AsTensor<float>().ToArray();
            }
            pastLen = totalLen;
        }
        Console.WriteLine($"decoder_step: {generated.Count} tokens");

        // ── Decode tokens to text ─────────────────────────────────────────
        string text = DecodeTokens(generated, idToToken, addedTokens, byteLevelDecode);

        Console.WriteLine();
        Console.WriteLine($"  ORT  transcript: {Quote(text)}");
        Console.WriteLine($"  Ref  transcript: {Quote(expectedText)}");
        bool match = text.Trim() == expectedText.Trim();
        Console.WriteLine($"  exact match: {match}");
        return match ? 0 : 1;
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static string Shape(int[] dims) => "(" + string.Join(", ", dims) + ")";

    private static string Quote(string s) => "\"" + s + "\"";

    private static long ArgmaxLastPosition(Tensor<float> logits, int seqLen, int vocab)
    {
        // logits shape: [1, seqLen, vocab]. Argmax along the vocab axis at position seqLen-1.
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

    /// <summary>
    /// GPT-2 ByteLevel decode table: a Unicode char inside a token string maps
    /// back to a single byte. The 256 byte values are mapped to:
    ///   * printable ASCII (33-126), Latin-1 supplement (161-172, 174-255) → themselves
    ///   * the remaining 68 bytes (0-32, 127, 128-160, 173) → U+0100, U+0101, …
    /// This table is the inverse: char -> byte.
    /// </summary>
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
                // Special tokens (e.g. <|begin_of_text|>) map to their literal content.
                // For ASR output we typically skip these, but this smoke decodes them
                // to keep parity with `tokenizer.decode(..., skip_special_tokens=False)`.
                // Since the smoke loop terminates on EOS we never see them, but include
                // the path for safety.
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
