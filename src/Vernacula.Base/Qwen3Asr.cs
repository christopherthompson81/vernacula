using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.Json;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// Qwen3-ASR 1.7B runtime for the split ONNX export produced by scripts/qwen3asr_export.
///
/// Pipeline:
///   1. Host-side Whisper-style log-mel frontend
///   2. encoder.onnx          mel [1,128,T] -> audio_features [1,N,2048]
///   3. decoder_init.onnx     prompt ids + audio features -> logits + KV cache
///   4. decoder_step.onnx     token embedding + KV cache -> logits + updated KV cache
///
/// The decoder token embedding matrix is kept in a memory-mapped file so we do not
/// need a 1.2 GB managed float[] on the LOH.
/// </summary>
public sealed class Qwen3Asr : IDisposable
{
    public const string EncoderFile = "encoder.onnx";
    public const string DecoderInitFile = "decoder_init.onnx";
    public const string DecoderStepFile = "decoder_step.onnx";
    public const string EmbedTokensFile = "embed_tokens.bin";
    public const string TokenizerFile = "tokenizer.json";
    public const string ConfigFile = "config.json";

    private const int SampleRate = 16_000;
    private const int NFft = 400;
    private const int HopLength = 160;
    private const int NMels = 128;
    private const float LogFloor = -10.0f;
    private const float LogClampSpan = 8.0f;
    private const float LogOffset = 4.0f;
    private const int ImStartTokenId = 151644;
    private const int ImEndTokenId = 151645;
    private const int AudioStartTokenId = 151669;
    private const int AudioEndTokenId = 151670;
    private const int AudioPadTokenId = 151676;
    private const int NewlineTokenId = 198;
    private const int SystemTokenId = 9125;
    private const int UserTokenId = 882;
    private const int AssistantTokenId = 77091;

    private static readonly float[,] MelFilterbank = CreateMelFilterbank();
    private static readonly double[] HannWindow = Window.HannPeriodic(NFft);

    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoderInit;
    private readonly InferenceSession _decoderStep;
    private readonly bool _useCudaIoBinding;
    private readonly int _hiddenSize;
    private readonly int _vocabSize;
    private readonly int _baseVocabSize;
    private readonly int[] _eosTokenIds;
    private readonly HashSet<int> _eosTokenIdSet;
    private readonly MemoryMappedFile _embedMmf;
    private readonly MemoryMappedViewAccessor _embedAccessor;
    private readonly string?[] _idToToken;
    private readonly Dictionary<int, string> _addedTokenContent;
    private readonly Dictionary<char, byte> _byteLevelDecode;

    public Qwen3Asr(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        string configJson = File.ReadAllText(Path.Combine(modelPath, ConfigFile));
        using var configDoc = JsonDocument.Parse(configJson);
        var root = configDoc.RootElement;
        var decoderConfig = root.GetProperty("decoder");
        var specialTokens = root.GetProperty("special_tokens");

        _hiddenSize = decoderConfig.GetProperty("hidden_size").GetInt32();
        _vocabSize = decoderConfig.GetProperty("vocab_size").GetInt32();
        _baseVocabSize = root.GetProperty("embed_tokens_shape")[0].GetInt32();
        _eosTokenIds = specialTokens.GetProperty("eos_token_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        _eosTokenIdSet = [.. _eosTokenIds];

        var encoderOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED, out bool encoderUsesCuda);
        var decoderOpts = MakeSessionOptions(ep, GraphOptimizationLevel.ORT_ENABLE_EXTENDED, out bool decoderUsesCuda);
        _useCudaIoBinding = encoderUsesCuda && decoderUsesCuda;

        _encoder = new InferenceSession(Path.Combine(modelPath, EncoderFile), encoderOpts);
        _decoderInit = new InferenceSession(Path.Combine(modelPath, DecoderInitFile), decoderOpts);
        _decoderStep = new InferenceSession(Path.Combine(modelPath, DecoderStepFile), decoderOpts);

        string embedPath = Path.Combine(modelPath, EmbedTokensFile);
        _embedMmf = MemoryMappedFile.CreateFromFile(embedPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _embedAccessor = _embedMmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        (_idToToken, _addedTokenContent) = LoadTokenizerVocab(Path.Combine(modelPath, TokenizerFile));
        _byteLevelDecode = BuildByteLevelDecode();
    }

    public IEnumerable<(int segId, string text)> Recognize(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        foreach (var result in RecognizeDetailed(segs, audio, maxNewTokens))
            yield return (result.SegmentId, result.Text);
    }

    public IEnumerable<QwenRecognitionResult> RecognizeDetailed(
        IReadOnlyList<(double start, double end, string spk)> segs,
        float[] audio,
        int maxNewTokens = 256)
    {
        for (int segId = 0; segId < segs.Count; segId++)
        {
            var (start, end, _) = segs[segId];
            int startSample = Math.Clamp((int)Math.Round(start * SampleRate), 0, audio.Length);
            int endSample = Math.Clamp((int)Math.Round(end * SampleRate), 0, audio.Length);
            if (endSample <= startSample)
            {
                yield return new QwenRecognitionResult(segId, "", [], [], []);
                continue;
            }

            int length = endSample - startSample;
            float[] mel = ComputeLogMelSpectrogram(audio, startSample, length, out int melFrames);
            float[] audioFeatures = RunEncoder(mel, melFrames, out int audioTokenCount);

            List<int> promptIds = BuildPromptIds(audioTokenCount);
            int audioOffset = GetAudioPadStart(promptIds);

            var (rawTokens, rawLogprobs) = _useCudaIoBinding
                ? DecodeWithIoBinding(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens)
                : DecodeOnCpu(promptIds, audioOffset, audioFeatures, audioTokenCount, maxNewTokens);

            var (textTokens, textLogprobs) = ExtractTextTokens(rawTokens, rawLogprobs);
            string text = DecodeTokens(textTokens);
            yield return new QwenRecognitionResult(segId, text, rawTokens, textTokens, textLogprobs);
        }
    }

    private static SessionOptions MakeSessionOptions(
        ExecutionProvider ep,
        GraphOptimizationLevel optimizationLevel,
        out bool usesCuda)
    {
        var opts = new SessionOptions { GraphOptimizationLevel = optimizationLevel };
        usesCuda = false;

        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                {
                    try
                    {
                        opts.AppendExecutionProvider_CUDA(0);
                        usesCuda = true;
                        break;
                    }
                    catch { }
                }
                try { opts.AppendExecutionProvider_DML(0); } catch { }
                break;
            case ExecutionProvider.Cuda:
                opts.AppendExecutionProvider_CUDA(0);
                usesCuda = true;
                break;
            case ExecutionProvider.DirectML:
                opts.AppendExecutionProvider_DML(0);
                break;
            case ExecutionProvider.Cpu:
                break;
        }

        return opts;
    }

    private float[] RunEncoder(float[] mel, int melFrames, out int audioTokenCount)
    {
        using var results = _encoder.Run(
        [
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(mel, [1, NMels, melFrames])),
        ]);

        var features = results.First(r => r.Name == "audio_features").AsTensor<float>();
        audioTokenCount = features.Dimensions[1];
        return ExtractTensor(features);
    }

    private (List<int> rawTokens, List<float> rawLogprobs) DecodeOnCpu(
        IReadOnlyList<int> promptIds,
        int audioOffset,
        float[] audioFeatures,
        int audioTokenCount,
        int maxNewTokens)
    {
        long[] inputIds = promptIds.Select(i => (long)i).ToArray();
        long[] positionIds = Enumerable.Range(0, promptIds.Count).Select(i => (long)i).ToArray();
        long[] audioOffsetTensor = [audioOffset];

        using var initOutputs = _decoderInit.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, [1, inputIds.Length])),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(positionIds, [1, positionIds.Length])),
            NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audioFeatures, [1, audioTokenCount, _hiddenSize])),
            NamedOnnxValue.CreateFromTensor("audio_offset", new DenseTensor<long>(audioOffsetTensor, [1])),
        ]);

        float[] logits = ExtractTensor(initOutputs.First(r => r.Name == "logits").AsTensor<float>());
        float[] pastKeys = ExtractTensor(initOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
        float[] pastValues = ExtractTensor(initOutputs.First(r => r.Name == "present_values").AsTensor<float>());
        int pastSeqLen = promptIds.Count;
        var tokenEmbed = new float[_hiddenSize];
        long[] stepPos = [0L];

        var rawTokens = new List<int>(maxNewTokens);
        var rawLogprobs = new List<float>(maxNewTokens);

        int nextToken = ArgMaxLastLogits(logits, promptIds.Count, out float nextLogprob);
        rawTokens.Add(nextToken);
        rawLogprobs.Add(nextLogprob);
        pastSeqLen = promptIds.Count;

        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tokenEmbed);
            stepPos[0] = promptIds.Count + rawTokens.Count - 1L;

            using var stepOutputs = _decoderStep.Run(
            [
                NamedOnnxValue.CreateFromTensor("input_embeds", new DenseTensor<float>(tokenEmbed, [1, 1, _hiddenSize])),
                NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(stepPos, [1, 1])),
                NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(pastKeys, [28, 1, 8, pastSeqLen, 128])),
                NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(pastValues, [28, 1, 8, pastSeqLen, 128])),
            ]);

            logits = ExtractTensor(stepOutputs.First(r => r.Name == "logits").AsTensor<float>());
            pastKeys = ExtractTensor(stepOutputs.First(r => r.Name == "present_keys").AsTensor<float>());
            pastValues = ExtractTensor(stepOutputs.First(r => r.Name == "present_values").AsTensor<float>());
            pastSeqLen++;

            nextToken = ArgMaxLastLogits(logits, 1, out nextLogprob);
            rawTokens.Add(nextToken);
            rawLogprobs.Add(nextLogprob);
        }

        return (rawTokens, rawLogprobs);
    }

    private (List<int> rawTokens, List<float> rawLogprobs) DecodeWithIoBinding(
        IReadOnlyList<int> promptIds,
        int audioOffset,
        float[] audioFeatures,
        int audioTokenCount,
        int maxNewTokens)
    {
        using var cpuMemInfo = new OrtMemoryInfo("Cpu", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cudaMemInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var runOpts = new RunOptions();

        long[] inputIds = promptIds.Select(i => (long)i).ToArray();
        long[] positionIds = Enumerable.Range(0, promptIds.Count).Select(i => (long)i).ToArray();
        long[] audioOffsetTensor = [audioOffset];

        using var inputIdsValue = OrtValue.CreateTensorValueFromMemory(inputIds, [1, inputIds.Length]);
        using var positionIdsValue = OrtValue.CreateTensorValueFromMemory(positionIds, [1, positionIds.Length]);
        using var audioFeaturesValue = OrtValue.CreateTensorValueFromMemory(audioFeatures, [1, audioTokenCount, _hiddenSize]);
        using var audioOffsetValue = OrtValue.CreateTensorValueFromMemory(audioOffsetTensor, [1]);

        using var initBinding = _decoderInit.CreateIoBinding();
        initBinding.BindInput("input_ids", inputIdsValue);
        initBinding.BindInput("position_ids", positionIdsValue);
        initBinding.BindInput("audio_features", audioFeaturesValue);
        initBinding.BindInput("audio_offset", audioOffsetValue);
        initBinding.BindOutputToDevice("logits", cpuMemInfo);
        initBinding.BindOutputToDevice("present_keys", cudaMemInfo);
        initBinding.BindOutputToDevice("present_values", cudaMemInfo);
        _decoderInit.RunWithBinding(runOpts, initBinding);

        var initOutputs = initBinding.GetOutputValues();
        var rawTokens = new List<int>(maxNewTokens);
        var rawLogprobs = new List<float>(maxNewTokens);
        var tokenEmbed = new float[_hiddenSize];
        long[] stepPos = [0L];

        int nextToken = ArgMaxLastLogits(initOutputs[0], promptIds.Count, out float nextLogprob);
        rawTokens.Add(nextToken);
        rawLogprobs.Add(nextLogprob);

        IDisposableReadOnlyCollection<OrtValue>? prevOutputs = initOutputs;
        int pastSeqLen = promptIds.Count;

        while (rawTokens.Count < maxNewTokens && !IsEos(nextToken))
        {
            ReadTokenEmbedding(nextToken, tokenEmbed);
            stepPos[0] = promptIds.Count + rawTokens.Count - 1L;

            using var tokenEmbedValue = OrtValue.CreateTensorValueFromMemory(tokenEmbed, [1, 1, _hiddenSize]);
            using var stepPosValue = OrtValue.CreateTensorValueFromMemory(stepPos, [1, 1]);
            using var stepBinding = _decoderStep.CreateIoBinding();
            stepBinding.BindInput("input_embeds", tokenEmbedValue);
            stepBinding.BindInput("position_ids", stepPosValue);
            stepBinding.BindInput("past_keys", prevOutputs![1]);
            stepBinding.BindInput("past_values", prevOutputs[2]);
            stepBinding.BindOutputToDevice("logits", cpuMemInfo);
            stepBinding.BindOutputToDevice("present_keys", cudaMemInfo);
            stepBinding.BindOutputToDevice("present_values", cudaMemInfo);
            _decoderStep.RunWithBinding(runOpts, stepBinding);

            var curOutputs = stepBinding.GetOutputValues();
            nextToken = ArgMaxLastLogits(curOutputs[0], 1, out nextLogprob);
            rawTokens.Add(nextToken);
            rawLogprobs.Add(nextLogprob);
            pastSeqLen++;

            prevOutputs.Dispose();
            prevOutputs = curOutputs;
        }

        prevOutputs?.Dispose();
        return (rawTokens, rawLogprobs);
    }

    private int ArgMaxLastLogits(float[] logits, int seqLen, out float logprob)
    {
        int vocabSize = logits.Length / seqLen;
        int offset = (seqLen - 1) * vocabSize;
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < vocabSize; i++)
        {
            float value = logits[offset + i];
            if (value > bestVal)
            {
                bestVal = value;
                best = i;
            }
        }

        double sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
            sumExp += Math.Exp(logits[offset + i] - bestVal);

        logprob = (float)(-Math.Log(sumExp));
        return best;
    }

    private int ArgMaxLastLogits(OrtValue logits, int seqLen, out float logprob)
    {
        var span = logits.GetTensorDataAsSpan<float>();
        int vocabSize = span.Length / seqLen;
        int offset = (seqLen - 1) * vocabSize;
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < vocabSize; i++)
        {
            float value = span[offset + i];
            if (value > bestVal)
            {
                bestVal = value;
                best = i;
            }
        }

        double sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
            sumExp += Math.Exp(span[offset + i] - bestVal);

        logprob = (float)(-Math.Log(sumExp));
        return best;
    }

    private void ReadTokenEmbedding(int tokenId, float[] destination)
    {
        long byteOffset = (long)tokenId * _hiddenSize * sizeof(float);
        _embedAccessor.ReadArray(byteOffset, destination, 0, _hiddenSize);
    }

    private static (List<int> textTokens, List<float> textLogprobs) ExtractTextTokens(
        IReadOnlyList<int> rawTokens,
        IReadOnlyList<float> rawLogprobs)
    {
        var textTokens = new List<int>(rawTokens.Count);
        var textLogprobs = new List<float>(rawTokens.Count);
        for (int i = 0; i < rawTokens.Count; i++)
        {
            int token = rawTokens[i];
            if (token >= 0 && token < 151643)
            {
                textTokens.Add(token);
                textLogprobs.Add(i < rawLogprobs.Count ? rawLogprobs[i] : 0f);
            }
        }

        return (textTokens, textLogprobs);
    }

    public string DecodeTokens(IReadOnlyList<int> tokens)
    {
        var bytes = new List<byte>(tokens.Count * 4);
        foreach (int token in tokens)
        {
            if (token < 0 || token >= _idToToken.Length)
                continue;

            string? raw = _idToToken[token];
            if (raw is null)
                continue;

            foreach (char ch in raw)
                if (_byteLevelDecode.TryGetValue(ch, out byte value))
                    bytes.Add(value);
        }

        string text = Encoding.UTF8.GetString(bytes.ToArray());
        return text.Length > 0 && text[0] == ' ' ? text[1..] : text;
    }

    private bool IsEos(int token) => _eosTokenIdSet.Contains(token);

    private static List<int> BuildPromptIds(int audioTokenCount)
    {
        var ids = new List<int>(audioTokenCount + 16)
        {
            ImStartTokenId,
            SystemTokenId,
            NewlineTokenId,
            ImEndTokenId,
            NewlineTokenId,
            ImStartTokenId,
            UserTokenId,
            NewlineTokenId,
            AudioStartTokenId,
        };

        for (int i = 0; i < audioTokenCount; i++)
            ids.Add(AudioPadTokenId);

        ids.Add(AudioEndTokenId);
        ids.Add(ImEndTokenId);
        ids.Add(NewlineTokenId);
        ids.Add(ImStartTokenId);
        ids.Add(AssistantTokenId);
        ids.Add(NewlineTokenId);
        return ids;
    }

    private static int GetAudioPadStart(IReadOnlyList<int> promptIds)
    {
        for (int i = 0; i < promptIds.Count; i++)
            if (promptIds[i] == AudioPadTokenId)
                return i;
        throw new InvalidOperationException("Prompt does not contain <|audio_pad|> tokens.");
    }

    private static float[] ComputeLogMelSpectrogram(float[] signal, int start, int length, out int framesOut)
    {
        int pad = NFft / 2;
        float[] padded = ReflectPad(signal, start, length, pad);
        int frameCount = ((padded.Length - NFft) / HopLength) + 1;
        int keptFrames = Math.Max(frameCount - 1, 1);
        int freqBins = (NFft / 2) + 1;
        var mel = new float[NMels * keptFrames];
        Parallel.For(
            0,
            frameCount,
            () => new Complex32[NFft],
            (frame, _, fft) =>
            {
                int startIndex = frame * HopLength;
                Array.Clear(fft, 0, fft.Length);
                for (int i = 0; i < NFft; i++)
                    fft[i] = new Complex32((float)(padded[startIndex + i] * HannWindow[i]), 0f);

                Fourier.Forward(fft, FourierOptions.NoScaling);

                if (frame < keptFrames)
                {
                    for (int m = 0; m < NMels; m++)
                    {
                        double sum = 0;
                        for (int k = 0; k < freqBins; k++)
                        {
                            float re = fft[k].Real;
                            float im = fft[k].Imaginary;
                            sum += MelFilterbank[m, k] * (re * re + im * im);
                        }

                        mel[m * keptFrames + frame] = MathF.Log10(MathF.Max((float)sum, 1e-10f));
                    }
                }

                return fft;
            },
            _ => { });

        float maxLog = float.NegativeInfinity;
        for (int i = 0; i < mel.Length; i++)
        {
            if (mel[i] > maxLog)
                maxLog = mel[i];
        }

        float floor = MathF.Max(maxLog - LogClampSpan, LogFloor);
        for (int i = 0; i < mel.Length; i++)
            mel[i] = (MathF.Max(mel[i], floor) + LogOffset) / LogOffset;

        framesOut = keptFrames;
        return mel;
    }

    private static float[] ReflectPad(float[] signal, int start, int length, int pad)
    {
        if (length == 0)
            return new float[pad * 2];

        var padded = new float[length + (pad * 2)];
        Array.Copy(signal, start, padded, pad, length);

        for (int i = 0; i < pad; i++)
        {
            int leftSrc = Math.Min(length - 1, pad - i);
            int rightSrc = Math.Max(0, length - 2 - i);
            padded[i] = signal[start + leftSrc];
            padded[pad + length + i] = signal[start + rightSrc];
        }

        return padded;
    }

    private static float[,] CreateMelFilterbank()
    {
        int freqBins = (NFft / 2) + 1;
        var fb = new float[NMels, freqBins];

        var fftFreqs = new double[freqBins];
        for (int k = 0; k < freqBins; k++)
            fftFreqs[k] = (double)k * SampleRate / NFft;

        double fminMel = AudioUtils.HzToMelSlaney(0.0);
        double fmaxMel = AudioUtils.HzToMelSlaney(SampleRate / 2.0);
        var melF = new double[NMels + 2];
        for (int i = 0; i <= NMels + 1; i++)
        {
            double m = fminMel + (fmaxMel - fminMel) * i / (NMels + 1);
            melF[i] = AudioUtils.MelToHzSlaney(m);
        }

        var fdiff = new double[NMels + 1];
        for (int i = 0; i <= NMels; i++)
            fdiff[i] = melF[i + 1] - melF[i];

        for (int i = 0; i < NMels; i++)
        {
            for (int k = 0; k < freqBins; k++)
            {
                double lower = (fftFreqs[k] - melF[i]) / fdiff[i];
                double upper = (melF[i + 2] - fftFreqs[k]) / fdiff[i + 1];
                fb[i, k] = (float)Math.Max(0.0, Math.Min(lower, upper));
            }
        }

        for (int i = 0; i < NMels; i++)
        {
            float enorm = (float)(2.0 / (melF[i + 2] - melF[i]));
            for (int k = 0; k < freqBins; k++)
                fb[i, k] *= enorm;
        }

        return fb;
    }

    private static float[] ExtractTensor(Tensor<float> tensor)
    {
        if (tensor is DenseTensor<float> dense)
            return dense.Buffer.ToArray();

        var result = new float[tensor.Length];
        int i = 0;
        foreach (float value in tensor)
            result[i++] = value;
        return result;
    }

    private static (string?[] idToToken, Dictionary<int, string> addedTokenContent) LoadTokenizerVocab(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var vocab = root.GetProperty("model").GetProperty("vocab");
        int maxId = -1;
        foreach (var kv in vocab.EnumerateObject())
            maxId = Math.Max(maxId, kv.Value.GetInt32());

        var added = new Dictionary<int, string>();
        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var token in addedTokens.EnumerateArray())
            {
                int id = token.GetProperty("id").GetInt32();
                string content = token.GetProperty("content").GetString() ?? "";
                added[id] = content;
                maxId = Math.Max(maxId, id);
            }
        }

        var idToToken = new string?[maxId + 1];
        foreach (var kv in vocab.EnumerateObject())
            idToToken[kv.Value.GetInt32()] = kv.Name;

        return (idToToken, added);
    }

    private static Dictionary<char, byte> BuildByteLevelDecode()
    {
        var bs = new List<int>();
        for (int i = (int)'!'; i <= (int)'~'; i++) bs.Add(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (bs.Contains(b)) continue;
            bs.Add(b);
            cs.Add(256 + extra);
            extra++;
        }

        var map = new Dictionary<char, byte>(256);
        for (int i = 0; i < bs.Count; i++)
            map[(char)cs[i]] = (byte)bs[i];
        return map;
    }

    public void Dispose()
    {
        _embedAccessor.Dispose();
        _embedMmf.Dispose();
        _decoderStep.Dispose();
        _decoderInit.Dispose();
        _encoder.Dispose();
    }
}

public sealed record QwenRecognitionResult(
    int SegmentId,
    string Text,
    IReadOnlyList<int> RawTokens,
    IReadOnlyList<int> TextTokens,
    IReadOnlyList<float> TextLogprobs);
