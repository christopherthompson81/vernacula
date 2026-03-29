using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Parakeet.Base;

/// <summary>
/// Silero VAD — C# wrapper for the silero_vad.onnx model.
/// Runs CPU-only windowed inference and post-processes probabilities
/// into speech segments (start, end) in seconds.
/// </summary>
public sealed class VadSegmenter : IDisposable
{
    private readonly InferenceSession _session;

    public VadSegmenter(string modelsDir)
    {
        var opts = new SessionOptions { IntraOpNumThreads = 1 };
        _session = new InferenceSession(Path.Combine(modelsDir, Config.VadFile), opts);
    }

    public void Dispose() => _session.Dispose();

    /// <summary>
    /// Runs Silero VAD on <paramref name="audio"/> and returns speech segments as
    /// (start, end) pairs in seconds.
    /// </summary>
    public List<(double start, double end)> GetSegments(float[] audio)
    {
        var probs = RunInference(audio);
        return PostProcess(probs, audio.Length);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    private List<float> RunInference(float[] audio)
    {
        int numWindows  = audio.Length / Config.VadWindowSamples;
        int fullSize    = Config.VadContextSamples + Config.VadWindowSamples;
        var probs       = new List<float>(numWindows);

        var state   = new float[2 * 1 * 128];
        var context = new float[Config.VadContextSamples];
        var fullInput = new float[fullSize];

        for (int i = 0; i < numWindows; i++)
        {
            context.CopyTo(fullInput, 0);
            audio.AsSpan(i * Config.VadWindowSamples, Config.VadWindowSamples)
                 .CopyTo(fullInput.AsSpan(Config.VadContextSamples));

            var inputTensor = new DenseTensor<float>(fullInput, [1, fullSize]);
            var stateTensor = new DenseTensor<float>(state,     [2, 1, 128]);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor),
                NamedOnnxValue.CreateFromTensor("state", stateTensor),
            };

            using var results = _session.Run(inputs);

            float prob = results.First(r => r.Name == "output").AsEnumerable<float>().First();
            probs.Add(prob);

            results.First(r => r.Name == "stateN").AsEnumerable<float>().ToArray().CopyTo(state, 0);

            fullInput.AsSpan(fullSize - Config.VadContextSamples).CopyTo(context);
        }

        return probs;
    }

    // ── Post-processing ───────────────────────────────────────────────────────

    private static List<(double start, double end)> PostProcess(List<float> probs, int totalSamples)
    {
        int minSpeechSamples  = Config.VadMinSpeechMs  * Config.SampleRate / 1000;
        int minSilenceSamples = Config.VadMinSilenceMs * Config.SampleRate / 1000;
        int speechPadSamples  = Config.VadSpeechPadMs  * Config.SampleRate / 1000;

        float onset  = Config.VadOnsetThreshold;
        float offset = Config.VadOffsetThreshold;

        var segments = new List<(double start, double end)>();
        bool triggered = false;
        int  speechStart = 0;
        int  tempEnd     = -1;

        for (int i = 0; i < probs.Count; i++)
        {
            int   pos = i * Config.VadWindowSamples;
            float p   = probs[i];

            if (p >= onset && tempEnd >= 0)
                tempEnd = -1;

            if (p >= onset && !triggered)
            {
                triggered   = true;
                speechStart = Math.Max(0, pos - speechPadSamples);
                tempEnd     = -1;
            }
            else if (triggered && p < offset)
            {
                if (tempEnd < 0)
                    tempEnd = pos;

                if (pos - tempEnd >= minSilenceSamples)
                {
                    int speechEnd = Math.Min(totalSamples, tempEnd + speechPadSamples);
                    if (speechEnd - speechStart >= minSpeechSamples)
                        segments.Add((speechStart / (double)Config.SampleRate,
                                      speechEnd   / (double)Config.SampleRate));
                    triggered = false;
                    tempEnd   = -1;
                }
            }
        }

        if (triggered)
        {
            int speechEnd = Math.Min(totalSamples, probs.Count * Config.VadWindowSamples);
            if (speechEnd - speechStart >= minSpeechSamples)
                segments.Add((speechStart / (double)Config.SampleRate,
                              speechEnd   / (double)Config.SampleRate));
        }

        return segments;
    }
}
