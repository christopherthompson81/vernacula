// OmniVoice TTS — C# end-to-end smoke test.
//
// Drives the full C# pipeline (Qwen3 tokenizer + duration estimate + text prep + the three
// ONNX graphs + the greedy diffusion loop in Chatterbox.Base.OmniVoiceTts) to synthesise a
// WAV, for the listen-test against the Python reference. Defaults reproduce the Phase-1
// capture (capture/ref_voice.wav + transcript, English, num_step 16) so outputs line up.
//
// Usage:
//   dotnet run --project tests/OmniVoiceSmoke -- \
//       --onnx-dir scripts/omnivoice_export/onnx \
//       --model    /mnt/data/models/omnivoice/k2-fsa-OmniVoice \
//       --ref-audio scripts/omnivoice_export/capture/ref_voice.wav \
//       --ref-text "This is a reference voice sample for testing the export pipeline." \
//       --text "Hello, ..." --out /tmp/omnivoice_cs.wav --ep cpu --num-step 16

using System.Diagnostics;
using Chatterbox.Base;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using Vernacula.Base;
using Vernacula.Base.Models;

namespace Vernacula.OmniVoiceSmoke;

internal static class Program
{
    private static int Main(string[] args)
    {
        string onnxDir = "scripts/omnivoice_export/onnx";
        string modelDir = "/mnt/data/models/omnivoice/k2-fsa-OmniVoice";
        string refAudio = "scripts/omnivoice_export/capture/ref_voice.wav";
        string? refText = null;   // defaults to <ref-audio>.txt sidecar
        string text = "Hello, this is a test of the OmniVoice O N N X export pipeline.";
        string lang = "en";
        string? instruct = null;
        string outPath = "/tmp/omnivoice_cs.wav";
        string ep = "cpu";
        int numStep = 16;
        int? durationTokens = null;
        string? transformerFile = null;  // e.g. omnivoice_transformer_fp16.onnx

        for (int i = 0; i < args.Length - 1; i++)
        {
            switch (args[i])
            {
                case "--onnx-dir": onnxDir = args[++i]; break;
                case "--model": modelDir = args[++i]; break;
                case "--ref-audio": refAudio = args[++i]; break;
                case "--ref-text": refText = args[++i]; break;
                case "--text": text = args[++i]; break;
                case "--lang": lang = args[++i]; break;
                case "--instruct": instruct = args[++i]; break;
                case "--out": outPath = args[++i]; break;
                case "--ep": ep = args[++i]; break;
                case "--num-step": numStep = int.Parse(args[++i]); break;
                case "--target-tokens": durationTokens = int.Parse(args[++i]); break;
                case "--transformer": transformerFile = args[++i]; break;
            }
        }

        var epEnum = ep.ToLowerInvariant() switch
        {
            "cpu" => ExecutionProvider.Cpu,
            "cuda" => ExecutionProvider.Cuda,
            _ => ExecutionProvider.Auto,
        };

        string tokenizerJson = Path.Combine(modelDir, "tokenizer.json");
        if (!File.Exists(tokenizerJson)) { Console.Error.WriteLine($"tokenizer.json not found: {tokenizerJson}"); return 1; }

        bool clone = !string.IsNullOrEmpty(refAudio) && File.Exists(refAudio);
        if (clone && refText is null)
        {
            var sidecar = Path.ChangeExtension(refAudio, ".txt");
            refText = File.Exists(sidecar) ? File.ReadAllText(sidecar).Trim() : null;
        }

        var sw = Stopwatch.StartNew();
        Console.WriteLine($"Loading OmniVoice ({ep}) ...");
        using var tts = new OmniVoiceTts(onnxDir, tokenizerJson, epEnum,
            e => Console.WriteLine($"  loaded {e.FileName} in {e.ElapsedMs} ms (cuda={e.UsedCuda})"),
            transformerFile);
        Console.WriteLine($"loaded in {sw.ElapsedMilliseconds} ms");

        long[,]? refCodes = null;
        int refTokens = 25;
        if (clone)
        {
            float[] refWav = Load24kMono(refAudio);
            refCodes = tts.EncodeReference(refWav);
            refTokens = refCodes.GetLength(1);
            Console.WriteLine($"reference: {refWav.Length} samples -> {refTokens} codes; ref_text={refText}");
        }

        int target = durationTokens
            ?? OmniVoiceDuration.EstimateTargetTokens(text, refText, clone ? refTokens : null);
        Console.WriteLine($"target tokens: {target}");

        sw.Restart();
        var tokens = tts.GenerateTokens(text, target, refText, refCodes, lang, instruct,
            new OmniVoiceTts.GenConfig(NumStep: numStep));
        Console.WriteLine($"diffusion ({numStep} steps): {sw.ElapsedMilliseconds} ms "
            + $"[transformer {tts.LastTransformerMs:f0} ms, host {tts.LastHostMs:f0} ms]");

        float[] audio = tts.DecodeTokens(tokens);
        Normalize(audio, 0.95f);

        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(OmniVoiceTts.SampleRate, 1);
        using (var w = new WaveFileWriter(outPath, fmt))
            w.WriteSamples(audio, 0, audio.Length);
        Console.WriteLine($"wrote {audio.Length / (double)OmniVoiceTts.SampleRate:f2}s -> {outPath}");
        return 0;
    }

    private static float[] Load24kMono(string path)
    {
        var (samples, sr, ch) = AudioUtils.ReadAudio(path);
        float[] mono = ch > 1 ? AudioUtils.DownmixToMono(samples, ch) : samples;
        if (sr == OmniVoiceTts.SampleRate) return mono;
        var src = new FloatArraySampleProvider(mono, WaveFormat.CreateIeeeFloatWaveFormat(sr, 1));
        var rs = new WdlResamplingSampleProvider(src, OmniVoiceTts.SampleRate);
        var outBuf = new List<float>(mono.Length * OmniVoiceTts.SampleRate / sr + 1024);
        var chunk = new float[4096];
        int n;
        while ((n = rs.Read(chunk, 0, chunk.Length)) > 0)
            outBuf.AddRange(chunk[..n]);
        return outBuf.ToArray();
    }

    private static void Normalize(float[] x, float peak)
    {
        float max = 0;
        foreach (var v in x) max = Math.Max(max, Math.Abs(v));
        if (max < 1e-6f) return;
        float g = peak / max;
        for (int i = 0; i < x.Length; i++) x[i] *= g;
    }
}
