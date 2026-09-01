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
using Vernacula.Base;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.OmniVoiceSmoke;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--fold-selftest")
            return FoldSelfTest(args[1], args[2], args[3], args.Length > 4 ? args[4] : "cpu");

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
        string? diffPath = null;         // ipa_diff.onnx: fold the IPA fine-tune onto the base at load

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
                case "--diff": diffPath = args[++i]; break;
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
        if (diffPath is not null) Console.WriteLine($"folding IPA diff at load: {diffPath}");
        using var tts = new OmniVoiceTts(onnxDir, tokenizerJson, epEnum,
            e => Console.WriteLine($"  loaded {e.FileName} in {e.ElapsedMs} ms (cuda={e.UsedCuda})"),
            transformerFile, diffPath);
        Console.WriteLine($"loaded in {sw.ElapsedMilliseconds} ms");

        long[,]? refCodes = null;
        int refTokens = 25;
        float? refRms = null;   // pre-boost ref RMS, for output volume un-boost (Python parity)
        if (clone)
        {
            float[] refWav = Load24kMono(refAudio);
            refRms = Rms(refWav);
            // add_punctuation on the ref transcript (Python create_voice_clone_prompt): the
            // punctuated ref_text must feed BOTH the duration estimate and the combined text.
            refText = refText is null ? null : OmniVoiceTextPrep.AddPunctuation(refText);
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
        // Output post-processing (Python _post_process_audio), in order: remove silence →
        // volume → fade-in/out + zero-pad the edges.
        audio = OmniVoiceAudioPost.RemoveSilence(audio, OmniVoiceTts.SampleRate,
            midSilMs: 500, leadSilMs: 100, trailSilMs: 100);
        // Volume (Python): with a reference, un-boost a previously-boosted quiet ref
        // (audio *= ref_rms/0.1 when ref_rms < 0.1; else leave the model's own level). Without a
        // reference, peak-normalise to 0.5.
        if (refRms is float rr) { if (rr < 0.1f) Scale(audio, rr / 0.1f); }
        else Normalize(audio, 0.5f);
        audio = OmniVoiceAudioPost.FadeAndPad(audio, OmniVoiceTts.SampleRate);

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
        return AudioUtils.ResampleMono(mono, sr, OmniVoiceTts.SampleRate);
    }

    private static void Normalize(float[] x, float peak)
    {
        float max = 0;
        foreach (var v in x) max = Math.Max(max, Math.Abs(v));
        if (max < 1e-6f) return;
        float g = peak / max;
        for (int i = 0; i < x.Length; i++) x[i] *= g;
    }

    private static void Scale(float[] x, float g)
    {
        for (int i = 0; i < x.Length; i++) x[i] *= g;
    }

    private static float Rms(float[] x)
    {
        double s = 0;
        foreach (var v in x) s += (double)v * v;
        return x.Length == 0 ? 0f : (float)Math.Sqrt(s / x.Length);
    }

    /// <summary>Validate the C# graph rewrite: run the BASE transformer with the diff applied as
    /// LoRA nodes, vs the Python-folded MERGED transformer, on the same input; they must agree.
    /// Args: base.onnx diff.onnx merged.onnx [ep]
    ///
    /// ⚠ PASS THE EP. This test previously ran only on an implicit-CPU `new SessionOptions()`,
    /// which is exactly why the old AddInitializer fold's total failure on CUDA went unnoticed
    /// until it reached a listening test. A device argument is the point of the test now.</summary>
    private static int FoldSelfTest(string baseOnnx, string diffOnnx, string mergedOnnx, string ep = "cpu")
    {
        Console.WriteLine($"diff self-test ({ep}): base+diff (C# graph rewrite) vs merged (Python)");
        var epEnum = ep.ToLowerInvariant() switch
        {
            "cuda" => ExecutionProvider.Cuda,
            "cpu" => ExecutionProvider.Cpu,
            _ => ExecutionProvider.Auto,
        };
        var diff = new Chatterbox.Base.OmniVoiceDiff();
        var opts = OrtSessionBuilder.Create(epEnum,
            Microsoft.ML.OnnxRuntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
            enableProfiling: false, out _, disableTf32: true);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var folded = diff.CreateSession(opts, baseOnnx, diffOnnx);
        Console.WriteLine($"  rewrite + load: {sw.ElapsedMilliseconds} ms "
            + $"({diff.PatchedModules} modules, {diff.PatchedEmbedRows} embed rows)");
        var mopts = OrtSessionBuilder.Create(epEnum,
            Microsoft.ML.OnnxRuntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
            enableProfiling: false, out _, disableTf32: true);
        using var merged = new Microsoft.ML.OnnxRuntime.InferenceSession(mergedOnnx, mopts);

        // small synthetic input; only need identical input to both sessions to compare outputs
        const int S = 16, C = 8, twoB = 2;
        var ids = new long[twoB * C * S];
        for (int i = 0; i < ids.Length; i++) ids[i] = (i * 7 + 3) % 1024;   // arbitrary valid ids
        var amask = new bool[twoB * S]; for (int i = 0; i < amask.Length; i++) amask[i] = i % 2 == 0;
        var attn = new bool[twoB * S * S]; Array.Fill(attn, true);
        Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long> IdsT() => new(ids, new[] { twoB, C, S });
        Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<bool> MaskT() => new(amask, new[] { twoB, S });
        Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<bool> AttnT() => new(attn, new[] { twoB, 1, S, S });
        float[] Run(Microsoft.ML.OnnxRuntime.InferenceSession s)
        {
            using var o = s.Run(new[]
            {
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("input_ids", IdsT()),
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("audio_mask", MaskT()),
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("attention_mask", AttnT()),
            });
            return o.First(v => v.Name == "logits").AsTensor<float>().ToArray();
        }
        float[] a = Run(folded), b = Run(merged);
        int V = 1025, n = a.Length / V, agree = 0;
        double maxd = 0;
        for (int p = 0; p < n; p++)
        {
            int am = 0, bm = 0; float av = float.MinValue, bv = float.MinValue;
            for (int v = 0; v < V; v++)
            {
                if (a[p * V + v] > av) { av = a[p * V + v]; am = v; }
                if (b[p * V + v] > bv) { bv = b[p * V + v]; bm = v; }
                maxd = Math.Max(maxd, Math.Abs(a[p * V + v] - b[p * V + v]));
            }
            if (am == bm) agree++;
        }
        Console.WriteLine($"  argmax agreement: {100.0 * agree / n:F3}%   max|Δlogit|: {maxd:E2}");
        bool pass = agree == n;
        Console.WriteLine(pass ? "  PASS" : "  CHECK");
        return pass ? 0 : 1;
    }
}
