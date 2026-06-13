using System.Runtime.InteropServices;
using System.Text.Json;
using Chatterbox.Base;
using Chatterbox.Base.Tokenization;
using Vernacula.Base.Models;
using Xunit;

namespace Chatterbox.Tests;

/// <summary>
/// Stage-D parity: the C# greedy diffusion loop reproduces the Phase-1 capture. Two checks:
///   (1) the conditioning built from text (with the reference codes) exactly equals the
///       captured step-0 cond — validates the ref-mode text prep, and
///   (2) the generated token field matches the captured final tokens.
/// The capture is from the PyTorch run; the C# loop uses the ONNX transformer, so a handful
/// of tokens may differ from numerical drift (cf. infer_onnx end-to-end log-spectral 0.0012).
/// We require a very high match rate rather than bit-exactness against the PyTorch reference.
/// Heavy (loads the 2.45 GB transformer); opt-in via OMNIVOICE_RUN_GRAPH_PARITY=1.
/// </summary>
public class OmniVoiceLoopParityTests
{
    private static string? RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Vernacula.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    private static string OnnxDir() =>
        Environment.GetEnvironmentVariable("OMNIVOICE_ONNX_DIR")
        ?? Path.Combine(RepoRoot() ?? ".", "scripts/omnivoice_export/onnx");

    private static string FixturesDir() =>
        Environment.GetEnvironmentVariable("OMNIVOICE_FIXTURES_DIR")
        ?? Path.Combine(RepoRoot() ?? ".", "scripts/omnivoice_export/capture/csharp_fixtures");

    private static string? FindTokenizerJson()
    {
        foreach (var dir in new[] { Environment.GetEnvironmentVariable("OMNIVOICE_MODEL_DIR"),
                                    "/mnt/data/models/omnivoice/k2-fsa-OmniVoice" })
        {
            if (string.IsNullOrEmpty(dir)) continue;
            var p = Path.Combine(dir, "tokenizer.json");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    private static long[] I64(string dir, string n) =>
        MemoryMarshal.Cast<byte, long>(File.ReadAllBytes(Path.Combine(dir, n + ".bin"))).ToArray();

    [Fact]
    public void Greedy_loop_reproduces_capture()
    {
        if (Environment.GetEnvironmentVariable("OMNIVOICE_RUN_GRAPH_PARITY") != "1")
            Assert.Skip("Set OMNIVOICE_RUN_GRAPH_PARITY=1 to run the heavy OmniVoice loop-parity test.");

        var fxDir = FixturesDir();
        var manifestPath = Path.Combine(fxDir, "manifest.json");
        var tokPath = FindTokenizerJson();
        if (!File.Exists(manifestPath) || tokPath is null
            || !File.Exists(Path.Combine(OnnxDir(), OmniVoice.TransformerFile)))
            Assert.Skip("OmniVoice fixtures / tokenizer / graphs not found.");

        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var arrays = doc.RootElement.GetProperty("arrays");
        var prm = doc.RootElement.GetProperty("params");
        int[] Shape(string n) => arrays.GetProperty(n).GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();

        string text = prm.GetProperty("text").GetString()!;
        string refText = prm.GetProperty("ref_text").GetString()!;
        string lang = prm.GetProperty("lang_resolved").GetString()!;
        int numStep = prm.GetProperty("num_step").GetInt32();
        float guidance = (float)prm.GetProperty("guidance_scale").GetDouble();

        // reference codes [8, Tref] and the captured final tokens [8, T]
        int tref = Shape("enc_audio_codes")[2];
        var encFlat = I64(fxDir, "enc_audio_codes");
        var refCodes = new long[OmniVoice.NumCodebooks, tref];
        for (int c = 0; c < OmniVoice.NumCodebooks; c++)
            for (int t = 0; t < tref; t++) refCodes[c, t] = encFlat[c * tref + t];

        int targetT = Shape("dec_audio_codes")[2];
        var finalFlat = I64(fxDir, "dec_audio_codes");

        // captured step-0 cond batch row 0 = the single-item cond [8, S]
        int condLen = Shape("tf_input_ids")[2];
        var tfIds = I64(fxDir, "tf_input_ids"); // [2,8,S]

        using var tts = new OmniVoiceTts(OnnxDir(), tokPath, ExecutionProvider.Cpu);

        // (1) cond built from text must equal the captured cond.
        var prep = new OmniVoiceTextPrep(new Qwen3Tokenizer(tokPath));
        var cond = prep.Prepare(text, targetT, refText, refCodes, lang, instruct: null, denoise: true);
        Assert.Equal(condLen, cond.Total);
        int condMismatch = 0;
        for (int c = 0; c < OmniVoice.NumCodebooks; c++)
            for (int p = 0; p < condLen; p++)
                if (cond.InputIds[c, p] != tfIds[(0 * OmniVoice.NumCodebooks + c) * condLen + p])
                    condMismatch++;
        Assert.True(condMismatch == 0, $"cond build differs from capture in {condMismatch} positions");

        // (2) greedy loop token field vs captured final tokens.
        var tokens = tts.RunDiffusion(cond,
            new OmniVoiceTts.GenConfig(NumStep: numStep, GuidanceScale: guidance,
                                       TShift: 0.1f, LayerPenaltyFactor: 5.0f, Denoise: true));

        int total = OmniVoice.NumCodebooks * targetT, match = 0;
        for (int c = 0; c < OmniVoice.NumCodebooks; c++)
            for (int t = 0; t < targetT; t++)
                if (tokens[c, t] == finalFlat[c * targetT + t]) match++;
        double rate = (double)match / total;
        Assert.True(rate > 0.98, $"token-field match rate {rate:f4} (vs PyTorch capture) below 0.98");
    }
}
