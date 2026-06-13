using System.Runtime.InteropServices;
using System.Text.Json;
using Chatterbox.Base;
using Vernacula.Base.Models;
using Xunit;

namespace Chatterbox.Tests;

/// <summary>
/// Stage-B parity: each C# OmniVoice ONNX graph reproduces the Phase-1 Python capture
/// on CPU / full fp32 (the TF32-compounding lesson — see docs/omnivoice_onnx_investigation.md).
/// Heavy integration test (loads the 2.45 GB transformer); gated on the exported graphs +
/// C#-readable fixtures (scripts/omnivoice_export/dump_csharp_fixtures.py), skipped if absent.
/// Override locations with OMNIVOICE_ONNX_DIR / OMNIVOICE_FIXTURES_DIR.
/// </summary>
public class OmniVoiceGraphParityTests
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

    private sealed record Fixtures(string Dir, Dictionary<string, int[]> Shapes)
    {
        public float[] F32(string n) => ReadF32(Path.Combine(Dir, n + ".bin"));
        public long[] I64(string n) => ReadI64(Path.Combine(Dir, n + ".bin"));
        public bool[] Bool(string n) => Array.ConvertAll(File.ReadAllBytes(Path.Combine(Dir, n + ".bin")), b => b != 0);
        public int[] Shape(string n) => Shapes[n];
    }

    private static float[] ReadF32(string p) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(p)).ToArray();
    private static long[] ReadI64(string p) => MemoryMarshal.Cast<byte, long>(File.ReadAllBytes(p)).ToArray();

    private static Fixtures? LoadFixtures()
    {
        var dir = FixturesDir();
        var manifestPath = Path.Combine(dir, "manifest.json");
        if (!File.Exists(manifestPath)) return null;
        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var shapes = new Dictionary<string, int[]>();
        foreach (var p in doc.RootElement.EnumerateObject())
            shapes[p.Name] = p.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        return new Fixtures(dir, shapes);
    }

    // Heavy (loads the 2.45 GB transformer on CPU): opt-in only, so the normal suite stays fast.
    private static void RequireOptIn()
    {
        if (Environment.GetEnvironmentVariable("OMNIVOICE_RUN_GRAPH_PARITY") != "1")
            Assert.Skip("Set OMNIVOICE_RUN_GRAPH_PARITY=1 to run the heavy OmniVoice graph-parity test.");
    }

    private static OmniVoice? TryLoad()
    {
        var onnxDir = OnnxDir();
        if (!File.Exists(Path.Combine(onnxDir, OmniVoice.TransformerFile))) return null;
        return new OmniVoice(onnxDir, ExecutionProvider.Cpu);
    }

    [Fact]
    public void Encoder_codes_match_capture_exactly()
    {
        RequireOptIn();
        var fx = LoadFixtures();
        if (fx is null) Assert.Skip("OmniVoice C# fixtures not found (run dump_csharp_fixtures.py).");
        using var om = TryLoad();
        if (om is null) Assert.Skip("OmniVoice ONNX graphs not found (run export_omnivoice.py).");

        var wav = fx.F32("enc_input_values");
        var got = om.EncodeRaw(wav, out int tc);
        var expected = fx.I64("enc_audio_codes");

        Assert.Equal(fx.Shape("enc_audio_codes")[2], tc);
        Assert.True(got.SequenceEqual(expected),
            $"encoder code mismatch: {got.Zip(expected).Count(p => p.First != p.Second)}/{expected.Length} differ");
    }

    [Fact]
    public void Decoder_waveform_matches_capture()
    {
        RequireOptIn();
        var fx = LoadFixtures();
        if (fx is null) Assert.Skip("OmniVoice C# fixtures not found.");
        using var om = TryLoad();
        if (om is null) Assert.Skip("OmniVoice ONNX graphs not found.");

        var codes = fx.I64("dec_audio_codes");
        int tc = fx.Shape("dec_audio_codes")[2];
        var got = om.DecodeRaw(codes, tc);
        var expected = fx.F32("dec_audio_values");

        Assert.Equal(expected.Length, got.Length);
        double maxAbs = got.Zip(expected).Max(p => Math.Abs(p.First - p.Second));
        Assert.True(maxAbs < 1e-2, $"decoder waveform max-abs {maxAbs:e3} exceeds 1e-2");
    }

    [Fact]
    public void Transformer_logits_match_capture()
    {
        RequireOptIn();
        var fx = LoadFixtures();
        if (fx is null) Assert.Skip("OmniVoice C# fixtures not found.");
        using var om = TryLoad();
        if (om is null) Assert.Skip("OmniVoice ONNX graphs not found.");

        var ids = fx.I64("tf_input_ids");
        var amask = fx.Bool("tf_audio_mask");
        var attn = fx.Bool("tf_attention_mask");
        int twoB = fx.Shape("tf_input_ids")[0], seq = fx.Shape("tf_input_ids")[2];

        var got = om.RunTransformer(ids, amask, attn, twoB, seq);
        var expected = fx.F32("tf_logits");
        Assert.Equal(expected.Length, got.Length);

        // argmax agreement over the vocab axis + max-abs (CPU↔CPU should be ~1e-4).
        int V = OmniVoice.AudioVocabSize;
        int slots = got.Length / V;
        int agree = 0;
        double maxAbs = 0;
        for (int s = 0; s < slots; s++)
        {
            int baseI = s * V;
            int aG = 0, aE = 0;
            float mG = float.NegativeInfinity, mE = float.NegativeInfinity;
            for (int v = 0; v < V; v++)
            {
                float g = got[baseI + v], e = expected[baseI + v];
                if (g > mG) { mG = g; aG = v; }
                if (e > mE) { mE = e; aE = v; }
                double d = Math.Abs(g - e);
                if (d > maxAbs) maxAbs = d;
            }
            if (aG == aE) agree++;
        }
        double agreement = (double)agree / slots;
        Assert.True(agreement > 0.9999 && maxAbs < 5e-3,
            $"transformer parity: argmax-agreement {agreement:f6}, max-abs {maxAbs:e3}");
    }
}
