using System.Text.Json;
using Chatterbox.Base;
using Chatterbox.Base.Tokenization;
using Xunit;

namespace Chatterbox.Tests;

/// <summary>
/// Stage-C parity: the C# duration estimator and text-prep match the Python reference.
/// Duration is a pure function (no model). The prepare-inputs cases need the Qwen3
/// tokenizer.json (gated on OMNIVOICE_MODEL_DIR / default path). Fixtures from
/// scripts/omnivoice_export/dump_textprep_fixtures.py.
/// </summary>
public class OmniVoiceTextPrepTests
{
    private static string FixturePath() =>
        Path.Combine(AppContext.BaseDirectory, "fixtures", "omnivoice_textprep_fixtures.json");

    private static JsonElement Root() =>
        JsonDocument.Parse(File.ReadAllText(FixturePath())).RootElement;

    private static string? FindTokenizerJson()
    {
        foreach (var dir in new[]
                 {
                     Environment.GetEnvironmentVariable("OMNIVOICE_MODEL_DIR"),
                     "/mnt/data/models/omnivoice/k2-fsa-OmniVoice",
                 })
        {
            if (string.IsNullOrEmpty(dir)) continue;
            var p = Path.Combine(dir, "tokenizer.json");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    [Fact]
    public void Duration_estimate_matches_python()
    {
        var failures = new List<string>();
        foreach (var c in Root().GetProperty("duration_cases").EnumerateArray())
        {
            string target = c.GetProperty("target").GetString()!;
            string refText = c.GetProperty("ref").GetString()!;
            int refTokens = c.GetProperty("ref_tokens").GetInt32();
            double expected = c.GetProperty("estimate").GetDouble();
            int expectedTokens = c.GetProperty("target_tokens").GetInt32();

            double got = OmniVoiceDuration.EstimateDuration(target, refText, refTokens);
            int gotTokens = Math.Max(1, (int)got);
            if (Math.Abs(got - expected) > 1e-4)
                failures.Add($"estimate {Trunc(target)}: expected {expected:f6}, got {got:f6}");
            if (gotTokens != expectedTokens)
                failures.Add($"tokens {Trunc(target)}: expected {expectedTokens}, got {gotTokens}");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void Prepare_inputs_row0_and_audiomask_match_python()
    {
        var tokPath = FindTokenizerJson();
        if (tokPath is null) Assert.Skip("OmniVoice tokenizer.json not found (set OMNIVOICE_MODEL_DIR).");
        var prep = new OmniVoiceTextPrep(new Qwen3Tokenizer(tokPath));

        var failures = new List<string>();
        foreach (var c in Root().GetProperty("prepare_cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            string? lang = c.GetProperty("lang").ValueKind == JsonValueKind.Null ? null : c.GetProperty("lang").GetString();
            string? instruct = c.GetProperty("instruct").ValueKind == JsonValueKind.Null ? null : c.GetProperty("instruct").GetString();
            bool denoise = c.GetProperty("denoise").GetBoolean();
            int numTarget = c.GetProperty("num_target").GetInt32();
            var expRow0 = c.GetProperty("row0_ids").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            var expMask = c.GetProperty("audio_mask").EnumerateArray().Select(x => x.GetInt32()).ToArray();

            var p = prep.Prepare(text, numTarget, refText: null, refCodes: null, lang, instruct, denoise);
            var gotRow0 = new long[p.Total];
            for (int i = 0; i < p.Total; i++) gotRow0[i] = p.InputIds[0, i];
            var gotMask = p.AudioMask.Select(b => b ? 1 : 0).ToArray();

            if (!gotRow0.SequenceEqual(expRow0))
                failures.Add($"row0 {Trunc(text)}: expected len {expRow0.Length} [{string.Join(",", expRow0.Take(12))}...], got len {gotRow0.Length} [{string.Join(",", gotRow0.Take(12))}...]");
            if (!gotMask.SequenceEqual(expMask))
                failures.Add($"audio_mask {Trunc(text)}: mismatch (expected sum {expMask.Sum()}, got {gotMask.Sum()})");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    private static string Trunc(string s) => "\"" + (s.Length > 24 ? s[..24] + "…" : s) + "\"";
}
