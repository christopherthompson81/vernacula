using System.Text.Json;
using Chatterbox.Base.Tokenization;
using Xunit;

namespace Chatterbox.Tests;

/// <summary>
/// Exact-match parity for the C# <see cref="Qwen3Tokenizer"/> against the Python
/// (HuggingFace) Qwen3 tokenizer used by OmniVoice. Fixtures are produced by
/// scripts/omnivoice_export/dump_tokenizer_fixtures.py. Needs the model's tokenizer.json;
/// gated on OMNIVOICE_MODEL_DIR (falls back to the default local path), skipped if absent.
/// </summary>
public class Qwen3TokenizerParityTests
{
    private static string? FindTokenizerJson()
    {
        var candidates = new[]
        {
            Environment.GetEnvironmentVariable("OMNIVOICE_MODEL_DIR"),
            "/mnt/data/models/omnivoice/k2-fsa-OmniVoice",
        };
        foreach (var dir in candidates)
        {
            if (string.IsNullOrEmpty(dir)) continue;
            var path = Path.Combine(dir, "tokenizer.json");
            if (File.Exists(path)) return path;
        }
        return null;
    }

    private static string FixturePath() =>
        Path.Combine(AppContext.BaseDirectory, "fixtures", "omnivoice_tokenizer_fixtures.json");

    private sealed record Case(string Text, int[] Ids);

    private static (List<Case> cases, List<Case> nonverbal) LoadFixtures()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(FixturePath()));
        var root = doc.RootElement;

        static List<Case> Read(JsonElement arr, string idsKey) =>
            arr.EnumerateArray().Select(c => new Case(
                c.GetProperty("text").GetString()!,
                c.GetProperty(idsKey).EnumerateArray().Select(x => x.GetInt32()).ToArray()))
            .ToList();

        return (Read(root.GetProperty("cases"), "ids"),
                Read(root.GetProperty("nonverbal_tag_cases"), "ids"));
    }

    [Fact]
    public void Encode_matches_python_exactly()
    {
        var tokPath = FindTokenizerJson();
        if (tokPath is null)
            Assert.Skip("OmniVoice tokenizer.json not found (set OMNIVOICE_MODEL_DIR).");

        var tok = new Qwen3Tokenizer(tokPath);
        var (cases, nonverbal) = LoadFixtures();

        var failures = new List<string>();
        foreach (var c in cases)
        {
            var got = tok.Encode(c.Text);
            if (!got.SequenceEqual(c.Ids))
                failures.Add($"Encode {Escape(c.Text)}\n  expected [{string.Join(",", c.Ids)}]\n  got      [{string.Join(",", got)}]");
        }
        foreach (var c in nonverbal)
        {
            var got = tok.EncodeWithNonverbalTags(c.Text);
            if (!got.SequenceEqual(c.Ids))
                failures.Add($"EncodeWithNonverbalTags {Escape(c.Text)}\n  expected [{string.Join(",", c.Ids)}]\n  got      [{string.Join(",", got)}]");
        }

        Assert.True(failures.Count == 0,
            $"{failures.Count}/{cases.Count + nonverbal.Count} tokenizer parity failures:\n"
            + string.Join("\n", failures));
    }

    [Fact]
    public void Special_token_ids_resolve()
    {
        var tokPath = FindTokenizerJson();
        if (tokPath is null)
            Assert.Skip("OmniVoice tokenizer.json not found (set OMNIVOICE_MODEL_DIR).");

        var tok = new Qwen3Tokenizer(tokPath);
        Assert.Equal(151669, tok.TokenId("<|denoise|>"));
        Assert.Equal(151674, tok.TokenId("<|text_start|>"));
        Assert.Equal(151675, tok.TokenId("<|text_end|>"));
    }

    private static string Escape(string s) =>
        "\"" + s.Replace("\\", "\\\\").Replace("\n", "\\n").Replace("\t", "\\t") + "\"";
}
