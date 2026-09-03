using System.Text.Json;

namespace Vernacula.Tts.CLI;

/// <summary>
/// A reference voice that is already encoded — Higgs codec codes plus the IPA transcript they were
/// encoded against — loaded from the web demo's voice library.
///
/// ⚠ THIS EXISTS SO THE CLI AND THE BROWSER CAN RENDER THE SAME VOICE. The demo ships 288 voices as
/// a few KB of codes each; the source WAVs are deliberately not in the repo (see docs/voice_sourcing.md),
/// so <c>--voice</c>, which encodes a WAV, cannot reach any of them. Without this, auditing what a
/// visitor actually hears means driving a browser once per clip.
///
/// ⚠ AND IT NEEDS NO ENCODER. <c>higgs_encoder.onnx</c> is 654 MB and is loaded only to turn a WAV
/// into codes; a stored voice skips that step entirely.
/// </summary>
public sealed class StoredVoice
{
    public required string Id { get; init; }
    public required string Lang { get; init; }
    public required string Label { get; init; }
    /// <summary>The transcript AS IPA — already phonemized when the voice was encoded.</summary>
    public required string RefIpa { get; init; }
    /// <summary>Codec codes, [8, refLen] row-major — the layout <c>EncodeReference</c> also returns.</summary>
    public required long[,] Codes { get; init; }
    /// <summary>RMS of the source audio before boosting, for the output un-boost.</summary>
    public required float RefRms { get; init; }

    private const int NumCodebooks = 8;

    /// <summary>
    /// Load one voice by id. Returns null when <paramref name="voiceId"/> is "list", having printed
    /// the library — a lookup failure on 288 entries is otherwise a guessing game.
    /// </summary>
    public static StoredVoice? Load(string dir, string voiceId)
    {
        string voicesPath = Path.Combine(dir, "voices.jsonc");
        string codesPath = Path.Combine(dir, "voice-codes.json");
        if (!File.Exists(voicesPath) || !File.Exists(codesPath))
            throw new FileNotFoundException(
                $"voice library not found: expected voices.jsonc and voice-codes.json in {Path.GetFullPath(dir)}. "
                + "Pass --voice-lib <dir>.");

        // ⚠ voices.jsonc is JSONC — the comments in it carry the provenance of each voice and are
        // meant to stay. Skip them at parse time rather than asking the file to become JSON.
        var opts = new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };
        using var voicesDoc = JsonDocument.Parse(File.ReadAllText(voicesPath), opts);

        if (voiceId == "list")
        {
            foreach (var e in voicesDoc.RootElement.EnumerateArray())
                Console.WriteLine($"  {e.GetProperty("id").GetString(),-32} {e.GetProperty("lang").GetString(),-8} "
                    + $"{e.GetProperty("label").GetString()}");
            Console.WriteLine($"\n{voicesDoc.RootElement.GetArrayLength()} voices in {Path.GetFullPath(dir)}");
            return null;
        }

        JsonElement? hit = null;
        foreach (var e in voicesDoc.RootElement.EnumerateArray())
            if (e.GetProperty("id").GetString() == voiceId) { hit = e; break; }
        if (hit is null)
            throw new ArgumentException($"no voice with id \"{voiceId}\" in {voicesPath}. "
                + "Use `--voice-id list` to see what is there.");
        var v = hit.Value;

        int refLen = v.GetProperty("refLen").GetInt32();
        using var codesDoc = JsonDocument.Parse(File.ReadAllText(codesPath));
        if (!codesDoc.RootElement.TryGetProperty(voiceId, out var flat))
            throw new ArgumentException($"voice \"{voiceId}\" is listed in voices.jsonc but has no codes "
                + $"in {Path.GetFileName(codesPath)} — the two files are out of step.");

        // ⚠ THE LENGTH IS CHECKED, NOT ASSUMED. A flat array reshaped against the wrong refLen still
        // produces a valid-looking [8, T] block, and the failure is audible rather than raised —
        // exactly the kind of silent mislabelling a voice audit exists to catch.
        int n = flat.GetArrayLength();
        if (n != refLen * NumCodebooks)
            throw new ArgumentException($"voice \"{voiceId}\": {n} codes is not refLen {refLen} × {NumCodebooks} "
                + $"codebooks ({refLen * NumCodebooks}). The library is inconsistent.");

        var codes = new long[NumCodebooks, refLen];
        int i = 0;
        foreach (var c in flat.EnumerateArray()) { codes[i / refLen, i % refLen] = c.GetInt64(); i++; }

        return new StoredVoice
        {
            Id = voiceId,
            Lang = v.GetProperty("lang").GetString() ?? "",
            Label = v.TryGetProperty("label", out var lb) ? lb.GetString() ?? "" : "",
            RefIpa = v.GetProperty("refIpa").GetString() ?? "",
            Codes = codes,
            RefRms = v.TryGetProperty("refRms", out var rr) ? (float)rr.GetDouble() : 0f,
        };
    }
}
