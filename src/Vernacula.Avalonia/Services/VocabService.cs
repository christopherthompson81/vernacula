using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using Avalonia.Media;
using Vernacula.Base;

namespace Vernacula.App.Services;

/// <summary>
/// Loads the Parakeet vocabulary and provides token→text decoding and
/// confidence-to-colour mapping for the transcript editor.
/// </summary>
internal class VocabService
{
    private enum VocabKind { Parakeet, Cohere, VibeVoice }

    private readonly Dictionary<int, string> _vocab;
    private readonly VocabKind _kind;
    private readonly Dictionary<int, string> _addedContent = [];
    private readonly Dictionary<char, byte> _byteLevelDecode = [];

    public VocabService(string modelsDir, string? asrModel = null)
    {
        if (string.Equals(asrModel, "CohereLabs/cohere-transcribe-03-2026", StringComparison.Ordinal))
        {
            _kind = VocabKind.Cohere;
            _vocab = LoadCohereVocab(Path.Combine(modelsDir, "cohere_transcribe", CohereTranscribe.VocabFile));
        }
        else if (string.Equals(asrModel, "vibevoice/vibevoice-asr", StringComparison.Ordinal))
        {
            _kind = VocabKind.VibeVoice;
            (_vocab, _addedContent) = LoadVibeVoiceVocab(Path.Combine(modelsDir, Config.VibeVoiceSubDir, VibeVoiceAsr.TokenizerFile));
            _byteLevelDecode = BuildByteLevelDecode();
        }
        else
        {
            _kind = VocabKind.Parakeet;
            _vocab = LoadParakeetVocab(Path.Combine(modelsDir, Config.VocabFile));
        }
    }

    // ── Vocab loading (mirrors Parakeet.GetVocab) ────────────────────────────

    private static Dictionary<int, string> LoadParakeetVocab(string vocabPath)
    {
        var vocab = new Dictionary<int, string>();
        if (!File.Exists(vocabPath)) return vocab;

        foreach (var line in File.ReadLines(vocabPath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            int lastSpace = line.LastIndexOf(' ');
            if (lastSpace < 0) continue;
            string token = line[..lastSpace].Replace("\u2581", " ");
            if (int.TryParse(line[(lastSpace + 1)..], out int id))
                vocab[id] = token;
        }
        return vocab;
    }

    private static Dictionary<int, string> LoadCohereVocab(string vocabPath)
    {
        var vocab = new Dictionary<int, string>();
        if (!File.Exists(vocabPath)) return vocab;

        string json = File.ReadAllText(vocabPath);
        var items = JsonSerializer.Deserialize<string[]>(json);
        if (items is null) return vocab;

        for (int i = 0; i < items.Length; i++)
            vocab[i] = items[i];
        return vocab;
    }

    private static (Dictionary<int, string> vocab, Dictionary<int, string> addedContent) LoadVibeVoiceVocab(string vocabPath)
    {
        var vocab = new Dictionary<int, string>();
        var addedContent = new Dictionary<int, string>();
        if (!File.Exists(vocabPath)) return (vocab, addedContent);

        using var doc = JsonDocument.Parse(File.ReadAllText(vocabPath));
        var root = doc.RootElement;

        foreach (var kv in root.GetProperty("model").GetProperty("vocab").EnumerateObject())
            vocab[kv.Value.GetInt32()] = kv.Name;

        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var at in addedTokens.EnumerateArray())
                addedContent[at.GetProperty("id").GetInt32()] = at.GetProperty("content").GetString() ?? "";
        }

        return (vocab, addedContent);
    }

    // ── Token decoding ────────────────────────────────────────────────────────

    public string DecodeTokens(IReadOnlyList<int> tokens)
    {
        return _kind switch
        {
            VocabKind.Cohere   => DecodeCohereTokens(tokens),
            VocabKind.VibeVoice => DecodeVibeVoiceTokens(tokens),
            _                  => DecodeParakeetTokens(tokens),
        };
    }

    /// <summary>Returns (text, logprob) pairs for each token.</summary>
    public IReadOnlyList<(string text, float logprob)> GetTokenRuns(
        IReadOnlyList<int> tokens, IReadOnlyList<float> logprobs)
    {
        if (_kind == VocabKind.Cohere)
            return GetCohereTokenRuns(tokens, logprobs);
        if (_kind == VocabKind.VibeVoice)
            return GetVibeVoiceTokenRuns(tokens, logprobs);

        var runs = new List<(string, float)>(tokens.Count);
        for (int i = 0; i < tokens.Count; i++)
        {
            string text = DecodeToken(tokens[i]);
            float  lp   = i < logprobs.Count ? logprobs[i] : 0f;
            runs.Add((text, lp));
        }
        return runs;
    }

    private IReadOnlyList<(string text, float logprob)> GetCohereTokenRuns(
        IReadOnlyList<int> tokens, IReadOnlyList<float> logprobs)
    {
        var runs = new List<(string, float)>(tokens.Count);
        Decoder decoder = Encoding.UTF8.GetDecoder();
        bool stripLeadingSpace = true;

        for (int i = 0; i < tokens.Count; i++)
        {
            string runText = DecodeCohereTokenIncrement(decoder, tokens[i], ref stripLeadingSpace);
            float lp = i < logprobs.Count ? logprobs[i] : 0f;
            runs.Add((runText, lp));
        }

        return runs;
    }

    private string DecodeParakeetTokens(IReadOnlyList<int> tokens)
    {
        var sb = new StringBuilder();
        foreach (var t in tokens)
            if (_vocab.TryGetValue(t, out var s)) sb.Append(s);
        return sb.ToString().Trim();
    }

    private string DecodeCohereTokens(IReadOnlyList<int> tokens)
    {
        var bytes = new List<byte>(tokens.Count * 4);
        foreach (int token in tokens)
        {
            if (!_vocab.TryGetValue(token, out var value)) continue;

            if (value.Length == 6 && value[0] == '<' && value[1] == '0' && value[2] == 'x' && value[5] == '>')
            {
                if (TryParseHexByte(value[3], value[4], out byte b))
                {
                    bytes.Add(b);
                    continue;
                }
            }

            bytes.AddRange(Encoding.UTF8.GetBytes(value.Replace('\u2581', ' ')));
        }

        string text = Encoding.UTF8.GetString(bytes.ToArray());
        return text.Length > 0 && text[0] == ' ' ? text[1..] : text;
    }

    private string DecodeVibeVoiceTokens(IReadOnlyList<int> tokens)
    {
        var bytes = new List<byte>(tokens.Count * 4);
        foreach (int token in tokens)
            bytes.AddRange(GetVibeVoiceTokenBytes(token));
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private string DecodeToken(int token)
    {
        if (!_vocab.TryGetValue(token, out var value))
            return "";

        if (_kind == VocabKind.Parakeet)
            return value;

        if (_kind == VocabKind.VibeVoice)
            return Encoding.UTF8.GetString(GetVibeVoiceTokenBytes(token));

        if (value.Length == 6 && value[0] == '<' && value[1] == '0' && value[2] == 'x' && value[5] == '>')
        {
            if (TryParseHexByte(value[3], value[4], out byte b))
                return Encoding.UTF8.GetString([b]);
        }

        return value.Replace('\u2581', ' ');
    }

    private string DecodeCohereTokenIncrement(Decoder decoder, int token, ref bool stripLeadingSpace)
    {
        byte[] tokenBytes = GetCohereTokenBytes(token);
        if (tokenBytes.Length == 0)
            return "";

        int charCount = decoder.GetCharCount(tokenBytes, 0, tokenBytes.Length, flush: false);
        if (charCount == 0)
            return "";

        char[] chars = new char[charCount];
        decoder.GetChars(tokenBytes, 0, tokenBytes.Length, chars, 0, flush: false);
        string text = new(chars);

        if (stripLeadingSpace && text.Length > 0)
        {
            stripLeadingSpace = false;
            if (text[0] == ' ')
                text = text[1..];
        }

        return text;
    }

    private byte[] GetCohereTokenBytes(int token)
    {
        if (!_vocab.TryGetValue(token, out var value))
            return [];

        if (value.Length == 6 && value[0] == '<' && value[1] == '0' && value[2] == 'x' && value[5] == '>')
        {
            if (TryParseHexByte(value[3], value[4], out byte b))
                return [b];
        }

        return Encoding.UTF8.GetBytes(value.Replace('\u2581', ' '));
    }

    private IReadOnlyList<(string text, float logprob)> GetVibeVoiceTokenRuns(
        IReadOnlyList<int> tokens, IReadOnlyList<float> logprobs)
    {
        var runs = new List<(string, float)>(tokens.Count);
        Decoder decoder = Encoding.UTF8.GetDecoder();

        for (int i = 0; i < tokens.Count; i++)
        {
            byte[] tokenBytes = GetVibeVoiceTokenBytes(tokens[i]);
            string runText;
            if (tokenBytes.Length == 0)
            {
                runText = "";
            }
            else
            {
                int charCount = decoder.GetCharCount(tokenBytes, 0, tokenBytes.Length, flush: false);
                char[] chars = new char[charCount];
                decoder.GetChars(tokenBytes, 0, tokenBytes.Length, chars, 0, flush: false);
                runText = new string(chars);
            }

            float lp = i < logprobs.Count ? logprobs[i] : 0f;
            runs.Add((runText, lp));
        }

        return runs;
    }

    private byte[] GetVibeVoiceTokenBytes(int token)
    {
        if (_addedContent.TryGetValue(token, out var special))
            return Encoding.UTF8.GetBytes(special);

        if (!_vocab.TryGetValue(token, out var raw))
            return [];

        var bytes = new List<byte>(raw.Length);
        foreach (char ch in raw)
            if (_byteLevelDecode.TryGetValue(ch, out byte b))
                bytes.Add(b);
        return bytes.ToArray();
    }

    private static bool TryParseHexByte(char hi, char lo, out byte value)
    {
        int h = HexVal(hi);
        int l = HexVal(lo);
        if (h < 0 || l < 0) { value = 0; return false; }
        value = (byte)((h << 4) | l);
        return true;
    }

    private static int HexVal(char c) => c switch
    {
        >= '0' and <= '9' => c - '0',
        >= 'a' and <= 'f' => c - 'a' + 10,
        >= 'A' and <= 'F' => c - 'A' + 10,
        _ => -1,
    };

    private static Dictionary<char, byte> BuildByteLevelDecode()
    {
        var printable = new HashSet<int>(
            Enumerable.Range(33, 94)
            .Concat(Enumerable.Range(161, 12))
            .Concat(Enumerable.Range(174, 82)));

        var dict = new Dictionary<char, byte>(280);

        foreach (int b in printable)
            dict[(char)b] = (byte)b;

        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!printable.Contains(b))
                dict[(char)(0x100 + extra++)] = (byte)b;
        }

        return dict;
    }

    // ── Confidence highlight ──────────────────────────────────────────────────

    /// <summary>
    /// Returns a semi-transparent background highlight colour for a token based on its logprob.
    /// High confidence → fully transparent; low confidence → opaque ConfidenceLowColor tint.
    /// logprob range [-2, 0]; values below -2 are clamped to maximum opacity.
    /// Uses a quadratic curve so only genuinely uncertain tokens get a visible tint.
    /// </summary>
    public static Color GetConfidenceHighlight(float logprob, Color highlightColor)
    {
        const float minLogprob = -2f;
        float linear = Math.Clamp((logprob - minLogprob) / (-minLogprob), 0f, 1f);
        // t = 1 high confidence, t = 0 low confidence
        float t     = (float)Math.Sqrt(linear);
        byte  alpha = (byte)((1f - t * t) * 200f); // quadratic: more opacity for very low confidence
        return Color.FromArgb(alpha, highlightColor.R, highlightColor.G, highlightColor.B);
    }
}
