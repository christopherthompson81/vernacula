using System.Text.RegularExpressions;
using Chatterbox.Base.Tokenization;

namespace Chatterbox.Base;

/// <summary>
/// Builds the diffusion transformer's conditioning inputs from text — the C# port of
/// OmniVoice's _combine_text and _prepare_inference_inputs (omnivoice/models/omnivoice.py).
///
/// Layout of input_ids [8, total] = style tokens ++ text tokens ++ [ref audio codes] ++
/// masked target. Style/text positions carry the same text-token id across all 8 codebook
/// rows (the graph reads row 0 for text); audio positions carry per-codebook codes (ref) or
/// the mask id (target). audio_mask is true over the ref+target span. Validated by
/// OmniVoiceTextPrepTests.
/// </summary>
public sealed partial class OmniVoiceTextPrep
{
    private readonly Qwen3Tokenizer _tok;

    public OmniVoiceTextPrep(Qwen3Tokenizer tokenizer) => _tok = tokenizer;

    public sealed record Prepared(long[,] InputIds, bool[] AudioMask, int TextLen, int RefLen, int TargetLen)
    {
        public int Total => TextLen + RefLen + TargetLen;
    }

    /// <summary>Build the conditioning input for one item. <paramref name="refCodes"/> is the
    /// reference audio codes [8, Tref] for voice cloning, or null for design/auto mode.</summary>
    public Prepared Prepare(string text, int numTargetTokens, string? refText,
                            long[,]? refCodes, string? lang, string? instruct, bool denoise)
    {
        int[] styleIds = _tok.Encode(BuildStyleText(lang, instruct, denoise, refCodes is not null));
        int[] textIds = _tok.EncodeWithNonverbalTags($"<|text_start|>{CombineText(text, refText)}<|text_end|>");

        int textLen = styleIds.Length + textIds.Length;
        int refLen = refCodes?.GetLength(1) ?? 0;
        int total = textLen + refLen + numTargetTokens;

        var ids = new long[OmniVoice.NumCodebooks, total];
        // style + text: same id across all 8 rows
        for (int c = 0; c < OmniVoice.NumCodebooks; c++)
        {
            int col = 0;
            foreach (var id in styleIds) ids[c, col++] = id;
            foreach (var id in textIds) ids[c, col++] = id;
        }
        // ref audio codes: per-codebook
        if (refCodes is not null)
            for (int c = 0; c < OmniVoice.NumCodebooks; c++)
                for (int t = 0; t < refLen; t++)
                    ids[c, textLen + t] = refCodes[c, t];
        // target: mask id across all rows
        for (int c = 0; c < OmniVoice.NumCodebooks; c++)
            for (int t = 0; t < numTargetTokens; t++)
                ids[c, textLen + refLen + t] = OmniVoice.AudioMaskId;

        var audioMask = new bool[total];
        for (int i = textLen; i < total; i++) audioMask[i] = true;

        return new Prepared(ids, audioMask, textLen, refLen, numTargetTokens);
    }

    /// <summary>The style prefix: [&lt;|denoise|&gt;] &lt;|lang_start|&gt;LANG&lt;|lang_end|&gt;
    /// &lt;|instruct_start|&gt;INSTRUCT&lt;|instruct_end|&gt;. denoise only when cloning.</summary>
    public static string BuildStyleText(string? lang, string? instruct, bool denoise, bool hasRef)
    {
        string s = denoise && hasRef ? "<|denoise|>" : "";
        s += $"<|lang_start|>{(string.IsNullOrEmpty(lang) ? "None" : lang)}<|lang_end|>";
        s += $"<|instruct_start|>{(string.IsNullOrEmpty(instruct) ? "None" : instruct)}<|instruct_end|>";
        return s;
    }

    /// <summary>Port of _combine_text: join ref+target, strip newlines, normalise full-width
    /// parens and runs of spaces, and drop spaces adjacent to CJK characters.</summary>
    public static string CombineText(string text, string? refText)
    {
        string full = !string.IsNullOrEmpty(refText)
            ? refText.Trim() + " " + text.Trim()
            : text.Trim();
        full = NewlineRegex().Replace(full, "");
        full = full.Replace('（', '(').Replace('）', ')');
        full = SpacesRegex().Replace(full, " ");
        full = CjkSpaceRegex().Replace(full, "");
        return full;
    }

    // End-of-sentence punctuation (utils/text.py END_PUNCTUATION) — if the text already ends in
    // one of these, add_punctuation is a no-op.
    private static readonly HashSet<char> EndPunctuation = new(
        new[] { ';', ':', ',', '.', '!', '?', '…', ')', ']', '}', '"', '\'', '“', '”',
                '‘', '’', '；', '：', '，', '。', '！', '？', '、', '）', '】' });

    /// <summary>Port of add_punctuation: append a period (or Chinese 。 if the text contains any
    /// CJK char) when it doesn't already end in sentence punctuation. Python applies this to the
    /// reference transcript in create_voice_clone_prompt, so the punctuated ref_text feeds BOTH
    /// the duration estimate and the combined text.</summary>
    public static string AddPunctuation(string text)
    {
        text = text.Trim();
        if (text.Length == 0) return text;
        if (EndPunctuation.Contains(text[^1])) return text;
        bool isChinese = false;
        foreach (var ch in text) if (ch >= '一' && ch <= '鿿') { isChinese = true; break; }
        return text + (isChinese ? "。" : ".");
    }

    [GeneratedRegex(@"[\r\n]+")] private static partial Regex NewlineRegex();
    [GeneratedRegex(@"[ \t]+")] private static partial Regex SpacesRegex();
    [GeneratedRegex(@"(?<=[一-鿿])\s+|\s+(?=[一-鿿])")] private static partial Regex CjkSpaceRegex();
}
