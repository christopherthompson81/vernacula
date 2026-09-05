using System.Globalization;
using System.Text;

namespace Vernacula.Tts.Base;

/// <summary>
/// C# port of OmniVoice's RuleDurationEstimator (omnivoice/utils/duration.py): a rule-based
/// estimate of how many audio tokens a target text needs, by weighting each character by its
/// script's relative speaking time and scaling by the reference's tokens-per-weight rate.
/// "Duration" here is measured in audio-token counts. Validated against the Python estimator
/// by OmniVoiceTextPrepTests.
/// </summary>
public static class OmniVoiceDuration
{
    private static readonly Dictionary<string, double> Weights = new(StringComparer.Ordinal)
    {
        ["cjk"] = 3.0, ["hangul"] = 2.5, ["kana"] = 2.2, ["ethiopic"] = 3.0, ["yi"] = 3.0,
        ["indic"] = 1.8, ["thai_lao"] = 1.5, ["khmer_myanmar"] = 1.8, ["arabic"] = 1.5,
        ["hebrew"] = 1.5, ["latin"] = 1.0, ["cyrillic"] = 1.0, ["greek"] = 1.0,
        ["armenian"] = 1.0, ["georgian"] = 1.0, ["punctuation"] = 0.5, ["space"] = 0.2,
        ["digit"] = 3.5, ["mark"] = 0.0, ["default"] = 1.0,
    };

    // Unicode block end-codepoints (ascending) -> script type. Mirrors duration.py's `ranges`.
    private static readonly (int End, string Type)[] Ranges =
    {
        (0x02AF, "latin"), (0x03FF, "greek"), (0x052F, "cyrillic"), (0x058F, "armenian"),
        (0x05FF, "hebrew"), (0x077F, "arabic"), (0x089F, "arabic"), (0x08FF, "arabic"),
        (0x097F, "indic"), (0x09FF, "indic"), (0x0A7F, "indic"), (0x0AFF, "indic"),
        (0x0B7F, "indic"), (0x0BFF, "indic"), (0x0C7F, "indic"), (0x0CFF, "indic"),
        (0x0D7F, "indic"), (0x0DFF, "indic"), (0x0EFF, "thai_lao"), (0x0FFF, "indic"),
        (0x109F, "khmer_myanmar"), (0x10FF, "georgian"), (0x11FF, "hangul"), (0x137F, "ethiopic"),
        (0x139F, "ethiopic"), (0x13FF, "default"), (0x167F, "default"), (0x169F, "default"),
        (0x16FF, "default"), (0x171F, "default"), (0x173F, "default"), (0x175F, "default"),
        (0x177F, "default"), (0x17FF, "khmer_myanmar"), (0x18AF, "default"), (0x18FF, "default"),
        (0x194F, "indic"), (0x19DF, "indic"), (0x19FF, "khmer_myanmar"), (0x1A1F, "indic"),
        (0x1AAF, "indic"), (0x1B7F, "indic"), (0x1BBF, "indic"), (0x1BFF, "indic"),
        (0x1C4F, "indic"), (0x1C7F, "indic"), (0x1C8F, "cyrillic"), (0x1CBF, "georgian"),
        (0x1CCF, "indic"), (0x1CFF, "indic"), (0x1D7F, "latin"), (0x1DBF, "latin"),
        (0x1DFF, "default"), (0x1EFF, "latin"), (0x309F, "kana"), (0x30FF, "kana"),
        (0x312F, "cjk"), (0x318F, "hangul"), (0x9FFF, "cjk"), (0xA4CF, "yi"),
        (0xA4FF, "default"), (0xA63F, "default"), (0xA69F, "cyrillic"), (0xA6FF, "default"),
        (0xA7FF, "latin"), (0xA82F, "indic"), (0xA87F, "default"), (0xA8DF, "indic"),
        (0xA8FF, "indic"), (0xA92F, "indic"), (0xA95F, "indic"), (0xA97F, "hangul"),
        (0xA9DF, "indic"), (0xA9FF, "khmer_myanmar"), (0xAA5F, "indic"), (0xAA7F, "khmer_myanmar"),
        (0xAADF, "indic"), (0xAAFF, "indic"), (0xAB2F, "ethiopic"), (0xAB6F, "latin"),
        (0xABBF, "default"), (0xABFF, "indic"), (0xD7AF, "hangul"), (0xFAFF, "cjk"),
        (0xFDFF, "arabic"), (0xFE6F, "default"), (0xFEFF, "arabic"), (0xFFEF, "latin"),
    };

    private static double CharWeight(int code)
    {
        if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) return Weights["latin"];
        if (code == 32) return Weights["space"];
        if (code == 0x0640) return Weights["mark"]; // Arabic Tatweel

        var cat = Rune.GetUnicodeCategory(new Rune(code));
        switch (cat)
        {
            case UnicodeCategory.NonSpacingMark:
            case UnicodeCategory.SpacingCombiningMark:
            case UnicodeCategory.EnclosingMark:
                return Weights["mark"];
            case UnicodeCategory.ConnectorPunctuation:
            case UnicodeCategory.DashPunctuation:
            case UnicodeCategory.OpenPunctuation:
            case UnicodeCategory.ClosePunctuation:
            case UnicodeCategory.InitialQuotePunctuation:
            case UnicodeCategory.FinalQuotePunctuation:
            case UnicodeCategory.OtherPunctuation:
            case UnicodeCategory.MathSymbol:
            case UnicodeCategory.CurrencySymbol:
            case UnicodeCategory.ModifierSymbol:
            case UnicodeCategory.OtherSymbol:
                return Weights["punctuation"];
            case UnicodeCategory.SpaceSeparator:
            case UnicodeCategory.LineSeparator:
            case UnicodeCategory.ParagraphSeparator:
                return Weights["space"];
            case UnicodeCategory.DecimalDigitNumber:
            case UnicodeCategory.LetterNumber:
            case UnicodeCategory.OtherNumber:
                return Weights["digit"];
        }

        // bisect_left over the block end-codepoints: first range whose End >= code.
        int idx = LowerBound(code);
        if (idx < Ranges.Length)
            return Weights.TryGetValue(Ranges[idx].Type, out var w) ? w : Weights["default"];
        if (code > 0x20000) return Weights["cjk"];
        return Weights["default"];
    }

    private static int LowerBound(int code)
    {
        int lo = 0, hi = Ranges.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (Ranges[mid].End < code) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public static double TotalWeight(string text)
    {
        double sum = 0;
        foreach (var r in text.EnumerateRunes())
            sum += CharWeight(r.Value);
        return sum;
    }

    /// <summary>
    /// Estimate target audio-token count for <paramref name="targetText"/> given a reference
    /// text and its token count. Short estimates are boosted by a power curve (matching Python).
    /// </summary>
    public static double EstimateDuration(string targetText, string refText, double refDuration,
                                          double lowThreshold = 50, double boostStrength = 3)
    {
        if (refDuration <= 0 || string.IsNullOrEmpty(refText)) return 0.0;
        double refWeight = TotalWeight(refText);
        if (refWeight == 0) return 0.0;

        double speedFactor = refWeight / refDuration;
        double estimated = TotalWeight(targetText) / speedFactor;
        if (estimated < lowThreshold)
        {
            double alpha = 1.0 / boostStrength;
            return lowThreshold * Math.Pow(estimated / lowThreshold, alpha);
        }
        return estimated;
    }

    /// <summary>
    /// Port of _estimate_target_tokens: falls back to a fixed reference when none is given,
    /// scales by speed, and floors to at least 1 token.
    /// </summary>
    public static int EstimateTargetTokens(string text, string? refText, int? numRefAudioTokens,
                                           double speed = 1.0)
    {
        if (numRefAudioTokens is null || string.IsNullOrEmpty(refText))
        {
            refText = "Nice to meet you.";
            numRefAudioTokens = 25;
        }
        double est = EstimateDuration(text, refText, numRefAudioTokens.Value);
        if (speed > 0 && speed != 1.0) est /= speed;
        return Math.Max(1, (int)est);
    }
}
