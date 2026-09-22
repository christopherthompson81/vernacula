using Vernacula.Phonemizer;

namespace Vernacula.Tts.Base;

/// <summary>One word's character range in the text it came from: <c>[Start, End)</c>.</summary>
public readonly record struct WordSpan(int Start, int End);

/// <summary>
/// What counts as a word, for a language that may not put spaces between them.
///
/// <para>
/// ⚠ ONE ANSWER, USED THREE TIMES. The reader builds a clickable word per span, the phonemizer's
/// group → word map indexes into the same spans, and the alignment sidecar names them. Those three
/// used to split on whitespace INDEPENDENTLY, which agreed by construction in English and produced
/// exactly one word per sentence in Japanese — so a whole paragraph lit up at once and clicking
/// anywhere in it seeked to its start.
/// </para>
///
/// <para>
/// ⚠ AND THE SEGMENTATION WAS ALREADY THERE. <c>PhonemizeTrace</c> reports an input span per token
/// and is built for this; the two callers above simply bucketed every span into the whitespace word
/// containing its start offset, which for a language without spaces is always word 0. This is that
/// information, kept.
/// </para>
/// </summary>
public static class WordSegmentation
{
    /// <summary>Languages whose words the trace has to find, because whitespace will not.</summary>
    public static bool NeedsTrace(string? lang) => lang is "ja" or "cmn";

    /// <summary>
    /// Word spans over <c>text[start..end)</c>. Whitespace unless <paramref name="lang"/> is one
    /// the trace has to segment, and whitespace again if the trace cannot do better — a language
    /// that does not space is not a reason to return something worse than what we have.
    /// </summary>
    public static IReadOnlyList<WordSpan> Segment(string text, int start, int end, string? lang)
    {
        if (!NeedsTrace(lang)) return Whitespace(text, start, end);
        try
        {
            var traced = FromTrace(text, start, end, lang!);
            return traced.Count > 0 ? traced : Whitespace(text, start, end);
        }
        catch (Exception)
        {
            // A missing data tree, an unregistered language: the words still have to be built.
            // ⚠ THIS IS WHY WHITESPACE STAYS THE FALLBACK. The reader segments when a document is
            // OPENED, which never needed a phonemizer before, so a tree that is absent must cost
            // granularity rather than correctness.
            return Whitespace(text, start, end);
        }
    }

    /// <summary>The split every other language uses, and the fallback for the two that cannot.</summary>
    public static IReadOnlyList<WordSpan> Whitespace(string text, int start, int end)
    {
        var words = new List<WordSpan>();
        var i = start;
        while (i < end)
        {
            while (i < end && char.IsWhiteSpace(text[i])) i++;
            if (i >= end) break;
            var from = i;
            while (i < end && !char.IsWhiteSpace(text[i])) i++;
            words.Add(new WordSpan(from, i));
        }
        return words;
    }

    /// <summary>
    /// The trace the Kokoro word path takes, in one place so its two callers read the same spans.
    /// </summary>
    /// <remarks>
    /// Not the only <c>PhonemizeTrace</c> in the codebase — <see cref="IpaAnnotator"/> and the
    /// OmniVoice IPA path take their own, for their own purposes. What this shares is narrower and
    /// load-bearing: <see cref="Segment"/> builds the words and
    /// <see cref="KokoroPhonemizer.Phonemize(string, string)"/> builds the group→word map onto
    /// those same words, so the two must be reading one trace's spans and not two.
    ///
    /// <para>
    /// This carried a retry until 2026-09-22: the first <c>ja</c> trace in a process returned every
    /// <c>InputSpan</c> null, because <c>Normalize</c>'s static constructor ran a tracked rewrite on
    /// its own first use — inside the traced window (vernacula-phonemizer#1408, fixed in #1417, C#
    /// only). ⚠ A REGRESSION HERE IS NOT VISIBLE FROM A TEST ASSEMBLY, because the defect is once
    /// per process and any earlier test warms the language; the gate is upstream's
    /// <c>csharp/tools/trace-cold</c>, which spawns its own process and swept 189 of 189 clean on
    /// this pin. See docs/investigations/audiocpp_multilingual_investigation.md, Run 8.
    /// </para>
    /// </remarks>
    public static PhonemeTrace Trace(string text, string lang)
        => Phonemizer.Phonemizer.PhonemizeTrace(text, lang);

    /// <summary>Han ideographs, which are the characters that map one-to-one onto syllables.</summary>
    private static bool IsHan(char c) => c is >= '一' and <= '鿿' or >= '㐀' and <= '䶿';

    /// <summary>
    /// Word spans from the phonemizer's own trace.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The two languages give different shapes, so this reads them differently rather than
    /// pretending one rule covers both.
    /// </para>
    /// <para>
    /// ⚠ JAPANESE GIVES PHRASES, NOT WORDS, and that is what is offered rather than a shortcoming
    /// here — <c>科学者たちが発表しました。</c> traces as <c>[科学者たちが]</c> and
    /// <c>[発表しました]</c>. Two units where whitespace sees one is the improvement; nobody should
    /// read this as morphological word segmentation.
    /// </para>
    /// <para>
    /// ⚠ MANDARIN GIVES ONE TOKEN FOR THE WHOLE SENTENCE and carries its segmentation in the IPA
    /// instead, one space-delimited group per syllable. One hanzi is one syllable, so the groups
    /// can be walked onto the characters — but ONLY when the counts agree. Measured over the
    /// phonemizer's 200 Mandarin goldens: 126 rows are pure hanzi and all 126 match exactly, while
    /// every one of the other 74 has digits or latin in it, where a run like <c>11</c> becomes two
    /// syllables and the positional correspondence silently shifts everything after it. So the
    /// count check is the whole safety argument, and a mismatch keeps the token whole.
    /// </para>
    /// </remarks>
    private static IReadOnlyList<WordSpan> FromTrace(string text, int start, int end, string lang)
    {
        var slice = text[start..end];
        var trace = Trace(slice, lang);
        if (!trace.Traced) return [];

        var words = new List<WordSpan>();
        foreach (var token in trace.Tokens)
        {
            if (token.InputSpan is not { } span) continue;
            var from = start + Math.Clamp(span.Start, 0, slice.Length);
            var to = start + Math.Clamp(span.End, 0, slice.Length);
            if (to <= from) continue;

            // A token that produced no spoken group is punctuation — the trailing 。of a Japanese
            // sentence traces as its own token with empty IPA. It is not a word to click.
            var groups = token.IpaSpan is { } ipa
                ? CountGroups(trace.Ipa[Math.Clamp(ipa.Start, 0, trace.Ipa.Length)..Math.Clamp(ipa.End, 0, trace.Ipa.Length)])
                : 0;
            if (groups == 0) continue;

            var body = text[from..to];
            if (groups > 1 && groups == body.Count(IsHan) && body.All(c => IsHan(c) || char.IsWhiteSpace(c)))
            {
                // One group per hanzi, and nothing else in the span to have eaten one: safe to
                // walk them onto the characters.
                for (var k = from; k < to; k++)
                    if (IsHan(text[k])) words.Add(new WordSpan(k, k + 1));
            }
            else
            {
                words.Add(new WordSpan(from, to));
            }
        }
        return words;
    }

    /// <summary>Space-delimited groups carrying a letter — the phonemizer's stand-alone
    /// punctuation tokens are not spoken units.</summary>
    private static int CountGroups(string ipa)
    {
        var n = 0;
        foreach (var g in ipa.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            if (g.Any(char.IsLetter)) n++;
        return n;
    }
}
