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
            return Segment(text, start, end, lang, Trace(text[start..end], lang!));
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

    /// <summary>
    /// Word spans over <c>text[start..end)</c>, read from a trace the caller already has.
    /// </summary>
    /// <remarks>
    /// ⚠ FOR A CALLER THAT NEEDS BOTH THE WORDS AND THE TRACE, so it cannot end up reading two of
    /// them. <see cref="KokoroPhonemizer.Phonemize(string, string)"/> builds the group→word map
    /// from a trace and the words it indexes into from another, and asserted in a comment that the
    /// two read "the same spans" — an invariant nothing enforced. vernacula-phonemizer#1408 was
    /// precisely a case of one trace of a text disagreeing with the next, so the assumption had
    /// already been wrong once. <paramref name="trace"/> must be a trace of <c>text[start..end)</c>.
    /// </remarks>
    public static IReadOnlyList<WordSpan> Segment(
        string text, int start, int end, string? lang, PhonemeTrace trace)
    {
        if (!NeedsTrace(lang)) return Whitespace(text, start, end);
        try
        {
            var traced = FromTrace(text, start, end, trace);
            return traced.Count > 0 ? traced : Whitespace(text, start, end);
        }
        catch (Exception)
        {
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
    private static IReadOnlyList<WordSpan> FromTrace(string text, int start, int end, PhonemeTrace trace)
    {
        var sliceLength = end - start;
        if (!trace.Traced) return [];

        var words = new List<WordSpan>();
        foreach (var token in trace.Tokens)
        {
            if (token.InputSpan is not { } span) continue;
            var from = start + Math.Clamp(span.Start, 0, sliceLength);
            var to = start + Math.Clamp(span.End, 0, sliceLength);
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
                    if (IsHan(text[k])) Append(words, new WordSpan(k, k + 1));
            }
            else
            {
                Append(words, new WordSpan(from, to));
            }
        }
        return words;
    }

    /// <summary>
    /// Adds a span, merging it into the previous one when the two overlap.
    /// </summary>
    /// <remarks>
    /// ⚠ TWO UNITS THAT CLAIM THE SAME CHARACTERS ARE ONE UNIT. A rewrite whose match covers more
    /// than the token it is rewriting stamps that match's span across every token it produced —
    /// <c>PDFファイルを開いてください</c> traced as three tokens (<c>ピーディーエフ</c>,
    /// <c>ファイルを</c>, <c>開いてください</c>) all claiming the whole sentence. Emitting one unit per token then offers the reader several clickable words that
    /// light identical text, and only the first of them receives any time at all, because
    /// <see cref="KokoroAlignment.WordsFromGroups"/> gives each later duplicate a zero-length marker.
    ///
    /// <para>
    /// Measured over the phonemizer's 200-row <c>ja</c> goldens: 14 of 123 distinct rows carry
    /// duplicate spans, ALL of them mixed-script (14 of the 49 rows containing latin or digits), and
    /// the worst offers eight units that each cover 30 of the sentence's 34 characters. ⚠ AND THE
    /// COUNT CHECK IN THE ALIGNER CANNOT SEE IT — groups and map entries agree, so the measured tier
    /// engages and trusts them; being degenerate is invisible to a count the way being one out is.
    /// </para>
    ///
    /// <para>
    /// Merging is the honest answer rather than a workaround: if two tokens cannot say which
    /// characters are theirs, those characters are one clickable unit whose time is the union of
    /// their groups. Coarser than the token count suggests, correct at the boundary it reports, and
    /// the same answer this already gives for an English rewrite where one written word becomes
    /// several tokens ("$3.14" → three, dollars, fourteen) — which is also why MERGING rather than
    /// DECLINING is the right response: tokens sharing a span is frequently correct, so a check
    /// that refused to align on it would throw away good expansions along with bad spans.
    /// </para>
    ///
    /// <para>
    /// ⚠ AND THE CAUSE WAS NOT THE JAPANESE PATH, which is why this guard is worth keeping even
    /// after the upstream fix. vernacula-phonemizer#1420: <c>normalizeRomans</c> rewrites on
    /// <c>\p{L}+</c> and runs over EVERY language, and in a script without spaces there is no word
    /// break for that match to stop at, so it matched the whole clause and stamped its span across
    /// the replacement. Its fast path skips text with no Roman letters — so the defect was absent
    /// from exactly the pure-kana sentences anyone reaches for first when testing Japanese, and
    /// present in exactly the ones containing a latin letter. The fix takes ja from 14 collapsed
    /// rows to 3; the residue is numeral/unit overlap ("83 m" and "83 mです"), a smaller and
    /// different thing that this merge still handles.
    /// </para>
    /// </remarks>
    private static void Append(List<WordSpan> words, WordSpan span)
    {
        if (words.Count > 0 && span.Start < words[^1].End)
        {
            // ⚠ THE UNION OF BOTH ENDS, not just the later one. The tokens arrive in order today —
            // measured, zero unordered pairs across the ja and cmn goldens — but keeping the
            // previous Start would drop the characters ahead of it out of every unit, and a
            // character belonging to no unit is not clickable and receives no highlight. That is a
            // silent loss, and it should not rest on an ordering this code does not enforce.
            var last = words[^1];
            words[^1] = new WordSpan(Math.Min(last.Start, span.Start), Math.Max(last.End, span.End));
            return;
        }
        words.Add(span);
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
