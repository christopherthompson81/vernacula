using Vernacula.Tts.Base;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// The text → Kokoro-phonemes path and its phoneme-group → source-word map, through the real
/// phonemizer. Needs the vernacula-phonemizer submodule's data/ tree; skips without it.
/// </summary>
public class KokoroPhonemizerTests
{
    private static KokoroPhonemizer? TryCreate()
        => PhonemizerData.Resolve(null) is null ? null : new KokoroPhonemizer();

    [Fact]
    public void OneGroupPerSpokenWord_PunctuationAttached()
    {
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        var r = g2p.Phonemize("Hello world. This is a test, isn't it?");
        // 8 source words → 8 groups; the three marks ride on their words, not as groups of their own.
        var groups = r.Phonemes.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(8, groups.Length);
        Assert.EndsWith(".", groups[1]);
        Assert.EndsWith(",", groups[5]);
        Assert.EndsWith("?", groups[7]);
        Assert.NotNull(r.GroupSourceWords);
        Assert.Equal(Enumerable.Range(0, 8), r.GroupSourceWords);
        foreach (var ch in r.Phonemes)
            Assert.True(ch == ' ' || KokoroVocab.Contains(ch), $"'{ch}' is not a Kokoro token");
    }

    [Fact]
    public void ExpandedNumbersCollapseOntoTheirWrittenWord()
    {
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        // "$3.14" reads as several words; every one of them maps back to source word 4.
        var r = g2p.Phonemize("I paid about it $3.14 yesterday.");
        Assert.NotNull(r.GroupSourceWords);
        var groups = r.Phonemes.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(groups.Length, r.GroupSourceWords!.Count);
        var fromPrice = r.GroupSourceWords.Count(w => w == 4);
        Assert.True(fromPrice >= 2, $"expected the price to read as several groups, map: {string.Join(",", r.GroupSourceWords)}");
        Assert.Equal(5, r.GroupSourceWords[^1]);   // "yesterday." is the last word
        Assert.True(r.GroupSourceWords.Zip(r.GroupSourceWords.Skip(1)).All(p => p.First <= p.Second), "map is monotone");
    }

    [Fact]
    public void BritishUsesTheGbReading()
    {
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        Assert.Equal("ɡˈO hˈOm", g2p.ToPhonemes("go home"));
        Assert.Equal("ɡˈQ hˈQm", g2p.ToPhonemes("go home", british: true));
    }

    /// <summary>
    /// The reduced de-/re-/pre- prefix vowel renders as ⟨ə⟩, the spelling Kokoro was trained on.
    ///
    /// <para>
    /// ⚠ EVERY CASE HERE IS A BOUNDARY THE RULE MUST NOT CROSS, not a demonstration that it works.
    /// The rule is keyed on the SOURCE WORD, which is what makes an orthographic carve-out possible;
    /// a phonological version needed a `dᵻd` key and a tie-bar guard, and would have moved the
    /// inflectional ⟨ᵻ⟩ that gold itself writes 1,503 times.
    /// </para>
    /// </summary>
    [Theory]
    // The target class: gold writes ə here, we wrote ᵻ, a listener preferred ə.
    [InlineData("determine", "dətˈɜɹmən")]
    [InlineData("describe", "dəskɹˈIb")]
    [InlineData("reduce", "ɹədˈus")]
    [InlineData("remember", "ɹəmˈɛmbəɹ")]
    // ⚠ ded- IS CARVED OUT: the deduce/deduct family is the one place gold attests a prefix ⟨ᵻ⟩,
    // so these are the words with the STRONGEST evidence in the whole class.
    [InlineData("deduce", "dᵻdˈus")]
    [InlineData("deduct", "dᵻdˈʌkt")]
    // ⚠ AND deg-/dej- ARE NOT: they only ever looked like ded- to a rule matching IPA, where the d
    // of /d͡ʒ/ is half a segment. Spelling never had the problem.
    [InlineData("dejection", "dəʤˈɛkʃən")]
    [InlineData("degeneracy", "dəʤˈɛnəɹəsi")]
    // ⚠ be- IS OUT OF SCOPE. Untested upstream, so it is left alone rather than swept in.
    [InlineData("before", "bᵻfˈɔɹ")]
    [InlineData("become", "bᵻkˈʌm")]
    public void PrefixVowelRendersAsSchwaExceptWhereItMustNot(string word, string expected)
    {
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        Assert.Equal(expected, g2p.Phonemize(word, "en").Phonemes);
    }

    [Fact]
    public void ARareCarveOutWordKeepsItsPrefixVowelWithoutPinningTheWholeReading()
    {
        // ⚠ `dedans` IS IN THE CARVE-OUT BUT CANNOT BE PINNED EXACTLY. It is rare enough to go
        // through the neural OOV path, so its tail is dᵻdˈæns or dᵻdˈænz depending on whether that
        // model resolved -- an exact expectation here would test the model's presence, not this
        // rule. Assert only what the rule is responsible for: the prefix vowel it must not move.
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        Assert.StartsWith("dᵻd", g2p.Phonemize("dedans", "en").Phonemes);
    }

    [Fact]
    public void AnInflectionalBarredISurvivesInTheSameWordAsAChangedPrefix()
    {
        // ⚠ THE CASE THAT PROVES THE RULE IS POSITIONAL AND NOT A SUBSTITUTION. `dejected` carries
        // both: the prefix vowel moves, and the -ed vowel -- which gold writes ⟨ᵻ⟩ 1,287 times --
        // stays. A rule replacing the first ⟨ᵻ⟩, or every ⟨ᵻ⟩, fails exactly here.
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        Assert.Equal("dəʤˈɛktᵻd", g2p.Phonemize("dejected", "en").Phonemes);
    }

    [Fact]
    public void ABarredIThatIsNotTheFirstVowelIsLeftAlone()
    {
        // `represent` is ɹˌɛpɹᵻzˈɛnt in gold: the ⟨ᵻ⟩ is in the SECOND syllable and has nothing to
        // do with the re- prefix, whose vowel is ɛ and unreduced. "First ⟨ᵻ⟩ in the token" would
        // move it; "first VOWEL in the token" does not.
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        var p = g2p.Phonemize("represent", "en").Phonemes;
        Assert.StartsWith("ɹ", p);
        Assert.DoesNotContain("ɹəp", p);
    }

    [Fact]
    public void TheRuleLeavesTheStreamAloneWhenItCannotIdentifyTheWord()
    {
        // A word-keyed rule that has lost the word must do nothing rather than guess from the
        // phonemes. Non-English is the reachable version of that: the same spellings exist in other
        // languages and the prefix claim is about English only.
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        foreach (var lang in new[] { "es", "fr", "it" })
        {
            var p = g2p.Phonemize("determinar", lang).Phonemes;
            Assert.DoesNotContain("ᵻ", p);   // the symbol is English-only; nothing to do either way
            Assert.NotEmpty(p);
        }
    }
}
