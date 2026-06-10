namespace Chatterbox.Base;

/// <summary>
/// Phoneme → token-id map for hexgrad/Kokoro-82M (the model's <c>vocab</c>,
/// 114 entries, ids sparse over 1..177; id 0 is the padding token used to
/// bracket the sequence). Keys are single Kokoro-alphabet phoneme codepoints
/// as produced by <c>Vernacula.Phonemizer.KokoroFormat.Render</c>.
///
/// GENERATED from <c>KModel.vocab</c> by
/// <c>scripts/kokoro_export/export_voices.py</c>'s sibling vocab dump
/// (scripts/kokoro_export gen_vocab); do not hand-edit. Phonemes not present
/// here are dropped during tokenization, mirroring misaki/KModel
/// (<c>filter(None, map(vocab.get, phonemes))</c>).
/// </summary>
public static class KokoroVocab
{
    /// <summary>Padding/boundary token id; the input sequence is [Pad, …ids…, Pad].</summary>
    public const long Pad = 0;

    private static readonly IReadOnlyDictionary<char, long> Map = new Dictionary<char, long>
    {
        [';'] = 1,
        [':'] = 2,
        [','] = 3,
        ['.'] = 4,
        ['!'] = 5,
        ['?'] = 6,
        ['—'] = 9,
        ['…'] = 10,
        ['"'] = 11,
        ['('] = 12,
        [')'] = 13,
        ['“'] = 14,
        ['”'] = 15,
        [' '] = 16,
        ['̃'] = 17,
        ['ʣ'] = 18,
        ['ʥ'] = 19,
        ['ʦ'] = 20,
        ['ʨ'] = 21,
        ['ᵝ'] = 22,
        ['ꭧ'] = 23,
        ['A'] = 24,
        ['I'] = 25,
        ['O'] = 31,
        ['Q'] = 33,
        ['S'] = 35,
        ['T'] = 36,
        ['W'] = 39,
        ['Y'] = 41,
        ['ᵊ'] = 42,
        ['a'] = 43,
        ['b'] = 44,
        ['c'] = 45,
        ['d'] = 46,
        ['e'] = 47,
        ['f'] = 48,
        ['h'] = 50,
        ['i'] = 51,
        ['j'] = 52,
        ['k'] = 53,
        ['l'] = 54,
        ['m'] = 55,
        ['n'] = 56,
        ['o'] = 57,
        ['p'] = 58,
        ['q'] = 59,
        ['r'] = 60,
        ['s'] = 61,
        ['t'] = 62,
        ['u'] = 63,
        ['v'] = 64,
        ['w'] = 65,
        ['x'] = 66,
        ['y'] = 67,
        ['z'] = 68,
        ['ɑ'] = 69,
        ['ɐ'] = 70,
        ['ɒ'] = 71,
        ['æ'] = 72,
        ['β'] = 75,
        ['ɔ'] = 76,
        ['ɕ'] = 77,
        ['ç'] = 78,
        ['ɖ'] = 80,
        ['ð'] = 81,
        ['ʤ'] = 82,
        ['ə'] = 83,
        ['ɚ'] = 85,
        ['ɛ'] = 86,
        ['ɜ'] = 87,
        ['ɟ'] = 90,
        ['ɡ'] = 92,
        ['ɥ'] = 99,
        ['ɨ'] = 101,
        ['ɪ'] = 102,
        ['ʝ'] = 103,
        ['ɯ'] = 110,
        ['ɰ'] = 111,
        ['ŋ'] = 112,
        ['ɳ'] = 113,
        ['ɲ'] = 114,
        ['ɴ'] = 115,
        ['ø'] = 116,
        ['ɸ'] = 118,
        ['θ'] = 119,
        ['œ'] = 120,
        ['ɹ'] = 123,
        ['ɾ'] = 125,
        ['ɻ'] = 126,
        ['ʁ'] = 128,
        ['ɽ'] = 129,
        ['ʂ'] = 130,
        ['ʃ'] = 131,
        ['ʈ'] = 132,
        ['ʧ'] = 133,
        ['ʊ'] = 135,
        ['ʋ'] = 136,
        ['ʌ'] = 138,
        ['ɣ'] = 139,
        ['ɤ'] = 140,
        ['χ'] = 142,
        ['ʎ'] = 143,
        ['ʒ'] = 147,
        ['ʔ'] = 148,
        ['ˈ'] = 156,
        ['ˌ'] = 157,
        ['ː'] = 158,
        ['ʰ'] = 162,
        ['ʲ'] = 164,
        ['↓'] = 169,
        ['→'] = 171,
        ['↗'] = 172,
        ['↘'] = 173,
        ['ᵻ'] = 177,
    };

    /// <summary>Number of phoneme entries (excludes the pad token).</summary>
    public static int Count => Map.Count;

    /// <summary>True if <paramref name="ch"/> is a known Kokoro phoneme token.</summary>
    public static bool Contains(char ch) => Map.ContainsKey(ch);

    /// <summary>
    /// Tokenize a Kokoro-alphabet phoneme string into the padded id sequence
    /// the ONNX graph expects: <c>[Pad, …ids…, Pad]</c>. Unknown codepoints are
    /// skipped (matching KModel). Surrogate pairs are not expected in the Kokoro
    /// alphabet (all phonemes are BMP), so per-char iteration is exact.
    /// </summary>
    public static long[] Encode(string phonemes)
    {
        var ids = new List<long>(phonemes.Length + 2) { Pad };
        foreach (var ch in phonemes)
            if (Map.TryGetValue(ch, out var id))
                ids.Add(id);
        ids.Add(Pad);
        return ids.ToArray();
    }
}
