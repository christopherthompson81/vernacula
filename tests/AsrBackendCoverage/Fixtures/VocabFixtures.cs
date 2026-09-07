using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Vernacula.Base;
using Vernacula.App.Services;

namespace Vernacula.Tests.AsrBackendCoverage.Fixtures;

/// <summary>
/// Tiny hand-built vocabularies, one per <see cref="VocabService.VocabKind"/>, written into a
/// throwaway models directory at the exact paths the loader probes.
///
/// <para>
/// They are built here rather than committed as blobs for two reasons: a real vocab is
/// megabytes of tokens irrelevant to what is being checked, and a fixture whose expected text
/// is known BY CONSTRUCTION is a stronger assertion than one whose expected text was copied
/// out of a previous run. Each vocab is written in its format's documented shape — the
/// SentencePiece word-boundary marker, the <c>&lt;0xNN&gt;</c> byte fallback, the HF
/// <c>tokenizer.json</c> object — so decoding it is a claim about the format, not about this
/// implementation.
/// </para>
/// </summary>
internal static class VocabFixtures
{
    /// <summary>The phrase every fixture encodes. Chosen for what it exercises, not for length:
    /// a word boundary, and an "ö" whose two UTF-8 bytes are deliberately split across two
    /// tokens so the per-token streaming decoders have to carry a partial character.</summary>
    public const string Phrase = "Hello wörld.";

    /// <summary>SentencePiece's word-boundary marker (U+2581), which the text loaders turn into a space.</summary>
    private const string WordStart = "▁";

    // ── GPT-2 ByteLevel, from the specification ──────────────────────────────
    //
    // ⚠ COMPUTED HERE, NOT BORROWED FROM VocabService. The HF-format loaders decode a token's
    // characters back to bytes through their own copy of this table; if the fixtures were
    // encoded with that same copy, a wrong table would round-trip and the tests would pass on
    // a broken decoder. Building it independently from the published mapping (printable ASCII
    // and Latin-1 ranges map to themselves; every other byte maps to U+0100 + its rank among
    // the non-printables, so 0x20 becomes U+0120 "Ġ") makes the tests check the table too.

    private static readonly Dictionary<byte, char> ByteToChar = BuildByteToChar();

    private static Dictionary<byte, char> BuildByteToChar()
    {
        var printable = new HashSet<int>(
            Enumerable.Range('!', '~' - '!' + 1)
            .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
            .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)));

        var map = new Dictionary<byte, char>(256);
        int extra = 0;
        for (int b = 0; b < 256; b++)
            map[(byte)b] = printable.Contains(b) ? (char)b : (char)(0x100 + extra++);
        return map;
    }

    /// <summary>A string's UTF-8 bytes as the characters a GPT-2 ByteLevel vocab spells them with.</summary>
    public static string ToByteLevel(string text) =>
        new(Encoding.UTF8.GetBytes(text).Select(b => ByteToChar[b]).ToArray());

    // ── Token ids, shared by every fixture ───────────────────────────────────
    // The same five ids spell the phrase in every format, so a test can name a sequence once.

    public const int Hello = 0, SpaceW = 1, OSplitHigh = 2, OSplitLowRld = 3, Dot = 4;
    /// <summary>An added ("special") token — the HF kinds disagree about what to do with it.</summary>
    public const int Special = 5;
    /// <summary>"Hello" with a leading space, for the kinds that strip one at the start.</summary>
    public const int SpaceHello = 6;
    /// <summary>A double quote, for the kind that trims them at the boundaries.</summary>
    public const int Quote = 7;

    public const string SpecialContent = "<|endoftext|>";

    // Cohere spends a token per raw byte, so "rld" and "." sit one slot further along.
    public const int CohereRld = 4, CohereDot = 5;

    /// <summary>
    /// The tokens that spell <see cref="Phrase"/> in a given fixture, and the directory holding
    /// it. The sequence is per kind because the formats genuinely differ in how few tokens can
    /// spell the phrase — a Cohere <c>&lt;0xNN&gt;</c> token is a whole byte and cannot also
    /// carry text, so its "ö" costs two tokens where a ByteLevel vocab folds the second byte
    /// into the next token.
    /// </summary>
    public sealed record Fixture(string Dir, int[] PhraseTokens) : IDisposable
    {
        public void Dispose() => Cleanup(Dir);
    }

    /// <summary>A throwaway models directory holding the fixture for <paramref name="kind"/>.</summary>
    public static Fixture Write(VocabService.VocabKind kind)
    {
        string dir = WriteModelsDir(kind);
        int[] tokens = kind == VocabService.VocabKind.Cohere
            ? [Hello, SpaceW, OSplitHigh, OSplitLowRld, CohereRld, CohereDot]
            : [Hello, SpaceW, OSplitHigh, OSplitLowRld, Dot];
        return new Fixture(dir, tokens);
    }

    /// <summary>A throwaway models directory holding the fixture for <paramref name="kind"/>.</summary>
    public static string WriteModelsDir(VocabService.VocabKind kind)
    {
        string dir = Path.Combine(Path.GetTempPath(), "vernacula-vocab-fixtures", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        switch (kind)
        {
            case VocabService.VocabKind.Parakeet:
                // "<token> <id>" per line; the loader splits on the LAST space.
                WriteFile(Path.Combine(dir, Config.VocabFile), string.Join('\n',
                [
                    $"{WordStart}Hello {Hello}",
                    $"{WordStart}w {SpaceW}",
                    $"ö {OSplitHigh}",           // text formats have no byte-level split to make
                    $"rld {OSplitLowRld}",
                    $". {Dot}",
                ]));
                break;

            case VocabService.VocabKind.IndicConformer:
                // One token per line; the id IS the line index.
                WriteFile(Path.Combine(dir, Config.IndicConformerSubDir, Config.VocabFile), string.Join('\n',
                [
                    $"{WordStart}Hello",
                    $"{WordStart}w",
                    "ö",
                    "rld",
                    ".",
                ]));
                break;

            case VocabService.VocabKind.Cohere:
                // A JSON array; the index is the id. "ö" is split into its two UTF-8 bytes
                // using the <0xNN> fallback, which is how this format spells raw bytes.
                WriteFile(Path.Combine(dir, "cohere_transcribe", CohereTranscribe.VocabFile),
                    $"[\"{WordStart}Hello\",\"{WordStart}w\",\"<0xC3>\",\"<0xB6>\",\"rld\",\".\"]");
                break;

            case VocabService.VocabKind.VibeVoice:
                WriteFile(Path.Combine(dir, Config.VibeVoiceSubDir, VibeVoiceAsr.TokenizerFile), HfTokenizerJson());
                break;

            case VocabService.VocabKind.Qwen3Asr:
                WriteFile(Path.Combine(dir, Config.Qwen3AsrSubDir, Qwen3Asr.TokenizerFile), HfTokenizerJson());
                break;

            case VocabService.VocabKind.GraniteSpeech:
                // The FP32 sibling; the loader falls back to it when the BF16 one is absent.
                WriteFile(Path.Combine(dir, Config.GraniteSpeechSubDir, GraniteSpeech.TokenizerFile), HfTokenizerJson());
                break;

            case VocabService.VocabKind.WhisperTurbo:
                WriteFile(Path.Combine(dir, Config.WhisperTurboSubDir, WhisperTurbo.TokenizerFile), HfTokenizerJson());
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(kind), kind,
                    "VocabFixtures has no fixture for this kind. Add one alongside the new VocabKind.");
        }

        return dir;
    }

    /// <summary>
    /// The HF <c>tokenizer.json</c> shape the ByteLevel kinds read: <c>model.vocab</c> maps a
    /// token's spelling to its id, <c>added_tokens</c> carries the specials. "ö" is split
    /// mid-character: its high byte is one token, its low byte opens the next.
    /// </summary>
    private static string HfTokenizerJson()
    {
        var vocab = new (string Spelling, int Id)[]
        {
            (ToByteLevel("Hello"),  Hello),
            (ToByteLevel(" w"),     SpaceW),
            (ByteLevelOf(0xC3),     OSplitHigh),
            (ByteLevelOf(0xB6) + ToByteLevel("rld"), OSplitLowRld),
            (ToByteLevel("."),      Dot),
            (ToByteLevel(" Hello"), SpaceHello),
            (ToByteLevel("\""),     Quote),
        };

        string entries = string.Join(",", vocab.Select(v => $"{Quoted(v.Spelling)}:{v.Id}"));
        string added = $"[{{\"id\":{Special},\"content\":{Quoted(SpecialContent)}}}]";
        return $"{{\"model\":{{\"vocab\":{{{entries}}}}},\"added_tokens\":{added}}}";
    }

    private static string ByteLevelOf(byte b) => ByteToChar[b].ToString();

    /// <summary>JSON string literal — the spellings hold quotes and backslashes.</summary>
    private static string Quoted(string s) => System.Text.Json.JsonSerializer.Serialize(s);

    private static void WriteFile(string path, string content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    /// <summary>Deletes a directory made by <see cref="WriteModelsDir"/>; best effort.</summary>
    public static void Cleanup(string dir)
    {
        try { Directory.Delete(dir, recursive: true); } catch { /* a leftover temp dir is harmless */ }
    }
}
