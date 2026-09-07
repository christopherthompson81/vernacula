using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Vernacula.Tts.Base.Alignment;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// The alignment sidecar is the on-disk contract between two producers (the CLI's
/// <c>--alignment-out</c> and the desktop app's TTS jobs) and the reader that opens either.
/// It used to be defined twice and had already drifted (#132). These pin the one definition:
/// the snake_case keys, that a CLI-shaped sidecar without the app-only fields still loads, and
/// the two file conventions that hang off it.
/// </summary>
public class AlignmentSidecarTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));

    public AlignmentSidecarTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private static AlignmentSidecar Sample() => new()
    {
        AudioPath = "/out/doc.wav",
        SampleRate = 24000,
        AudioDurationSeconds = 2.5,
        Aligner = "kokoro_duration",
        SourceText = "# Title\n\nHello world.",
        Chunks =
        {
            new ChunkRecord { Index = 0, AudioStartSeconds = 0, AudioEndSeconds = 1.0, Text = "Title.", WordCount = 1,
                              AudioFile = "seg_0000.wav", BlockKind = "Heading", BlockLevel = 1 },
            new ChunkRecord { Index = 1, AudioStartSeconds = 1.0, AudioEndSeconds = 2.5, Text = "Hello world.", WordCount = 2,
                              AudioFile = "seg_0001.wav", BlockKind = "Paragraph" },
        },
        Words =
        {
            new AlignedWord { Text = "Title.", StartSeconds = 0.1, EndSeconds = 0.9, ChunkIndex = 0 },
            new AlignedWord { Text = "Hello",  StartSeconds = 1.1, EndSeconds = 1.6, ChunkIndex = 1 },
            new AlignedWord { Text = "world.", StartSeconds = 1.7, EndSeconds = 2.4, ChunkIndex = 1 },
        },
    };

    /// <summary>The keys are the contract. A renamed C# property must not be able to rename one.</summary>
    [Fact]
    public void KeysAreSnakeCaseAndStable()
    {
        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(Sample()));
        var root = doc.RootElement;
        foreach (var key in new[] { "audio_path", "sample_rate", "audio_duration_seconds", "aligner", "chunks", "words", "source_text" })
            Assert.True(root.TryGetProperty(key, out _), $"missing top-level key {key}");
        var chunk = root.GetProperty("chunks")[0];
        foreach (var key in new[] { "index", "audio_start_seconds", "audio_end_seconds", "text", "word_count", "audio_file", "block_kind", "block_level" })
            Assert.True(chunk.TryGetProperty(key, out _), $"missing chunk key {key}");
        var word = root.GetProperty("words")[0];
        foreach (var key in new[] { "text", "start_seconds", "end_seconds", "chunk_index" })
            Assert.True(word.TryGetProperty(key, out _), $"missing word key {key}");
        // No PascalCase leaked through: every key is lower-case.
        Assert.All(root.EnumerateObject(), p => Assert.Equal(p.Name, p.Name.ToLowerInvariant()));
    }

    [Fact]
    public void SaveThenLoadRoundTripsAndLeavesNoTempFile()
    {
        string path = Path.Combine(_dir, "doc_tts.json");
        Sample().Save(path, indented: true);

        Assert.True(File.Exists(path));
        Assert.False(File.Exists(path + ".tmp"), "atomic write must rename its .tmp away");

        var back = AlignmentSidecar.Load(path);
        Assert.Equal("kokoro_duration", back.Aligner);
        Assert.Equal(2, back.Chunks.Count);
        Assert.Equal("Heading", back.Chunks[0].BlockKind);
        Assert.Equal(1, back.Chunks[0].BlockLevel);
        Assert.Equal("seg_0001.wav", back.Chunks[1].AudioFile);
        Assert.Equal(3, back.Words.Count);
        Assert.Equal(1.7, back.Words[2].StartSeconds);
        Assert.Equal("# Title\n\nHello world.", back.SourceText);
        Assert.Null(back.NfaBundle);

        // A nullable field the producer did not set is absent on disk, not null: the sample has
        // no nfa_bundle. (Value-typed fields such as block_level are always written — the second
        // chunk carries block_level 0 — which is why BlockLevel is an int rather than int?: 0 is a
        // meaningful "not a heading", not a missing value.)
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        Assert.False(doc.RootElement.TryGetProperty("nfa_bundle", out _), "null fields must not be written");
        Assert.Equal(0, doc.RootElement.GetProperty("chunks")[1].GetProperty("block_level").GetInt32());
    }

    /// <summary>
    /// What the CLI writes: no source_text, no per-chunk audio_file / block_kind, but an
    /// nfa_bundle. The reader has to open this too — it is the case that was broken.
    /// </summary>
    [Fact]
    public void CliShapedSidecarLoadsWithAppOnlyFieldsAbsent()
    {
        string path = Path.Combine(_dir, "cli.json");
        File.WriteAllText(path, """
            {
              "audio_path": "/out/cli.wav",
              "sample_rate": 24000,
              "audio_duration_seconds": 1.0,
              "aligner": "nemo_nfa",
              "nfa_bundle": "/models/nfa",
              "chunks": [ { "index": 0, "audio_start_seconds": 0, "audio_end_seconds": 1.0, "text": "Hi there.", "word_count": 2 } ],
              "words": [ { "text": "Hi", "start_seconds": 0.1, "end_seconds": 0.4, "chunk_index": 0 },
                         { "text": "there.", "start_seconds": 0.5, "end_seconds": 0.9, "chunk_index": 0 } ]
            }
            """);

        var s = AlignmentSidecar.Load(path);
        Assert.Equal("nemo_nfa", s.Aligner);
        Assert.Equal("/models/nfa", s.NfaBundle);
        Assert.Null(s.SourceText);
        var c = Assert.Single(s.Chunks);
        Assert.Null(c.AudioFile);
        Assert.Null(c.BlockKind);
        Assert.Equal(0, c.BlockLevel);
        Assert.Equal(2, s.Words.Count);
    }

    [Fact]
    public void LoadRejectsSomethingThatIsNotASidecar()
    {
        string path = Path.Combine(_dir, "null.json");
        File.WriteAllText(path, "null");
        Assert.Throws<InvalidDataException>(() => AlignmentSidecar.Load(path));
    }

    /// <summary>The folder and file conventions, spelled once.</summary>
    [Fact]
    public void SegmentConventionsAreDerivedFromTheSidecarPath()
    {
        string sidecar = Path.Combine("/jobs", "abc_tts.json");
        Assert.Equal(Path.Combine("/jobs", "abc_tts_segments"), AlignmentSidecar.SegmentsDirFor(sidecar));
        Assert.Equal("seg_0000.wav", AlignmentSidecar.SegmentFileName(0));
        Assert.Equal("seg_0042.wav", AlignmentSidecar.SegmentFileName(42));
        // Sorts lexically in index order, which is what a directory listing relies on.
        var names = Enumerable.Range(0, 1200).Select(AlignmentSidecar.SegmentFileName).ToList();
        Assert.Equal(names, names.OrderBy(n => n, StringComparer.Ordinal).ToList());
    }
}
