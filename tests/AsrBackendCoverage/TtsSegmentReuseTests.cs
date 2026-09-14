using Vernacula.App.Services.Tts;
using Vernacula.Tts.Base.Alignment;
using Vernacula.Tts.Base.Markdown;
using Xunit;

namespace AsrBackendCoverage;

/// <summary>
/// Paragraph-local re-synthesis: which paragraphs a given edit actually costs, and that an unchanged
/// one comes back off disk rather than through the engine.
/// </summary>
public class TtsSegmentReuseTests
{
    private const string Doc = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";

    /// <summary>A sidecar as a finished job would have left it, with one WAV per segment.</summary>
    private static AlignmentSidecar SidecarFor(string text, string? segDir = null)
    {
        var sidecar = new AlignmentSidecar { SampleRate = 24000 };
        double cursor = 0;
        foreach (var seg in ParagraphSegmenter.Segment(text))
        {
            string file = AlignmentSidecar.SegmentFileName(seg.Index);
            sidecar.Chunks.Add(new ChunkRecord
            {
                Index = seg.Index,
                AudioStartSeconds = cursor,
                AudioEndSeconds = cursor + 1,
                Text = seg.Text,
                WordCount = seg.WordCount,
                AudioFile = file,
                BlockKind = seg.Kind.ToString(),
                BlockLevel = seg.Level,
            });
            // One word per segment is enough to check the absolute→relative conversion.
            sidecar.Words.Add(new AlignedWord
            {
                Text = seg.Text.Split(' ')[0],
                StartSeconds = cursor + 0.25,
                EndSeconds = cursor + 0.75,
                ChunkIndex = seg.Index,
            });
            if (segDir is not null) WriteWav(Path.Combine(segDir, file), [0.5f, -0.5f, 0.25f], 24000);
            cursor += 1;
        }
        return sidecar;
    }

    [Fact]
    public void AnUnchangedDocumentCostsNothing()
        => Assert.Empty(TtsSegmentReuse.ChangedSegments(Doc, SidecarFor(Doc)));

    [Fact]
    public void EditingOneParagraphChangesOnlyThatOne()
    {
        var changed = TtsSegmentReuse.ChangedSegments(
            "First paragraph.\n\nSecond paragraph EDITED.\n\nThird paragraph.", SidecarFor(Doc));
        Assert.Equal([1], changed);
    }

    /// <summary>⚠ The point of the whole feature: inserting shifts every later index, and the
    /// paragraphs that merely MOVED must still be reused rather than re-rendered.</summary>
    [Fact]
    public void InsertingAParagraphOnlyCostsTheInsertedOne()
    {
        var changed = TtsSegmentReuse.ChangedSegments(
            "First paragraph.\n\nBrand new.\n\nSecond paragraph.\n\nThird paragraph.", SidecarFor(Doc));
        Assert.Equal([1], changed);
    }

    [Fact]
    public void DeletingAParagraphCostsNothing()
        => Assert.Empty(TtsSegmentReuse.ChangedSegments("First paragraph.\n\nThird paragraph.", SidecarFor(Doc)));

    /// <summary>The same words as a heading are not the same audio, so kind is part of the identity.</summary>
    [Fact]
    public void TheSameTextAsADifferentBlockKindIsAChange()
    {
        var changed = TtsSegmentReuse.ChangedSegments("# First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            SidecarFor(Doc));
        Assert.Contains(0, changed);
    }

    [Fact]
    public void AnUnchangedSegmentComesFromDiskAndItsWordsAreMadeSegmentRelative()
    {
        var dir = Directory.CreateTempSubdirectory("seg-reuse");
        try
        {
            var previous = SidecarFor(Doc, dir.FullName);
            int engineCalls = 0;
            var decorated = TtsSegmentReuse.Decorate(
                (seg, warn) => { engineCalls++; return ([1f, 1f], []); },
                previous, dir.FullName);

            // Segment 2 starts at 2.0s in the old audio and its word sits at 2.25–2.75.
            var third = ParagraphSegmenter.Segment(Doc)[2];
            var (audio, words) = decorated(third, _ => { });

            Assert.Equal(0, engineCalls);                       // reused, not rendered
            Assert.Equal(3, audio.Length);                      // the WAV's own samples
            Assert.Equal(0.25, words[0].StartSeconds, 3);       // 2.25 − 2.00
            Assert.Equal(0.75, words[0].EndSeconds, 3);
            Assert.Equal(third.Index, words[0].ChunkIndex);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void AChangedSegmentFallsThroughToTheEngine()
    {
        var dir = Directory.CreateTempSubdirectory("seg-reuse");
        try
        {
            var previous = SidecarFor(Doc, dir.FullName);
            int engineCalls = 0;
            var decorated = TtsSegmentReuse.Decorate(
                (seg, warn) => { engineCalls++; return ([9f], []); }, previous, dir.FullName);
            var edited = ParagraphSegmenter.Segment("First paragraph.\n\nSomething else entirely.")[1];
            var (audio, _) = decorated(edited, _ => { });
            Assert.Equal(1, engineCalls);
            Assert.Equal([9f], audio);
        }
        finally { dir.Delete(recursive: true); }
    }

    /// <summary>⚠ A missing segment file must RENDER, not throw or return silence — the reuse map is
    /// an optimisation and every miss has to have a correct answer.</summary>
    [Fact]
    public void AMissingSegmentFileFallsThroughToTheEngine()
    {
        var dir = Directory.CreateTempSubdirectory("seg-reuse");
        try
        {
            var previous = SidecarFor(Doc);   // records audio files that were never written
            int engineCalls = 0;
            var decorated = TtsSegmentReuse.Decorate(
                (seg, warn) => { engineCalls++; return ([7f], []); }, previous, dir.FullName);
            decorated(ParagraphSegmenter.Segment(Doc)[0], _ => { });
            Assert.Equal(1, engineCalls);
        }
        finally { dir.Delete(recursive: true); }
    }

    private static void WriteWav(string path, float[] samples, int rate)
    {
        using var w = new BinaryWriter(File.Create(path));
        int bytes = samples.Length * 4;
        w.Write("RIFF"u8); w.Write(36 + bytes); w.Write("WAVE"u8);
        w.Write("fmt "u8); w.Write(16); w.Write((short)3); w.Write((short)1);   // 3 = IEEE float
        w.Write(rate); w.Write(rate * 4); w.Write((short)4); w.Write((short)32);
        w.Write("data"u8); w.Write(bytes);
        foreach (var s in samples) w.Write(s);
    }
}
