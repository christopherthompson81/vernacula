using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Vernacula.App.Models;
using Vernacula.App.Services.Tts;
using Vernacula.Tts.Base.Alignment;
using Vernacula.Tts.Base.Markdown;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// Kokoro renders a GROUP of paragraphs per ONNX call (#190). The driver that arranges that
/// lives in <see cref="SegmentedSynthesis"/> and is shared by every engine, so what it must not
/// change is the part a user sees: paragraphs stream out in document order, each carrying its
/// own audio, whether or not the engine batches.
///
/// These use fake synthesizers rather than a model, so they pin the arrangement itself —
/// grouping, ordering and the solo first segment — which is exactly what an end-to-end run
/// through a real engine cannot isolate.
/// </summary>
public class SegmentedSynthesisBatchTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));

    public SegmentedSynthesisBatchTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    private const string Doc = """
        One alpha.

        Two bravo.

        Three charlie.

        Four delta.

        Five echo.

        Six foxtrot.
        """;

    // Audio a test can trace back to its segment: length encodes the paragraph's word count.
    private static (float[], IReadOnlyList<AlignedWord>) Render(TextSegment seg)
    {
        var words = seg.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var audio = new float[100 * words.Length];
        var aligned = words.Select((w, i) => new AlignedWord
        {
            Text = w, StartSeconds = i, EndSeconds = i + 1,
        }).ToList();
        return (audio, aligned);
    }

    private TtsRequest NewRequest(string tag) =>
        new(Doc, Path.Combine(_dir, $"{tag}.wav"), "af_heart");

    private (List<int> order, List<int> audioLengths, List<string> texts) Run(
        int batchSize, bool batched, List<int> groupSizes)
    {
        var order = new List<int>();
        var lens = new List<int>();
        var texts = new List<string>();
        SegmentedSynthesis.SegmentBatchSynthesizer? batchFn = null;
        if (batched)
            batchFn = (segs, _) =>
            {
                groupSizes.Add(segs.Count);
                return segs.Select(Render).ToList();
            };

        SegmentedSynthesis.Run(
            NewRequest(batched ? "batched" : "plain"), 24000, "test",
            (seg, _) => Render(seg),
            ev => { order.Add(ev.ChunkIndex); lens.Add(ev.Audio24k.Length); texts.Add(ev.ChunkText); },
            null, CancellationToken.None, batchFn, batchSize);
        return (order, lens, texts);
    }

    [Fact]
    public void BatchingChangesNothingAUserSees()
    {
        var plainGroups = new List<int>();
        var plain = Run(1, batched: false, plainGroups);
        var batchGroups = new List<int>();
        var batched = Run(4, batched: true, batchGroups);

        Assert.Equal(plain.order, batched.order);                 // same streaming order
        Assert.Equal(plain.audioLengths, batched.audioLengths);   // each paragraph keeps its own audio
        Assert.Equal(plain.texts, batched.texts);
        Assert.Equal(Enumerable.Range(0, plain.order.Count), plain.order);
    }

    [Fact]
    public void TheFirstSegmentIsRenderedAloneSoPlaybackStartsNoLater()
    {
        var groups = new List<int>();
        var r = Run(4, batched: true, groups);
        // 6 paragraphs, batch 4: segment 0 solo (never handed to the batch fn), then 4, then 1.
        Assert.Equal([4, 1], groups);
        Assert.Equal(6, r.order.Count);
    }

    [Fact]
    public void ASingleParagraphDocumentDoesNotBatch()
    {
        var groups = new List<int>();
        var order = new List<int>();
        SegmentedSynthesis.Run(
            new TtsRequest("Only one.", Path.Combine(_dir, "one.wav"), "af_heart"), 24000, "test",
            (seg, _) => Render(seg),
            ev => order.Add(ev.ChunkIndex), null, CancellationToken.None,
            (segs, _) => { groups.Add(segs.Count); return segs.Select(Render).ToList(); }, 4);
        Assert.Empty(groups);
        Assert.Equal([0], order);
    }

    [Fact]
    public void AShortReturnFromTheBatchSynthesizerSaysSoPlainly()
    {
        // The delegate's contract is one result per segment, in order. Three backends can
        // implement it; a short return must not surface as an IndexOutOfRange deep inside.
        var ex = Assert.Throws<InvalidOperationException>(() =>
            SegmentedSynthesis.Run(
                NewRequest("short"), 24000, "test",
                (seg, _) => Render(seg), null, null, CancellationToken.None,
                (segs, _) => segs.Take(segs.Count - 1).Select(Render).ToList(), 4));
        Assert.Contains("results for", ex.Message);
    }
}
