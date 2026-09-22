using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using NAudio.Wave;
using Vernacula.App.Models;
using Vernacula.App.Services.Tts;
using Vernacula.Tts.Base.Alignment;
using Vernacula.Tts.Base.Markdown;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The merged WAV is written paragraph by paragraph rather than assembled at the end.
///
/// <para>
/// ⚠ THIS EXISTS SO A WORD CAN BE CLICKED WHILE THE DOCUMENT IS STILL RENDERING. The reader has
/// word timings for every finished paragraph long before the job ends, but a timing is only
/// useful if there is something to seek INTO — and until this, the only seekable copy of the
/// audio appeared when the run finished, so clicking a word mid-render moved the highlight and
/// not the playhead. What these pin is the property the reader depends on: at any point during a
/// run, the file on disk is readable and describes exactly the paragraphs already rendered.
/// </para>
/// </summary>
public class SegmentedSynthesisGrowingWavTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));

    public SegmentedSynthesisGrowingWavTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    private const int Rate = 24_000;

    private const string Doc = """
        One alpha.

        Two bravo.

        Three charlie.

        Four delta.
        """;

    /// <summary>One second of audio per paragraph, so a duration maps to a paragraph count.</summary>
    private static (float[], IReadOnlyList<AlignedWord>) Render(TextSegment seg)
    {
        var words = seg.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return (new float[Rate], words.Select((w, i) => new AlignedWord
        {
            Text = w, StartSeconds = 0, EndSeconds = 1,
        }).ToList());
    }

    /// <summary>
    /// The duration another component would see, read exactly the way PlaybackService reads it
    /// while the render still holds the file open for writing.
    /// </summary>
    private static double DurationSeenByAReader(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using var reader = new WaveFileReader(stream);
        return reader.TotalTime.TotalSeconds;
    }

    [Fact]
    public void TheFileIsReadableMidRunAndDescribesOnlyWhatIsRendered()
    {
        var wav = Path.Combine(_dir, "grow.wav");
        var seen = new List<double>();

        SegmentedSynthesis.Run(
            new TtsRequest(Doc, wav, "af_heart"), Rate, "test",
            (seg, _) => Render(seg),
            // ⚠ READ FROM THE CHUNK CALLBACK, which is the reader's own vantage point: the event is
            // raised after the write, so whatever a chunk says has been rendered must already be
            // in the file. Reading anywhere else would be testing a race instead of the contract.
            _ => seen.Add(DurationSeenByAReader(wav)),
            null, CancellationToken.None);

        Assert.Equal(4, seen.Count);
        // One second per paragraph: the file grows in step with the chunks, and never leads them.
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0 }, seen.Select(d => Math.Round(d, 3)));
    }

    [Fact]
    public void TheFinishedFileIsStillTheWholeRender()
    {
        // The artifact a finished job leaves behind must not change because of how it is written.
        var wav = Path.Combine(_dir, "final.wav");
        var result = SegmentedSynthesis.Run(
            new TtsRequest(Doc, wav, "af_heart"), Rate, "test",
            (seg, _) => Render(seg), null, null, CancellationToken.None);

        Assert.Equal(wav, result.AudioPath);
        Assert.Equal(4.0, DurationSeenByAReader(wav), 3);
        Assert.Equal(4.0, result.Alignment.AudioDurationSeconds, 3);
        using var reader = new AudioFileReader(wav);
        Assert.Equal(Rate, reader.WaveFormat.SampleRate);
        Assert.Equal(1, reader.WaveFormat.Channels);
    }

    [Fact]
    public void ACancelledRunLeavesThePartialAudioReadable()
    {
        // A cancelled render is exactly when a half-written file matters most: the reader is still
        // showing it, and the paragraphs that did render are the ones the user already heard.
        var wav = Path.Combine(_dir, "cancelled.wav");
        using var cts = new CancellationTokenSource();

        Assert.ThrowsAny<OperationCanceledException>(() => SegmentedSynthesis.Run(
            new TtsRequest(Doc, wav, "af_heart"), Rate, "test",
            (seg, _) => Render(seg),
            ev => { if (ev.ChunkIndex == 1) cts.Cancel(); },
            null, cts.Token));

        Assert.True(File.Exists(wav), "the partial render is still on disk");
        Assert.Equal(2.0, DurationSeenByAReader(wav), 3);
    }
}
