using NAudio.Wave;
using Vernacula.Tts.Base.Alignment;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Paragraph-local re-synthesis: after an edit, render only the paragraphs that actually changed
/// and take the rest from the audio already on disk.
///
/// <para>This is a DECORATOR over the engine's own <see cref="SegmentedSynthesis.SegmentSynthesizer"/>
/// rather than a second rebuild path, and that is the whole design. A segment whose text, kind and
/// level match a chunk in the previous sidecar is answered from that chunk's WAV; everything else
/// falls through to the engine. <see cref="SegmentedSynthesis.Run"/> is then run UNCHANGED, so the
/// concatenation, the absolute word timings, the per-segment files and the sidecar are all produced
/// by exactly the code that produced them the first time — a re-render cannot drift from a first
/// render, because there is only one of them.</para>
///
/// <para>⚠ THE CALLER MUST RENDER TO A TEMPORARY OUTPUT AND SWAP ON SUCCESS. The reuse map reads the
/// OLD segment files, and a run writes `seg_NNNN.wav` under the SAME names — inserting a paragraph
/// shifts every later index, so a run writing in place would overwrite the very files the next
/// reused segment is about to read. Rendering elsewhere also means a failed or cancelled re-render
/// leaves the existing audio intact, which matters because this is triggered automatically by a
/// debounce rather than by someone pressing a button.</para>
/// </summary>
internal static class TtsSegmentReuse
{
    /// <summary>What a re-render did, for the UI to report and for tests to assert on.</summary>
    public sealed record Stats(int Total, int Reused, int Rendered);

    /// <summary>A segment's identity for reuse. Kind and level are part of it because the same words
    /// as a heading and as a paragraph are not the same audio.</summary>
    private readonly record struct Key(string Text, string Kind, int Level);

    private static Key KeyOf(TextSegment s) => new(s.Text, s.Kind.ToString(), s.Level);
    private static Key KeyOf(ChunkRecord c) => new(c.Text, c.BlockKind ?? "", c.BlockLevel);

    /// <summary>
    /// Wraps <paramref name="render"/> so that segments already present in <paramref name="previous"/>
    /// come back from disk. <paramref name="previousSegmentsDir"/> is where that sidecar's per-segment
    /// WAVs live; a chunk with no audio file, or whose file is missing, simply falls through to the
    /// engine.
    /// </summary>
    public static SegmentedSynthesis.SegmentSynthesizer Decorate(
        SegmentedSynthesis.SegmentSynthesizer render,
        AlignmentSidecar previous,
        string previousSegmentsDir,
        Action<Stats>? onProgress = null)
    {
        // ⚠ A key may repeat: duplicating a paragraph is an ordinary edit, and both copies should be
        // answered from the one file rather than re-rendered. So this is a lookup, not a queue —
        // entries are never consumed.
        var byKey = new Dictionary<Key, ChunkRecord>();
        foreach (var c in previous.Chunks)
            if (c.AudioFile is not null) byKey.TryAdd(KeyOf(c), c);

        // Words, grouped by the chunk they belong to and made SEGMENT-RELATIVE. The sidecar stores
        // them absolute; the synthesizer contract is relative to the segment's own start, and
        // SegmentedSynthesis re-offsets them to wherever the segment now lands.
        var wordsByChunk = new Dictionary<int, List<AlignedWord>>();
        foreach (var w in previous.Words)
        {
            if (!wordsByChunk.TryGetValue(w.ChunkIndex, out var list))
                wordsByChunk[w.ChunkIndex] = list = new List<AlignedWord>();
            list.Add(w);
        }

        int reused = 0, rendered = 0;
        return (segment, warn) =>
        {
            if (byKey.TryGetValue(KeyOf(segment), out var chunk) &&
                LoadSegment(Path.Combine(previousSegmentsDir, chunk.AudioFile!)) is { } audio)
            {
                var local = new List<AlignedWord>();
                if (wordsByChunk.TryGetValue(chunk.Index, out var abs))
                    foreach (var w in abs)
                        local.Add(new AlignedWord
                        {
                            Text = w.Text,
                            StartSeconds = w.StartSeconds - chunk.AudioStartSeconds,
                            EndSeconds   = w.EndSeconds   - chunk.AudioStartSeconds,
                            ChunkIndex   = segment.Index,
                        });
                reused++;
                onProgress?.Invoke(new Stats(reused + rendered, reused, rendered));
                return (audio, local);
            }

            var result = render(segment, warn);
            rendered++;
            onProgress?.Invoke(new Stats(reused + rendered, reused, rendered));
            return result;
        };
    }

    /// <summary>The segment's samples, or null when the file is absent or unreadable — either way the
    /// caller renders it instead, which is always a correct answer.</summary>
    private static float[]? LoadSegment(string path)
    {
        if (!File.Exists(path)) return null;
        try
        {
            using var reader = new AudioFileReader(path);
            // ⚠ AS ISampleProvider, and read in blocks. AudioFileReader carries BOTH Read overloads
            // (byte[] from IWaveProvider, float[] from ISampleProvider) so the float one has to be
            // reached through the interface; and `Length` is the source's BYTE length, which is not
            // the sample count for any format but 32-bit float. Accumulating what Read actually
            // returns is right for every input the reader can open.
            // ⚠ NAudio 3: ISampleProvider.Read takes a Span<float>, not (float[], offset, count).
            ISampleProvider samples = reader;
            var block = new float[16384];
            var all = new List<float>((int)Math.Min(reader.Length / sizeof(float) + 1, 1 << 22));
            int read;
            while ((read = samples.Read(block)) > 0)
                all.AddRange(read == block.Length ? block : block[..read]);
            return all.Count == 0 ? null : all.ToArray();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[TtsSegmentReuse] could not reuse {path}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Which segments of <paramref name="newText"/> would be rendered rather than reused. Pure, and
    /// separate from <see cref="Decorate"/> so the UI can say "3 paragraphs will re-render" before
    /// committing to any work — and so the matching rule can be tested without an engine.
    /// </summary>
    public static IReadOnlyList<int> ChangedSegments(string newText, AlignmentSidecar previous)
    {
        var have = new HashSet<Key>();
        foreach (var c in previous.Chunks)
            if (c.AudioFile is not null) have.Add(KeyOf(c));

        var changed = new List<int>();
        foreach (var seg in ParagraphSegmenter.Segment(newText))
            if (!have.Contains(KeyOf(seg))) changed.Add(seg.Index);
        return changed;
    }
}
