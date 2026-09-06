using NAudio.Wave;
using Vernacula.App.Models;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// The synthesis loop every engine shares: cut the document into segments (one per markdown
/// block — see <see cref="ParagraphSegmenter"/>), render each through the engine's own
/// per-segment function, stream it out, keep its audio as its own file, and at the end write
/// the concatenated WAV plus the alignment sidecar. Engines differ only in how a segment
/// becomes audio + word timings (and in how they split a segment that is too long for one
/// pass, which stays their business — a segment always comes back whole).
/// </summary>
internal static class SegmentedSynthesis
{
    /// <summary>Renders one segment: audio at the engine's sample rate + words timed from the segment's start.</summary>
    public delegate (float[] Audio, IReadOnlyList<AlignedWord> Words) SegmentSynthesizer(TextSegment segment, Action<string> warn);

    public static SynthesisResult Run(
        TtsRequest request,
        int sampleRate,
        string aligner,
        SegmentSynthesizer synthesize,
        Action<ChunkProducedEvent>? onChunkProduced,
        Action<ProgressEvent>? onProgress,
        CancellationToken ct)
    {
        var extract = MarkdownTextExtractor.Extract(request.Text);
        if (string.IsNullOrWhiteSpace(extract.Text))
            throw new InvalidOperationException("Text is empty after markdown extraction.");
        var segments = ParagraphSegmenter.Segment(extract);
        int total = segments.Count;
        onProgress?.Invoke(new ProgressEvent($"{total} paragraph{(total == 1 ? "" : "s")}", null, total));

        if (request.SegmentsDir is { } segDir) Directory.CreateDirectory(segDir);

        var audios = new float[total][];
        var records = new List<ChunkRecord>(total);
        var allWords = new List<AlignedWord>();
        int sampleCursor = 0;

        for (int idx = 0; idx < total; idx++)
        {
            ct.ThrowIfCancellationRequested();
            var seg = segments[idx];
            onProgress?.Invoke(new ProgressEvent("synthesizing", idx + 1, total));

            var (audio, localWords) = synthesize(seg,
                msg => onProgress?.Invoke(new ProgressEvent(msg, idx + 1, total)));

            double startSec = sampleCursor / (double)sampleRate;
            double endSec   = (sampleCursor + audio.Length) / (double)sampleRate;
            sampleCursor += audio.Length;
            audios[idx] = audio;

            var absWords = new List<AlignedWord>(localWords.Count);
            foreach (var w in localWords)
                absWords.Add(new AlignedWord
                {
                    Text = w.Text,
                    StartSeconds = startSec + w.StartSeconds,
                    EndSeconds   = startSec + w.EndSeconds,
                    ChunkIndex   = idx,
                });
            allWords.AddRange(absWords);

            string? audioFile = null;
            if (request.SegmentsDir is { } dir)
            {
                audioFile = $"seg_{idx:D4}.wav";
                WriteWav(Path.Combine(dir, audioFile), audio, sampleRate);
            }

            records.Add(new ChunkRecord
            {
                Index = idx,
                AudioStartSeconds = startSec,
                AudioEndSeconds = endSec,
                Text = seg.Text,
                WordCount = absWords.Count,
                AudioFile = audioFile,
                BlockKind = seg.Kind.ToString(),
                BlockLevel = seg.Level,
            });

            onProgress?.Invoke(new ProgressEvent($"paragraph {idx + 1}/{total} ready", idx + 1, total));
            onChunkProduced?.Invoke(new ChunkProducedEvent(idx, total, audio, seg.Text, startSec, absWords));
        }

        ChatterboxSynthesisService.WriteWavFromChunks(request.OutWavPath, audios, sampleRate);

        var sidecar = new AlignmentSidecar
        {
            AudioPath = request.OutWavPath,
            SampleRate = sampleRate,
            AudioDurationSeconds = sampleCursor / (double)sampleRate,
            Aligner = aligner,
            Chunks = records,
            Words = allWords,
        };
        return new SynthesisResult(request.OutWavPath, sidecar);
    }

    /// <summary>Joins sub-chunk results into one segment result, shifting each word by the audio before it.</summary>
    public static (float[] Audio, IReadOnlyList<AlignedWord> Words) Join(
        IReadOnlyList<(float[] Audio, IReadOnlyList<(string Text, double Start, double End)> Words)> parts, int sampleRate)
    {
        int total = 0;
        foreach (var p in parts) total += p.Audio.Length;
        var audio = new float[total];
        var words = new List<AlignedWord>();
        int off = 0;
        foreach (var p in parts)
        {
            double offSec = off / (double)sampleRate;
            foreach (var (text, start, end) in p.Words)
                words.Add(new AlignedWord { Text = text, StartSeconds = offSec + start, EndSeconds = offSec + end });
            Array.Copy(p.Audio, 0, audio, off, p.Audio.Length);
            off += p.Audio.Length;
        }
        return (audio, words);
    }

    public static void WriteWav(string path, float[] samples, int sampleRate)
    {
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        using var writer = new WaveFileWriter(path, fmt);
        writer.WriteSamples(samples, 0, samples.Length);
    }
}
