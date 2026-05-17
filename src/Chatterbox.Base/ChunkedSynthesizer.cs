using System.Diagnostics;
using System.Threading.Channels;

namespace Chatterbox.Base;

/// <summary>
/// Per-chunk timing record for callers that want to log + investigate.
/// All durations are wall-clock at the producer/consumer thread that
/// produced them — under pipelined execution, voc starts *while* the
/// next chunk's LM is running, so summing the columns OVER-counts
/// total wall time. Use <see cref="ChunkSynthesisResult.TotalWallMs"/>
/// for the actual end-to-end number.
/// </summary>
/// <param name="ChunkIndex">0-based index in input order.</param>
/// <param name="LmMs">LM rollout wall-clock for this chunk.</param>
/// <param name="LmSteps">Number of autoregressive steps the LM took before STOP_SPEECH (or maxSteps cap).</param>
/// <param name="VocoderMs">Vocoder wall-clock for this chunk.</param>
/// <param name="AudioSamples">Length of the chunk's waveform in samples (24 kHz mono).</param>
public sealed record ChunkTiming(int ChunkIndex, long LmMs, int LmSteps, long VocoderMs, int AudioSamples);

/// <summary>
/// Output of <see cref="ChunkedSynthesizer.Synthesize"/>: waveforms in
/// input order, plus per-chunk timings and a true end-to-end wall.
/// </summary>
public sealed record ChunkSynthesisResult(
    IReadOnlyList<float[]> Waveforms,
    IReadOnlyList<ChunkTiming> ChunkTimings,
    long TotalWallMs);

/// <summary>
/// Long-form chunk synthesizer that overlaps LM rollout for chunk N+1
/// with vocoder synthesis for chunk N. Constructed over an existing
/// <see cref="ChatterboxPipeline"/>; uses its <see cref="AcousticLM"/>
/// and <see cref="Vocoder"/> sessions directly.
///
/// Concurrency model: one producer task running LM rollouts in input
/// order, one consumer task running the vocoder. A bounded
/// <see cref="Channel{T}"/> hands the LM's <c>speech_tokens</c> output
/// to the vocoder; bound is intentionally small so memory doesn't
/// balloon if the vocoder is faster than the LM (it never is for our
/// chunk sizes, but defensive).
///
/// Safety: LM and Vocoder own different ORT <c>InferenceSession</c>
/// objects. ORT sessions are not internally synchronized for concurrent
/// <c>Run()</c> calls on the SAME session, but DIFFERENT sessions can
/// run on different threads concurrently — that's exactly the
/// configuration here. The shared <see cref="SpeakerEmbedding"/> is
/// effectively read-only after construction (CPU arrays; no internal
/// state mutated by either consumer).
/// </summary>
public sealed class ChunkedSynthesizer
{
    private readonly AcousticLM _lm;
    private readonly Vocoder _vocoder;

    /// <summary>Construct over already-loaded LM + Vocoder sessions.
    /// Caller retains ownership; ChunkedSynthesizer doesn't dispose them.</summary>
    public ChunkedSynthesizer(AcousticLM lm, Vocoder vocoder)
    {
        _lm = lm;
        _vocoder = vocoder;
    }

    /// <summary>Convenience overload — pulls the LM + Vocoder from a pipeline.</summary>
    public ChunkedSynthesizer(ChatterboxPipeline pipeline)
        : this(pipeline.Lm, pipeline.Vocoder) { }

    /// <summary>
    /// Synthesize each <paramref name="textTokenIdsPerChunk"/> entry
    /// in order. The same <paramref name="speaker"/> is used for every
    /// chunk (synthesizing N chunks in one voice is the long-form
    /// audiobook case). Returns waveforms in input order.
    /// </summary>
    /// <param name="useIoBinding">
    /// Forwarded to <see cref="AcousticLM.Generate"/>; null = auto-detect
    /// from the LM session's effective EP.
    /// </param>
    public ChunkSynthesisResult Synthesize(
        SpeakerEmbedding speaker,
        IReadOnlyList<long[]> textTokenIdsPerChunk,
        bool? useIoBinding = null,
        float exaggeration = ChatterboxConstants.DefaultExaggeration,
        int maxSteps = ChatterboxConstants.DefaultMaxLmSteps)
    {
        int n = textTokenIdsPerChunk.Count;
        if (n == 0)
            return new ChunkSynthesisResult(Array.Empty<float[]>(), Array.Empty<ChunkTiming>(), 0);

        // Bounded channel between LM and vocoder. Bound=1 because the
        // vocoder is faster than the LM for typical chunks; the LM is
        // the critical path, so having more buffered LM outputs in
        // flight wouldn't help (vocoder is never starved) and would
        // just waste memory (each speech_tokens[] is ~few KB but adds up).
        var channel = Channel.CreateBounded<(int idx, long[] speechTokens, long lmMs, int lmSteps)>(
            new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = true,
            });

        var waveforms = new float[n][];
        var timings = new ChunkTiming[n];
        var totalSw = Stopwatch.StartNew();

        // ── Producer: LM ─────────────────────────────────────────────
        var lmTask = Task.Run(() =>
        {
            try
            {
                for (int i = 0; i < n; i++)
                {
                    var sw = Stopwatch.StartNew();
                    var lm = _lm.Generate(
                        speaker.CondEmb, textTokenIdsPerChunk[i],
                        useIoBinding: useIoBinding,
                        exaggeration: exaggeration,
                        maxSteps: maxSteps);
                    sw.Stop();
                    var speechTokens = lm.BuildSpeechTokens(speaker.AudioTokens);
                    // Block if the vocoder hasn't drained the previous one yet.
                    channel.Writer.WriteAsync((i, speechTokens, sw.ElapsedMilliseconds, lm.Steps))
                                  .AsTask().GetAwaiter().GetResult();
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        });

        // ── Consumer: Vocoder ─────────────────────────────────────────
        // Runs on the current thread. Each iteration may block waiting
        // for the next LM output, but the LM is on another thread, so
        // we're not stalling its work.
        foreach (var (idx, speechTokens, lmMs, lmSteps) in
                 channel.Reader.ReadAllAsync().ToBlockingEnumerable())
        {
            var sw = Stopwatch.StartNew();
            var wav = _vocoder.Synthesize(
                speechTokens, speaker.SpeakerEmbeddings, speaker.SpeakerFeatures);
            sw.Stop();
            waveforms[idx] = wav;
            timings[idx] = new ChunkTiming(idx, lmMs, lmSteps, sw.ElapsedMilliseconds, wav.Length);
        }

        // Surface any LM exception that the channel.Complete() hid.
        lmTask.GetAwaiter().GetResult();
        totalSw.Stop();

        return new ChunkSynthesisResult(waveforms, timings, totalSw.ElapsedMilliseconds);
    }
}
