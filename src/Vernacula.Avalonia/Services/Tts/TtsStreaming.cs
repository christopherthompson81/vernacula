using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vernacula.App.Models;

namespace Vernacula.App.Services.Tts;

/// <summary>Which TTS engine drives synthesis.</summary>
public enum TtsBackendKind
{
    /// <summary>Chatterbox LM + vocoder, voice-WAV prompt, attention word alignment.</summary>
    Chatterbox,

    /// <summary>Kokoro-82M, named voice + speed, duration-based word alignment.</summary>
    Kokoro,

    /// <summary>OmniVoice IPA fine-tune via vernacula-phonemizer: any of its languages, a stored
    /// voice from the web demo's library, proportional (estimated) word alignment.</summary>
    OmniVoice,
}

/// <summary>Final synthesis output: the written WAV path + the alignment sidecar.</summary>
public sealed record SynthesisResult(string AudioPath, AlignmentSidecar Alignment);

/// <summary>Per-chunk streaming event payload (shared by all backends).</summary>
/// <param name="ChunkIndex">0-based index in input order.</param>
/// <param name="TotalChunks">Total chunks the synthesis will produce.</param>
/// <param name="Audio24k">Raw float32 mono samples at 24 kHz.</param>
/// <param name="ChunkText">The reference text for this chunk (post-markdown-extraction).</param>
/// <param name="AudioStartSeconds">Absolute start in the concatenated audio timeline.</param>
/// <param name="Words">Per-word alignments with absolute timings (may be empty).</param>
public sealed record ChunkProducedEvent(
    int ChunkIndex,
    int TotalChunks,
    float[] Audio24k,
    string ChunkText,
    double AudioStartSeconds,
    IReadOnlyList<AlignedWord> Words);

/// <summary>Status-line events for the UI. Phase is a free-form short string.</summary>
public sealed record ProgressEvent(string Phase, int? ChunkIndex = null, int? TotalChunks = null);

/// <summary>
/// A backend-agnostic synthesis request. <see cref="Voice"/> is interpreted by the
/// backend: a voice-WAV path for Chatterbox, a named voice (e.g. "af_heart") for Kokoro, a
/// library voice id for OmniVoice. <see cref="Speed"/> applies to Kokoro; <see cref="Lang"/>
/// (a vernacula-phonemizer code) and <see cref="NumStep"/> (diffusion steps) to OmniVoice.
/// </summary>
public sealed record TtsRequest(string Text, string OutWavPath, string Voice, float Speed = 1.0f,
                                string? Lang = null, int NumStep = 32);

/// <summary>
/// A streaming TTS backend. Implementations lazily load their models on the first call
/// and reuse them. Chunks stream back via <c>onChunkProduced</c>; the final concatenated
/// WAV + alignment is returned when done.
/// </summary>
public interface ITtsBackend : IDisposable
{
    /// <summary>Output sample rate of the produced audio (Hz).</summary>
    int SampleRate { get; }

    Task<SynthesisResult> SynthesizeStreamingAsync(
        TtsRequest request,
        Action<ChunkProducedEvent>? onChunkProduced = null,
        Action<ProgressEvent>? onProgress = null,
        CancellationToken cancellationToken = default);
}
