namespace Vernacula.Base.Alignment;

/// <summary>
/// Model-agnostic interface for forced-alignment backends. Given audio
/// and the known text it contains, returns per-word audio-time
/// alignments. Multiple implementations are expected (per issue #36):
/// <see cref="NemoNfaAligner"/> for NeMo CTC checkpoints landed first;
/// future MMS / wav2vec2 implementations plug behind the same interface.
///
/// Threading contract: implementations may hold expensive resources
/// (ORT sessions, in-memory tokenizers). One instance per concurrent
/// caller; the interface does not promise re-entrancy on a single
/// instance.
/// </summary>
public interface IForcedAligner : IDisposable
{
    /// <summary>
    /// Align <paramref name="text"/> to <paramref name="audio16k"/>.
    /// </summary>
    /// <param name="audio16k">Mono 16 kHz float32 PCM in [-1, 1].
    /// Implementations resample if their underlying model expects
    /// a different rate, but 16 kHz is the universal default.</param>
    /// <param name="text">The reference text to align. Implementations
    /// tokenize with whatever encoder the underlying model uses.</param>
    /// <param name="languageIso">BCP-47 / ISO-639 hint, e.g. "en", "ja-Hira".
    /// English-only aligners (<see cref="NemoNfaAligner"/> with the
    /// recommended NVIDIA checkpoints) ignore this and assume English.
    /// Multilingual aligners (future MMS) use it to gate phoneme inventory.</param>
    /// <returns>Word timings in input order. Empty when text contains no
    /// alignable tokens (e.g. punctuation-only). May omit words the
    /// aligner couldn't place — callers should treat the result as a
    /// best-effort alignment, not a 1:1 mapping of input words.</returns>
    IReadOnlyList<WordTiming> Align(float[] audio16k, string text, string languageIso);
}
