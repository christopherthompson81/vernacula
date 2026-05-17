namespace Vernacula.Base.Alignment;

/// <summary>
/// One word's audio-time alignment as produced by an
/// <see cref="IForcedAligner"/>. Times are seconds from the start of
/// the audio buffer the aligner was given — callers translating to
/// playback-absolute time (e.g. for long-form concatenated audio)
/// add the chunk's audio offset themselves.
///
/// Source-position mapping (which character range in some source
/// document this word came from) is deliberately NOT here — that's
/// caller-side concern. The word-highlighting consumer in the
/// Avalonia app combines these timings with the
/// <see cref="Chatterbox.Base.Markdown.MarkdownTextExtractor"/>'s
/// source-range index by matching the aligner's emitted word text
/// against the extracted text's word stream.
/// </summary>
/// <param name="Text">The word as the aligner saw it. Sentencepiece-style
/// tokens with the leading <c>▁</c> word-boundary marker have been
/// joined and the marker stripped; this is human-readable.</param>
/// <param name="StartSeconds">Start of the word in audio time.</param>
/// <param name="EndSeconds">End of the word in audio time. May equal
/// <see cref="StartSeconds"/> for zero-frame degenerate words; consumers
/// should not assume strict inequality.</param>
public sealed record WordTiming(string Text, double StartSeconds, double EndSeconds);
