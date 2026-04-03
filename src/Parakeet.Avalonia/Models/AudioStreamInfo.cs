namespace ParakeetCSharp.Models;

/// <summary>Metadata about a single audio stream within a media file.</summary>
public record AudioStreamInfo(
    int     StreamIndex,
    string  CodecName,
    string? Language,        // ISO 639-2 tag, e.g. "eng", "fra" — null if untagged
    string? Title,           // optional stream title tag
    int     Channels,
    int     SampleRate,
    double  DurationSeconds);
