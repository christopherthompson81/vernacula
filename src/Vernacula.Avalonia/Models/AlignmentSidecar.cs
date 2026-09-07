using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Vernacula.App.Models;

/// <summary>
/// Deserialized shape of the JSON sidecar emitted by Vernacula.Tts.Backends.CLI
/// (`--alignment-out` flag, PR #72). The snake_case JSON keys are
/// mapped via <see cref="JsonPropertyNameAttribute"/> rather than a
/// global naming policy so the consumer contract is explicit per
/// field.
/// </summary>
public sealed class AlignmentSidecar
{
    [JsonPropertyName("audio_path")]
    public string AudioPath { get; set; } = "";

    [JsonPropertyName("sample_rate")]
    public int SampleRate { get; set; }

    [JsonPropertyName("audio_duration_seconds")]
    public double AudioDurationSeconds { get; set; }

    [JsonPropertyName("aligner")]
    public string Aligner { get; set; } = "";

    [JsonPropertyName("chunks")]
    public List<ChunkRecord> Chunks { get; set; } = new();

    [JsonPropertyName("words")]
    public List<AlignedWord> Words { get; set; } = new();

    /// <summary>
    /// The markdown/text the job rendered, verbatim. Written by the desktop app's TTS jobs so
    /// the reader can rebuild the structured view without the input file; absent in sidecars
    /// from the CLI. The word sequence in <see cref="Words"/> is 1:1 with the whitespace-split
    /// output of MarkdownTextExtractor.Extract(SourceText), which is what lets the reader attach
    /// timing by running index.
    /// </summary>
    [JsonPropertyName("source_text")]
    public string? SourceText { get; set; }
}

/// <summary>
/// One rendered segment (a markdown paragraph / heading / list item / quote). The desktop app
/// adds the per-segment fields: which file holds this segment's audio on its own, and what
/// kind of block it was — the handles a later per-paragraph re-render needs.
/// </summary>
public sealed class ChunkRecord
{
    [JsonPropertyName("index")]              public int Index { get; set; }
    [JsonPropertyName("audio_start_seconds")] public double AudioStartSeconds { get; set; }
    [JsonPropertyName("audio_end_seconds")]   public double AudioEndSeconds { get; set; }
    [JsonPropertyName("text")]                public string Text { get; set; } = "";
    [JsonPropertyName("word_count")]          public int WordCount { get; set; }
    /// <summary>File name of this segment's own WAV inside the job's segments folder, when written.</summary>
    [JsonPropertyName("audio_file")]          public string? AudioFile { get; set; }
    /// <summary>BlockKind name: "Paragraph", "Heading", "ListItem", "Quote".</summary>
    [JsonPropertyName("block_kind")]          public string? BlockKind { get; set; }
    [JsonPropertyName("block_level")]         public int BlockLevel { get; set; }
}

public sealed class AlignedWord
{
    [JsonPropertyName("text")]          public string Text { get; set; } = "";
    [JsonPropertyName("start_seconds")] public double StartSeconds { get; set; }
    [JsonPropertyName("end_seconds")]   public double EndSeconds { get; set; }
    [JsonPropertyName("chunk_index")]   public int ChunkIndex { get; set; }
}
