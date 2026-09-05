using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Vernacula.Tts.App.Models;

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
}

public sealed class ChunkRecord
{
    [JsonPropertyName("index")]              public int Index { get; set; }
    [JsonPropertyName("audio_start_seconds")] public double AudioStartSeconds { get; set; }
    [JsonPropertyName("audio_end_seconds")]   public double AudioEndSeconds { get; set; }
    [JsonPropertyName("text")]                public string Text { get; set; } = "";
    [JsonPropertyName("word_count")]          public int WordCount { get; set; }
}

public sealed class AlignedWord
{
    [JsonPropertyName("text")]          public string Text { get; set; } = "";
    [JsonPropertyName("start_seconds")] public double StartSeconds { get; set; }
    [JsonPropertyName("end_seconds")]   public double EndSeconds { get; set; }
    [JsonPropertyName("chunk_index")]   public int ChunkIndex { get; set; }
}
