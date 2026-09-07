using System.Text.Json;
using System.Text.Json.Serialization;

namespace Vernacula.Tts.Base.Alignment;

/// <summary>
/// The alignment sidecar: the JSON written beside a rendered WAV that says which words were
/// spoken when, and how the input was cut into segments. One definition, here, for both
/// producers — Vernacula.Tts.Backends.CLI (<c>--alignment-out</c>) and the desktop app's TTS
/// jobs — and for the reader that opens either (#132).
/// <para>
/// The snake_case keys are the on-disk contract and are spelled per field with
/// <see cref="JsonPropertyNameAttribute"/> rather than a naming policy, so renaming a C#
/// property cannot silently rename a key. Fields the CLI does not produce
/// (<see cref="SourceText"/>, <see cref="ChunkRecord.AudioFile"/>,
/// <see cref="ChunkRecord.BlockKind"/>) are nullable; a reader must cope with their absence.
/// </para>
/// <para>
/// The two file-naming conventions that hang off a sidecar — where its per-segment WAVs live,
/// and what each is called — are defined here too, so that the writer, the reader and the job
/// list all agree without each spelling them out.
/// </para>
/// </summary>
public sealed class AlignmentSidecar
{
    [JsonPropertyName("audio_path")]
    public string AudioPath { get; set; } = "";

    [JsonPropertyName("sample_rate")]
    public int SampleRate { get; set; }

    [JsonPropertyName("audio_duration_seconds")]
    public double AudioDurationSeconds { get; set; }

    /// <summary>Which aligner timed the words: "nemo_nfa", "chatterbox_attention", "kokoro_duration", …</summary>
    [JsonPropertyName("aligner")]
    public string Aligner { get; set; } = "";

    /// <summary>The NeMo forced-aligner bundle the CLI used, when it did; absent otherwise.</summary>
    [JsonPropertyName("nfa_bundle")]
    public string? NfaBundle { get; set; }

    [JsonPropertyName("chunks")]
    public List<ChunkRecord> Chunks { get; set; } = new();

    /// <summary>Every word in playback order, absolute-timed.</summary>
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

    // ── File conventions ─────────────────────────────────────────────────────

    /// <summary>The folder of per-segment WAVs beside a sidecar: <c>{stem}_segments/</c>.</summary>
    public static string SegmentsDirFor(string sidecarPath) =>
        Path.Combine(Path.GetDirectoryName(sidecarPath) ?? "",
                     Path.GetFileNameWithoutExtension(sidecarPath) + "_segments");

    /// <summary>The file inside that folder holding segment <paramref name="index"/>'s own audio.</summary>
    public static string SegmentFileName(int index) => $"seg_{index:D4}.wav";

    // ── Serialization ────────────────────────────────────────────────────────

    /// <summary>
    /// Writes the sidecar atomically: to a sibling <c>.tmp</c>, then renamed over the target, so
    /// an interrupted write cannot leave a truncated file for a reader to choke on.
    /// </summary>
    public void Save(string path, bool indented = false)
    {
        // Nulls are written as absent keys, so "this producer did not make that field" reads the
        // same on disk whichever producer wrote it, and a CLI sidecar stays as lean as it was.
        var options = new JsonSerializerOptions
        {
            WriteIndented          = indented,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };
        string json = JsonSerializer.Serialize(this, options);
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, json);
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>Reads a sidecar written by either producer. Throws on unreadable JSON.</summary>
    public static AlignmentSidecar Load(string path) =>
        JsonSerializer.Deserialize<AlignmentSidecar>(File.ReadAllText(path))
        ?? throw new InvalidDataException($"\"{path}\" is not an alignment sidecar.");
}

/// <summary>
/// One rendered segment (a markdown paragraph / heading / list item / quote). The desktop app
/// adds the per-segment fields: which file holds this segment's audio on its own, and what
/// kind of block it was — the handles a later per-paragraph re-render needs.
/// </summary>
public sealed class ChunkRecord
{
    [JsonPropertyName("index")]               public int Index { get; set; }
    [JsonPropertyName("audio_start_seconds")] public double AudioStartSeconds { get; set; }
    [JsonPropertyName("audio_end_seconds")]   public double AudioEndSeconds { get; set; }
    [JsonPropertyName("text")]                public string Text { get; set; } = "";
    [JsonPropertyName("word_count")]          public int WordCount { get; set; }
    /// <summary>File name of this segment's own WAV inside the sidecar's segments folder, when written.</summary>
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
