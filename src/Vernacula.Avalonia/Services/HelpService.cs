using System.Reflection;
using System.Text;
using Avalonia.Platform;

namespace Vernacula.App.Services;

/// <summary>
/// Represents a single help article, keyed by its stable topic_id.
/// </summary>
/// <param name="TopicId">Stable snake_case identifier from the YAML frontmatter.</param>
/// <param name="Title">Display name shown in the sidebar.</param>
/// <param name="ResourcePath">
///   Language-neutral path relative to the per-language Help folder,
///   e.g. "first_steps/model_precision.md" or "index.md".
///   The full manifest resource name is built at load time:
///   "parakeet_csharp.Help.{lang}." + path.Replace('/','.')
/// </param>
internal record HelpTopic(string TopicId, string Title, string ResourcePath);

/// <summary>A named group of related topics shown as a section in the help sidebar.</summary>
internal record HelpGroup(string GroupTitle, IReadOnlyList<HelpTopic> Topics);

internal static class HelpService
{
    // ── Index / landing page ─────────────────────────────────────────────────
    internal static readonly HelpTopic IndexTopic =
        new("help_index", "Help", "index.md");

    // ── Navigation tree ──────────────────────────────────────────────────────
    internal static readonly IReadOnlyList<HelpGroup> Groups =
    [
        new HelpGroup("help_group_first_steps",
        [
            new HelpTopic("first_steps_settings_window",    "Settings",                      "first_steps/settings_window.md"),
            new HelpTopic("first_steps_language",           "Language",                      "first_steps/language.md"),
            new HelpTopic("first_steps_theme",              "Theme",                         "first_steps/theme.md"),
            new HelpTopic("first_steps_downloading_models", "Downloading Models",            "first_steps/downloading_models.md"),
            new HelpTopic("first_steps_model_precision",    "Model Weight Precision",        "first_steps/model_precision.md"),
            new HelpTopic("first_steps_cuda_installation",  "CUDA & cuDNN Installation",     "first_steps/cuda_installation.md"),
        ]),
        new HelpGroup("help_group_operations",
        [
            new HelpTopic("operations_new_transcription",         "New Transcription",             "operations/new_transcription.md"),
            new HelpTopic("operations_monitoring_jobs",           "Monitoring Jobs",               "operations/monitoring_jobs.md"),
            new HelpTopic("operations_loading_completed_jobs",    "Loading Completed Jobs",        "operations/loading_completed_jobs.md"),
            new HelpTopic("operations_editing_transcripts",       "Editing Transcripts",           "operations/editing_transcripts.md"),
            new HelpTopic("operations_editing_speaker_names",     "Editing Speaker Names",         "operations/editing_speaker_names.md"),
            new HelpTopic("operations_exporting_results",         "Exporting Results",             "operations/exporting_results.md"),
            new HelpTopic("operations_pausing_resuming_removing", "Pausing / Resuming / Removing", "operations/pausing_resuming_removing.md"),
            new HelpTopic("operations_bulk_add_jobs",             "Bulk Add Jobs",                 "operations/bulk_add_jobs.md"),
        ]),
    ];

    /// <summary>All leaf topics flattened across all groups.</summary>
    internal static readonly IReadOnlyList<HelpTopic> AllTopics =
        Groups.SelectMany(g => g.Topics).ToArray();

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Loads and returns the Markdown content of <paramref name="topic"/> with YAML
    /// frontmatter stripped.  Tries the current UI language first and falls back to
    /// English if no localised resource exists.
    /// </summary>
    internal static string LoadMarkdown(HelpTopic topic)
    {
        string lang = Loc.Instance.CurrentLanguage;

        using var stream = OpenHelpResource(topic.ResourcePath, lang)
            ?? OpenHelpResource(topic.ResourcePath, "en")
            ?? throw new InvalidOperationException($"Help resource not found: {topic.ResourcePath}");

        using var reader = new StreamReader(stream, Encoding.UTF8);
        return StripFrontmatter(reader.ReadToEnd());
    }

    private static Stream? OpenHelpResource(string relativePath, string lang)
    {
        var uri = new Uri($"avares://Vernacula.Avalonia/Help/{lang}/{relativePath}");
        return AssetLoader.Exists(uri) ? AssetLoader.Open(uri) : null;
    }
    /// <summary>Returns the topic matching <paramref name="topicId"/>, or null.</summary>
    internal static HelpTopic? FindById(string topicId) =>
        topicId == IndexTopic.TopicId
            ? IndexTopic
            : AllTopics.FirstOrDefault(t => t.TopicId == topicId);
    /// <summary>
    /// Resolves a relative Markdown link (e.g. "../first_steps/downloading_models.md")
    /// from the directory of <paramref name="current"/> to a <see cref="HelpTopic"/>.
    /// Both <paramref name="current"/>.ResourcePath and the stored paths in AllTopics are
    /// language-neutral (e.g. "operations/monitoring_jobs.md"), so the resolution works
    /// regardless of which language is currently loaded.
    /// Returns null if no matching topic is found.
    /// </summary>
    internal static HelpTopic? ResolveRelativeLink(HelpTopic current, string relativeLink)
    {
        // Get the directory portion of the language-neutral path (forward-slash)
        string dir = current.ResourcePath.Contains('/')
            ? current.ResourcePath[..current.ResourcePath.LastIndexOf('/')]
            : string.Empty;
        // Combine and normalise using a path stack (handles ".." segments)
        string combined = dir.Length > 0 ? $"{dir}/{relativeLink}" : relativeLink;
        var    stack    = new Stack<string>();
        foreach (string seg in combined.Split('/'))
        {
            if (seg == "..")      { if (stack.Count > 0) stack.Pop(); }
            else if (seg != ".") { stack.Push(seg); }
        }
        string resolved = string.Join("/", stack.Reverse());
        // Match against the index topic first, then all leaf topics
        if (resolved == IndexTopic.ResourcePath) return IndexTopic;
        return AllTopics.FirstOrDefault(t => t.ResourcePath == resolved);
    }
    // ── Internal helpers ─────────────────────────────────────────────────────
    private static string StripFrontmatter(string text)
    {
        if (!text.StartsWith("---", StringComparison.Ordinal)) return text;
        int closingDashes = text.IndexOf("\n---", 3, StringComparison.Ordinal);
        if (closingDashes < 0) return text;
        // Advance past the closing "---" line (skip to the character after the newline)
        int afterLine = text.IndexOf('\n', closingDashes + 1);
        return afterLine < 0 ? string.Empty : text[(afterLine + 1)..].TrimStart('\r', '\n');
    }
}
