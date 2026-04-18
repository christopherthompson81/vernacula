namespace Vernacula.App.Models;

/// <summary>
/// A user-selectable language for ASR backends that support forced language
/// (e.g. Cohere, Qwen3-ASR). <see cref="Code"/> is an ISO 639-1 code ("en", "fr"),
/// or empty string to mean "auto-detect". <see cref="DisplayName"/> is the UI label.
/// </summary>
public record AsrLanguageOption(string Code, string DisplayName)
{
    public override string ToString() => DisplayName;
}
