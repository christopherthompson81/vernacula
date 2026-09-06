using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform.Storage;

namespace Vernacula.App.Services;

/// <summary>
/// File/folder pickers for view models that have no window of their own (the Settings TTS
/// tab, the New TTS Job dialog). The picker is owned by the active window so it opens over
/// the dialog the user is looking at, not behind it over the main window.
/// </summary>
internal static class StoragePickers
{
    private static TopLevel? Owner()
    {
        if (Application.Current?.ApplicationLifetime is not IClassicDesktopStyleApplicationLifetime desktop)
            return null;
        var window = desktop.Windows.FirstOrDefault(w => w.IsActive) ?? desktop.MainWindow;
        return window is null ? null : TopLevel.GetTopLevel(window);
    }

    public static async Task<string?> PickFolderAsync(string title)
    {
        var top = Owner();
        if (top is null) return null;
        var picks = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
        });
        return picks.Count > 0 ? picks[0].TryGetLocalPath() : null;
    }

    public static async Task<string?> PickFileAsync(string title, params FilePickerFileType[] filters)
    {
        var top = Owner();
        if (top is null) return null;
        var picks = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = filters,
        });
        return picks.Count > 0 ? picks[0].TryGetLocalPath() : null;
    }

    public static readonly FilePickerFileType TextDocuments =
        new("Text and markdown") { Patterns = ["*.md", "*.markdown", "*.txt"] };

    public static readonly FilePickerFileType AudioClips =
        new("Audio") { Patterns = ["*.wav", "*.flac", "*.mp3"] };

    public static readonly FilePickerFileType AllFiles =
        new("All files") { Patterns = ["*"] };
}
