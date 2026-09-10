using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Avalonia.Platform;

namespace Vernacula.App;

/// <summary>
/// Sets the macOS Dock icon at runtime.
/// </summary>
/// <remarks>
/// macOS takes an app's Dock icon from its <c>.app</c> bundle's <c>Info.plist</c>, and this
/// project builds a bare executable — so <c>dotnet run</c> shows the generic .NET rocket no
/// matter what <c>ApplicationIcon</c> says (that is a Windows PE resource, and `icon.ico` is
/// not a format macOS reads anyway). Avalonia's <c>Window.Icon</c> does not help either: on
/// macOS that is the title-bar/proxy icon, not the Dock tile.
///
/// The one thing that works for an un-bundled process is asking AppKit directly, which is
/// what this does — `[[NSApplication sharedApplication] setApplicationIconImage:]` through
/// the Objective-C runtime. It also wins over a bundle's icon when both exist, so it stays
/// correct if the app is bundled later.
///
/// Everything here is best-effort: a missing asset, a machine that is not a Mac, or an
/// AppKit that will not load leaves the default icon rather than failing startup. A dock
/// icon is not worth a crash.
/// </remarks>
[SupportedOSPlatform("macos")]
internal static class MacDockIcon
{
    private const string Objc = "/usr/lib/libobjc.dylib";
    private const string AppKitPath = "/System/Library/Frameworks/AppKit.framework/AppKit";

    [DllImport(Objc, EntryPoint = "objc_getClass", CharSet = CharSet.Ansi)]
    private static extern IntPtr GetClass(string name);

    [DllImport(Objc, EntryPoint = "sel_registerName", CharSet = CharSet.Ansi)]
    private static extern IntPtr Selector(string name);

    [DllImport(Objc, EntryPoint = "objc_msgSend")]
    private static extern IntPtr Send(IntPtr receiver, IntPtr selector);

    [DllImport(Objc, EntryPoint = "objc_msgSend")]
    private static extern IntPtr Send(IntPtr receiver, IntPtr selector, IntPtr arg);

    [DllImport(Objc, EntryPoint = "objc_msgSend", CharSet = CharSet.Ansi)]
    private static extern IntPtr SendString(IntPtr receiver, IntPtr selector, string arg);

    [DllImport("/usr/lib/libSystem.dylib", EntryPoint = "dlopen", CharSet = CharSet.Ansi)]
    private static extern IntPtr LoadLibrary(string path, int mode);

    /// <summary>
    /// Points the Dock tile at <paramref name="assetUri"/>, an <c>avares://</c> image.
    /// Silently does nothing off macOS or if anything along the way is unavailable.
    /// </summary>
    public static void Set(string assetUri)
    {
        if (!OperatingSystem.IsMacOS())
            return;

        try
        {
            var uri = new Uri(assetUri);
            if (!AssetLoader.Exists(uri))
                return;

            // NSImage reads a file path, and the asset is embedded in the assembly rather
            // than on disk, so it has to be spilled somewhere first. Keyed by name so
            // repeated launches reuse one file instead of littering the temp directory.
            string path = Path.Combine(Path.GetTempPath(),
                                       "vernacula-dock-" + Path.GetFileName(uri.AbsolutePath));
            using (var src = AssetLoader.Open(uri))
            using (var dst = File.Create(path))
                src.CopyTo(dst);

            // AppKit is already loaded in an Avalonia process; this is belt and braces for
            // the case where the icon is set before the platform is up.
            LoadLibrary(AppKitPath, 0x2 /* RTLD_NOW */);

            IntPtr nsString = GetClass("NSString");
            IntPtr nsImage = GetClass("NSImage");
            IntPtr nsApplication = GetClass("NSApplication");
            if (nsString == IntPtr.Zero || nsImage == IntPtr.Zero || nsApplication == IntPtr.Zero)
                return;

            IntPtr pathObj = SendString(nsString, Selector("stringWithUTF8String:"), path);
            IntPtr image = Send(Send(nsImage, Selector("alloc")),
                                Selector("initWithContentsOfFile:"), pathObj);
            if (image == IntPtr.Zero)
                return;

            IntPtr app = Send(nsApplication, Selector("sharedApplication"));
            if (app != IntPtr.Zero)
                Send(app, Selector("setApplicationIconImage:"), image);
        }
        catch
        {
            // A dock icon is never worth failing startup over.
        }
    }
}
