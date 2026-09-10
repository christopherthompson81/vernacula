using System.Text.RegularExpressions;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// One <c>SessionOptions</c> may back exactly one <c>InferenceSession</c>.
/// </summary>
/// <remarks>
/// Sharing one across sessions segfaults the process on the WebGPU EP: the second
/// <c>Dispose</c> lands in <c>webgpu::BufferManager::Release</c> on already-freed buffers.
/// It dies at teardown — after the work, before the caller writes its output — so the
/// symptom is a crash and a MISSING result rather than a wrong one, and nothing on the
/// CPU or CUDA path ever complains.
///
/// This was found in <c>Parakeet</c> and then in five more backends, all written the same
/// natural-looking way:
/// <code>
///     var opts = OrtSessionBuilder.Create(ep);      // ← one object
///     _mel     = new InferenceSession(melPath,     opts);
///     _encoder = new InferenceSession(encoderPath, opts);   // ← boom, later
/// </code>
/// A reviewer will not catch the sixth one either, so it is checked here. `Create` is a
/// factory: call it once per session.
/// </remarks>
public class SharedSessionOptionsTests
{
    /// <summary>
    /// A local, field or property holding a SessionOptions. Fields matter as much as
    /// locals: `_opts = OrtSessionBuilder.Create(ep);` used by two sessions in two different
    /// methods is the same bug, and the likeliest way it comes back.
    /// </summary>
    private static readonly Regex OptionsDecl = new(
        @"\b(?:var|SessionOptions)\s+(\w+)\s*=\s*(?:OrtSessionBuilder\.Create|new\s+SessionOptions)\b",
        RegexOptions.Compiled);

    /// <summary>
    /// A SessionOptions passed to a session — in ANY argument position. Requiring it to be
    /// the last argument missed `new InferenceSession(path, opts, prepackedWeights)`.
    /// </summary>
    private static Regex UseOf(string name) => new(
        @"new\s+InferenceSession\s*\([^;]*?\b" + Regex.Escape(name) + @"\b[^;]*?\)",
        RegexOptions.Singleline);

    /// <summary>Walks up from the test binary to the repo root.</summary>
    private static DirectoryInfo RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Vernacula.slnx")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return dir!;
    }

    /// <summary>
    /// Method-ish blocks, so a name reused in two methods is not mistaken for one shared
    /// object. Crude, and deliberately so: it only has to be good enough to localise a
    /// declaration and its uses.
    /// </summary>
    private static IEnumerable<string> Blocks(string source)
    {
        var starts = Regex.Matches(source, @"\n    (?:public|private|internal|protected|static)[^\n]*\n")
                          .Select(m => m.Index).ToList();
        starts.Insert(0, 0);
        starts.Add(source.Length);
        for (int i = 0; i < starts.Count - 1; i++)
            yield return source[starts[i]..starts[i + 1]];
    }

    [Fact]
    public void NoSessionOptionsObjectBacksMoreThanOneSession()
    {
        var root = RepoRoot();
        var offenders = new List<string>();

        foreach (var file in Directory.EnumerateFiles(
                     Path.Combine(root.FullName, "src"), "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") ||
                file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
                continue;

            string source = File.ReadAllText(file);
            if (!source.Contains("InferenceSession"))
                continue;

            // Locals are scoped to their method; a field is not, so its uses have to be
            // counted across the whole file. Doing both is what catches the field case the
            // first version of this test walked straight past.
            foreach (string block in Blocks(source))
                foreach (Match decl in OptionsDecl.Matches(block))
                {
                    string name = decl.Groups[1].Value;
                    int uses = UseOf(name).Matches(block).Count;
                    if (uses > 1)
                        offenders.Add($"{Path.GetRelativePath(root.FullName, file)}: " +
                                      $"'{name}' backs {uses} sessions");
                }

            foreach (Match decl in Regex.Matches(
                         source,
                         @"\b(?:private|internal|protected|public)\s+(?:readonly\s+)?SessionOptions\s+(\w+)\b"))
            {
                string name = decl.Groups[1].Value;
                int uses = UseOf(name).Matches(source).Count;
                if (uses > 1)
                    offenders.Add($"{Path.GetRelativePath(root.FullName, file)}: " +
                                  $"field '{name}' backs {uses} sessions");
            }
        }

        Assert.True(offenders.Count == 0,
            "A SessionOptions may back only one InferenceSession — sharing one segfaults the "
            + "process on the WebGPU EP at teardown, losing the caller's output. Call "
            + "OrtSessionBuilder.Create() once per session:\n  " + string.Join("\n  ", offenders));
    }
}
