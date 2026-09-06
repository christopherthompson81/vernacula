# Dependency upgrade investigation

A running log of the upgrade queue: what each bump was validated against, and
what the raw output said. Entries are chronological.

## Run 1 — 2026-09-05 — Markdig 0.34.0 → 1.3.2

**Question.** The karaoke view maps highlight positions through
`TextRange.OutputStart/OutputLength` and `SourceStart/SourceLength` from
`MarkdownTextExtractor`. A major-version bump of the parser could shift source
spans (block start offsets, inline delimiter inclusion) without failing a
single test, because the 24 existing tests assert invariants on constructs they
name rather than on a broad corpus. Does 1.3.2 produce the same extraction as
0.34.0?

**First finding, before running anything.** The pin comment in
`Vernacula.Tts.Base.csproj` justified the version by "cf. Markdown.Avalonia →
Markdig". That is wrong. `project.assets.json` shows Markdown.Avalonia 11.0.3
pulling only its own sub-packages plus ColorTextBlock/ColorDocument, Avalonia,
HtmlAgilityPack and Avalonia.AvaloniaEdit — no Markdig. It parses markdown with
its own engine. The only Markdig in the graph is ours, from
`Vernacula.Tts.Base` and `Vernacula.Avalonia`. The two pins still need to match
each other, so the constraint survives; only the stated reason changed.

**Command.** A throwaway console harness in the scratchpad referencing
`Vernacula.Tts.Base` (with `ImportDirectoryBuildProps=false` so it doesn't
inherit the EP guards), dumping `Text` plus every collection property of
`MarkdownExtractionResult` — `Ranges` and `Blocks` — with full record
formatting, over a corpus exercising: ATX and setext headings, bold/italic,
inline code, links, backtick and tilde fences, an indented code block, tight
and nested lists, an ordered list, a blockquote, a pipe table, a footnote
reference and definition, an image, a two-space hard break, a raw HTML block,
an autolink, an entity, a backslash escape, and a thematic break. Run once on
`chore/markdig-1.3.2`, then `git stash` back to 0.34.0 and run again.

```
dotnet run --project <harness> -p:EP=Cpu -c Release -- corpus.md
diff out-0.34.0.txt out-1.3.2.txt
```

**Raw finding.** `IDENTICAL` — byte-for-byte, text and every span. Notable
values that held across both: inline code's source span covers the backticks
while the output omits them (`OutputLength = 11, SourceLength = 13`), the link
range covers the label only and skips the destination (source jumps 78→104
across the URL), and fenced/indented code and table cells contribute no output
text at all.

**Implication.** No span-semantics change to absorb; the bump is a
straight version change. The gap the exercise did expose was documentary, not
behavioural, so the pin comment is corrected in the same branch. Nothing here
argues for new tests — the corpus asserts equivalence between two versions, not
a property worth freezing, and the 24 existing tests already cover the
constructs by name.
