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

**Raw finding (superseded — see Run 2, which found this conclusion too
narrow).** `IDENTICAL` — byte-for-byte, text and every span. Notable
values that held across both: inline code's source span covers the backticks
while the output omits them (`OutputLength = 11, SourceLength = 13`), the link
range covers the label only and skips the destination (source jumps 78→104
across the URL), and fenced/indented code and table cells contribute no output
text at all.

**Implication (wrong — see Run 2).** Read at the time as "no span-semantics
change to absorb; the bump is a straight version change", and as an argument
that no new tests were needed. Both conclusions were drawn from a corpus that
did not contain the one construct that moved. The corpus was written from the
list of constructs the extractor already handles, which is exactly the wrong
generator for a question about *new* parser behaviour: an extension added in
1.x recognizes syntax that 0.34.0 saw as something else, so the construct at
risk is by definition one the old version had no concept of.

The documentary finding above stands: the pin comment is corrected in the same
branch.

## Run 2 — 2026-09-05 — the same bump, corpus extended to constructs 1.x newly recognizes

**Question.** Run 1 asked "does the same input parse the same way", and got
yes. The question it failed to ask: **did `UseAdvancedExtensions()` pick up new
extensions in 1.x?** Both consumers call it bare —
`MarkdownTextExtractor.cs:108` and `MarkdownFlowBuilder.cs:20` — so any
extension added to that bundle is opted into silently.

**Command.** A corpus of GitHub alert blockquotes through the same harness, on
both versions:

```
> [!NOTE]
> Useful information that users should know.

> [!WARNING]
> Careful.

> plain quote
> second line

> [!FOO]
> unknown kind
```

**Raw finding.** Not identical. Markdig 1.x adds `AlertExtension` to
`UseAdvancedExtensions()`; `AlertBlock` subclasses `QuoteBlock` and consumes
the `[!KIND]` marker into the block rather than leaving it as a
`LiteralInline`.

| | 0.34.0 | 1.3.2 |
|---|---|---|
| `Text` (first alert) | `[!NOTE] Useful information that users should know.` | `Useful information that users should know.` |
| `Ranges` over the corpus | 11 | 5 |
| first `BlockSpan` length | 50 | 42 |

Every alert form tested diverges — `[!NOTE]`, `[!WARNING]`, and unknown kinds
such as `[!FOO]`, which 1.x also strips. Plain quotes and `> [not an alert]`
are unchanged.

**Scope — top-level alerts only.** Markdig's `AlertParser` fires on a
top-level quote. An alert indented under a list item is unchanged from 0.34.0:
`- item one\n  > [!NOTE]\n  > body` still extracts as
`item one\n\n[!NOTE] body` on both versions. GitHub *does* render that as an
alert, so the two disagree, and a README whose install steps carry a
`> [!WARNING]` under a numbered step still has the marker spoken while an
identical marker two paragraphs earlier at top level is stripped. Pinned by a
test rather than worked around: a future Markdig that closes the gap would
shift offsets again, silently, for exactly the reason this run exists. Two
related forms, measured and consistent with the above: `> [!NOTE] trailing`
on one line is not an alert on either version (spec-correct), and a `> >`
alert nested inside an alert is not stripped (`AllowNestedAlerts` off).

An alert with an empty body (`> [!NOTE]` alone) is a third asymmetry: on
1.3.2 it yields no text *and no `BlockSpan`*, where 0.34.0 gave one quote
block containing `[!NOTE]`. The speech outcome is right, but a blockquote in
the source now contributes nothing to the block index. Also pinned.

**Implication.** The new behaviour is the one we want: the synthesizer no
longer speaks "bracket bang NOTE bracket" when reading a top-level alert. But it moves
every output offset after an alert by the marker's length, so it is a
behaviour change to land deliberately, not a silent one — and it is exactly
the kind of thing that should be frozen by a test, since neither the build nor
the 24 existing tests noticed. Nine tests added in
`MarkdownTextExtractorTests.cs` (marker stripped for each of the five GitHub
kinds and for an unknown kind; the block is still a `QuoteBlock` — which is
what keeps the extractor's `case QuoteBlock` arm matching it — spanning
exactly the body; following text is not shifted; non-alert brackets survive;
and the two asymmetries above). Seven of the original nine fail against
0.34.0, which is the check that they pin the change rather than restate the
parser. The two that pass on both earn their place differently: one guards
against over-broad future alert recognition, the other against `AlertBlock`
being reparented away from `QuoteBlock`.

**Method note for the next bump.** Diffing a corpus of what the code already
handles only proves the old behaviour is intact. Ask separately what the new
version *added* — for a Markdig-style bundle, diff the extension list of the
built pipeline between versions, and write corpus cases for whatever appears.
`CjkFriendlyEmphasis`, `AllowDomainWithoutPeriod`, `AllowNestedAlerts` and
`InferColumnWidthsFromSeparator` are also new in 1.x, and are opt-in and
verified off. `MaximumNestingDepth` is new but *not* off — it defaults to 128.
It is harmless for a different reason: 0.34.0 had the same limit as a
hardcoded constant, and 1.x only made it configurable. Measured on both
versions, 100 nested `>` parses and 130 throws the identical
`ArgumentException`. Worth stating precisely, since this paragraph is the
recipe for the next bump: "new option" and "changed behaviour" are not the
same question.

## Open, not caused by any bump — `SourceStart` is wrong inside container continuation lines

Surfaced while reading Run 2's output. `TextRange.SourceStart` is documented as
a character offset into the original markdown, and is that for paragraphs. For
a literal on a continuation line of a blockquote or list it is an offset into
Markdig's re-assembled container content instead. In the Run 2 corpus,
`plain quote` reports `SourceStart = 0` — document offset 0 is the `>` of the
first alert — and `second line` reports 12, both offsets into
`"plain quote\nsecond line"`. Identical on 0.34.0 and 1.3.2, so it predates
this work; filed separately.
