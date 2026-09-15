# Markdown source-span investigation (#124)

`TextRange.SourceStart` is documented as a character offset into the original
markdown. For literals on a container's continuation lines it was an offset
into a different string entirely. This log covers finding the real cause and
measuring the blast radius.

## Run 1 — 2026-09-05 — where does the wrong offset come from?

**Question.** `MarkdownTextExtractor` recorded `srcStart = lit.Content.Start`.
`LiteralInline.Content` is a `StringSlice`, and a slice carries the buffer it
points into. Is that buffer always the original document?

**Command.** A probe over Markdig 1.3.2 printing, for every `LiteralInline`:
the slice text, `ReferenceEquals(slice.Text, markdown)`, the document
substring at the slice offsets, and the same at `lit.Span` offsets.

**Raw finding.** `sameBuf` is `False` for exactly the shapes in the bug
report, and in those the slice offsets point at unrelated text while
`lit.Span` is right:

```
=== > plain quote\n> second line
  lit=plain quote   sameBuf=False  slice[0,11]="> plain quo"   span[2,11]="plain quote"
  lit=second line   sameBuf=False  slice[12,11]="e\n> second"  span[16,11]="second line"
=== - item one\ncontinued lazily
  lit=item one          sameBuf=False slice[0,8]="- item o"        span[2,8]="item one"
  lit=continued lazily  sameBuf=False slice[9,16]="e\ncontinued lazi" span[11,16]="continued lazily"
```

`sameBuf=True` for ordinary paragraphs, where slice and span agree.

**Implication.** Markdig re-assembles a container's lines into a fresh buffer
and slices *that*; `slice.Start` is an offset into the re-assembled content.
`lit.Span` is in document coordinates unconditionally. The fix is to read the
position from the span. Note `CodeInline` was already doing this
(`code.Span.Start`) — the literal case was the outlier.

## Run 2 — 2026-09-05 — does the span hold up on the awkward inputs?

**Question.** Before switching, is `lit.Span` correct everywhere, and does its
length still match the emitted text?

**Command.** The same probe over escapes, HTML entities, hard breaks, CJK,
astral-plane emoji, CRLF documents, emphasis spanning a line break, and
quotes/lists containing all of the above.

**Raw finding.** Span offsets correct in every case. Lengths match the output
except for backslash escapes, where the span is one character wider:

```
  lit=*not italic   slice[8,11]="*not italic"   span[7,12]="\*not italic"  [len slice=11 span=12]
```

**Implication.** That widening is right, and the record's own doc comment
already allows it — output is a *subsequence* of the source slice, not an
equal-length one, the same way inline code's span covers its backticks. So
`SourceLength` moves from "however long the output happened to be" to the
span's real extent, which is strictly more accurate.

## Run 3 — 2026-09-05 — blast radius, on real documents

**Question.** How much of the index was actually wrong, and does the fix
close it?

**Command.** Walk every range of all 425 tracked markdown files (excluding
`external/`), slice the source at `SourceStart..+SourceLength`, and require it
to contain the output text. Whitespace-normalized, because inline code turns
an internal line break into a space.

```
git ls-files '*.md' | grep -v '^external/'
```

**Raw finding.**

| | files with a bad range | bad ranges | total ranges |
|---|---|---|---|
| before | 410 / 425 | 19,894 | 50,397 |
| after | 1 / 425 | 1 | 50,397 |

**Implication.** This was not an edge case — 96% of the repo's own markdown
had a broken index, because any document long enough to contain one quote or
one wrapped list item has them.

**Why nobody noticed, stated accurately:** `TextRange.SourceStart` and
`SourceLength` have *no consumer in the tree*. The only reader of `Ranges` is
`FindStyle` in `MainViewModel`, which uses `Style` and compares `OutputStart`
/`OutputLength` only. The source index is being built ahead of forced
alignment, so none of the 19,894 wrong entries was ever rendered. An earlier
draft of this note said the karaoke highlight "lands on the wrong source
text"; that overstated it — the highlight does not read these fields yet.
The fix is therefore low-risk rather than urgent, which is worth saying in
that direction: it corrects the index before anything depends on it.

The single remaining case is not a defect: an inline code span inside a block
quote, whose document extent legitimately includes the `> ` continuation
marker that the output drops. The check can't express "subsequence" without
re-implementing the extractor, so it stays as a known-good outlier rather than
a weakened assertion.

**Test note.** The pre-existing tests asserted `Assert.Contains(outSlice,
srcSlice)` on one construct each, which passes by luck whenever the wrong
offset still lands inside a long enough source slice. The new theory asserts
the same property across the container shapes that were broken, and three
`[Fact]`s pin the bug report's exact offsets (16, not 12; 2, not 0; 13, not
0). After the additions in Runs 4 and 5, 17 of the 21 new cases fail against
the unfixed extractor.

## Run 4 — 2026-09-05 — review follow-up: the output side of the same index

**Question.** Review pointed out that the new invariant test slices
`r.Text.Substring(range.OutputStart, range.OutputLength)` without asserting
output bounds first. Can a recorded range overrun `Text`?

**Command.** Probe documents whose last literal is followed by markup that
emits nothing.

**Raw finding.** Yes, and it predates this work:

```
"hello ![img](/x.png)"  → Text="hello" (5), last range Output[0,6]   OVERRUN
"text <span>"           → Text="text"  (4), last range Output[0,5]   OVERRUN
```

`Run` trims trailing whitespace off the builder but never shortens the last
range, so anything dropped at the tail — an image, an inline tag — leaves a
range pointing past the end. A consumer walking the index to slice `Text`
gets an `ArgumentOutOfRangeException`, not a wrong highlight.

**Blast radius: latent, not live.** Across the same 425 files: **0** overrunning
ranges before *and* after. No document in this repo ends that way. The shape
that triggers it is ordinary though — a README whose last line is a badge —
and the invariant test was one `[InlineData]` away from dying with a throw
instead of a readable assert. Fixed at the source (`TrimRangesToOutput`) rather
than only asserted around, with three such inputs added to the theory.

**Two more from the same review, both pre-existing and neither fixed here:**

- A synthesized inline can carry a *degenerate* span: `"+-\n[1]"` produces a
  literal whose `Span` has `End < Start`. `SourceStartOf`'s clamp turns that
  into a harmless in-bounds zero-length entry. The extractor comment claimed
  `lit.Span` was in document coordinates "in every case"; that overstated it,
  and the comment now says what the bounds check is actually for.
- `lit.Span` is off by one for tab-led heading text — `"# \tabc"` records
  `"\tab"` for output `"abc"`. Identical on both versions, needs a tab
  immediately after the heading marker, and correcting it would mean
  second-guessing Markdig's own position. Left alone, recorded here.

`CodeInline` was the last path still writing raw span arithmetic; it now goes
through the same helper, so "every recorded range is sliceable" is total
rather than per-case. To be precise about what that buys: the reroute is
provably a no-op on real input (code-inline ranges are byte-identical across
all 425 files), and it does *not* fix the one surviving bad range, which is
correct as recorded. It closes the last path that could produce an
unsliceable one.

## Run 5 — 2026-09-05 — review round two: the same overrun in the block index

**Question.** `TrimRangesToOutput` clamps `_ranges`. `_blocks` is built from
the same builder offsets and documented in the same output-text terms
("spans are in output order"). Does it overrun on the same inputs?

**Raw finding.** Yes, on all of them:

```
"hello ![img](/x.png)"  → Text="hello" (5)   Paragraph[0,6]  OVERRUN
"> quoted ![badge](/b)" → Text="quoted"(6)   Quote[0,7]      OVERRUN
"a *b ![x](y)*"         → Text="a b"   (3)   Paragraph[0,4]  OVERRUN
```

**Implication.** Same defect, same shapes, one list over. Latent for the same
reason and one step further removed — the only consumer, `FindBlockIndex`,
does a containment lookup and never slices `Text`. Fixed alongside rather than
left as the twin of a bug fixed inches away, and the invariant test now walks
`Blocks` as well as `Ranges`.

`TrimBlocksToOutput` walks the whole list instead of stopping at the first
entry that fits. The ranges version can stop early because both `_ranges.Add`
calls sit immediately after an append to `_sb`, which is only truncated after
the walk — so output order is monotonic by construction. Block spans are
recorded as their blocks complete, which is a weaker guarantee, and the list
is short enough that relying on it buys nothing.

## Run 6 — 2026-09-15 14:10 — "cards can change their own kind" reported not working

**Question.** The card extent was widened over the block marker (PR #203) and
the unit tests pass, but the feature was reported as not working against a real
job. Is the span model wrong, or is the failure downstream of it?

**Command.** A throwaway xunit probe dumping every card's editable text for a
composite document (ATX headings, wrapped and nested bullets, an ordered list,
a multi-line quote).

**Raw finding.** The model is right on every shape:

```
segments=10 spans=10
[0] Heading/1  editor=<# A document title>
[2] Heading/2  editor=<## A section heading>
[3] ListItem/0 editor=<- first bullet>
[4] ListItem/0 editor=<- second bullet that\n  wraps onto another line>
[5] ListItem/0 editor=<  - a nested bullet>
[6] ListItem/0 editor=<1. numbered one>
[8] Quote/0    editor=<> a quoted line\n> continuing the quote>
[9] Paragraph/0 editor=<A closing paragraph.>
```

**Implication.** Not the spans. Reading the load path for what rebuilds the
cards found the actual answer: `BuildDisplayStructure` is reached only from
`SetText`, and after the initial load the ONLY caller is `LoadCompleted` — which
runs at the end of a successful re-render. So a committed card edit changes the
document and nothing on screen, for the ten-second debounce plus a synthesis
round trip. Worse, `ReRenderAsync` returns early when `ChangedSegments` is
empty, so an edit that changes no segment key (a `-` bullet retyped as `*`, a
list item's indentation) never rebuilds the cards at all. From outside, both are
indistinguishable from the edit being ignored.

**Negative result worth keeping:** two theories were checked and both were
wrong. (1) That `AlignmentSidecar.SourceText` might hold extracted text rather
than markdown — it is set from `File.ReadAllText(documentPath)` verbatim, so the
markers are there. (2) That heading LEVEL might not be drawn, making an H2→H3
edit invisible by construction — `WordItemViewModel` maps level to font size
(30/24/20/18/16), so it is drawn.

**Fix.** Rebuild the cards from the edited markdown as soon as a card commits,
re-attaching timings from the current sidecar. Gated on the structure actually
having changed: a rebuild replaces every card object, and the card the user is
moving TO (committing one card by clicking the next) would become an orphan
mid-click and swallow the click. `OnBlockEditRequested` re-resolves its block by
index for the same reason.

**⚠ NOT CONFIRMED AGAINST THE REPORT.** This explains a delay and a permanent
miss for key-preserving edits, but the reporter may have waited out the
re-render and seen nothing, which would mean a second cause. Needs their
observation: whether the marker appeared in the card's editor at all.

## Run 7 — 2026-09-15 14:40 — the real cause: markup between the marker and the first word

**Question.** Run 6's fix explains a delay, but the reporter answered that the
marker never appeared in the card's editor at all. The synthetic probe widens
every shape. What is different about a real document?

**Command.** The same probe pointed at the sidecar of an actual finished job,
printing structure only — kind, level, offsets, and the prefix between the line
start and the card's first word with letters masked to `a` and digits to `9`, so
markup survives and content cannot.

**Raw finding.**

```
[ 0] Heading/1  lineStart=    0 srcStart=    4 prefix=<# **>
[ 4] ListItem/0 lineStart=  888 srcStart=  893 prefix=<9. **>
[ 1] Paragraph/0 lineStart=  80 srcStart=   82 prefix=<**>
```

**Implication.** The document writes its heading as `# **Title**` and its list
items as `1. **Lead-in** …`. The `**` sits BETWEEN the block marker and the first
word the extractor emits, so the anchored `BlockMarker` pattern — which allowed
only marker characters — rejected the prefix and widened nothing. Every document
written to test the feature had bare headings; none of the real one's did. The
pattern now allows inline openers after the marker, and the marker itself is
optional, which also fixes the third row above: a paragraph with a bold lead-in
had its opening `**` outside the extent while the closing one fell inside, so its
editor opened on text carrying a `**` that closed nothing.

**Second defect, found by the first test written for the fix.** Widening only
left leaves the pair broken the other way: `# **Title**` is bold to its last
word, so the closing `**` lies past the last emitted text and the editor showed
`# **Title`. The widening is now symmetric, bounded by the next card's first word
the way the left side is bounded by the previous card's end.

**Third, found by re-probing after that.** The trailing widening swallowed the
two spaces that end several list items — a markdown HARD LINE BREAK, invisible in
an editor and easy to delete by accident. Both directions now require the run to
contain at least one markup character rather than being blank, which also keeps a
paragraph's indentation out of the box.

**Verified.** Re-probed against the same document: every card carries its marker
and every inline pair is balanced.
