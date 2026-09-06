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
one wrapped list item has them. It went unnoticed because nothing downstream
of `SourceStart` fails loudly; the karaoke highlight just lands on the wrong
source text.

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
0). 13 of the 17 new cases fail against the unfixed extractor.

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

`CodeInline` was the last path still writing raw span arithmetic, and the
source of the single surviving bad range repo-wide; it now goes through the
same helper, so "every recorded range is sliceable" is total rather than
per-case.
