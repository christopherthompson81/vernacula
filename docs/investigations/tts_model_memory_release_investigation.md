# Releasing TTS model memory after synthesis

**Report.** *"vernacula should free the TTS models VRAM or RAM after bulk synthesis."*

## Run 1 — 2026-09-18 20:10 — what is actually held, and by what

**Question.** Where is the memory, and does disposing the backend return it?

**How it is held today.** `TtsJobRunner` caches one `ITtsBackend` keyed on the model locations
(`EnsureBackend`). The cache is dropped on exactly three events: the settings change the cache key,
`Invalidate()` is called from Settings, or the app exits. **A completed job drops nothing** — the
ONNX sessions stay resident for the life of the process. That is by design, and the design predates
bulk synthesis being a normal thing to do.

Scale of what is resident, from the model trees on this machine:

```
kokoro_publish    947 MB      omnivoice   3.1 GB      chatterbox_export  16 GB
```

⚠ **First a false lead, cleared.** `SessionLoader` calls `OrtSessionBuilder.CreateCachedSession`, and
a session cache that outlived the backend would have made disposal pointless. It does not: the cache
is a DISK cache of pre-optimized `.onnx` files keyed on EP + ORT version + source mtime. Nothing
holds sessions in memory across a dispose.

**Measured** (CPU EP, Kokoro, RSS of the probe process):

```
baseline                 rss    26 MB   managed     0 MB
loaded                   rss   469 MB   managed     4 MB
5 synths                 rss  1111 MB   managed   366 MB
disposed (no gc)         rss   853 MB   managed   366 MB
plain GC                 rss   843 MB   managed   322 MB
aggressive + LOH compact rss   750 MB   managed   322 MB
```

Disposing everything leaves **750 MB resident**, of which **322 MB is managed and still reachable**.

**Three load/dispose cycles say it is not a leak:**

```
cycle 1: loaded 469 → 5 synths 1112 → disposed 816
cycle 2: loaded 905 → 5 synths 1125 → disposed 920
cycle 3: loaded 926 → 5 synths 1132 → disposed 927
```

It plateaus. Cycle 2's load costs +89 MB where cycle 1's cost +443, so the model's memory IS being
freed and reused — the allocators simply do not return it to the OS.

**And the 322 MB of managed memory is not the model at all.** Isolating the phonemizer:

```
baseline                 rss    26 MB   managed     0 MB
phonemizer constructed   rss    52 MB   managed     4 MB   ← lazy, costs nothing yet
phonemizer warmed (en)   rss   503 MB   managed   358 MB   ← ONE English sentence
after GC                 rss   494 MB   managed   321 MB
+ kokoro loaded          rss   901 MB   managed   321 MB   ← the model is the +400
+ 5 synths               rss  1010 MB   managed   330 MB
kokoro disposed          rss   866 MB   managed   322 MB   ← back to the phonemizer's share
```

Phonemizing one English sentence costs **~450 MB**, held in the phonemizer's STATIC caches — the
manifests and the 141k-row accent lexicon — for the life of the process. `KokoroTts` builds its own
`KokoroPhonemizer`, and loading it on top of an existing one added nothing, which confirms the data
is shared static rather than per-instance.

**Implication — the request splits in two.**

1. **The model session (~400 MB for Kokoro, more for OmniVoice, far more for Chatterbox; VRAM under
   CUDA).** Freed by disposing the backend. This is the part the report asks for and the part a
   release policy can deliver.
2. **The phonemizer's static lexicon (~450 MB).** No disposal reaches it: it is static by design,
   shared with the reader's IPA annotation, and costs seconds to reload. Releasing it needs an
   explicit cache-clear API upstream in vernacula-phonemizer, and a decision about whether the
   reader wants to pay the reload. **Out of scope here, recorded so the remaining ~450 MB after a
   release is not mistaken for the policy failing.**

⚠ And RSS is the wrong number to promise on. The allocator plateau means process RSS after a release
will not fall all the way back; what the release genuinely returns is the session's own memory —
which, under CUDA, is **device** memory, where the allocator slack does not apply and the win is
real and full.

## Run 2 — 2026-09-18 21:05 — what the release returns, on the device where it matters

**Question.** Run 1 measured the CPU EP and found process RSS barely moves, because the host
allocators keep what they free. The report names VRAM first. Does the device behave differently?

**Measured** (CUDA EP, RTX 3090, Kokoro, `nvidia-smi --query-compute-apps` for this PID):

```
baseline                 vram     0 MB   rss    41 MB
loaded (cuda)            vram   782 MB   rss   776 MB
5 synths                 vram   826 MB   rss  1725 MB
disposed (the release)   vram   272 MB   rss  1711 MB
```

**The release returns 554 MB of VRAM — 67% of what was held — and 14 MB of RSS.** That is the whole
argument for the feature in one table, and it is also the reason not to advertise it in RSS terms:
on the host the allocator keeps the pages, on the device it does not.

The 272 MB that stays is the CUDA context and the cuDNN kernels, which go only when the process
does. Kokoro is the SMALLEST of the four backends; OmniVoice's tree is 3.1 GB and Chatterbox's 16 GB.

## Run 3 — 2026-09-18 21:20 — the policy, and why not "release when the job ends"

**The obvious implementation is wrong for this app.** The reader re-renders one paragraph at a time
as the user edits, through the same cached backend — and that happens *immediately after* a synthesis
finishes, because finishing is what puts the document in front of the user. Releasing at the end of
the job would make the first edit pay a full model load, which for Chatterbox is tens of seconds.

**What shipped: an idle timer, default 60 s**, on `TtsModelIdleReleaseSeconds`. 0 releases as soon as
the work stops; negative keeps the old hold-forever behaviour for anyone who would rather spend the
memory than the reload.

Three things the implementation has to get right, all of them about not disposing a model someone is
using:

- **A lease count, not a flag.** The queue runs one job at a time *by default*, but its slot count is
  a constructor parameter and the reader's re-render can overlap a queued job. The timer is armed by
  the last lease to close, not the first.
- **Re-checked under the lock after the delay.** A job can start between the delay expiring and the
  release running; disposing mid-synthesis is the one outcome this must never produce.
- **The model is built OUTSIDE the lock.** A load is seconds to tens of seconds of I/O and graph
  optimization; holding the gate across it would block every other caller and the timer's re-check
  for that whole time. Two callers racing to build the same key is handled by keeping the first and
  disposing the loser, rather than leaking it.

**Tests** assert the transition on `IsModelLoaded` rather than on a memory number — a test cannot
hold RSS still, and the thing under test is the policy, not the allocator. All three run unskipped on
this machine (idle release, zero-second release, negative keeps it), so they exercised a real Kokoro
load and dispose.

**Still not released, and out of scope: the phonemizer's ~450 MB of static lexicon** (Run 1). Nothing
in this change touches it. After a release the process still holds it, and that is the expected
number, not a failure of the policy.
