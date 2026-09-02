/**
 * Word timings for karaoke highlighting, ESTIMATED rather than measured.
 *
 * ⚠ OmniVoice emits no alignment — the diffusion loop unmasks all target positions at once, so
 * there is nothing to read out of it. What we do know is that the raw decoded audio is exactly
 * `targetTokens / 25` seconds long, and that `targetTokens` was chosen by `duration.ts` as a sum of
 * per-character script weights over the IPA. Attributing the raw duration back to each
 * whitespace-separated IPA token in proportion to those same weights is therefore self-consistent
 * with the length the model was asked for: the words divide up the time the way the estimator
 * divided up the tokens. It is not a forced alignment and will drift within a sentence — a few
 * hundred milliseconds is typical — but it is right at the ends, monotone, and costs nothing.
 *
 * Two post-processing steps move samples after generation and must be threaded through, or the
 * highlight leads the audio by the total silence removed: silence removal (via `KeptRun[]`, in
 * ORIGINAL coordinates) and the zero-pad `fadeAndPad` adds at the front.
 */
import { charWeight } from "./duration.ts";
import type { KeptRun } from "./audioPost.ts";

export interface WordTiming {
  ipa: string;
  /** Seconds in the FINAL audio. */
  start: number;
  end: number;
}

/** Whitespace-separated IPA tokens with [start,end) cumulative weight — spaces included in the
 *  running total, exactly as `totalWeight` counted them for the duration estimate. */
function tokenWeights(ipa: string): { ipa: string; w0: number; w1: number }[] {
  const out: { ipa: string; w0: number; w1: number }[] = [];
  let acc = 0, cur = "", w0 = 0;
  const flush = () => { if (cur) out.push({ ipa: cur, w0, w1: acc }); cur = ""; };
  for (const ch of ipa) {
    const isSpace = /\s/u.test(ch);
    if (isSpace) flush();
    else if (!cur) w0 = acc;
    acc += charWeight(ch.codePointAt(0)!);
    if (!isSpace) cur += ch;
  }
  flush();
  return out;
}

/** Map an ORIGINAL sample index to its position in the silence-removed output. Samples inside a
 *  removed span snap to the seam, so a word ending in a cut pause ends where the pause was cut. */
function mapSample(s: number, runs: KeptRun[]): number {
  let off = 0;
  for (const r of runs) {
    if (s < r.src) return off;                       // in a removed span before this run
    if (s < r.src + r.len) return off + (s - r.src);
    off += r.len;
  }
  return off;
}

/**
 * @param ipa        the target IPA exactly as fed to the model
 * @param rawSamples length of the decoded audio BEFORE post-processing
 * @param runs       from `removeSilenceMapped`
 * @param padSamples leading zero-pad `fadeAndPad` added
 */
export function attributeWords(ipa: string, rawSamples: number, sr: number,
                               runs: KeptRun[], padSamples: number): WordTiming[] {
  const toks = tokenWeights(ipa);
  if (toks.length === 0 || rawSamples === 0) return [];
  const total = Math.max(toks[toks.length - 1].w1, 1e-9);
  // The last token's w1 is the running total through the last non-space char; any trailing space
  // weight is dropped so the final word reaches the end of the audio.
  const toRaw = (w: number) => Math.min(rawSamples, Math.round((w / total) * rawSamples));
  return toks.map((t) => ({
    ipa: t.ipa,
    start: (mapSample(toRaw(t.w0), runs) + padSamples) / sr,
    end: (mapSample(toRaw(t.w1), runs) + padSamples) / sr,
  }));
}
