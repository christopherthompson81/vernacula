/**
 * Word timings for karaoke highlighting, ESTIMATED rather than measured.
 *
 * ⚠ OmniVoice emits no alignment — the diffusion loop unmasks all target positions at once, so
 * there is nothing to read out of it. The estimate has two ingredients:
 *
 * 1. WHICH share of the speech each word gets: its share of the per-character script weights that
 *    `duration.ts` summed to choose the token count. The words divide up the speech the way the
 *    estimator divided up the tokens, which is the one thing about the timing we actually know.
 * 2. WHERE the speech is: a 10 ms energy envelope over the final audio. Word shares are placed on
 *    the cumulative count of SPEECH frames, not on clock time, so onset silence, pauses, and the
 *    low-level tail the model tends to leave consume no word weight.
 *
 * The first cut placed shares on clock time over the raw generation and was visibly early at the
 * start and late at the end for exactly that reason. Punctuation tokens get zero weight here: they
 * were weighted for the duration estimate because pauses take time, but a pause is precisely the
 * thing the envelope already accounts for. It is still not a forced alignment — within a run of
 * speech the distribution is proportional — but its errors are now bounded by a word, not a pause.
 */
import { charWeight } from "./duration.ts";

export interface WordTiming {
  ipa: string;
  /** Seconds in the FINAL audio. */
  start: number;
  end: number;
}

const PUNCT_TOKEN = /^[\p{P}]+$/u;
const FRAME_MS = 10;
/** A frame counts as speech when its RMS is within this many dB of the loudest frame. */
const SPEECH_REL_DB = -35;

/** Whitespace-separated IPA tokens with [start,end) cumulative weight. Punctuation-only tokens
 *  are kept (they are shown, and they seek) but carry no weight. */
function tokenWeights(ipa: string): { ipa: string; w0: number; w1: number }[] {
  const raw = ipa.split(/\s+/u).filter(Boolean);
  let acc = 0;
  return raw.map((t) => {
    const w0 = acc;
    if (!PUNCT_TOKEN.test(t)) for (const ch of t) acc += charWeight(ch.codePointAt(0)!);
    return { ipa: t, w0, w1: acc };
  });
}

/** Indices of the frames that are speech. */
function speechFrames(audio: Float32Array, sr: number): { frames: number[]; frameLen: number } {
  const frameLen = Math.max(1, Math.floor((FRAME_MS * sr) / 1000));
  const n = Math.floor(audio.length / frameLen);
  const rms = new Float64Array(n);
  let max = 0;
  for (let f = 0; f < n; f++) {
    let s = 0;
    for (let i = f * frameLen, e = i + frameLen; i < e; i++) s += audio[i] * audio[i];
    rms[f] = Math.sqrt(s / frameLen);
    if (rms[f] > max) max = rms[f];
  }
  const thr = max * Math.pow(10, SPEECH_REL_DB / 20);
  const frames: number[] = [];
  for (let f = 0; f < n; f++) if (rms[f] >= thr) frames.push(f);
  return { frames, frameLen };
}

/**
 * @param ipa   the target IPA exactly as fed to the model
 * @param audio the FINAL audio, after silence removal, fade and pad — timings are in its seconds
 */
export function attributeWords(ipa: string, audio: Float32Array, sr: number): WordTiming[] {
  const toks = tokenWeights(ipa);
  if (toks.length === 0 || audio.length === 0) return [];
  const { frames, frameLen } = speechFrames(audio, sr);
  const total = toks[toks.length - 1].w1;
  if (frames.length === 0 || total <= 0) {
    // Nothing above threshold (or no weighted tokens): fall back to an even spread.
    const dur = audio.length / sr;
    return toks.map((_, i) => ({ ipa: toks[i].ipa, start: (i / toks.length) * dur, end: ((i + 1) / toks.length) * dur }));
  }
  const n = frames.length;
  const at = (k: number) => frames[Math.min(n - 1, Math.max(0, k))];
  return toks.map((t) => {
    const f0 = t.w0 / total, f1 = t.w1 / total;
    // First speech frame at or after the start share; last speech frame before the end share.
    const k0 = Math.min(n - 1, Math.floor(f0 * n));
    const k1 = Math.max(k0, Math.ceil(f1 * n) - 1);
    const start = (at(k0) * frameLen) / sr;
    const end = t.w1 === t.w0 ? start : ((at(k1) + 1) * frameLen) / sr;
    return { ipa: t.ipa, start, end };
  });
}
