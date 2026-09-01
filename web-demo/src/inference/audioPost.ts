/**
 * Output post-processing — the TypeScript port of `Chatterbox.Base.OmniVoiceAudioPost`
 * (OmniVoice's `_post_process_audio`, via pydub).
 *
 * Order matters and is the Python order: remove silence -> volume -> fade + pad. Silence detection
 * mirrors pydub: a 10 ms chunk is "silent" when its RMS is below −50 dBFS.
 */

const SILENCE_THRESH_DB = -50;
const THRESH_RMS = Math.pow(10, SILENCE_THRESH_DB / 20); // ≈0.0031623

function chunkRms(a: Float32Array, start: number, len: number): number {
  const end = Math.min(start + len, a.length);
  if (end <= start) return 0;
  let s = 0;
  for (let i = start; i < end; i++) s += a[i] * a[i];
  return Math.sqrt(s / (end - start));
}

/** pydub detect_leading_silence: advance in 10 ms chunks while below threshold. */
function detectLeadingSilence(a: Float32Array, sr: number, chunkMs = 10): number {
  const chunk = Math.max(1, Math.floor((chunkMs * sr) / 1000));
  let trim = 0;
  while (trim < a.length && chunkRms(a, trim, chunk) < THRESH_RMS) trim += chunk;
  return Math.min(trim, a.length);
}

function detectSilence(a: Float32Array, sr: number, minSilMs: number, seekMs = 10): [number, number][] {
  const minSil = Math.floor((minSilMs * sr) / 1000);
  const seek = Math.max(1, Math.floor((seekMs * sr) / 1000));
  const ranges: [number, number][] = [];
  if (a.length < minSil) return ranges;

  const starts: number[] = [];
  const lastStart = a.length - minSil;
  for (let i = 0; i <= lastStart; i += seek) if (chunkRms(a, i, minSil) <= THRESH_RMS) starts.push(i);
  if (starts.length === 0) return ranges;

  let rangeStart = starts[0], prev = starts[0];
  for (let j = 1; j < starts.length; j++) {
    const si = starts[j];
    const contiguous = si === prev + seek;
    const gap = si > prev + minSil;
    if (!contiguous && gap) { ranges.push([rangeStart, prev + minSil]); rangeStart = si; }
    prev = si;
  }
  ranges.push([rangeStart, prev + minSil]);
  return ranges;
}

function detectNonsilent(a: Float32Array, sr: number, minSilMs: number, seekMs = 10): [number, number][] {
  const sil = detectSilence(a, sr, minSilMs, seekMs);
  const non: [number, number][] = [];
  if (sil.length === 0) { if (a.length > 0) non.push([0, a.length]); return non; }
  let cur = 0;
  for (const [s, e] of sil) { if (s > cur) non.push([cur, s]); cur = e; }
  if (cur < a.length) non.push([cur, a.length]);
  return non;
}

/** pydub split_on_silence + reconcatenation, with keep_silence padding and the midpoint clamp on
 *  overlapping expanded ranges. */
function splitAndConcat(a: Float32Array, sr: number, minSilMs: number, keepMs: number, seekMs = 10): Float32Array {
  const ranges = detectNonsilent(a, sr, minSilMs, seekMs);
  if (ranges.length === 0) return new Float32Array(0);
  const keep = Math.floor((keepMs * sr) / 1000);
  const outR = ranges.map(([s, e]) => [s - keep, e + keep] as [number, number]);
  for (let i = 0; i + 1 < outR.length; i++) {
    const lastEnd = outR[i][1], nextStart = outR[i + 1][0];
    if (nextStart < lastEnd) {
      const mid = Math.trunc((lastEnd + nextStart) / 2);
      outR[i][1] = mid;
      outR[i + 1][0] = mid;
    }
  }
  let n = 0;
  for (const [s0, e0] of outR) n += Math.min(a.length, e0) - Math.max(0, s0);
  const buf = new Float32Array(Math.max(0, n));
  let w = 0;
  for (const [s0, e0] of outR) {
    const s = Math.max(0, s0), e = Math.min(a.length, e0);
    for (let i = s; i < e; i++) buf[w++] = a[i];
  }
  return buf.subarray(0, w);
}

/** Port of `remove_silence`: collapse mid-silences longer than `midSilMs` down to that length,
 *  then trim edge silences keeping lead/trail ms. */
export function removeSilence(audio: Float32Array, sr: number,
                              midSilMs: number, leadSilMs: number, trailSilMs: number): Float32Array {
  if (audio.length === 0) return audio;
  const a = midSilMs > 0 ? splitAndConcat(audio, sr, midSilMs, midSilMs) : audio;
  if (a.length === 0) return a;
  const lead = Math.floor((leadSilMs * sr) / 1000), trail = Math.floor((trailSilMs * sr) / 1000);
  const start = Math.max(0, detectLeadingSilence(a, sr) - lead);
  const rev = Float32Array.from(a).reverse();
  const trailStart = Math.max(0, detectLeadingSilence(rev, sr) - trail);
  const end = a.length - trailStart;
  if (end <= start) return new Float32Array(0);
  return a.slice(start, end);
}

/**
 * Port of `fade_and_pad_audio`: linear fade-in/out over `fadeSec`, then zero-pad `padSec` both ends.
 *
 * ⚠ This runs AFTER normalization, which is why the fine-tune's leading transient costs headroom:
 * peak-normalizing to 0.5 can pick a transient inside the first 0.1 s, and the fade then removes
 * it, leaving the speech well below the nominal peak. Faithful to the Python order; see
 * docs/vernacula_tts_investigation.md before "fixing" it.
 */
export function fadeAndPad(audio: Float32Array, sr: number, padSec = 0.1, fadeSec = 0.1): Float32Array {
  if (audio.length === 0) return audio;
  const fade = Math.trunc(fadeSec * sr), pad = Math.trunc(padSec * sr);
  const proc = Float32Array.from(audio);
  const k = Math.min(fade, Math.trunc(proc.length / 2));
  for (let i = 0; i < k; i++) {
    const g = i / (k - 1 === 0 ? 1 : k - 1);
    proc[i] *= g;
    proc[proc.length - 1 - i] *= g;
  }
  if (pad <= 0) return proc;
  const out = new Float32Array(pad + proc.length + pad);
  out.set(proc, pad);
  return out;
}

export function peakNormalize(x: Float32Array, peak: number): void {
  let max = 0;
  for (const v of x) max = Math.max(max, Math.abs(v));
  if (max < 1e-6) return;
  const g = peak / max;
  for (let i = 0; i < x.length; i++) x[i] *= g;
}

export function scale(x: Float32Array, g: number): void {
  for (let i = 0; i < x.length; i++) x[i] *= g;
}

/** 32-bit float mono WAV, for an <audio> element or a download. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 4);
  const dv = new DataView(buf);
  const put = (o: number, s: string) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  put(0, "RIFF"); dv.setUint32(4, 36 + samples.length * 4, true); put(8, "WAVE");
  put(12, "fmt "); dv.setUint32(16, 16, true);
  dv.setUint16(20, 3, true);                 // IEEE float
  dv.setUint16(22, 1, true);                 // mono
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 4, true);
  dv.setUint16(32, 4, true); dv.setUint16(34, 32, true);
  put(36, "data"); dv.setUint32(40, samples.length * 4, true);
  new Float32Array(buf, 44).set(samples);
  return new Blob([buf], { type: "audio/wav" });
}
