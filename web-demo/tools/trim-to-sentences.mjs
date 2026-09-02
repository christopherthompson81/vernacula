/**
 * Cut a long clip down to a usable reference length — WITHOUT desynchronising its transcript.
 *
 * Several corpora only offer long clips: Omnilingual's spontaneous speech runs 25-80 s, and a
 * reference that long lengthens every sentence the demo later speaks in that language. Trimming the
 * audio alone is not an option, because the transcript is fed to the model beside the codes and a
 * reference whose text does not match its audio is worse than no reference at all.
 *
 * ⚠ SO THE CUT MUST BE PROVABLE, and this is the proof it uses: a long pause is a sentence boundary.
 * Count the sentences in the transcript, count the speech runs separated by pauses of at least
 * `gapMs`, and proceed ONLY when the two counts agree. When they agree, run k corresponds to
 * sentence k, so cutting after run k and keeping sentences 1..k yields a pair that still matches.
 * When they disagree — a pause inside a sentence, a sentence read without a pause after it — the
 * clip is rejected rather than guessed at.
 *
 * ⚠ EQUAL COUNTS ARE NOT PROOF ON THEIR OWN, and the first version of this claimed they were. In
 * spontaneous speech a speaker pauses mid-sentence and runs two sentences together, which keeps the
 * counts equal while shifting the mapping: a Hawaiian clip came back with 8 words of transcript
 * against 13.2 s of audio. So the pair is also checked against SPEAKING RATE, calibrated on the 174
 * references already shipped — those run 6.4 to 13.7 IPA characters per second (p10-p90, median
 * 9.8). A trimmed pair outside a generous band around that is rejected: it means the text and the
 * audio describe different amounts of speech, whatever the counts said.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

const SR = 24000;

/** 10 ms frame energies in dB, from a 24 kHz mono 16-bit WAV. */
function frames(path) {
  const b = readFileSync(path);
  let off = 12, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString("ascii", off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === "data") data = b.subarray(off + 8, off + 8 + sz);
    off += 8 + sz + (sz & 1);
  }
  const n = data.length / 2, fr = SR / 100, m = Math.floor(n / fr), db = new Float64Array(m);
  for (let f = 0; f < m; f++) {
    let s = 0;
    for (let i = f * fr; i < (f + 1) * fr; i++) { const v = data.readInt16LE(i * 2) / 32768; s += v * v; }
    db[f] = 10 * Math.log10(s / fr + 1e-12);
  }
  return db;
}

/** Speech runs as [startFrame, endFrame], split on pauses of at least `gapMs`. */
export function speechRuns(db, gapMs = 400, relDb = 30) {
  const top = Math.max(...db), thr = top - relDb, gap = gapMs / 10;
  const runs = [];
  let start = -1, silence = 0;
  for (let i = 0; i < db.length; i++) {
    if (db[i] >= thr) {
      if (start < 0) start = i;
      silence = 0;
    } else if (start >= 0 && ++silence >= gap) {
      runs.push([start, i - silence + 1]);
      start = -1; silence = 0;
    }
  }
  if (start >= 0) runs.push([start, db.length]);
  return runs;
}

/** Sentences, on explicit line breaks first and terminal punctuation second. */
export function sentences(text) {
  const lines = text.split(/\n+/u).map((s) => s.trim()).filter(Boolean);
  const out = [];
  for (const l of lines)
    for (const s of l.split(/(?<=[.!?…。।۔፡።؟])\s+/u).map((x) => x.trim()).filter(Boolean)) out.push(s);
  return out;
}

/**
 * @returns {{wav: string, text: string, sec: number} | null} the trimmed pair, or null when the
 * sentence and pause counts disagree or no prefix lands in the target window.
 */
/** Non-space characters per second of audio, the shape the rate check works on. `phonemize` is not
 *  available here, so this counts the SOURCE text — coarser than IPA, hence the wide band. */
const RATE_MIN = 4, RATE_MAX = 30;

export function trimToSentences(wavPath, text, { minSec = 6, maxSec = 14, gapMs = 400, out } = {}) {
  const db = frames(wavPath);
  const runs = speechRuns(db, gapMs);
  const sents = sentences(text);
  if (runs.length < 2 || runs.length !== sents.length) return null;

  // The cut goes in the middle of the pause after run k, so neither side clips a consonant.
  for (let k = 1; k < runs.length; k++) {
    const end = (runs[k - 1][1] + (runs[k][0] - runs[k - 1][1]) / 2) / 100;
    if (end < minSec) continue;
    if (end > maxSec) return null;
    const kept = sents.slice(0, k).join(" ");
    const chars = [...kept].filter((ch) => !/\s/u.test(ch)).length;
    const rate = chars / end;
    if (rate < RATE_MIN || rate > RATE_MAX) return null;   // text and audio disagree about length
    const trimmed = out ?? wavPath.replace(/\.wav$/, ".trim.wav");
    execFileSync("ffmpeg", ["-v", "error", "-y", "-i", wavPath, "-t", end.toFixed(3),
                            "-ac", "1", "-ar", String(SR), "-sample_fmt", "s16", trimmed]);
    return { wav: trimmed, text: kept, sec: end, rate };
  }
  return null;
}
