#!/usr/bin/env node
/**
 * Measure each stored voice's median F0 and classify the speaker as male or female.
 *
 * ⚠ 391 OF 527 VOICES CARRY NO GENDER LABEL, so the metadata cannot say where the gaps are. Common
 * Voice's `gender` field is optional and mostly unset; FLEURS supplies one only sometimes; the
 * OpenSLR and Omnilingual paths supply none at all. The audio can answer, and the reference codes
 * decode back to audio at exactly 960 samples per frame — no source WAV, no encoder.
 *
 * ⚠ AND THE ANSWER IS A PITCH MEASUREMENT, NOT A FACT ABOUT THE SPEAKER. Adult F0 distributions
 * overlap: ~85-180 Hz male, ~165-255 Hz female, with real people either side. This reports the
 * number and a band, and anything between LOW and HIGH is left UNCERTAIN rather than guessed —
 * a wrong label is worse than a missing one when the point is to balance a voice list.
 *
 *   node tools/measure-voice-gender.mjs                  # measure all, write tools/data/voice-f0.json
 *   node tools/measure-voice-gender.mjs --ids en,ab-29828647
 *   node tools/measure-voice-gender.mjs --from /tmp/tr-cv   # classify CANDIDATES before merging
 *
 * ⚠ `--from` is what makes gender-targeted sourcing possible at all. Common Voice's `gender` field is
 * unset on most contributors and OpenSLR/Omnilingual supply none, so a corpus filter cannot fill a
 * gap reliably — measuring the candidate's own audio can.
 */
import * as ort from "onnxruntime-node";
import { readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i < 0 ? d : args[i + 1]; };
const DECODER = opt("decoder", "/mnt/data/omnivoice_ipa/onnx/higgs_decoder.onnx");
const OUT = opt("out", args.includes("--from") ? `${opt("from")}/f0.json` : "tools/data/voice-f0.json");
const SR = 24000, DS = 3, FSR = SR / DS;        // F0 is found on 8 kHz — 3x less work, same answer

/** Median F0 in Hz over voiced frames, by autocorrelation. Null when nothing is voiced enough. */
function medianF0(x) {
  // Decimate to 8 kHz with a cheap 3-tap average, which also suppresses the HF the codec adds.
  const n = Math.floor(x.length / DS), d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = (x[i * DS] + x[i * DS + 1] + x[i * DS + 2]) / 3;
  const lo = Math.floor(FSR / 300), hi = Math.floor(FSR / 65);   // 65-300 Hz search range
  const W = Math.floor(FSR * 0.04), HOP = Math.floor(FSR * 0.02);
  const peaks = [];
  let energies = [];
  for (let s = 0; s + W + hi < n; s += HOP) {
    let e = 0;
    for (let i = s; i < s + W; i++) e += d[i] * d[i];
    energies.push(e / W);
  }
  if (!energies.length) return null;
  const sorted = [...energies].sort((a, b) => a - b);
  const thr = sorted[Math.floor(sorted.length * 0.6)];           // top 40% of frames by energy
  let fi = 0;
  for (let s = 0; s + W + hi < n; s += HOP, fi++) {
    if (energies[fi] < thr || energies[fi] <= 0) continue;
    let best = 0, bestLag = 0, zero = 0;
    for (let i = s; i < s + W; i++) zero += d[i] * d[i];
    for (let lag = lo; lag <= hi; lag++) {
      let c = 0;
      for (let i = s; i < s + W; i++) c += d[i] * d[i + lag];
      if (c > best) { best = c; bestLag = lag; }
    }
    // ⚠ Require a strong periodic peak. An unvoiced frame still has a maximum; taking it would fill
    // the distribution with noise and drag every median toward the middle of the search range.
    if (bestLag && best / zero > 0.35) peaks.push(FSR / bestLag);
  }
  if (peaks.length < 8) return null;
  peaks.sort((a, b) => a - b);
  return peaks[Math.floor(peaks.length / 2)];
}

const LOW = 155, HIGH = 185;   // below LOW -> male, above HIGH -> female, between -> uncertain
const classify = (f0) => f0 == null ? "unknown" : f0 < LOW ? "M" : f0 > HIGH ? "F" : "uncertain";

const FROM = opt("from");
let voices, codes;
if (FROM) {
  const entries = JSON.parse(readFileSync(`${FROM}/voices.json`, "utf8"));
  voices = entries.map((e) => e.voice);
  codes = Object.fromEntries(entries.map((e) => [e.voice.id, e.codes]));
} else {
  const raw = readFileSync("public/models/voices.jsonc", "utf8");
  voices = JSON.parse(raw.replace(/^\s*\/\/.*$/gmu, ""));
  codes = JSON.parse(readFileSync("public/models/voice-codes.json", "utf8"));
}
const only = opt("ids")?.split(",").map((s) => s.trim());
const dec = await ort.InferenceSession.create(DECODER);

const out = {};
let i = 0;
for (const v of voices) {
  if (only && !only.includes(v.id)) continue;
  const flat = codes[v.id];
  if (!flat || flat.length !== v.refLen * 8) { console.log(`  ${v.id}: codes/refLen mismatch`); continue; }
  const audio = (await dec.run({ audio_codes: new ort.Tensor("int64", BigInt64Array.from(flat, BigInt), [1, 8, v.refLen]) })).audio_values.data;
  const f0 = medianF0(Float32Array.from(audio));
  out[v.id] = { lang: v.lang, f0: f0 == null ? null : Math.round(f0), sex: classify(f0),
                labelled: (((v.source ?? {}).gender ?? "") || null) };
  if (++i % 40 === 0) console.log(`  ${i} measured…`);
}
writeFileSync(OUT, JSON.stringify(out, null, 1) + "\n");
const c = {};
for (const r of Object.values(out)) c[r.sex] = (c[r.sex] ?? 0) + 1;
console.log(`\n${Object.keys(out).length} voices measured -> ${OUT}`);
console.log("classification:", c);
// Agreement with the labels that DO exist — the only check available on the measurement itself.
let ok = 0, bad = 0;
for (const r of Object.values(out)) {
  if (!r.labelled || r.sex === "unknown" || r.sex === "uncertain") continue;
  const lab = /female/i.test(r.labelled) ? "F" : /male/i.test(r.labelled) ? "M" : null;
  if (!lab) continue;
  if (lab === r.sex) ok++; else bad++;
}
console.log(`agreement with existing labels: ${ok} agree, ${bad} disagree`
  + (ok + bad ? ` (${(100 * ok / (ok + bad)).toFixed(0)}%)` : ""));
