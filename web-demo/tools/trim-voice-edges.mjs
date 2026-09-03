#!/usr/bin/env node
/**
 * Trim non-speech from the START and END of a stored reference voice — a throat-clear, a click, a
 * breath, room noise after the last word — WITHOUT re-encoding and without the source audio.
 *
 * ⚠ THE REFERENCE'S EDGES ARE COPIED INTO EVERY SENTENCE THE DEMO SPEAKS. Cloning reproduces what
 * it is given, so a reference that opens with a throat-clear makes a language that clears its throat
 * before every utterance. A listening pass over all 318 clips found this on 19 voices, and it is the
 * one class of defect that needs no new audio: the noise is at the edge, and the edge can be cut.
 *
 * ⚠ NO SOURCE WAV IS NEEDED, AND THAT IS THE POINT. The demo ships voices as codec codes only; the
 * WAVs are deliberately not in the repo. But the codec is invertible — `higgs_decoder` turns
 * [8, refLen] codes back into audio at EXACTLY 960 samples per frame — so the edges can be measured
 * on decoded audio and cut on the code frames they correspond to. 40 ms of granularity, no encoder.
 *
 *   node tools/trim-voice-edges.mjs --list                    # what would be trimmed, write nothing
 *   node tools/trim-voice-edges.mjs --ids en,sr,ca --apply
 *   node tools/trim-voice-edges.mjs --from-manifest /tmp/listen-all/manifest.tsv --apply
 *
 * Writes before/after WAVs to --out (default /tmp/trim-preview) so the cut can be HEARD. Nothing
 * here can tell a breath from a plosive; the guards below only bound the damage.
 */
import * as ort from "onnxruntime-node";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i < 0 ? d : args[i + 1]; };
const has = (k) => args.includes(`--${k}`);
const DECODER = opt("decoder", "/mnt/data/omnivoice_ipa/onnx/higgs_decoder.onnx");
const OUT = opt("out", "/tmp/trim-preview");
const SR = 24000, HOP = 960;             // one code frame = 960 samples = 40 ms, exactly

const VP = "public/models/voices.jsonc", CP = "public/models/voice-codes.json";
const raw = readFileSync(VP, "utf8");
const voices = JSON.parse(raw.replace(/^\s*\/\/.*$/gmu, ""));
const codes = JSON.parse(readFileSync(CP, "utf8"));

/** Which voices to consider. */
let ids = opt("ids")?.split(",").map((s) => s.trim()).filter(Boolean);
if (!ids && opt("from-manifest")) {
  // Take the voices whose reviewed verdict names an EDGE problem — the class this tool can fix.
  const EDGE = /trimming|throat|vocaliz|noise at the start|starts with (a little )?noise|ends with noise|noise at the end|silence/iu;
  const seen = new Set();
  ids = [];
  for (const [i, line] of readFileSync(opt("from-manifest"), "utf8").split("\n").entries()) {
    if (i === 0 || !line.trim()) continue;
    const c = line.split("\t");
    if (EDGE.test(c[9] ?? "") && !seen.has(c[4])) { seen.add(c[4]); ids.push(c[4]); }
  }
}
// Default to EVERY voice. The reviewed list names the ones a listener noticed; the measurement
// found 28 more with over a quarter silence that nobody flagged, because a slow start does not
// sound broken — it sounds like a slow speaker, in every sentence that voice ever says.
if (!ids?.length) ids = voices.map((v) => v.id);

/** 10 ms frame energies in dB. */
function envelope(x) {
  const fr = SR / 100, m = Math.floor(x.length / fr), db = new Float64Array(m);
  for (let f = 0; f < m; f++) {
    let s = 0;
    for (let i = f * fr; i < (f + 1) * fr; i++) s += x[i] * x[i];
    db[f] = 10 * Math.log10(s / fr + 1e-12);
  }
  return db;
}

/** Speech runs as [startFrame, endFrame) over 10 ms frames, split on pauses of at least gapMs. */
function runs(db, gapMs, relDb) {
  const thr = Math.max(...db) - relDb, gap = gapMs / 10, out = [];
  let start = -1, sil = 0;
  for (let i = 0; i < db.length; i++) {
    if (db[i] >= thr) { if (start < 0) start = i; sil = 0; }
    else if (start >= 0 && ++sil >= gap) { out.push([start, i - sil + 1]); start = -1; sil = 0; }
  }
  if (start >= 0) out.push([start, db.length]);
  return out;
}

/**
 * ⚠ THE RULE IS "SHORT AND DETACHED", NOT "QUIET". A throat-clear is not quiet — it is loud, brief,
 * and separated from the utterance by a pause. A plain energy gate keeps it and eats the first
 * consonant instead.
 *
 * ⚠ AND THE DISCRIMINATOR THAT MATTERS IS RUN LENGTH, which the first version left out and which
 * broke it in both directions at once. Stripping every leading run under 400 ms marched inward
 * through real words — en, fi and cmn each wanted 41-49% of the clip cut, and only the damage guards
 * stopped it. Meanwhile `ca`'s trailing silence survived, because three stray frames of room tone
 * above the threshold read as a speech run. Requiring a run to last at least MIN_RUN_MS before it
 * counts as speech fixes both: the blips stop anchoring the span, and the body stops being eaten.
 *
 * At most ONE event is dropped per side. A throat-clear is one event; a loop that removes several is
 * removing words.
 *
 * ⚠ SILENCE TRIMMING AND EVENT DROPPING ARE SEPARATE OPERATIONS, and only the first is structurally
 * safe. Cutting frames that lie OUTSIDE the first and last qualifying speech run cannot remove speech
 * — that is what "qualifying" means — so it needs no fraction guard however much it takes, and on
 * FLEURS references it takes a lot: 28 of 102 are more than a quarter silence, `sv` is 55%. Dropping a
 * detached event is a judgement about what a short burst IS, so it stays opt-in behind --drop-events.
 */
const MAX_EVENT_MS = 400, GAP_MS = 150, REL_DB = 32, MIN_RUN_MS = 150, PAD_FRAMES = 6;

function plan(x) {
  const db = envelope(x);
  const all = runs(db, GAP_MS, REL_DB);
  const speech = all.filter(([a, b]) => (b - a) * 10 >= MIN_RUN_MS);
  if (!speech.length) return null;
  let rs = speech, dropLead = 0, dropTrail = 0;
  // One detached event per side, and only when a real pause separates it from the body. Opt-in.
  if (has("drop-events")) {
    if (rs.length > 1 && (rs[0][1] - rs[0][0]) * 10 <= MAX_EVENT_MS && (rs[1][0] - rs[0][1]) * 10 >= 200) {
      rs = rs.slice(1); dropLead = 1;
    }
    if (rs.length > 1 && (rs.at(-1)[1] - rs.at(-1)[0]) * 10 <= MAX_EVENT_MS
        && (rs.at(-1)[0] - rs.at(-2)[1]) * 10 >= 200) {
      rs = rs.slice(0, -1); dropTrail = 1;
    }
  }
  // Keep a little air either side so a cut never clips an onset or a release.
  const s = Math.max(0, rs[0][0] - PAD_FRAMES), e = Math.min(db.length, rs.at(-1)[1] + PAD_FRAMES);
  return { startFrame: Math.floor(s / 4), endFrame: Math.ceil(e / 4), dropLead, dropTrail };  // 10 ms -> 40 ms
}

const rms = (x) => Math.sqrt(x.reduce((a, v) => a + v * v, 0) / Math.max(1, x.length));

function wav(path, x) {
  const b = Buffer.alloc(44 + x.length * 2);
  b.write("RIFF", 0); b.writeUInt32LE(36 + x.length * 2, 4); b.write("WAVEfmt ", 8);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(SR, 24); b.writeUInt32LE(SR * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write("data", 36); b.writeUInt32LE(x.length * 2, 40);
  for (let i = 0; i < x.length; i++) b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767))), 44 + i * 2);
  writeFileSync(path, b);
}

mkdirSync(OUT, { recursive: true });
const dec = await ort.InferenceSession.create(DECODER);
const byId = new Map(voices.map((v) => [v.id, v]));
const changes = [];

for (const id of ids) {
  const v = byId.get(id);
  if (!v) { console.log(`  ${id}: not in voices.jsonc`); continue; }
  const flat = codes[id];
  if (!flat || flat.length !== v.refLen * 8) { console.log(`  ${id}: codes/refLen mismatch`); continue; }
  const audio = (await dec.run({ audio_codes: new ort.Tensor("int64", BigInt64Array.from(flat, BigInt), [1, 8, v.refLen]) })).audio_values.data;
  const x = Float32Array.from(audio);
  const p = plan(x);
  if (!p) { console.log(`  ${id}: no speech found — skipped`); continue; }
  const { startFrame, endFrame, dropLead, dropTrail } = p;
  const newLen = endFrame - startFrame;
  const cutMs = (v.refLen - newLen) * 40;
  // ⚠ A MINIMUM CUT, so a fleet-wide pass does not rewrite 89 voices to shave 40 ms off each. Below
  // this the change is inaudible and the churn — a new refLen and refRms on every line — costs more
  // in review than it buys in audio.
  const MIN_CUT_MS = Number(opt("min-cut", 200));
  if (v.refLen - newLen < MIN_CUT_MS / 40) {
    if (has("verbose")) console.log(`  ${id}: ${(v.refLen - newLen) * 40} ms at the edges — under --min-cut`);
    continue;
  }

  // ⚠ GUARDS. The transcript is not re-derived, so a cut that removes SPEECH desynchronises the
  // reference from its own IPA — worse than the noise it fixes. Reject a cut that takes too much,
  // or that pushes the speaking rate outside the band the shipped references occupy.
  const ipaChars = [...v.refIpa].filter((c) => !/\s/u.test(c)).length;
  const rate = ipaChars / (newLen / 25);
  const cutFrac = (v.refLen - newLen) / v.refLen;
  // ⚠ NO FRACTION GUARD ON A SILENCE-ONLY TRIM. The span runs from the first qualifying speech run to
  // the last, so what is removed is by construction not speech, and `sv` legitimately loses 55%. The
  // guards that remain are about the RESULT being usable, plus a rate check that only bites when an
  // event was dropped — the one operation that can take speech.
  let reject = null;
  if (newLen < 75) reject = `would leave only ${(newLen / 25).toFixed(1)}s of reference`;
  else if ((dropLead || dropTrail) && rate > 22)
    reject = `dropping an edge event would put the rate at ${rate.toFixed(1)} IPA ch/s — it was speech`;

  const kept = x.subarray(startFrame * HOP, endFrame * HOP);
  wav(join(OUT, `${id}.before.wav`), x);
  wav(join(OUT, `${id}.after.wav`), kept);
  const tag = reject ? `REJECTED — ${reject}` : `cut ${cutMs} ms (lead ${dropLead}, trail ${dropTrail})`;
  console.log(`  ${id.padEnd(34)} ${(v.refLen / 25).toFixed(1)}s -> ${(newLen / 25).toFixed(1)}s  ${rate.toFixed(1)} ch/s  ${tag}`);
  if (reject) continue;
  changes.push({ id, startFrame, newLen,
    // refRms describes the SOURCE audio; scale it by what the trim did to the decoded level so it
    // keeps meaning the same thing relative to the clip that is actually stored.
    refRms: Number((v.refRms * (rms(kept) / Math.max(1e-9, rms(x)))).toFixed(5)) });
}

if (!has("apply")) { console.log(`\n${changes.length} would change. Listen in ${OUT}, then re-run with --apply.`); process.exit(0); }

let text = raw;
for (const c of changes) {
  const v = byId.get(c.id);
  const flat = codes[c.id];
  const out = [];
  for (let cb = 0; cb < 8; cb++)
    for (let t = c.startFrame; t < c.startFrame + c.newLen; t++) out.push(flat[cb * v.refLen + t]);
  codes[c.id] = out;
  // Rewrite refLen and refRms in place, on this voice's line only.
  const line = new RegExp(`(\\{"id":"${c.id.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&")}"[^\\n]*?)"refLen":\\d+([^\\n]*?)"refRms":[\\d.]+`, "u");
  text = text.replace(line, `$1"refLen":${c.newLen}$2"refRms":${c.refRms}`);
}
writeFileSync(VP, text);
writeFileSync(CP, JSON.stringify(codes));
console.log(`\napplied ${changes.length} trims. Re-render those languages and LISTEN — the guards bound the damage, they do not prove the cut is right.`);
