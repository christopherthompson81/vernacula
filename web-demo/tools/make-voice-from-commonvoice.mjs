#!/usr/bin/env node
/**
 * Source a reference voice from Mozilla Common Voice (CC0) for a language FLEURS does not cover.
 *
 * The demo clones a voice for every generation, so each of the 193 languages needs a reference clip.
 * 102 come from the FLEURS-derived fine-tune corpus; the rest are read by a donor from a neighbouring
 * language until a native clip exists. Common Voice 22.0 covers 137 languages and is CC0, which
 * makes it the obvious first source — this automates the whole path from dataset to voices.jsonc.
 *
 *   node tools/make-voice-from-commonvoice.mjs --cv ab --lang ab --n 3 [--split test] [--write]
 *
 * `--cv` is the Common Voice locale, `--lang` the demo/phonemizer code (they differ: cv/chv,
 * quy/qu, nan-tw/nan, hy-AM/hy…). Without `--write` it prints the entries and changes nothing.
 *
 * ⚠ SELECTION IS MEASURED, NOT TRUSTED. Common Voice is crowd-recorded: levels, room and mic vary
 * enormously, and the up/down votes only say the reading matches the sentence. So metadata narrows
 * the field (8-9 s, 2+ up-votes, no down-votes) and the AUDIO decides: every candidate is scored for
 * noise floor, speech fraction and peak, and the top few are encoded for a listening test. A noisy
 * reference is cloned faithfully — the noise comes out in every sentence the demo ever speaks.
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i < 0 ? d : args[i + 1]; };
const CV = opt("cv"), LANG = opt("lang", CV), SPLIT = opt("split", "test");
const N = Number(opt("n", 3)), WRITE = args.includes("--write");
const MIN_MS = Number(opt("min", 7800)), MAX_MS = Number(opt("max", 9200));
/** Force specific clips instead of ranking — this is what makes a committed voice REPRODUCIBLE:
 *  the shipped set names its files, and re-running with them re-derives the same codes. */
const CLIPS = (opt("clip", "") || "").split(",").filter(Boolean);
const ENCODER = opt("encoder", "/mnt/data/omnivoice_ipa/onnx_base/higgs_encoder.onnx");
const PHONEMIZER = "../external/vernacula-phonemizer";
const WORK = opt("work", `${process.env.TMPDIR ?? "/tmp"}/cv-voices/${CV}`);
if (!CV) { console.error("usage: --cv <locale> [--lang <demo code>] [--n 3] [--split test] [--write]"); process.exit(2); }

const HF = "https://huggingface.co/datasets/fsicoli/common_voice_22_0/resolve/main";
mkdirSync(join(WORK, "wav"), { recursive: true });

const fetchTo = (url, path) => {
  if (existsSync(path)) return path;
  console.log(`  fetching ${url.split("/").pop()}…`);
  execFileSync("curl", ["-sfL", "--max-time", "1800", url, "-o", path]);
  return path;
};

const tsv = (path) => {
  const [head, ...rows] = readFileSync(path, "utf8").split("\n").filter(Boolean);
  const cols = head.split("\t");
  return rows.map((r) => Object.fromEntries(r.split("\t").map((v, i) => [cols[i], v])));
};

console.log(`Common Voice ${CV} -> demo language ${LANG}`);
const durs = Object.fromEntries(tsv(fetchTo(`${HF}/transcript/${CV}/clip_durations.tsv`, join(WORK, "durations.tsv")))
  .map((r) => [r.clip, Number(r["duration[ms]"])]));
const rows = tsv(fetchTo(`${HF}/transcript/${CV}/${SPLIT}.tsv`, join(WORK, `${SPLIT}.tsv`)));

// Metadata pass: right length, validated by at least two listeners, rejected by none.
const shortlist = CLIPS.length ? CLIPS.map((c) => rows.find((r) => r.path === c)).filter(Boolean)
  : rows.filter((r) => {
      const ms = durs[r.path] ?? 0;
      return ms >= MIN_MS && ms <= MAX_MS && Number(r.up_votes) >= 2 && Number(r.down_votes) === 0;
    }).slice(0, 40);
if (CLIPS.length && shortlist.length !== CLIPS.length)
  console.warn(`  ⚠ ${CLIPS.length - shortlist.length} named clip(s) not in ${SPLIT}.tsv`);
console.log(`  ${shortlist.length} candidates by metadata (${MIN_MS}-${MAX_MS} ms, 2+ up, 0 down)`);
if (!shortlist.length) { console.error("  nothing matched — widen --min/--max or try --split train"); process.exit(1); }

const tar = fetchTo(`${HF}/audio/${CV}/${SPLIT}/${CV}_${SPLIT}_0.tar`, join(WORK, `${SPLIT}_0.tar`));
console.log("  extracting…");
execFileSync("tar", ["-xf", tar, "-C", WORK, "--wildcards", ...shortlist.map((r) => `*${r.path}`)],
             { stdio: ["ignore", "ignore", "ignore"] });

/** Noise floor, speech fraction and peak, from 10 ms frames of the decoded clip. */
function score(wavPath) {
  const b = readFileSync(wavPath);
  let off = 12, data = null, bits = 16;
  while (off + 8 <= b.length) {
    const id = b.toString("ascii", off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === "fmt ") bits = b.readUInt16LE(off + 22);
    if (id === "data") data = b.subarray(off + 8, off + 8 + sz);
    off += 8 + sz + (sz & 1);
  }
  const n = data.length / (bits / 8);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = data.readInt16LE(i * 2) / 32768;
  const fr = 240, m = Math.floor(n / fr), db = new Float64Array(m);
  let peak = 0;
  for (let f = 0; f < m; f++) {
    let s = 0;
    for (let i = f * fr; i < (f + 1) * fr; i++) { s += x[i] * x[i]; if (Math.abs(x[i]) > peak) peak = Math.abs(x[i]); }
    db[f] = 10 * Math.log10(s / fr + 1e-12);
  }
  const sorted = [...db].sort((a, b2) => a - b2);
  const floor = sorted[Math.floor(m * 0.1)], top = sorted[m - 1];
  const speech = db.filter((v) => v > top - 25);
  const mean = speech.reduce((a, b2) => a + b2, 0) / (speech.length || 1);
  return { snr: mean - floor, speechFrac: speech.length / m, peak };
}

const scored = [];
for (const r of shortlist) {
  const mp3 = readdirSync(WORK, { recursive: true }).find((f) => String(f).endsWith(r.path));
  if (!mp3) continue;
  const wav = join(WORK, "wav", r.path.replace(/\.mp3$/, ".wav"));
  execFileSync("ffmpeg", ["-v", "error", "-y", "-i", join(WORK, String(mp3)), "-ac", "1", "-ar", "24000",
                          "-sample_fmt", "s16", wav]);
  const s = score(wav);
  // Clipping is unrecoverable and a mostly-silent clip wastes the reference; both are hard rejects.
  if (!CLIPS.length && (s.peak > 0.98 || s.speechFrac < 0.45)) continue;
  scored.push({ ...r, wav, ...s, ms: durs[r.path] });
}
// Quiet is fine (the output chain normalises); noisy is not, and neither is a clip that is half pause.
if (!CLIPS.length) scored.sort((a, b) => (b.snr + 40 * b.speechFrac) - (a.snr + 40 * a.speechFrac));
console.log(`  ${scored.length} scored; best:`);
for (const s of scored.slice(0, N))
  console.log(`    ${s.path}  snr ${s.snr.toFixed(0)} dB  speech ${(s.speechFrac * 100).toFixed(0)}%  peak ${s.peak.toFixed(2)}  ${s.gender || "?"} ${s.age || "?"}`);

const chosen = scored.slice(0, N);
const entries = [];
for (const [i, c] of chosen.entries()) {
  const ipaPath = join(WORK, `${c.path}.ipa`);
  // Phonemize in the phonemizer repo, through the same async path the demo uses.
  const probe = join(PHONEMIZER, "cv-phonemize.tmp.mts");
  writeFileSync(probe, `import { phonemizeAsync } from "./src/index.ts";\n`
    + `console.log(await phonemizeAsync(${JSON.stringify(c.sentence)}, ${JSON.stringify(LANG)}));\n`);
  let ipa;
  try { ipa = execFileSync("npx", ["tsx", "cv-phonemize.tmp.mts"], { cwd: PHONEMIZER, encoding: "utf8" }).trim(); }
  finally { execFileSync("rm", ["-f", probe]); }
  writeFileSync(ipaPath, ipa);

  // The clip's own number, so an id traces straight back to the dataset row. `replace(/\D/g,"")`
  // over the whole filename folds in the "22" of common_voice and the "3" of mp3.
  const id = `${LANG}-${c.path.match(/(\d+)\.mp3$/u)?.[1] ?? c.path}`;
  const out = execFileSync("node", ["tools/make-voices.mjs", ENCODER, c.wav, ipaPath, id, id], { encoding: "utf8" });
  const v = JSON.parse(out)[0];
  const gender = { female_feminine: "FEMALE", male_masculine: "MALE" }[c.gender] ?? "";
  entries.push({
    voice: {
      id, lang: LANG, ...(i === 0 ? { default: true } : {}),
      label: `${LANG} · ${gender || "unknown"} · ${(c.ms / 1000).toFixed(1)}s`,
      refIpa: v.refIpa, refLen: v.refLen, refRms: Number(v.refRms.toFixed(5)),
      source: {
        dataset: "mozilla/common_voice_22_0", lang: CV, file: c.path, split: SPLIT,
        // ⚠ The 64-hex sentence id is truncated and client_id is dropped: neither belongs in a
        // published file, and 12 hex is enough to find the row again in the dataset.
        sentenceId: (c.sentence_id ?? "").slice(0, 12), gender,
        durationS: Number((c.ms / 1000).toFixed(1)), candidateIndex: i, text: c.sentence,
      },
    },
    codes: v.codes,
  });
}

const json = (o) => JSON.stringify(o, null, 0);
console.log("\n// paste into public/models/voices.jsonc (codes go to voice-codes.json):");
for (const e of entries) console.log(`  // "${e.voice.source.text}"\n  ${json(e.voice)},`);

if (WRITE) {
  const codesPath = "public/models/voice-codes.json";
  const codes = JSON.parse(readFileSync(codesPath, "utf8"));
  for (const e of entries) codes[e.voice.id] = e.codes;
  writeFileSync(codesPath, JSON.stringify(codes));
  const vp = "public/models/voices.jsonc";
  let t = readFileSync(vp, "utf8");
  const block = entries.map((e) => `  // "${e.voice.source.text}"\n  ${json(e.voice)},\n`).join("");
  const header = `  // ${LANG} — Common Voice ${CV} (CC0). Sourced because FLEURS has no ${LANG} speaker.\n`;
  t = t.replace("  // af — af_za", header + block + "  // af — af_za");
  writeFileSync(vp, t);
  console.log(`\nwrote ${entries.length} voices to ${vp} and ${codesPath}`);
  console.log("next: drop the donor in tools/data/language-meta.json, re-run make-language-catalog.mjs, then LISTEN");
}
