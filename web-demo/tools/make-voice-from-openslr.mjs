#!/usr/bin/env node
/**
 * Source a reference voice from an OpenSLR "index file + wav files" corpus.
 *
 * Several of the languages the demo lacks have exactly one acceptable open corpus, and it is on
 * OpenSLR rather than HuggingFace: UK/Ireland English dialects (SLR83, CC BY-SA 4.0), Tibetan
 * (SLR158, CC BY 4.0), Sundanese (SLR44), Sinhala (SLR52). They share a shape — a zip of wavs plus a
 * TSV/CSV index of `<id>\t<transcript>` — so one downloader serves them all.
 *
 *   node tools/make-voice-from-openslr.mjs --url https://.../southern_english_male.zip \
 *       --lang en-GB --n 3 [--index line_index.csv] [--sep ,]
 *
 * ⚠ These are whole-corpus downloads (SLR83's largest dialect zip is hundreds of MB) — unlike the
 * HuggingFace path, there is no way to fetch three clips. Run it once per language and keep the
 * extracted directory if you may want alternates.
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, basename } from "node:path";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i < 0 ? d : args[i + 1]; };
const URL = opt("url"), LANG = opt("lang");
const N = Number(opt("n", 3)), SEP = opt("sep", "\t");
const MIN_SEC = Number(opt("min-sec", 4)), MAX_SEC = Number(opt("max-sec", 14));
const ENCODER = opt("encoder", "/mnt/data/omnivoice_ipa/onnx_base/higgs_encoder.onnx");
const PHONEMIZER = "../external/vernacula-phonemizer";
const WORK = opt("work", `/tmp/slr-voices/${basename(URL ?? "x").replace(/\.\w+$/, "")}`);
if (!URL || !LANG) { console.error("usage: --url <zip> --lang <demo code> [--n 3]"); process.exit(2); }

mkdirSync(join(WORK, "wav"), { recursive: true });
const archive = join(WORK, basename(URL));
if (!existsSync(archive)) {
  console.log(`  fetching ${basename(URL)}…`);
  execFileSync("curl", ["-sfL", "--max-time", "3600", URL, "-o", archive]);
}
console.log(`  ${(statSync(archive).size / 1e6).toFixed(0)} MB; extracting…`);
if (!existsSync(join(WORK, "x"))) {
  mkdirSync(join(WORK, "x"), { recursive: true });
  if (archive.endsWith(".zip")) execFileSync("unzip", ["-qo", archive, "-d", join(WORK, "x")]);
  else execFileSync("tar", ["-xf", archive, "-C", join(WORK, "x")]);
}

/** Every file under a directory, recursively. */
const walk = (d) => readdirSync(d, { withFileTypes: true }).flatMap((e) =>
  e.isDirectory() ? walk(join(d, e.name)) : [join(d, e.name)]);
const files = walk(join(WORK, "x"));
const wavs = new Map(files.filter((f) => /\.(wav|flac|mp3|opus)$/i.test(f))
  .map((f) => [basename(f).replace(/\.\w+$/, ""), f]));
const indexes = files.filter((f) => /line_index|\.tsv$|\.csv$|metadata/i.test(f) && !/\.(wav|flac|mp3)$/i.test(f));
console.log(`  ${wavs.size} audio files, ${indexes.length} index file(s)`);

// id -> transcript, from whichever index files the corpus ships.
const text = new Map();
for (const idx of indexes) {
  for (const line of readFileSync(idx, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const parts = line.split(SEP === "," && line.includes('","') ? /","|^"|"$/ : SEP).map((s) => s.trim().replace(/^"|"$/g, ""));
    // The id is whichever column matches a wav we have; the transcript is the longest other column.
    const id = parts.find((p) => wavs.has(p));
    if (!id) continue;
    const t = parts.filter((p) => p !== id).sort((a, b) => b.length - a.length)[0];
    if (t && t.length > 3) text.set(id, t);
  }
}
console.log(`  ${text.size} transcripts matched to audio`);
if (!text.size) { console.error("  index/audio mismatch — check --sep or the index format"); process.exit(1); }

function score(wavPath) {
  const b = readFileSync(wavPath);
  let off = 12, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString("ascii", off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === "data") data = b.subarray(off + 8, off + 8 + sz);
    off += 8 + sz + (sz & 1);
  }
  const n = data.length / 2, x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = data.readInt16LE(i * 2) / 32768;
  const fr = 240, m = Math.floor(n / fr), db = new Float64Array(m);
  let peak = 0;
  for (let f = 0; f < m; f++) {
    let s = 0;
    for (let i = f * fr; i < (f + 1) * fr; i++) { s += x[i] * x[i]; if (Math.abs(x[i]) > peak) peak = Math.abs(x[i]); }
    db[f] = 10 * Math.log10(s / fr + 1e-12);
  }
  const sorted = [...db].sort((a, b2) => a - b2);
  const speech = db.filter((v) => v > sorted[m - 1] - 25);
  return { snr: speech.reduce((a, b2) => a + b2, 0) / (speech.length || 1) - sorted[Math.floor(m * 0.1)],
           speechFrac: speech.length / m, peak, sec: n / 24000 };
}

const scored = [];
for (const [id, t] of text) {
  if (scored.length >= 12) break;
  const wav = join(WORK, "wav", `${id}.wav`);
  try {
    if (!existsSync(wav))
      execFileSync("ffmpeg", ["-v", "error", "-y", "-i", wavs.get(id), "-ac", "1", "-ar", "24000",
                              "-sample_fmt", "s16", wav]);
  } catch { continue; }
  const s = score(wav);
  if (s.sec < MIN_SEC || s.sec > MAX_SEC || s.peak > 0.98 || s.speechFrac < 0.45) continue;
  scored.push({ id, text: t, wav, ...s });
}
scored.sort((a, b) => (b.snr + 40 * b.speechFrac) - (a.snr + 40 * a.speechFrac));
console.log(`  ${scored.length} scored; best:`);
for (const s of scored.slice(0, N))
  console.log(`    ${s.id}  ${s.sec.toFixed(1)}s  snr ${s.snr.toFixed(0)} dB  speech ${(s.speechFrac * 100).toFixed(0)}%`);

const entries = [];
for (const [i, c] of scored.slice(0, N).entries()) {
  const ipaPath = join(WORK, `${i}.ipa`);
  const probe = join(PHONEMIZER, "slr-phonemize.tmp.mts");
  writeFileSync(probe, `import { phonemizeAsync } from "./src/index.ts";\n`
    + `console.log(await phonemizeAsync(${JSON.stringify(c.text)}, ${JSON.stringify(LANG)}));\n`);
  let ipa;
  try { ipa = execFileSync("npx", ["tsx", "slr-phonemize.tmp.mts"], { cwd: PHONEMIZER, encoding: "utf8" }).trim(); }
  finally { execFileSync("rm", ["-f", probe]); }
  writeFileSync(ipaPath, ipa);
  const id = `${LANG}-${c.id}`.slice(0, 40);
  const out = execFileSync("node", ["tools/make-voices.mjs", ENCODER, c.wav, ipaPath, id, id], { encoding: "utf8" });
  const v = JSON.parse(out)[0];
  entries.push({
    voice: {
      id, lang: LANG, ...(i === 0 ? { default: true } : {}),
      label: `${LANG} · ${c.sec.toFixed(1)}s`,
      refIpa: v.refIpa, refLen: v.refLen, refRms: Number(v.refRms.toFixed(5)),
      source: { dataset: `openslr:${basename(URL)}`, lang: LANG, file: c.id, split: "all",
                sentenceId: null, gender: "", durationS: Number(c.sec.toFixed(1)),
                candidateIndex: i, text: c.text },
    },
    codes: v.codes,
  });
}
writeFileSync(join(WORK, "voices.json"), JSON.stringify(entries));
console.log(`\n${entries.length} voices -> ${join(WORK, "voices.json")}`);
