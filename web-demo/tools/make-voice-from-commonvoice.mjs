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
/** Minimum up-votes. ⚠ Lower this only with the split in mind: `other` is UNVALIDATED, so its clips
 *  carry no votes at all and nothing but the audio score stands between a bad read and the demo. */
const VOTES = Number(opt("votes", 2));
/**
 * Regex over Common Voice's `accents` column.
 *
 * ⚠ Needed wherever the locale is broader than the demo language. Common Voice `es` is Spanish
 * everywhere — the largest labelled group in the test split is Mexico, not Spain — so sourcing a
 * CASTILIAN reference without filtering would likely re-import the Latin American accent the demo
 * already has from FLEURS. Same for `pt`, which mixes Brazil and Portugal.
 */
const ACCENT = opt("accent") ? new RegExp(opt("accent"), "iu") : null;
/**
 * Regex over Common Voice's `variant` column — the RIGHT filter for a language whose locale covers
 * more than one standard.
 *
 * ⚠ European Portuguese was written off as unsourceable on the strength of the `accents` column,
 * which is nearly empty for `pt` and labelled only by region-of-Brazil. `variant` is the column that
 * separates the standards, and it holds readable labels rather than codes: "Portuguese (Portugal)",
 * not "pt-PT". 595 European rows sit in the train split alone, 65 of them in the length band.
 */
const VARIANT = opt("variant") ? new RegExp(opt("variant"), "iu") : null;
/**
 * Fetch only the first N MB of the split archive, with a Range request.
 *
 * ⚠ For the big locales the archive is most of an hour on a throttled connection (Kinyarwanda's
 * test split is 676 MB, Uyghur's 553 MB) and all of it is thrown away but three clips. A tar is
 * sequential, so a prefix is a valid archive up to the member it cuts through: extract what is
 * whole, ignore the truncation, and score whatever candidates landed inside. It only works when the
 * language has candidates to spare — with 924 qualifying clips spread over 16,213, a tenth of the
 * archive still holds ~90 of them — so the shortlist is widened in this mode rather than capped.
 */
const PREFIX_MB = Number(opt("prefix-mb", 0));
/** Force specific clips instead of ranking — this is what makes a committed voice REPRODUCIBLE:
 *  the shipped set names its files, and re-running with them re-derives the same codes. */
const CLIPS = (opt("clip", "") || "").split(",").filter(Boolean);
const ENCODER = opt("encoder", "/mnt/data/omnivoice_ipa/onnx_base/higgs_encoder.onnx");
const PHONEMIZER = "../external/vernacula-phonemizer";
const WORK = opt("work", `${process.env.TMPDIR ?? "/tmp"}/cv-voices/${CV}`);
if (!CV) { console.error("usage: --cv <locale> [--lang <demo code>] [--n 3] [--split test] [--write]"); process.exit(2); }

const HF = "https://huggingface.co/datasets/fsicoli/common_voice_22_0/resolve/main";
mkdirSync(join(WORK, "wav"), { recursive: true });

/**
 * The scripts the phonemizer's engine declares for this language.
 *
 * ⚠ A Common Voice locale is not guaranteed to be written in the script the engine reads. `pa-IN` is
 * Gurmukhi while `pnb` is Shahmukhi; `zgh` ships both Tifinagh and Latin; `nan-tw` mixes Han with
 * romanisation. Phonemizing a sentence in the wrong script produces confident nonsense for the
 * reference transcript, and the reference transcript is fed to the model beside the codes — so a
 * clip whose sentence is not in the engine's script is rejected rather than encoded.
 */
function declaredScripts(lang) {
  const dir = PHONEMIZER;
  const src = readFileSync(join(dir, "src/core/scripts.ts"), "utf8");
  const tbl = src.slice(src.indexOf("MANIFESTLESS_SCRIPTS"));
  for (const m of tbl.slice(0, tbl.indexOf("\n};")).matchAll(/"?([\w-]+)"?:\s*\[([^\]]*)\]/g))
    if (m[1] === lang) return [...m[2].matchAll(/"([\w ]+)"/g)].map((x) => x[1].replace(/ /gu, "_"));
  for (const d of readdirSync(join(dir, "data/languages"))) {
    for (const f of readdirSync(join(dir, "data/languages", d)).filter((x) => x.endsWith(".jsonc"))) {
      const t = readFileSync(join(dir, "data/languages", d, f), "utf8");
      if (t.match(/"language":\s*"([\w-]+)"/)?.[1] !== lang) continue;
      const sc = t.match(/"script":\s*\[([^\]]*)\]/)?.[1];
      if (sc) return [...sc.matchAll(/"([\w ]+)"/g)].map((x) => x[1].replace(/ /gu, "_"));
    }
  }
  return [];
}

const scriptRe = (n) => n === "Kana" ? /\p{Script=Hiragana}|\p{Script=Katakana}/u
                                     : new RegExp(`\\p{Script=${n}}`, "u");

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
const SCRIPTS = declaredScripts(LANG).map(scriptRe);
if (!SCRIPTS.length) console.warn(`  ⚠ ${LANG} declares no script — sentences not script-checked`);
const inScript = (s) => !SCRIPTS.length || SCRIPTS.some((re) => re.test(s));
const durs = Object.fromEntries(tsv(fetchTo(`${HF}/transcript/${CV}/clip_durations.tsv`, join(WORK, "durations.tsv")))
  .map((r) => [r.clip, Number(r["duration[ms]"])]));
const rows = tsv(fetchTo(`${HF}/transcript/${CV}/${SPLIT}.tsv`, join(WORK, `${SPLIT}.tsv`)));

// Metadata pass: right length, validated by at least two listeners, rejected by none.
const shortlist = CLIPS.length ? CLIPS.map((c) => rows.find((r) => r.path === c)).filter(Boolean)
  : rows.filter((r) => {
      const ms = durs[r.path] ?? 0;
      if (ACCENT && !ACCENT.test(r.accents ?? "")) return false;
      if (VARIANT && !VARIANT.test(r.variant ?? "")) return false;
      return ms >= MIN_MS && ms <= MAX_MS && Number(r.up_votes || 0) >= VOTES && Number(r.down_votes || 0) === 0;
    }).filter((r) => inScript(r.sentence)).slice(0, PREFIX_MB ? 4000 : 40);
for (const r of shortlist) if (!inScript(r.sentence))
  console.warn(`  ⚠ ${r.path}: sentence is not in ${LANG}'s script`);
if (CLIPS.length && shortlist.length !== CLIPS.length)
  console.warn(`  ⚠ ${CLIPS.length - shortlist.length} named clip(s) not in ${SPLIT}.tsv`);
console.log(`  ${shortlist.length} candidates by metadata (${MIN_MS}-${MAX_MS} ms, ${VOTES}+ up, 0 down)`);
if (!shortlist.length) { console.error("  nothing matched — widen --min/--max or try --split train"); process.exit(1); }

console.log(`  script check: ${SCRIPTS.length ? declaredScripts(LANG).join("+") : "none"}`
          + (ACCENT ? `, accent filter ${ACCENT}` : "")
          + (VARIANT ? `, variant filter ${VARIANT}` : ""));
const tarUrl = `${HF}/audio/${CV}/${SPLIT}/${CV}_${SPLIT}_0.tar`;
const tar = join(WORK, `${SPLIT}_0.tar`);
if (!existsSync(tar)) {
  if (PREFIX_MB) {
    console.log(`  fetching first ${PREFIX_MB} MB of ${SPLIT}_0.tar…`);
    execFileSync("curl", ["-sfL", "--max-time", "1800", "-r", `0-${PREFIX_MB * 1000000 - 1}`, tarUrl, "-o", tar]);
  } else fetchTo(tarUrl, tar);
}
console.log("  extracting…");
try {
  // A truncated tar makes tar exit non-zero AFTER extracting every whole member — which is exactly
  // what the prefix mode wants, so the failure is expected and ignored there.
  execFileSync("tar", ["-xf", tar, "-C", WORK, "--wildcards", ...(PREFIX_MB ? ["*.mp3"] : shortlist.map((r) => `*${r.path}`))],
               { stdio: ["ignore", "ignore", "ignore"] });
} catch (e) { if (!PREFIX_MB) throw e; }
// The split archives run to hundreds of MB each and 26 languages will not fit on a scratch disk.
if (!args.includes("--keep-tar")) execFileSync("rm", ["-f", tar]);

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

const present = new Set(readdirSync(WORK, { recursive: true }).map(String).filter((f) => f.endsWith(".mp3")));
const byName = new Map([...present].map((f) => [f.split("/").pop(), f]));
if (PREFIX_MB) console.log(`  ${present.size} clips inside the prefix, ${shortlist.filter((r) => byName.has(r.path)).length} of them candidates`);
const scored = [];
let considered = 0;
for (const r of shortlist) {
  const mp3 = byName.get(r.path);
  if (!mp3) continue;
  if (PREFIX_MB && ++considered > 40) break;
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

// ⚠ ALWAYS persist the codes, even on a dry run. Encoding is the expensive half (the split archive
// is hundreds of MB and is deleted after extraction), so a run that only PRINTS entries throws away
// the one artifact that cannot be cheaply recomputed. `tools/merge-cv-voices.mjs` reads these.
writeFileSync(join(WORK, "voices.json"), JSON.stringify(entries));
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
