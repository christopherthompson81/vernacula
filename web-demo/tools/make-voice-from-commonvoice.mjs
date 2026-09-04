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
if (!CV) { console.error("usage: --cv <locale> [--lang <demo code>] [--n 3] [--split test] [--from-dir <cv tree>] [--write]"); process.exit(2); }

const HF = "https://huggingface.co/datasets/fsicoli/common_voice_22_0/resolve/main";
/**
 * Read an ALREADY-DOWNLOADED Common Voice tree instead of fetching CV 22 from HuggingFace.
 *
 * ⚠ Mozilla Data Collective has no API route for accepting terms — they are accepted per dataset in
 * a browser — so a CV 26 corpus arrives as a local tarball rather than a URL. The alternative was
 * `make-voice-from-openslr.mjs --dir`, which reads a local tree but knows nothing about Common
 * Voice's columns: no up/down votes, no `variant`, and critically NO `client_id`, so its "top N by
 * audio score" can return three clips from one contributor and call them alternates. Everything
 * that makes CV selection sound lives in THIS file; only the transport differs.
 *
 * The layout is Common Voice's own: <root>/clip_durations.tsv, <root>/<split>.tsv, <root>/clips/*.mp3.
 */
const FROM_DIR = opt("from-dir");
/**
 * What gets recorded as the voice's provenance.
 *
 * ⚠ THIS WAS HARDCODED to `mozilla/common_voice_22_0`. With --from-dir that is simply false — an MDC
 * export is Common Voice 26 — and provenance that lies is worse than none: the licence, the version
 * and the row a clip came from are all read off this field. Defaults to the HF corpus this tool
 * fetches, and to the local tree's own name otherwise.
 */
const DATASET = opt("dataset", FROM_DIR ? `mdc:${FROM_DIR.replace(/\/+$/, "").split("/").pop()}` : "mozilla/common_voice_22_0");
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
const cvFile = (name, cached) => FROM_DIR ? join(FROM_DIR, name) : fetchTo(`${HF}/transcript/${CV}/${name}`, join(WORK, cached));
const rows = tsv(cvFile(`${SPLIT}.tsv`, `${SPLIT}.tsv`));

/**
 * clip -> duration in ms.
 *
 * ⚠ AN MDC EXPORT HAS NO `clip_durations.tsv`. The HuggingFace mirror ships one; the Mozilla Data
 * Collective tarballs do not, and the duration window is what narrows 73,020 rows to a shortlist —
 * without it every clip looks eligible and the audio scorer decodes the whole corpus. ffprobe can
 * answer, but not 73,020 times, so probe a SPEAKER-DIVERSE prefix instead of the file order: Common
 * Voice is dominated by a handful of prolific contributors, and probing the first N rows would spend
 * the whole budget on two or three voices.
 */
// Where the mp3s actually are: extracted under WORK for the HF path, already on disk for --from-dir.
// ⚠ The two paths index at DIFFERENT times. The HF path shortlists first and extracts only the
// chosen clips, so the index cannot exist until after extraction. --from-dir has every clip on disk
// from the start and NEEDS the index first, to probe durations. One function, called at whichever
// point the path allows.
const CLIP_ROOT = FROM_DIR ?? WORK;
const indexClips = () =>
  new Set(readdirSync(CLIP_ROOT, { recursive: true }).map(String).filter((f) => f.endsWith(".mp3")));

const PROBE_MAX = Number(opt("probe", 800));
function probeDurations(cands) {
  const bySpk = new Map();
  for (const r of cands) (bySpk.get(r.client_id) ?? bySpk.set(r.client_id, []).get(r.client_id)).push(r);
  const queues = [...bySpk.values()];
  const order = [];
  for (let i = 0; order.length < Math.min(PROBE_MAX, cands.length); i++) {
    let any = false;
    for (const q of queues) { if (q[i]) { order.push(q[i]); any = true; if (order.length >= PROBE_MAX) break; } }
    if (!any) break;
  }
  console.log(`  probing ${order.length} clips for duration across ${bySpk.size} contributors (no clip_durations.tsv)`);
  const out = {};
  for (const r of order) {
    const f = byName.get(r.path);
    if (!f) continue;
    try {
      out[r.path] = Math.round(1000 * Number(execFileSync("ffprobe",
        ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", join(CLIP_ROOT, String(f))],
        { encoding: "utf8" }).trim()));
    } catch { /* unreadable clip: leave it out, the duration filter then drops it */ }
  }
  return out;
}
const HAVE_DURATIONS = !FROM_DIR || existsSync(join(FROM_DIR, "clip_durations.tsv"));
let byName;   // clip basename -> path under CLIP_ROOT; set once, by whichever path gets there first
let durs;
if (HAVE_DURATIONS) {
  durs = Object.fromEntries(tsv(cvFile("clip_durations.tsv", "durations.tsv"))
    .map((r) => [r.clip, Number(r["duration[ms]"])]));
} else {
  byName = new Map([...indexClips()].map((f) => [f.split("/").pop(), f]));
  durs = probeDurations(rows.filter((r) => Number(r.up_votes || 0) >= VOTES
                                        && Number(r.down_votes || 0) === 0 && inScript(r.sentence)));
}

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
if (FROM_DIR) {
  console.log(`  local tree: ${FROM_DIR} (no download, no extract)`);
} else if (!existsSync(tar)) {
  if (PREFIX_MB) {
    console.log(`  fetching first ${PREFIX_MB} MB of ${SPLIT}_0.tar…`);
    execFileSync("curl", ["-sfL", "--max-time", "1800", "-r", `0-${PREFIX_MB * 1000000 - 1}`, tarUrl, "-o", tar]);
  } else fetchTo(tarUrl, tar);
}
if (!FROM_DIR) {
console.log("  extracting…");
try {
  // A truncated tar makes tar exit non-zero AFTER extracting every whole member — which is exactly
  // what the prefix mode wants, so the failure is expected and ignored there.
  execFileSync("tar", ["-xf", tar, "-C", WORK, "--wildcards", ...(PREFIX_MB ? ["*.mp3"] : shortlist.map((r) => `*${r.path}`))],
               { stdio: ["ignore", "ignore", "ignore"] });
} catch (e) { if (!PREFIX_MB) throw e; }
// The split archives run to hundreds of MB each and 26 languages will not fit on a scratch disk.
if (!args.includes("--keep-tar")) execFileSync("rm", ["-f", tar]);
}


/**
 * Is this clip actually CLIPPED, or merely normalised to full scale?
 *
 * ⚠ `peak > 0.98` rejects both, and that threw away a whole corpus: every Vaani clip peaks at
 * exactly 1.0 because the corpus is peak-normalised, yet only 0.001% of samples touch the ceiling
 * and the longest flat run is a single sample. Real clipping FLATTENS the waveform — consecutive
 * samples pinned at full scale — so that is what to measure.
 */
function clipping(x) {
  let atCeiling = 0, run = 0, worst = 0;
  for (let i = 0; i < x.length; i++) {
    if (Math.abs(x[i]) >= 0.999) { atCeiling++; worst = Math.max(worst, ++run); }
    else run = 0;
  }
  return { frac: atCeiling / x.length, run: worst };
}

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
  let sq = 0;
  for (let i = 0; i < n; i++) sq += x[i] * x[i];
  return { snr: mean - floor, speechFrac: speech.length / m, peak, rms: Math.sqrt(sq / n), clip: clipping(x) };
}

byName ??= new Map([...indexClips()].map((f) => [f.split("/").pop(), f]));
const present = byName;
if (PREFIX_MB) console.log(`  ${present.size} clips inside the prefix, ${shortlist.filter((r) => byName.has(r.path)).length} of them candidates`);
const scored = [];
let considered = 0;
for (const r of shortlist) {
  const mp3 = byName.get(r.path);
  if (!mp3) continue;
  // ⚠ SCORE UNTIL THERE ARE ENOUGH DISTINCT SPEAKERS, not until a flat count. In a thin locale one
  // contributor dominates the archive — every one of the first forty `pa-IN` clips was the same man —
  // so a fixed cap returns three sentences in one voice and calls them alternates.
  if (PREFIX_MB && ++considered > 40
      && (new Set(scored.map((x) => x.client_id)).size >= N || considered > 400)) break;
  const wav = join(WORK, "wav", r.path.replace(/\.mp3$/, ".wav"));
  execFileSync("ffmpeg", ["-v", "error", "-y", "-i", join(CLIP_ROOT, String(mp3)), "-ac", "1", "-ar", "24000",
                          "-sample_fmt", "s16", wav]);
  const s = score(wav);
  // Clipping is unrecoverable and a mostly-silent clip wastes the reference; both are hard rejects.
  // ⚠ AND A LEVEL FLOOR. Not a PEAK test — that is the check that silently discarded the whole Vaani
  // corpus, because peak-normalised audio all peaks at 1.0 — but an RMS one. `tr-17344946` scored 71 dB
  // and came back at peak 0.02 / rms 0.0035: 30 dB below every usable reference, and nothing in the
  // screen could see it. The quiet FLEURS voices (ar 0.030, hi 0.020, am 0.007) are the same class.
  if (!CLIPS.length && (s.clip.run >= 3 || s.clip.frac > 0.0005 || s.speechFrac < 0.45 || s.rms < 0.02)) continue;
  scored.push({ ...r, wav, ...s, ms: durs[r.path] });
}
// Quiet is fine (the output chain normalises); noisy is not, and neither is a clip that is half pause.
/**
 * ⚠ CAP THE SNR TERM. A digitally GATED clip reports 100+ dB because its 10th-percentile frame is
 * true zero, and uncapped that outranks every genuinely clean recording — it put a 45%-speech clip
 * top for `pa-IN` and a peak-0.99 one top for `tr`. Above ~45 dB the number stops describing the
 * recording and starts describing the noise gate, so it is worth nothing more.
 */
const SNR_CAP = 45;
if (!CLIPS.length) scored.sort((a, b) =>
  (Math.min(b.snr, SNR_CAP) + 40 * b.speechFrac) - (Math.min(a.snr, SNR_CAP) + 40 * a.speechFrac));
console.log(`  ${scored.length} scored`);

/**
 * ⚠ ALTERNATES SHOULD BE DIFFERENT PEOPLE. The picker offers a language's voices as a choice, and a
 * choice between three recordings of one speaker is not one. `client_id` identifies the contributor,
 * so at most `--per-speaker` clips are taken from any one of them, best first, and the cap is only
 * relaxed if that cannot fill N.
 *
 * ⚠ THE ID IS USED HERE AND STORED NOWHERE. It is a stable per-speaker handle and does not belong in
 * this repo — see the source block below, which records the clip and drops the contributor.
 */
const PER_SPEAKER = Number(opt("per-speaker", 1));
function pick(list, cap) {
  const seen = new Map(), out = [];
  for (const c of list) {
    const k = c.client_id ?? `?${out.length}`;
    const n = seen.get(k) ?? 0;
    if (n >= cap) continue;
    seen.set(k, n + 1); out.push(c);
    if (out.length >= N) break;
  }
  return out;
}
let chosen = CLIPS.length ? scored.slice(0, N) : pick(scored, PER_SPEAKER);
if (chosen.length < N && !CLIPS.length) {
  const relaxed = pick(scored, PER_SPEAKER + 1);
  if (relaxed.length > chosen.length) {
    console.log(`  only ${new Set(scored.map((x) => x.client_id)).size} distinct speaker(s) available — `
      + `allowing ${PER_SPEAKER + 1} clip(s) each to fill ${N}`);
    chosen = relaxed;
  }
}
// ⚠ REPORT WHAT WAS CHOSEN, NOT WHAT SCORED BEST. These are different lists once the per-speaker cap
// applies, and printing the score order while encoding the picked order made the tool describe clips
// it had not taken — I selected four ids off the printed list and two of them did not exist.
console.log(`  chosen: ${chosen.length} clip(s) from ${new Set(chosen.map((c) => c.client_id)).size} speaker(s)`);
for (const s2 of chosen)
  console.log(`    ${s2.path}  snr ${s2.snr.toFixed(0)} dB  speech ${(s2.speechFrac * 100).toFixed(0)}%  `
    + `peak ${s2.peak.toFixed(2)}  rms ${s2.rms.toFixed(3)}  ${s2.gender || "?"} ${s2.age || "?"}`);
const entries = [];
for (const [i, c] of chosen.entries()) {
  const ipaPath = join(WORK, `${c.path}.ipa`);
  // Phonemize in the phonemizer repo, through the same async path the demo uses.
  // ⚠ PER-PROCESS FILENAME. Two sourcing runs in parallel share this directory, and a fixed
  // name let one delete the temp module while the other was importing it — the Maithili
  // encode died on "cannot find module" while an Awadhi run finished normally.
  const probe = join(PHONEMIZER, `cv-phonemize.${process.pid}.tmp.mts`);
  writeFileSync(probe, `import { phonemizeAsync } from "./src/index.ts";\n`
    + `console.log(await phonemizeAsync(${JSON.stringify(c.sentence)}, ${JSON.stringify(LANG)}));\n`);
  let ipa;
  try { ipa = execFileSync("npx", ["tsx", `cv-phonemize.${process.pid}.tmp.mts`], { cwd: PHONEMIZER, encoding: "utf8" }).trim(); }
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
        dataset: DATASET, lang: CV, file: c.path, split: SPLIT,
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
