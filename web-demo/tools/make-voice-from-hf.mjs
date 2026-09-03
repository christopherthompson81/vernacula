#!/usr/bin/env node
/**
 * Source a reference voice from ANY ungated HuggingFace audio dataset.
 *
 * `make-voice-from-commonvoice.mjs` knows Common Voice's split archives and TSVs. Most other corpora
 * are parquet, and downloading a parquet shard to reach three clips is absurd when the datasets
 * server will hand over individual rows — transcript, duration and a direct audio URL — for any
 * dataset that is not gated. That is what this uses, so the cost of sourcing a voice is a few MB
 * rather than the corpus.
 *
 *   node tools/make-voice-from-hf.mjs --dataset facebook/omnilingual-asr-corpus \
 *       --config ary_Arab --split train --lang ary --n 3 [--write]
 *
 * ⚠ SELECTION IS STILL MEASURED. The row metadata narrows by duration and transcript shape; the
 * AUDIO decides, on the same noise-floor / speech-fraction / peak screen the Common Voice tool uses.
 * And the last word is a listening test: nothing here can hear a bad read or a second speaker.
 *
 * ⚠ THE TRANSCRIPT MUST MATCH THE AUDIO EXACTLY. It is fed to the model beside the codes, so a clip
 * may never be trimmed to fit a length target — the whole clip is encoded or none of it is. Where a
 * corpus only has long clips (Omnilingual's spontaneous speech runs 25-80 s) that means a long
 * reference, which costs generation time on every sentence the demo later speaks in that language.
 * `--max-sec` is therefore a real quality knob, not a filter of convenience.
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { trimToSentences } from "./trim-to-sentences.mjs";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i < 0 ? d : args[i + 1]; };
const DATASET = opt("dataset"), CONFIG = opt("config", "default"), SPLIT = opt("split", "train");
const LANG = opt("lang");
const N = Number(opt("n", 3)), WRITE = args.includes("--write");
const MIN_SEC = Number(opt("min-sec", 4)), MAX_SEC = Number(opt("max-sec", 14));
const SCAN = Number(opt("scan", 300));            // rows to look at
const TEXT_FIELD = opt("text"), AUDIO_FIELD = opt("audio", "audio"), DUR_FIELD = opt("duration");
const FILTER_FIELD = opt("filter-field"), FILTER_RE = opt("filter") ? new RegExp(opt("filter"), "iu") : null;
/**
 * ⚠ REQUIRE THE TRANSCRIPT TO STAY INSIDE THE LANGUAGE'S OWN ALPHABET.
 *
 * A reference is a claim that THIS audio realises THIS IPA. A foreign proper noun breaks the claim
 * in a way no audio metric can see: Hawaiian has no /d/ and no /s/, so the phonemizer correctly
 * nativises "Indonesia" to *inkonekia* — and the speaker, just as correctly, says it the Indonesian
 * way. The reference then teaches the model that Hawaiian /k/ sounds like [d] and [ʒ]. The stored
 * clip was flagged by ear as "a lot of noise in the middle"; the mismatch is a likelier cause than
 * the microphone.
 *
 * The check cannot be derived from the phonemizer's own manifests — those deliberately MAP the loan
 * letters (t→k, s→k, d/g→k) so that loanwords read sensibly, so harvesting their grapheme keys hands
 * back the whole Latin alphabet. It has to be the language's NATIVE inventory, passed in.
 *
 *   --letters "aeiouāēīōūhklmnpwʻ"
 */
const LETTERS = opt("letters") ? new Set([...opt("letters").toLowerCase()]) : null;
const outsideAlphabet = (t) => {
  if (!LETTERS) return [];
  const bad = new Set();
  for (const ch of t.toLowerCase()) if (/\p{L}/u.test(ch) && !LETTERS.has(ch)) bad.add(ch);
  return [...bad];
};
/**
 * Cut over-long clips at a pause, keeping the sentences that precede it.
 *
 * ⚠ Only where the cut is PROVABLE — `trim-to-sentences.mjs` proceeds when the transcript's sentence
 * count equals the count of speech runs separated by a real pause, so run k is sentence k. Without
 * that agreement the clip is dropped, because a reference whose text does not match its audio is
 * the one failure that stays invisible afterwards.
 */
const TRIM = args.includes("--trim");
const ENCODER = opt("encoder", "/mnt/data/omnivoice_ipa/onnx_base/higgs_encoder.onnx");
const PHONEMIZER = "../external/vernacula-phonemizer";
const WORK = opt("work", `${process.env.TMPDIR ?? "/tmp"}/hf-voices/${(DATASET ?? "x").replace(/\//g, "_")}_${CONFIG}`);
if (!DATASET || !LANG) { console.error("usage: --dataset <id> --lang <demo code> [--config c] [--split s] [--n 3] [--write]"); process.exit(2); }

mkdirSync(join(WORK, "wav"), { recursive: true });
const ROWS = "https://datasets-server.huggingface.co/rows";

/**
 * HuggingFace token, for datasets behind an accept-the-terms gate.
 *
 * ⚠ PASSED THROUGH A CURL CONFIG FILE, NEVER ON THE COMMAND LINE. Anything in argv is visible in
 * `ps` to every process on the machine, and this is a credential. The file is written 0600 into the
 * work directory and removed on exit. It is only ever sent to huggingface.co hosts.
 */
const TOKEN = (() => {
  if (process.env.HF_TOKEN) return process.env.HF_TOKEN.trim();
  const f = join(process.env.HOME ?? "", ".cache/huggingface/token");
  try { return readFileSync(f, "utf8").trim(); } catch { return ""; }
})();
const CURLRC = join(WORK, ".curlrc");
if (TOKEN) {
  writeFileSync(CURLRC, `header = "Authorization: Bearer ${TOKEN}"\n`, { mode: 0o600 });
  process.on("exit", () => { try { execFileSync("rm", ["-f", CURLRC]); } catch { /* best effort */ } });
}
/** curl args for a huggingface.co or datasets-server URL. */
const auth = (url) => TOKEN && /(^https:\/\/)([\w.-]*\.)?(huggingface\.co|hf\.co)\//u.test(url)
  ? ["--config", CURLRC] : [];

/** The scripts the phonemizer declares for this language — a transcript in the wrong script would
 *  become confident nonsense as the reference text. Same check the Common Voice tool makes. */
function declaredScripts(lang) {
  const src = readFileSync(join(PHONEMIZER, "src/core/scripts.ts"), "utf8");
  const tbl = src.slice(src.indexOf("MANIFESTLESS_SCRIPTS"));
  for (const m of tbl.slice(0, tbl.indexOf("\n};")).matchAll(/"?([\w-]+)"?:\s*\[([^\]]*)\]/g))
    if (m[1] === lang) return [...m[2].matchAll(/"([\w ]+)"/g)].map((x) => x[1].replace(/ /gu, "_"));
  for (const d of readdirSync(join(PHONEMIZER, "data/languages"))) {
    for (const f of readdirSync(join(PHONEMIZER, "data/languages", d)).filter((x) => x.endsWith(".jsonc"))) {
      const t = readFileSync(join(PHONEMIZER, "data/languages", d, f), "utf8");
      if (t.match(/"language":\s*"([\w-]+)"/)?.[1] !== lang) continue;
      const sc = t.match(/"script":\s*\[([^\]]*)\]/)?.[1];
      if (sc) return [...sc.matchAll(/"([\w ]+)"/g)].map((x) => x[1].replace(/ /gu, "_"));
    }
  }
  return [];
}
const SCRIPTS = declaredScripts(LANG).map((n) =>
  n === "Kana" ? /\p{Script=Hiragana}|\p{Script=Katakana}/u : new RegExp(`\\p{Script=${n}}`, "u"));
if (!SCRIPTS.length) console.warn(`  ⚠ ${LANG} declares no script — transcripts not script-checked`);
const inScript = (s) => !SCRIPTS.length || SCRIPTS.some((re) => re.test(s));

console.log(`${DATASET} [${CONFIG}/${SPLIT}] -> demo language ${LANG}`);
const candidates = [];
const skippedForeign = [];
for (let offset = 0; offset < SCAN; offset += 100) {
  const url = `${ROWS}?dataset=${encodeURIComponent(DATASET)}&config=${encodeURIComponent(CONFIG)}`
            + `&split=${encodeURIComponent(SPLIT)}&offset=${offset}&length=100`;
  let page;
  try { page = JSON.parse(execFileSync("curl", ["-sfL", "--max-time", "120", ...auth(url), url],
                                        { encoding: "utf8", maxBuffer: 1 << 28 })); }
  catch { break; }
  const rows = page.rows ?? [];
  if (!rows.length) break;
  for (const { row } of rows) {
    const text = String(row[TEXT_FIELD ?? guessText(row)] ?? "").trim();
    const audio = row[AUDIO_FIELD];
    const src = Array.isArray(audio) ? audio[0]?.src : audio?.src;
    if (!text || !src) continue;
    if (FILTER_RE && !FILTER_RE.test(String(row[FILTER_FIELD] ?? ""))) continue;
    if (!inScript(text)) continue;
    const foreign = outsideAlphabet(text);
    if (foreign.length) { skippedForeign.push([text.slice(0, 60), foreign.join("")]); continue; }
    const dur = Number(row[DUR_FIELD ?? "duration"] ?? row.duration_seconds ?? 0);
    // With --trim a long clip is a candidate: it may still yield a short prefix that ends on a pause.
    if (dur && (dur < MIN_SEC || (dur > MAX_SEC && !TRIM))) continue;
    candidates.push({ src, text, dur, id: `${row.segment_id ?? row.id ?? offset}-${candidates.length}` });
  }
  if (candidates.length >= 40) break;
}
function guessText(row) {
  for (const k of ["raw_text", "text", "sentence", "transcription", "transcript", "normalized_text"])
    if (typeof row[k] === "string") return k;
  return "text";
}
if (skippedForeign.length)
  console.log(`  ${skippedForeign.length} rejected for letters outside --letters `
    + `(e.g. "${skippedForeign[0][0]}" -> ${skippedForeign[0][1]})`);
console.log(`  ${candidates.length} candidates in ${MIN_SEC}-${MAX_SEC}s`);
if (!candidates.length) { console.error("  nothing matched — widen --min-sec/--max-sec or raise --scan"); process.exit(1); }


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

/** Noise floor, speech fraction and peak from 10 ms frames — the screen that actually rejects. */
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
  const floor = sorted[Math.floor(m * 0.1)], top = sorted[m - 1];
  const speech = db.filter((v) => v > top - 25);
  return { snr: speech.reduce((a, b2) => a + b2, 0) / (speech.length || 1) - floor,
           speechFrac: speech.length / m, peak, sec: n / 24000, clip: clipping(x) };
}

const scored = [];
for (const [i, c] of candidates.entries()) {
  const wav = join(WORK, "wav", `c${i}.wav`);
  try {
    if (!existsSync(wav)) {
      const raw = join(WORK, `c${i}.src`);
      execFileSync("curl", ["-sfL", "--max-time", "180", ...auth(c.src), c.src, "-o", raw]);
      execFileSync("ffmpeg", ["-v", "error", "-y", "-i", raw, "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", wav]);
    }
  } catch { continue; }
  let wavUse = wav, textUse = c.text;
  if (TRIM) {
    const t = trimToSentences(wav, c.text, { minSec: MIN_SEC, maxSec: MAX_SEC });
    if (!t) continue;                       // sentence count and pause count disagree — unprovable
    wavUse = t.wav; textUse = t.text;
  }
  const s = score(wavUse);
  // ⚠ ENFORCE THE LENGTH BOUND ON THE DECODED AUDIO, not only on a row's duration field. Corpora
  // that ship no duration (WaxalNLP) skipped the metadata filter entirely, and a 48 s Ewe clip went
  // through: `make-voices.mjs` then removed 40 s of silence from it, leaving 7.5 s of audio paired
  // with a transcript for the whole 48 s. A reference whose text does not match its audio is worse
  // than no reference.
  if (s.sec < MIN_SEC || s.sec > MAX_SEC) continue;
  if (s.clip.run >= 3 || s.clip.frac > 0.0005 || s.speechFrac < 0.45) continue;
  scored.push({ ...c, wav: wavUse, text: textUse, ...s });
  if (scored.length >= 12) break;
}
scored.sort((a, b) => (b.snr + 40 * b.speechFrac) - (a.snr + 40 * a.speechFrac));
console.log(`  ${scored.length} scored; best:`);
for (const s of scored.slice(0, N))
  console.log(`    ${s.id}  ${s.sec.toFixed(1)}s  snr ${s.snr.toFixed(0)} dB  speech ${(s.speechFrac * 100).toFixed(0)}%  peak ${s.peak.toFixed(2)}`);

const entries = [];
for (const [i, c] of scored.slice(0, N).entries()) {
  const ipaPath = join(WORK, `${i}.ipa`);
  // ⚠ PER-PROCESS FILENAME. Two sourcing runs in parallel share this directory, and a fixed
  // name let one delete the temp module while the other was importing it — the Maithili
  // encode died on "cannot find module" while an Awadhi run finished normally.
  const probe = join(PHONEMIZER, `hf-phonemize.${process.pid}.tmp.mts`);
  writeFileSync(probe, `import { phonemizeAsync } from "./src/index.ts";\n`
    + `console.log(await phonemizeAsync(${JSON.stringify(c.text)}, ${JSON.stringify(LANG)}));\n`);
  let ipa;
  try { ipa = execFileSync("npx", ["tsx", `hf-phonemize.${process.pid}.tmp.mts`], { cwd: PHONEMIZER, encoding: "utf8" }).trim(); }
  finally { execFileSync("rm", ["-f", probe]); }
  writeFileSync(ipaPath, ipa);

  const id = `${LANG}-${DATASET.split("/")[1].slice(0, 6).toLowerCase()}${i}`;
  const out = execFileSync("node", ["tools/make-voices.mjs", ENCODER, c.wav, ipaPath, id, id], { encoding: "utf8" });
  const v = JSON.parse(out)[0];
  entries.push({
    voice: {
      id, lang: LANG, ...(i === 0 ? { default: true } : {}),
      label: `${LANG} · ${c.sec.toFixed(1)}s`,
      refIpa: v.refIpa, refLen: v.refLen, refRms: Number(v.refRms.toFixed(5)),
      source: { dataset: DATASET, lang: CONFIG, file: c.id, split: SPLIT, sentenceId: null,
                gender: "", durationS: Number(c.sec.toFixed(1)), candidateIndex: i, text: c.text },
    },
    codes: v.codes,
  });
}
writeFileSync(join(WORK, "voices.json"), JSON.stringify(entries));
console.log(`\n${entries.length} voices -> ${join(WORK, "voices.json")}`);
console.log("merge with: node tools/merge-cv-voices.mjs " + WORK);
