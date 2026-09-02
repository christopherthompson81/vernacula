#!/usr/bin/env node
/**
 * Fill in `sample` for every language in tools/data/language-meta.json, from REAL text.
 *
 * ⚠ THE SENTENCES ARE SOURCED, NOT WRITTEN. A demo that ships 193 prefilled sentences cannot have
 * them invented by whoever wired the picker: a subtly wrong sentence in a language the author does
 * not read is indistinguishable, to that author, from a right one — and it is the first thing a
 * native speaker sees. Two sources, both already on disk and both real published text:
 *
 *   1. FLEURS transcripts (google/fleurs, CC BY 4.0) for the 102 corpus languages, taken from a
 *      DIFFERENT clip than the one the language's reference voice was cut from — otherwise the demo
 *      would ask the model to say the same sentence it was just given as the reference.
 *   2. The phonemizer's mined normalization corpora (tools/corpus/mined/<code>.jsonc), which are
 *      Wikipedia-dump paragraphs, for the 61 further languages that have one.
 *
 * ⚠ SIX LANGUAGES ARE NOT MINEABLE and their sentences are entered by hand in language-meta.json:
 * K'iche', Kalaallisut and Totontepec Mixe from the UDHR (public domain — the first two via the
 * NLTK udhr2 corpus, Mixe's Article 1 as reproduced by Omniglot), and Lule Sami, Nogai and Nama
 * from Wikimedia Incubator (CC-BY-SA), which is the only running text those three have. None has a
 * Wikipedia dump, which is exactly why the phonemizer's mined corpora do not cover them either.
 *
 * Hand-written entries already in language-meta.json are KEPT (--force overrides): the 30 curated
 * sentences the demo shipped with are short, parallel across languages and better demo material
 * than a random encyclopaedia line.
 *
 *   node tools/make-samples.mjs [--fleurs <dir>] [--force]
 */
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const FLEURS = args.includes("--fleurs") ? args[args.indexOf("--fleurs") + 1]
                                         : "/mnt/data/omnivoice_ipa/corpus/fleurs_transcripts/data";
const FORCE = args.includes("--force");
const MINED = "../external/vernacula-phonemizer/tools/corpus/mined";
const META = "tools/data/language-meta.json";

const meta = JSON.parse(readFileSync(META, "utf8"));

/** JSONC -> value. String-aware, and tolerant of trailing commas (the mined artifacts have them). */
function parseJsonc(t) {
  let out = "", inStr = false, esc = false, line = false, block = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i], n = t[i + 1];
    if (line) { if (c === "\n") { line = false; out += c; } continue; }
    if (block) { if (c === "*" && n === "/") { block = false; i++; } continue; }
    if (inStr) { out += c; if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') { inStr = true; out += c; continue; }
    if (c === "/" && n === "/") { line = true; i++; continue; }
    if (c === "/" && n === "*") { block = true; i++; continue; }
    out += c;
  }
  return JSON.parse(out.replace(/,(\s*[}\]])/g, "$1"));
}

// Wiki markup leftovers, references, digits (they expand to spoken numbers and blow up the length),
// and any bracketing that would read aloud as nothing.
const BAD = /[[\]{}<>|&#*=_/\\@~^()"“”„«»–—《》「」『』]|&\w+;|\d|https?:/u;

/**
 * ⚠ ONE SCRIPT PER SENTENCE, and this is not tidiness. Wikipedia dumps are full of embedded English
 * — the first Tibetan candidate opened "UEFA Champions League", the Tigrinya one began with a whole
 * English clause, and the Zhuang one carried Chinese book-title brackets. Prefilling the demo with
 * those would make the model read English in a Tibetan voice on first click. Letters must all come
 * from a single script (punctuation and spaces are script-neutral and do not count).
 */
const SCRIPTS = ["Latin", "Cyrillic", "Arabic", "Greek", "Hebrew", "Devanagari", "Bengali", "Tamil",
  "Telugu", "Kannada", "Malayalam", "Gujarati", "Gurmukhi", "Oriya", "Sinhala", "Khmer", "Lao",
  "Thai", "Tibetan", "Myanmar", "Ethiopic", "Armenian", "Georgian", "Hangul", "Han", "Hiragana",
  "Katakana", "Tifinagh", "Cherokee", "Ol_Chiki", "Adlam", "Nko", "Syloti_Nagri", "Javanese",
  "Sundanese"].map((n) => [n, new RegExp(`\\p{Script=${n}}`, "u")]);

function singleScript(s) {
  const hit = SCRIPTS.filter(([, re]) => re.test(s)).map(([n]) => n);
  // Japanese legitimately mixes Han with the kana; nothing else here does.
  const jp = hit.every((n) => n === "Han" || n === "Hiragana" || n === "Katakana");
  return hit.length === 1 || (jp && hit.length > 1);
}
const END = /[.!?。।።།၊။]$/u;
const CJK = /[぀-ヿ㐀-鿿가-힯]/u;
const SPLIT = /(?<=[.!?。।።།၊။])\s+/u;

/** Closest to `want` characters, in the language's own script. */
function pick(sentences, want = 58, want_scripts = [], bounds) {
  const ok = sentences.filter((s) => {
    if (!s || BAD.test(s) || !END.test(s) || !singleScript(s)) return false;
    if (want_scripts.length && !want_scripts.some(([, re]) => re.test(s))) return false;
    const n = [...s].length, cjk = CJK.test(s);
    const [lo, hi] = bounds ?? (cjk ? [12, 30] : [38, 85]);
    return n >= lo && n <= hi && (s.match(/,/gu) ?? []).length <= (bounds ? 2 : 1);
  });
  ok.sort((a, b) => Math.abs([...a].length - want) - Math.abs([...b].length - want));
  return ok[0];
}

/** demo code -> FLEURS directory + the clip the reference voice came from, read from voices.jsonc. */
function fleursMap() {
  const t = readFileSync("public/models/voices.jsonc", "utf8");
  const out = {};
  for (const m of t.matchAll(/"lang":"([a-z-]+)"[^\n]*?"lang":"([a-z_]+)","file":"([^"]+)"/g))
    out[m[1]] ??= { dir: m[2], voiceFile: m[3] };
  return out;
}

/**
 * code -> the scripts its engine declares, from the phonemizer's own manifests (plus the
 * MANIFESTLESS_SCRIPTS table for varieties and aliases).
 *
 * ⚠ Without this the filter accepts an ENGLISH sentence as a Tibetan sample. The mined corpora are
 * Wikipedia dumps and carry English boilerplate ("UEFA Champions League Season Squad"), which is
 * single-script Latin and passes every other check. Tibetan, Tigrinya, Uyghur, Sinhala and Santali
 * all came back with an English sentence before the script of the LANGUAGE was checked.
 */
function scriptsByCode() {
  const dir = "../external/vernacula-phonemizer";
  const out = {};
  const src = readFileSync(join(dir, "src/core/scripts.ts"), "utf8");
  const tbl = src.slice(src.indexOf("MANIFESTLESS_SCRIPTS"));
  for (const m of tbl.slice(0, tbl.indexOf("\n};")).matchAll(/"?([\w-]+)"?:\s*\[([^\]]*)\]/g))
    out[m[1]] = [...m[2].matchAll(/"(\w+)"/g)].map((x) => x[1]);
  for (const d of readdirSync(join(dir, "data/languages"))) {
    for (const f of readdirSync(join(dir, "data/languages", d)).filter((f) => f.endsWith(".jsonc"))) {
      const t = readFileSync(join(dir, "data/languages", d, f), "utf8");
      const code = t.match(/"language":\s*"([\w-]+)"/)?.[1];
      const scripts = t.match(/"script":\s*\[([^\]]*)\]/)?.[1];
      // ⚠ `[\w ]+`, not `\w+`: two manifests name a two-word script ("Ol Chiki", "Syloti Nagri"),
      // and \w+ matched neither — so Santali and Sylheti came back with NO declared script, skipped
      // the script check, and were prefilled with English sentences from their Wikipedia dumps.
      if (code && scripts) out[code] ??= [...scripts.matchAll(/"([\w ]+)"/g)].map((x) => x[1]);
    }
  }
  return out;
}
const SCRIPT_OF = scriptsByCode();

let filled = 0, kept = 0;
const fl = existsSync(FLEURS) ? fleursMap() : {};
if (!existsSync(FLEURS)) console.warn(`  ⚠ ${FLEURS} not found — FLEURS languages keep their current sample`);

for (const [code, entry] of Object.entries(meta)) {
  if (entry.sample && !FORCE) { kept++; continue; }
  // ⚠ With --force, DROP the old value first. Keeping it when the re-run finds nothing silently
  // preserves exactly the samples a tightened filter has just rejected — which is how five English
  // sentences survived the script check that was written to remove them.
  if (FORCE) delete entry.sample;
  let sample;
  const declared = SCRIPT_OF[code] ?? [];
  // Manifests spell two of them with a space ("Ol Chiki", "Syloti Nagri"); Unicode uses underscores.
  const norm = declared.map((n) => n.replace(/ /gu, "_"));
  const want = SCRIPTS.filter(([n]) => norm.includes(n) || (norm.includes("Kana") && (n === "Hiragana" || n === "Katakana")));
  if (!want.length) console.warn(`  ⚠ ${code}: no declared script, sample not script-checked`);
  let lastPool = [];
  const f = fl[code];
  if (f && existsSync(join(FLEURS, f.dir))) {
    const rows = [];
    for (const split of ["dev.tsv", "test.tsv", "train.tsv"]) {
      const p = join(FLEURS, f.dir, split);
      if (!existsSync(p)) continue;
      for (const line of readFileSync(p, "utf8").split("\n")) {
        const c = line.split("\t");
        if (c.length > 2 && c[1] !== f.voiceFile) rows.push(c[2].trim());
      }
    }
    lastPool = rows.flatMap((r) => r.split(SPLIT));
    sample = pick(lastPool, 58, want);
  }
  if (!sample && existsSync(join(MINED, `${code}.jsonc`))) {
    const d = parseJsonc(readFileSync(join(MINED, `${code}.jsonc`), "utf8"));
    lastPool = (d.sample ?? []).flatMap((s) => s.split(SPLIT)).map((s) => s.trim())
                                  // ⚠ "starts with a capital" must be expressed as "does not start with a LOWERCASE
                 // letter". `s[0] !== s[0].toLowerCase()` is false for every caseless script, so it
                 // silently rejected every candidate in Ethiopic, Tibetan, Sinhala, Myanmar, Arabic,
                 // Ol Chiki and Syloti Nagri — 14 languages came back with no sentence at all.
                 .filter((s) => s && s[0] === s[0].toUpperCase());
    sample = pick(lastPool, 58, want);
  }
  // A second pass at wider bounds for the languages whose corpus is thin — better a long or short
  // sentence than an empty box.
  if (!sample && lastPool.length) sample = pick(lastPool, 58, want, [22, 130]);
  if (sample) { entry.sample = sample; filled++; }
}

/**
 * Varieties and accent variants inherit their parent's sentence.
 *
 * ⚠ BY SCRIPT, NOT BY VOICE DONOR. Western Punjabi's donor VOICE is Punjabi, but `pa` is Gurmukhi
 * and `pnb` is Shahmukhi — inheriting there would prefill the box with text in the wrong script.
 * These pairs are the ones that read the same written language.
 */
const SAMPLE_PARENT = {
  "en-GB": "en", "en-IN": "en", "es-419": "es", "fr-CA": "fr", "pt-BR": "pt", zsm: "ms", pbt: "ps",
  pnb: "ur", skr: "ur", arz: "ar", apc: "ar", ajp: "ar", apd: "ar", acm: "ar", afb: "ar", acw: "ar",
  ary: "ar", ayl: "ar", hyw: "hy", bgc: "hi", bho: "hi", hne: "hi", mag: "hi", mai: "hi", awa: "hi",
  rkt: "bn", bpy: "bn", grc: "el",
  cjy: "cmn", gan: "cmn", hsn: "cmn", wuu: "cmn", hak: "yue",
};
let inherited = 0;
for (const [code, parent] of Object.entries(SAMPLE_PARENT)) {
  if (meta[code] && !meta[code].sample && meta[parent]?.sample) { meta[code].sample = meta[parent].sample; inherited++; }
}

writeFileSync(META, JSON.stringify(meta, null, 1) + "\n");
const none = Object.entries(meta).filter(([, e]) => !e.sample).map(([c]) => c);
console.log(`${filled} samples filled, ${inherited} inherited from a parent variety, ${kept} kept`);
if (none.length) console.log(`  no sourced sentence for: ${none.join(" ")}`);
