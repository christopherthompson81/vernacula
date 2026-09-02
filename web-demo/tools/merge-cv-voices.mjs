#!/usr/bin/env node
/**
 * Merge the voices produced by `make-voice-from-commonvoice.mjs` into the shipped files.
 *
 * Sourcing 26 languages one at a time and hand-pasting each block is how ids and codes drift apart,
 * so the encode step persists `<work>/voices.json` (entries + codes) and this reads them:
 *
 *   node tools/merge-cv-voices.mjs /tmp/cv-voices/*      [--dry]
 *
 * It writes voices.jsonc + voice-codes.json, and drops the donor entry in language-meta.json for
 * every language that now has a native voice. Re-running is safe: an id already present is replaced,
 * not duplicated.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const dirs = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const DRY = process.argv.includes("--dry");
const VP = "public/models/voices.jsonc", CP = "public/models/voice-codes.json";
const MP = "tools/data/language-meta.json";

const all = [];
for (const d of dirs) {
  const p = join(d, "voices.json");
  if (!existsSync(p)) continue;
  const entries = JSON.parse(readFileSync(p, "utf8"));
  if (entries.length) all.push(...entries);
}
if (!all.length) { console.error("no voices.json found in the given directories"); process.exit(1); }

const byLang = {};
for (const e of all) (byLang[e.voice.lang] ??= []).push(e);

let text = readFileSync(VP, "utf8");
const codes = JSON.parse(readFileSync(CP, "utf8"));
const meta = JSON.parse(readFileSync(MP, "utf8"));

const json = (o) => JSON.stringify(o, null, 0);
/**
 * One line, for the human-readable comment above each entry.
 *
 * ⚠ A transcript may contain NEWLINES — Omnilingual's spontaneous-speech rows separate sentences
 * that way — and a `// "…"` comment carrying one leaves its tail on the next line as bare text,
 * which makes voices.jsonc unparseable. This broke the file on the first Arabic merge.
 */
const oneLine = (t) => String(t).replace(/\s+/gu, " ").trim();
let added = 0, replaced = 0;
for (const [lang, entries] of Object.entries(byLang).sort()) {
  // Replace any existing block for this language rather than appending a second one.
  const existing = new Set([...text.matchAll(new RegExp(`\\{"id":"(${lang}-[\\w-]+)"`, "g"))].map((m) => m[1]));
  for (const id of existing) {
    const line = new RegExp(`^.*"id":"${id}".*\\n`, "m");
    const before = text;
    text = text.replace(line, "");
    if (before !== text) replaced++;
  }
  const cv = entries[0].voice.source.lang;
  const block = `  // ${lang} — Common Voice ${cv} (CC0). Sourced because FLEURS has no ${lang} speaker.\n`
    + entries.map((e) => `  // "${oneLine(e.voice.source.text)}"\n  ${json(e.voice)},\n`).join("");
  text = text.replace("  // af — af_za", block + "  // af — af_za");
  for (const e of entries) { codes[e.voice.id] = e.codes; added++; }
  if (meta[lang]) delete meta[lang].voice;
}

if (DRY) {
  console.log(`${added} voices across ${Object.keys(byLang).length} languages (dry run, nothing written)`);
  for (const [l, e] of Object.entries(byLang)) console.log(`  ${l}: ${e.map((x) => x.voice.id).join(" ")}`);
  process.exit(0);
}
writeFileSync(VP, text);
writeFileSync(CP, JSON.stringify(codes));
writeFileSync(MP, JSON.stringify(meta, null, 1) + "\n");
console.log(`${added} voices across ${Object.keys(byLang).length} languages merged`
          + (replaced ? ` (${replaced} existing entries replaced)` : ""));
console.log("next: node tools/make-language-catalog.mjs, rebuild, and LISTEN to each one");
