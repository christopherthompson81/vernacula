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
  // ⚠ `[\w-]+` DOES NOT MATCH A DOT, and every Common Voice voice id ends in `.mp3`. The id was
  // captured up to the dot, the removal regex below then matched nothing, and a re-source APPENDED a
  // second copy instead of replacing the first — 288 voices became 309, with all 21 client_id
  // entries still in the file after a run that reported success. Match to the closing quote.
  // ⚠ A LANGUAGE'S FIRST VOICE IS NAMED FOR THE LANGUAGE ITSELF. make-voices-from-corpus.mjs writes
  // `ar`, `en`, `cmn` for the first FLEURS voice and `ar-1`, `ar-2` for alternates, so a pattern of
  // `<lang>-...` misses exactly the one that is usually the DEFAULT. Re-sourcing Arabic left the old
  // muffled `ar` in place beside its replacement, with two entries claiming `"default":true`.
  const existing = new Set([...text.matchAll(new RegExp(`\\{"id":"(${lang}|${lang}-[^"]+)"`, "g"))].map((m) => m[1]));
  for (const id of existing) {
    // ⚠ REMOVE THE COMMENT LINE WITH THE ENTRY. Each voice is written as a `// "<transcript>"` line
    // followed by its JSON line, and dropping only the JSON left the old transcript behind as an
    // orphan comment. That is normally cosmetic drift; it was not when the 21 client_id voices were
    // repaired, because the SPEAKER IDENTIFIER being removed lives in that comment too.
    // ⚠ And the id is interpolated INTO a regex, so its dots must be escaped or `.mp3` matches
    // any character — which would delete a different voice on a near-miss id.
    const esc = id.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
    const line = new RegExp(`(?:^\\s*//[^\\n]*\\n)?^.*"id":"${esc}".*\\n`, "m");
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
// ⚠ Prune codes whose voice is no longer listed. Replacing a language's block leaves its old ids in
// voice-codes.json, where nothing reads them and nothing reports them — the file simply grows, and a
// stale id looks live to any tool that reads the codes file alone.
// ⚠ PARSE, DO NOT PATTERN-MATCH, BEFORE DELETING ANYTHING. This started as a regex over the file
// text, which cannot see an entry written with different spacing — a hand-added voice serialized as
// `{"id": "acw-s01-6"` was invisible to it, so the prune classified a LIVE voice's codes as orphaned
// and deleted them. A prune step must know exactly what is live; the strip-comments parser already
// used at the top of this file does.
const live = new Set(JSON.parse(text.replace(/^\s*\/\/.*$/gmu, "")).map((v) => v.id));
let pruned = 0;
for (const id of Object.keys(codes)) if (!live.has(id)) { delete codes[id]; pruned++; }
if (pruned) console.log(`  pruned ${pruned} orphaned code entr${pruned === 1 ? "y" : "ies"}`);

writeFileSync(VP, text);
writeFileSync(CP, JSON.stringify(codes));
writeFileSync(MP, JSON.stringify(meta, null, 1) + "\n");
console.log(`${added} voices across ${Object.keys(byLang).length} languages merged`
          + (replaced ? ` (${replaced} existing entries replaced)` : ""));
console.log("next: node tools/make-language-catalog.mjs, rebuild, and LISTEN to each one");
