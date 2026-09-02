#!/usr/bin/env node
/**
 * Verify the type-ahead can actually find every language — by English name, by code, and by the
 * language's OWN name for itself.
 *
 * ⚠ THIS EXISTS BECAUSE THE ENDONYM SEARCH IS EASY TO SHIP BROKEN AND HARD TO SPOT. JS `\b` is
 * ASCII-only, so a word-boundary matcher silently fails on every non-Latin script while a Latin-script
 * spot check ("Boarisch", "Kreyòl") looks perfect. A native speaker typing 中文 or ไทย would get "no
 * language matches" and there is no other test that would say so.
 *
 *   node --experimental-strip-types tools/check-language-search.mjs
 */
import { LANGUAGES } from "../src/inference/languages.ts";
import { search } from "../src/langSearch.ts";

let fail = 0;
const bad = (msg) => { console.error(`  ✗ ${msg}`); fail++; };

// 1. Every language is findable by its own name, its code, and its endonym — and is RANKED FIRST.
for (const l of LANGUAGES) {
  for (const [what, q] of [["name", l.name], ["code", l.code], ...(l.native ? [["native", l.native]] : [])]) {
    const hits = search(LANGUAGES, q);
    if (!hits.some((h) => h.code === l.code)) bad(`${l.code}: not found by ${what} "${q}"`);
    else if (hits[0].code !== l.code)
      bad(`${l.code}: found by ${what} "${q}" but ranked behind ${hits[0].code}`);
  }
}

// 2. A PREFIX of each endonym finds it too — that is what a person actually types.
for (const l of LANGUAGES) {
  if (!l.native) continue;
  const chars = [...l.native];
  if (chars.length < 2) continue;
  const q = chars.slice(0, Math.min(3, chars.length)).join("");
  if (!search(LANGUAGES, q).some((h) => h.code === l.code))
    bad(`${l.code}: endonym prefix "${q}" does not find ${l.native}`);
}

// 3. No two languages share an endonym — that would make the picker ambiguous rather than helpful.
const seen = new Map();
for (const l of LANGUAGES) {
  if (!l.native) continue;
  const k = l.native.toLowerCase();
  if (seen.has(k)) bad(`${l.code} and ${seen.get(k)} share the endonym "${l.native}"`);
  seen.set(k, l.code);
}

const withNative = LANGUAGES.filter((l) => l.native).length;
console.log(`${LANGUAGES.length} languages searchable by name and code; ${withNative} also by endonym`);
if (fail) { console.error(`\n${fail} failure(s)`); process.exit(1); }
console.log("all checks pass");
