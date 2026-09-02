#!/usr/bin/env node
/**
 * Replay every offered language against ONLY the staged files, and fail if any key is missing.
 *
 * ⚠ THIS EXISTS BECAUSE "RECORDED" IS NOT THE SAME AS "COMPLETE". The prefetch lists are recorded
 * by running the engine, which is far better than a hand-kept list — but a recording only captures
 * what the probe text reached, and several tables load lazily on first USE. Japanese's pitch-accent
 * table sits behind a `??=` that a Latin probe never touches, so it was recorded as absent and the
 * demo failed at run time with "phonemizer data not prefetched".
 *
 * A staged set is only trustworthy if the exact text the demo ships can be phonemized from it and
 * nothing else. That is what this checks, and it is the gate that turns the upstream warning into
 * an enforced property.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";

const REPO = "../external/vernacula-phonemizer";
const DATA = "public/vphon-data";

const manifest = JSON.parse(fs.readFileSync(`${DATA}/_keys.json`, "utf8"));
// The catalogue is generated, so the gate reads the SHIPPED text rather than a copy of it.
const src = fs.readFileSync("src/inference/languages.ts", "utf8");
const samples = {};
for (const m of src.matchAll(/code:\s*"([\w-]+)"[^}]*?sample:\s*"((?:[^"\\]|\\.)*)"/g))
  samples[m[1]] = JSON.parse(`"${m[2]}"`);

// The check runs inside the phonemizer repo so it can import the engine through its browser seams,
// exactly as the demo does — a frozen Map as the data source, nothing else reachable.
const script = `
import fs from "node:fs";
import path from "node:path";
import { setDataSource, loadEngine } from "./src/browser.ts";
const DATA = ${JSON.stringify(process.cwd() + "/" + DATA)};
const manifest = JSON.parse(fs.readFileSync(path.join(DATA, "_keys.json"), "utf8"));
const samples = ${JSON.stringify(samples)};
const bytes = new Map();
const load = (keys) => { for (const k of keys) { const p = path.join(DATA, k);
  if (fs.existsSync(p)) bytes.set(k, new Uint8Array(fs.readFileSync(p))); } };
const missing = [];
// Script name -> detector, built from the routing table's own keys. "Kana" is the engine's name for
// the two Japanese syllabaries and is not a Unicode script property, so it is spelled out.
const SCRIPT_RE = Object.fromEntries(Object.keys(manifest.foreign.defaults).map((n) => [n,
  n === "Kana" ? /\\p{Script=Hiragana}|\\p{Script=Katakana}/u
               : new RegExp("\\\\p{Script=" + n + "}", "u")]));
load(manifest.engine);
setDataSource({ read(k) { const b = bytes.get(k); if (!b) { missing.push(k); throw new Error("missing " + k); } return b; } });
const { phonemizeAsync } = await loadEngine();
const results = [];
const keysFor = (code) => {
  const e = manifest.languages[code];
  if (!e) return [];
  const skip = new Set(e.exclude ?? []);
  return [...e.dirs.flatMap((d) => manifest.dirs[d] ?? []), ...e.core].filter((k) => !skip.has(k));
};
for (const [code, text] of Object.entries(samples)) {
  load(keysFor(code));
  // The demo resolves embedded foreign scripts to their reader and fetches that language too;
  // mirror it here or a sample with a Latin quotation inside fails the gate for the wrong reason.
  for (const [script, target] of Object.entries(manifest.foreign.defaults)) {
    const re = SCRIPT_RE[script];
    if (!re || !re.test(text)) continue;
    const t = manifest.foreign.overrides[code]?.[script] ?? target;
    if (t !== code) load(keysFor(t));
  }
  const before = missing.length;
  try { const ipa = await phonemizeAsync(text, code);
        // ⚠ A key the ENGINE asks for that does not exist upstream either is an OPTIONAL probe, not
        // a staging gap: some loaders try a lexicon that a language simply does not ship (Saraiki's
        // is optional:true) and handle the throw. Only a key that exists in data/ and was not
        // staged is a real failure — that is the class that produced silently wrong readings.
        const real = missing.slice(before).filter((k) => fs.existsSync(path.join("data", k)));
        results.push({ code, ok: real.length === 0, ipa: ipa.slice(0, 40), missing: real }); }
  catch (e) { results.push({ code, ok: false, err: String(e).slice(0, 90), missing: missing.slice(before) }); }
}
console.log("@@" + JSON.stringify(results));
`;
fs.writeFileSync(`${REPO}/verify.tmp.mts`, script);
let out = "";
try {
  out = execFileSync("npx", ["tsx", "verify.tmp.mts"], { cwd: REPO, encoding: "utf8", maxBuffer: 1 << 28 });
} finally { fs.rmSync(`${REPO}/verify.tmp.mts`, { force: true }); }

const results = JSON.parse(out.split("@@")[1]);
let bad = 0;
for (const r of results) {
  if (r.ok) { console.log(`  ok    ${r.code.padEnd(5)} ${r.ipa}…`); continue; }
  bad++;
  const uniq = [...new Set(r.missing)];
  console.error(`  FAIL  ${r.code.padEnd(5)} ${r.err ?? ""}`);
  for (const k of uniq) console.error(`          missing: ${k}`);
}
console.log(`\n${results.length - bad}/${results.length} languages phonemize from the staged data alone`);
process.exit(bad ? 1 : 0);
