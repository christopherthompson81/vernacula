#!/usr/bin/env node
/**
 * Stage only the data files the engine actually reads, into public/vphon-data/<key>.
 *
 * The full asset tree is 151 MB. What a browser needs is far less and is RECORDED rather than declared:
 * upstream's tools/browser-prefetch.mts runs the engine and reports the keys it read, in two phases —
 * `engine` (182 manifests, 4.5 MB, needed whatever language you pick) and `languages[code]` on top.
 * A hand-maintained list would rot silently, because a missing optional table is not an error to the
 * engine: it is an empty Map, and every word then takes the OOV path to a plausible wrong reading.
 *
 *   node tools/stage-phonemizer-data.mjs es cy en
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, copyFileSync, existsSync, rmSync, writeFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";

const REPO = "../external/vernacula-phonemizer";
const OUT = "public/vphon-data";
const langs = process.argv.slice(2);
if (!langs.length) { console.error("usage: stage-phonemizer-data.mjs <lang>..."); process.exit(2); }

rmSync(OUT, { recursive: true, force: true });
const keys = new Set();
for (const code of langs) {
    // ⚠ One child process per language — upstream's tool does the same, because the table loaders
    //   memoize and a second language in the same process records no read for a file the first already
    //   pulled, attributing shared data to whichever went first.
    const out = execFileSync("npx", ["tsx", "tools/browser-prefetch.mts", code],
                             { cwd: REPO, encoding: "utf8", maxBuffer: 1 << 28 });
    const d = JSON.parse(out);
    for (const k of d.engine.keys) keys.add(k);
    for (const v of Object.values(d.languages)) for (const k of v.keys) keys.add(k);
}
let bytes = 0;
for (const k of [...keys].sort()) {
    const src = join(REPO, "data", k), dst = join(OUT, k);
    if (!existsSync(src)) { console.warn("  missing:", k); continue; }
    mkdirSync(dirname(dst), { recursive: true });
    copyFileSync(src, dst); bytes += statSync(src).size;
}
writeFileSync(join(OUT, "_keys.json"), JSON.stringify([...keys].sort()));
console.log(`staged ${keys.size} files, ${(bytes / 1e6).toFixed(1)} MB for [${langs.join(" ")}] -> ${OUT}`);
