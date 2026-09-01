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
import { mkdirSync, copyFileSync, existsSync, rmSync, writeFileSync, statSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

/**
 * ⚠ RECORD WITH REPRESENTATIVE TEXT, NOT THE DEFAULT PROBE. Some tables load lazily on first USE
 * rather than at construction — Japanese's pitch-accent and kanji readings sit behind a `??=` that
 * a Latin-script probe never reaches. Staging with the default probe produced a Japanese list that
 * was missing `languages/japanese/pitch-accent.tsv`, and the demo failed at run time with
 * "phonemizer data not prefetched". Upstream's tool says this in its header; ignoring it cost a
 * user-visible bug.
 *
 * The sample sentences the demo ships are exactly the representative text, so they are read from
 * config.ts rather than duplicated here.
 */
function samplesFromConfig() {
    const src = readFileSync("src/inference/config.ts", "utf8");
    const out = {};
    for (const m of src.matchAll(/code:\s*"([\w-]+)"[^}]*?sample:\s*"((?:[^"\\]|\\.)*)"/g))
        out[m[1]] = JSON.parse(`"${m[2]}"`);
    return out;
}

const REPO = "../external/vernacula-phonemizer";
const OUT = "public/vphon-data";
const langs = process.argv.slice(2);
const SAMPLES = samplesFromConfig();
if (!langs.length) { console.error("usage: stage-phonemizer-data.mjs <lang>..."); process.exit(2); }

rmSync(OUT, { recursive: true, force: true });

// ⚠ TWO PHASES, KEPT SEPARATE ON PURPOSE. `engine` is what importing the engine reads — every
// language's manifest, 182 files / 4.5 MB — and is needed whatever language you pick. Everything
// else is per-language and fetched only when that language is chosen, which is what keeps first
// load at ~4.5 MB instead of the ~80 MB all 28 would cost together.
const engine = new Set();
const byLang = {};
for (const code of langs) {
    // One child process per language — upstream's tool does the same, because the table loaders
    // memoize and a second language in the same process records no read for a file the first
    // already pulled, attributing shared data to whichever ran first.
    const text = SAMPLES[code];
    const argv = text ? ["tsx", "tools/browser-prefetch.mts", "--text", text, code]
                      : ["tsx", "tools/browser-prefetch.mts", code];
    if (!text) console.warn(`  ⚠ ${code}: no sample text in config.ts — recording with the default probe,`
                          + ` which may miss tables that load lazily in this language's script`);
    const out = execFileSync("npx", argv, { cwd: REPO, encoding: "utf8", maxBuffer: 1 << 28 });
    const d = JSON.parse(out);
    for (const k of d.engine.keys) engine.add(k);
    const own = new Set();
    for (const v of Object.values(d.languages)) for (const k of v.keys) own.add(k);
    byLang[code] = [...own].sort();
}

// ⚠ Count each file ONCE. Summing per-language totals double-counts everything shared — and
// almost every language shares English's tables, because phonemizeAsync prewarms the English
// tagger. That inflated the on-disk figure to 202 MB when the real answer is a third of that.
const copied = new Set();
let bytes = 0;
const copy = (k) => {
    if (copied.has(k)) return;
    const src = join(REPO, "data", k), dst = join(OUT, k);
    if (!existsSync(src)) { console.warn("  missing:", k); return; }
    mkdirSync(dirname(dst), { recursive: true });
    copyFileSync(src, dst); bytes += statSync(src).size; copied.add(k);
};
for (const k of [...engine].sort()) copy(k);
for (const ks of Object.values(byLang)) for (const k of ks) copy(k);

const sizeOf = (ks) => ks.reduce((n, k) => n + (existsSync(join(REPO, "data", k)) ? statSync(join(REPO, "data", k)).size : 0), 0);
writeFileSync(join(OUT, "_keys.json"), JSON.stringify({ engine: [...engine].sort(), languages: byLang }));
console.log(`engine: ${engine.size} files, ${(sizeOf([...engine]) / 1e6).toFixed(1)} MB (always fetched)`);
for (const [c, ks] of Object.entries(byLang))
    console.log(`  ${c}: ${ks.length} files, ${(sizeOf(ks) / 1e6).toFixed(2)} MB`);
console.log(`total staged ${copied.size} UNIQUE files, ${(bytes / 1e6).toFixed(1)} MB -> ${OUT}`);
console.log("(per-language figures above overlap heavily — English's tables are shared by most)");
