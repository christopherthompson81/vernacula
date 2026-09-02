#!/usr/bin/env node
/**
 * Stage the phonemizer's data tree into public/vphon-data/, and record WHICH DIRECTORIES each
 * language needs.
 *
 * ⚠ WHOLE DIRECTORIES, NOT RECORDED FILE LISTS — and that is a correctness decision, not laziness.
 * The previous version staged only the keys a probe run actually read. That is exact for the probe
 * and wrong for everything else: several tables load lazily on first USE (Japanese pitch-accent
 * behind a `??=`), so the recording missed them and the demo failed at run time. With 30 languages
 * a representative sentence per language could be written by hand; with 193 it cannot, and a
 * missing optional table is not an error to the engine — it is an empty Map, and every word then
 * takes the OOV path to a plausible wrong reading.
 *
 * So the recording is used for what it is genuinely good at — discovering the DIRECTORY set a
 * language reaches, including the structural parents no one would guess (Assamese reads Bengali's
 * dir, Bosnian reads Serbian's, Bashkir reads Russian's) — and every file in those directories is
 * then shipped. The whole tree is 142 MB in 347 files; a visitor still downloads only their own
 * language's directories.
 *
 *   node tools/stage-phonemizer-data.mjs            # every language in the registry
 *   node tools/stage-phonemizer-data.mjs es cy en   # a subset, for a fast dev build
 */
import { execFile } from "node:child_process";
import { registryTargets, neuralTargets, dirsFor } from "./lang-dirs.mjs";
import { promisify } from "node:util";
import { mkdirSync, copyFileSync, existsSync, rmSync, writeFileSync, statSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";

const exec = promisify(execFile);
const REPO = "../external/vernacula-phonemizer";
const OUT = "public/vphon-data";
const DATA = join(REPO, "data");

/** Every code the registry can build. */
function registryCodes() {
  const src = readFileSync(join(REPO, "src/registry.ts"), "utf8");
  const body = src.slice(src.indexOf("function build(lang: string)"));
  return [...new Set([...body.matchAll(/case "([\w-]+)":/g)].map((m) => m[1]))];
}

/**
 * The foreign-run routing table, lifted from the engine rather than retyped.
 *
 * ⚠ THIS IS THE ONE EDGE THAT IS NOT STATIC. A run of text in a script the host language does not
 * own is delegated to another language chosen from the SCRIPT (core/scripts.ts `readerFor`), so any
 * language can reach any of these ~34 readers depending on what the visitor types. Worse, the
 * delegation is wrapped in try/catch: a missing data key is swallowed and the run silently degrades
 * to the Latin path. That is invisible, so the client resolves these targets from the input text
 * before phonemizing (see phonemizer.ts) and this table is what it uses.
 */
function foreignRouting() {
  const src = readFileSync(join(REPO, "src/core/scripts.ts"), "utf8");
  const grab = (name) => {
    const i = src.indexOf(`export const ${name}`);
    const s = src.indexOf("{", i), e = src.indexOf("\n};", s);
    return src.slice(s, e);
  };
  const defaults = Object.fromEntries(
    [...grab("DEFAULT_READER").matchAll(/(\w+):\s*"([\w-]+)"/g)].map((m) => [m[1], m[2]]));
  const overrides = {};
  for (const m of grab("OVERRIDES").matchAll(/(\w+):\s*\{([^}]*)\}/g))
    overrides[m[1]] = Object.fromEntries([...m[2].matchAll(/(\w+):\s*"([\w-]+)"/g)].map((x) => [x[1], x[2]]));
  return { defaults, overrides };
}

/**
 * Per-language exclusions inside an otherwise whole-directory set.
 *
 * ⚠ Only for files whose consumer is UNAMBIGUOUS in the source, since the point of shipping whole
 * directories is that lazily-loaded tables cannot go missing. `diacritizer-egy.onnx` (12 MB) is
 * loaded only when the Arabic engine is built with `variety === "egyptian"`
 * (languages/arabic/diacritizer.ts:167), i.e. for `arz` alone — charging the other ten Arabic codes
 * 12 MB for a model they can never load is not a defensible default.
 */
const excludeFor = (code, key) =>
  key.endsWith("arabic/diacritizer-egy.onnx") && code !== "arz";

const filesUnder = (dir) => {
  const abs = join(DATA, dir);
  if (!existsSync(abs)) return [];
  return readdirSync(abs, { withFileTypes: true })
    .filter((d) => d.isFile()).map((d) => `${dir}/${d.name}`).sort();
};

const dirOf = (key) => key.slice(0, key.lastIndexOf("/"));
const sizeOf = (keys) => keys.reduce((n, k) => n + (existsSync(join(DATA, k)) ? statSync(join(DATA, k)).size : 0), 0);

const argv = process.argv.slice(2);
const langs = argv.length ? argv : registryCodes();

// ⚠ ONE recording, for the ENGINE phase only. Importing the engine reads every language's manifest
// at module scope (182 files, 4.5 MB) whatever language you then pick, and that set is not derivable
// from the module graph — it is what the manifest loaders do at import time.
const { stdout } = await exec("npx", ["tsx", "tools/browser-prefetch.mts", "en"],
                              { cwd: REPO, maxBuffer: 1 << 28 });
const engine = new Set(JSON.parse(stdout).engine.keys);

const reg = registryTargets(), neural = neuralTargets();
const languages = {};
const dirs = {};
for (const code of langs) {
  if (!reg[code]) { console.warn(`  ⚠ ${code}: not in the registry, skipped`); continue; }
  const { dirs: own, coreFiles } = dirsFor(reg[code], neural[code]);
  for (const d of own) dirs[d] ??= filesUnder(d);
  const keys = [...own.flatMap((d) => dirs[d]), ...coreFiles].filter((k) => !excludeFor(code, k));
  languages[code] = { dirs: own, core: coreFiles, bytes: sizeOf(keys),
                      exclude: [...own.flatMap((d) => dirs[d])].filter((k) => excludeFor(code, k)) };
}
// Every language directory is staged, not only the ones some language claims: a foreign-script run
// is routed to a reader chosen from the TEXT (see foreignRouting), so the client may ask for any.
for (const d of readdirSync(join(DATA, "languages"))) dirs[`languages/${d}`] ??= filesUnder(`languages/${d}`);
dirs["core"] = filesUnder("core");

rmSync(OUT, { recursive: true, force: true });
let bytes = 0;
const copied = new Set();
const copy = (k) => {
  if (copied.has(k)) return;
  const src = join(DATA, k), dst = join(OUT, k);
  if (!existsSync(src)) { console.warn("  missing:", k); return; }
  mkdirSync(dirname(dst), { recursive: true });
  copyFileSync(src, dst); bytes += statSync(src).size; copied.add(k);
};
for (const k of [...engine].sort()) copy(k);
for (const ks of Object.values(dirs)) for (const k of ks) copy(k);

const manifest = { engine: [...engine].sort(), dirs, languages, foreign: foreignRouting() };
writeFileSync(join(OUT, "_keys.json"), JSON.stringify(manifest));
// Committed, so the language catalogue can report download sizes without re-running the probe.
writeFileSync("tools/data/lang-dirs.json", JSON.stringify(languages, null, 1));

console.log(`${Object.keys(languages).length} languages catalogued`);
console.log(`engine: ${engine.size} files, ${(sizeOf([...engine]) / 1e6).toFixed(1)} MB (always fetched)`);
console.log(`staged ${copied.size} files, ${(bytes / 1e6).toFixed(1)} MB -> ${OUT}`);
const big = Object.entries(languages).sort((a, b) => b[1].bytes - a[1].bytes).slice(0, 8);
for (const [c, v] of big) console.log(`  heaviest: ${c} ${(v.bytes / 1e6).toFixed(1)} MB (${v.dirs.join(" ")})`);
