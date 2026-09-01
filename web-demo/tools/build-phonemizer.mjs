#!/usr/bin/env node
/**
 * Transpile vernacula-phonemizer to plain ESM under public/vphon/src/, preserving the module tree.
 *
 * ⚠ PRESERVING THE TREE IS THE WHOLE POINT, NOT A STYLE CHOICE. The engine derives every data key from
 * `import.meta.url` (core/dataPath.ts: it slices after the last "/src/"), so a bundler that rewrites module
 * URLs to chunk names erases the only thing naming the data — and the engine throws rather than guessing.
 * That rules out letting Vite bundle it. Transpiling per-file into a directory that still contains "/src/"
 * keeps `import.meta.url` = "/vphon/src/languages/welsh/welsh.js", whose key is "languages/welsh".
 *
 * The app loads it with a dynamic import of an ABSOLUTE URL (see src/inference/phonemizer.ts), which Vite
 * leaves alone, so this output is served verbatim in dev and copied as-is into dist by the public/ dir.
 */
import * as esbuild from "esbuild";
import { readdirSync, statSync, readFileSync, writeFileSync, rmSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";

const SRC = "../external/vernacula-phonemizer/src";
const OUT = "public/vphon/src";

function walk(dir, acc = []) {
    for (const e of readdirSync(dir)) {
        const p = join(dir, e);
        if (statSync(p).isDirectory()) walk(p, acc);
        else if (e.endsWith(".ts") && !e.endsWith(".d.ts") && !e.includes(".scratch.")) acc.push(p);
    }
    return acc;
}

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });
const entryPoints = walk(SRC);

await esbuild.build({
    entryPoints,
    outdir: OUT,
    outbase: SRC,
    format: "esm",
    platform: "browser",
    target: "es2022",
    bundle: false,          // per-file transpile; keeps one output module per source module
    sourcemap: false,
    logLevel: "warning",
});

// esbuild leaves import specifiers verbatim, and this codebase writes them with explicit `.ts`.
let rewritten = 0;
for (const f of walk(OUT).concat(walkJs(OUT))) {
    const before = readFileSync(f, "utf8");
    const after = before.replace(/(from\s*["'][^"']+?)\.ts(["'])/g, "$1.js$2")
                        .replace(/(import\(\s*["'][^"']+?)\.ts(["'])/g, "$1.js$2");
    if (after !== before) { writeFileSync(f, after); rewritten++; }
}
function walkJs(dir, acc = []) {
    for (const e of readdirSync(dir)) {
        const p = join(dir, e);
        if (statSync(p).isDirectory()) walkJs(p, acc);
        else if (e.endsWith(".js")) acc.push(p);
    }
    return acc;
}

console.log(`transpiled ${entryPoints.length} modules -> ${OUT} (${rewritten} with .ts specifiers rewritten)`);
