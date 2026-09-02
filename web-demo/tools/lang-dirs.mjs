/**
 * Which data directories does each language read?
 *
 * ⚠ ANSWERED BY STATIC MODULE-GRAPH WALK, NOT BY RECORDING A RUN. Recording (upstream's
 * browser-prefetch) reports what a probe text actually touched, and that is systematically
 * incomplete in both directions: Basque's own tables load lazily on first use so a Latin probe
 * records NO directory for it, while Assamese's real dependency on Bengali's directory never
 * appears because the probe never reaches those tables either. Both were measured.
 *
 * The graph is exact because data keys are derived mechanically: `core/dataPath.ts` slices
 * `import.meta.url` after "/src/", so ANY module that mentions `import.meta.url` reads from the
 * directory mirroring its own source directory. Walking a language's import closure and collecting
 * those directories therefore yields the complete set — including the structural parents no one
 * would guess (Assamese→Bengali, Bosnian/Croatian→Serbian, Bashkir→Russian, Xhosa→Zulu,
 * Gujarati→Hindi, Saraiki→Punjabi, W. Armenian→Armenian, Kirundi→Kinyarwanda).
 */
import { readFileSync, existsSync } from "node:fs";
import { dirname, join, normalize } from "node:path";

const SRC = "../external/vernacula-phonemizer/src";

const read = (f) => readFileSync(f, "utf8");

/** code -> the module that implements it, from the registry's imports + its build() switch. */
export function registryTargets() {
  const src = read(join(SRC, "registry.ts"));
  const fnFile = {};
  for (const m of src.matchAll(/import\s*\{([^}]*)\}\s*from\s*"(\.\/[^"]+)"/g))
    for (const n of m[1].split(",").map((s) => s.trim()).filter(Boolean))
      fnFile[n] = join(SRC, m[2].replace(/\.ts$/, "") + ".ts");
  const body = src.slice(src.indexOf("function build(lang: string)"));
  const cases = [...body.matchAll(/case "([\w-]+)":/g)];
  const rets = [...body.matchAll(/return\s+([A-Za-z0-9_]+)\(/g)];
  const out = {};
  for (const c of cases) {
    const r = rets.find((x) => x.index > c.index);
    if (r && fnFile[r[1]]) out[c[1]] = fnFile[r[1]];
  }
  return out;
}

/**
 * code -> the module implementing its ASYNC path, from neuralRegistry's NEURAL table.
 *
 * ⚠ Needed separately: the demo calls `phonemizeAsync`, whose neural entries are built directly
 * rather than through the registry, so they are NOT in the factory module's import closure. Urdu
 * and Western Punjabi reach `core/riderDiacritizer.onnx` (15 MB) only this way.
 */
export function neuralTargets() {
  const src = read(join(SRC, "neuralRegistry.ts"));
  const fnFile = {};
  for (const m of src.matchAll(/import\s*\{([^}]*)\}\s*from\s*"(\.\/[^"]+)"/g))
    for (const n of m[1].split(",").map((s) => s.trim()).filter(Boolean))
      fnFile[n] = join(SRC, m[2].replace(/\.ts$/, "") + ".ts");
  const table = src.slice(src.indexOf("const NEURAL"), src.indexOf("\n};", src.indexOf("const NEURAL")));
  const out = {};
  for (const m of table.matchAll(/^\s{4}([\w-]+):\s*(?:\(t\)\s*=>\s*)?(\w+)/gm))
    if (fnFile[m[2]]) out[m[1]] = fnFile[m[2]];
  return out;
}

function resolve(from, spec) {
  if (!spec.startsWith(".")) return null;
  const p = normalize(join(dirname(from), spec.replace(/\.ts$/, "") + ".ts"));
  return existsSync(p) ? p : null;
}

/** Transitive import closure of a module, within the engine's source tree. */
function closure(entry) {
  const seen = new Set(), stack = [entry];
  while (stack.length) {
    const f = stack.pop();
    if (!f || seen.has(f)) continue;
    seen.add(f);
    const src = read(f);
    // ⚠ `import type` is erased at compile time and must NOT be followed. Nearly every language
    // module imports `type { Phonemizer } from "../../registry.ts"`, and following that edge pulls
    // the registry — which imports all 193 language modules, making every language's directory set
    // "all of them". Measured: Basque came back with 175 directories.
    for (const m of src.matchAll(/(?:^|\n)\s*import\s+(type\s+)?[^;]*?from\s+"(\.[^"]+)"/g)) {
      if (m[1]) continue;
      const r = resolve(f, m[2]);
      if (r && !r.endsWith("registry.ts") && !r.endsWith("src/index.ts")) stack.push(r);
    }
  }
  return seen;
}

const keyOf = (file) => dirname(file).slice(SRC.length + 1);

/**
 * @returns { dirs: string[], coreFiles: string[] } — language data directories (shipped whole,
 * because tables inside them load lazily) and the individual shared files under `core`.
 *
 * ⚠ `core` is NOT shipped whole: `core/riderDiacritizer.onnx` is 15 MB and serves only the
 * Perso-Arabic rider languages. Adding it to every language would put 15 MB on Welsh.
 */
export function dirsFor(...entries) {
  const dirs = new Set(), coreFiles = new Set();
  const all = new Set();
  for (const e of entries) if (e) for (const f of closure(e)) all.add(f);
  for (const f of all) {
    if (!read(f).includes("import.meta.url")) continue;
    const key = keyOf(f);
    if (key === "core") {
      if (f.endsWith("phonology.ts")) coreFiles.add("core/phonology.jsonc");
      if (f.endsWith("riderDiacritizer.ts"))
        { coreFiles.add("core/riderDiacritizer.onnx"); coreFiles.add("core/riderDiacritizer.meta.json"); }
    } else if (key.startsWith("languages/")) dirs.add(key);
  }
  return { dirs: [...dirs].sort(), coreFiles: [...coreFiles].sort() };
}
