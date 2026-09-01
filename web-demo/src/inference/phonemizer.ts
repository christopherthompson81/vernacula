/**
 * Browser glue for vernacula-phonemizer.
 *
 * ⚠ The engine is loaded by DYNAMIC IMPORT OF AN ABSOLUTE URL, deliberately. Its data keys come
 * from `import.meta.url` (core/dataPath.ts slices after the last "/src/"), so a bundler that
 * rewrites module URLs to chunk names erases the only thing naming the data — the engine throws
 * rather than guess, because a wrong key would surface as a missing lexicon, i.e. a plausible wrong
 * reading. `tools/build-phonemizer.mjs` transpiles it per-file into /vphon/src/ preserving the
 * tree; Vite leaves an absolute-URL import alone.
 *
 * ⚠ setDataSource() must run BEFORE loadEngine(): importing the engine reads 182 manifests at
 * module scope, so a source installed afterwards has already missed them.
 */
// ⚠ ASSEMBLED AT RUNTIME, AND THAT IS NOT STYLE. Vite constant-folds a literal specifier, sees it
// resolves into public/, and refuses to serve it in dev: "This file is in /public and will be
// copied as-is during build ... should not be imported from source code." `/* @vite-ignore */`
// alone does not help, because the path is still statically analyzable. Building it from parts
// makes it opaque to the bundler, which is exactly what we want — the engine must NOT be bundled
// (its data keys come from import.meta.url; see the note below).
const ENGINE_URL = ["", "vphon", "src", "browser.js"].join("/");
const DATA_BASE = "/vphon-data";
import { getOrt } from "./ortInit.ts";

type PhonemizeAsync = (text: string, lang: string) => Promise<string>;

interface KeyManifest { engine: string[]; languages: Record<string, string[]> }

/**
 * ⚠ TWO PHASES, AND THE SPLIT IS WHAT KEEPS FIRST LOAD SMALL. `engine` is what importing the engine
 * reads — every language's manifest, 182 files / 4.5 MB — and is needed whatever language you pick.
 * Per-language tables are fetched only when that language is chosen; all 28 together are ~202 MB,
 * which is not a first-load budget.
 *
 * ⚠ Expect English's data in most languages' lists. `phonemizeAsync` prewarms the English tagger
 * for mixed-Latin text, and a run in a script the host language does not own is delegated
 * (core/foreign.ts) — so a Thai page containing an English phrase loads English's tables too. The
 * lists are RECORDED from the engine rather than declared, so this is captured rather than guessed.
 */
const bytes = new Map<string, Uint8Array>();
let manifest: KeyManifest | undefined;
const fetched = new Set<string>();

async function fetchKeys(keys: string[], onProgress?: (d: string) => void, label = ""): Promise<void> {
  const missing = keys.filter((k) => !bytes.has(k));
  if (missing.length === 0) return;
  let done = 0;
  await Promise.all(missing.map(async (k) => {
    const r = await fetch(`${DATA_BASE}/${k}`);
    if (!r.ok) throw new Error(`phonemizer data ${k} -> ${r.status}`);
    bytes.set(k, new Uint8Array(await r.arrayBuffer()));
    if (++done % 20 === 0) onProgress?.(`${label}${done}/${missing.length} files`);
  }));
}

let enginePromise: Promise<PhonemizeAsync> | undefined;

async function init(onProgress?: (d: string) => void): Promise<PhonemizeAsync> {
  onProgress?.("fetching phonemizer data");
  manifest = await (await fetch(`${DATA_BASE}/_keys.json`)).json();
  await fetchKeys(manifest!.engine, onProgress, "phonemizer ");

  onProgress?.(`loading phonemizer (${bytes.size} files)`);
  const vp = await import(/* @vite-ignore */ ENGINE_URL);
  vp.setDataSource({
    read(key: string): Uint8Array {
      const b = bytes.get(key);
      // Throwing is required: returning empty bytes would turn an absent lexicon into a silently
      // empty one, and every word would take the OOV path to a plausible wrong reading.
      if (!b) throw new Error(`phonemizer data not prefetched: ${key}`);
      return b;
    },
  });
  // Same module the TTS side uses, loaded from the CDN — NOT bundled. See ortInit.ts.
  vp.setOrtLoader(() => getOrt());
  const engine = await vp.loadEngine();
  return engine.phonemizeAsync as PhonemizeAsync;
}

/** Text -> canonical IPA. Uses phonemizeAsync, which routes through each language's neural model
 *  where one exists; the sync entry is the fallback and gives different (worse) IPA. */
export async function phonemize(text: string, lang: string,
                                onProgress?: (d: string) => void): Promise<string> {
  // ⚠ Clear the cached promise on failure. Caching the PROMISE means one transient fetch error
  // would poison the page: every later click re-awaits the same rejection and reports the same
  // message, with no way to retry short of a reload.
  enginePromise ??= init(onProgress).catch((e) => { enginePromise = undefined; throw e; });
  const phon = await enginePromise;
  if (!fetched.has(lang)) {
    const keys = manifest?.languages[lang];
    if (keys?.length) {
      onProgress?.(`fetching ${lang} data`);
      await fetchKeys(keys, onProgress, `${lang} `);
    }
    fetched.add(lang);
  }
  return phon(text, lang);
}

export function phonemizerReady(): boolean {
  return enginePromise !== undefined;
}
