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
const ENGINE_URL = "/vphon/src/browser.js";
const DATA_BASE = "/vphon-data";

type PhonemizeAsync = (text: string, lang: string) => Promise<string>;

let enginePromise: Promise<PhonemizeAsync> | undefined;

async function init(onProgress?: (d: string) => void): Promise<PhonemizeAsync> {
  onProgress?.("fetching phonemizer data");
  const keys: string[] = await (await fetch(`${DATA_BASE}/_keys.json`)).json();
  const bytes = new Map<string, Uint8Array>();
  await Promise.all(keys.map(async (k) => {
    const r = await fetch(`${DATA_BASE}/${k}`);
    if (!r.ok) throw new Error(`phonemizer data ${k} -> ${r.status}`);
    bytes.set(k, new Uint8Array(await r.arrayBuffer()));
  }));

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
  // The neural tier (English's BiLSTM and friends) reaches ORT through this seam. Each neural
  // path DEGRADES to the rule engine when its model is absent rather than throwing.
  vp.setOrtLoader(() => import("onnxruntime-web"));
  const engine = await vp.loadEngine();
  return engine.phonemizeAsync as PhonemizeAsync;
}

/** Text -> canonical IPA. Uses phonemizeAsync, which routes through each language's neural model
 *  where one exists; the sync entry is the fallback and gives different (worse) IPA. */
export async function phonemize(text: string, lang: string,
                                onProgress?: (d: string) => void): Promise<string> {
  enginePromise ??= init(onProgress);
  return (await enginePromise)(text, lang);
}

export function phonemizerReady(): boolean {
  return enginePromise !== undefined;
}
