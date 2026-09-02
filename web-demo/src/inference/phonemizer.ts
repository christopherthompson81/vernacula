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

interface LangEntry { dirs: string[]; core: string[]; bytes: number; exclude?: string[] }
interface KeyManifest {
  engine: string[];
  /** Every data directory, listed file by file — a language ships whole directories. */
  dirs: Record<string, string[]>;
  languages: Record<string, LangEntry>;
  /** The foreign-run routing table, lifted from the engine at staging time. */
  foreign: { defaults: Record<string, string>; overrides: Record<string, Record<string, string>> };
}

/**
 * Scripts the engine can route a foreign run to, as a JavaScript-side detector.
 *
 * ⚠ THE ENGINE PICKS A READER FROM THE TEXT, NOT FROM THE SELECTED LANGUAGE. A run in a script the
 * host language does not own is handed to another language's phonemizer, chosen by script
 * (core/scripts.ts). So typing an English phrase inside a Thai sentence pulls ENGLISH's tables,
 * and a Devanagari quotation inside anything pulls Hindi's. The delegation is wrapped in try/catch
 * upstream, which means a data key we failed to prefetch does not raise — it silently degrades to a
 * wrong reading. That is invisible, so the scripts present in the input are resolved to languages
 * HERE and fetched before phonemizing.
 *
 * ⚠ This changes which phoneme tables load. It does not change the VOICE: one voice renders the
 * whole utterance, code-switching included.
 */
const SCRIPT_RE: [string, RegExp][] = [
  ["Latin", /\p{Script=Latin}/u], ["Cyrillic", /\p{Script=Cyrillic}/u], ["Arabic", /\p{Script=Arabic}/u],
  ["Greek", /\p{Script=Greek}/u], ["Hebrew", /\p{Script=Hebrew}/u], ["Devanagari", /\p{Script=Devanagari}/u],
  ["Bengali", /\p{Script=Bengali}/u], ["Tamil", /\p{Script=Tamil}/u], ["Telugu", /\p{Script=Telugu}/u],
  ["Kannada", /\p{Script=Kannada}/u], ["Malayalam", /\p{Script=Malayalam}/u], ["Gujarati", /\p{Script=Gujarati}/u],
  ["Gurmukhi", /\p{Script=Gurmukhi}/u], ["Oriya", /\p{Script=Oriya}/u], ["Sinhala", /\p{Script=Sinhala}/u],
  ["Khmer", /\p{Script=Khmer}/u], ["Lao", /\p{Script=Lao}/u], ["Thai", /\p{Script=Thai}/u],
  ["Tibetan", /\p{Script=Tibetan}/u], ["Myanmar", /\p{Script=Myanmar}/u], ["Ethiopic", /\p{Script=Ethiopic}/u],
  ["Armenian", /\p{Script=Armenian}/u], ["Georgian", /\p{Script=Georgian}/u], ["Hangul", /\p{Script=Hangul}/u],
  ["Han", /\p{Script=Han}/u], ["Kana", /\p{Script=Hiragana}|\p{Script=Katakana}/u],
  ["Tifinagh", /\p{Script=Tifinagh}/u], ["Cherokee", /\p{Script=Cherokee}/u], ["Ol_Chiki", /\p{Script=Ol_Chiki}/u],
  ["Adlam", /\p{Script=Adlam}/u], ["Nko", /\p{Script=Nko}/u], ["Syloti_Nagri", /\p{Script=Syloti_Nagri}/u],
  ["Javanese", /\p{Script=Javanese}/u], ["Sundanese", /\p{Script=Sundanese}/u],
];

/** Languages whose data the engine may reach for this text, host first. */
function languagesForText(text: string, host: string, m: KeyManifest): string[] {
  const out = new Set([host]);
  for (const [script, re] of SCRIPT_RE) {
    if (!re.test(text)) continue;
    const target = m.foreign.overrides[host]?.[script] ?? m.foreign.defaults[script];
    if (target && target !== host) out.add(target);
  }
  return [...out];
}

/**
 * ⚠ TWO PHASES, AND THE SPLIT IS WHAT KEEPS FIRST LOAD SMALL. `engine` is what importing the engine
 * reads — every language's manifest, 182 files / 4.5 MB — and is needed whatever language you pick.
 * A language's own directories come on top and only when it is chosen: the whole tree is 151 MB,
 * but the median language is 10 KB and 152 of 193 are under 1 MB. The heavy ones are Arabic
 * (24 MB, a 15 MB diacritizer), Urdu/W. Punjabi (30 MB, the shared rider model), English (14 MB),
 * Persian, Russian and Japanese.
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
  for (const code of languagesForText(text, lang, manifest!)) {
    if (fetched.has(code)) continue;
    const entry = manifest!.languages[code];
    if (!entry) { fetched.add(code); continue; }
    const skip = new Set(entry.exclude ?? []);
    const keys = [...entry.dirs.flatMap((d) => manifest!.dirs[d] ?? []), ...entry.core]
      .filter((k) => !skip.has(k));
    if (keys.length) {
      onProgress?.(code === lang ? `fetching ${lang} data` : `fetching ${code} data (embedded script)`);
      await fetchKeys(keys, onProgress, `${code} `);
    }
    fetched.add(code);
  }
  return phon(text, lang);
}

export function phonemizerReady(): boolean {
  return enginePromise !== undefined;
}
