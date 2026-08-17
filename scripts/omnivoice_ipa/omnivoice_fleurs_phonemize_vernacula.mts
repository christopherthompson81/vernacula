/**
 * OmniVoice IPA corpus — re-phonemize FLEURS transcripts with vernacula-phonemizer.
 *
 * Companion to `omnivoice_flieurs_phonemize.ts`, which produced the CURRENT corpus
 * via espeak-ng-portable in canonical-IPA mode. This script reads the same FLEURS
 * train.tsv inputs and writes the same `${id}\t${ipa}` byid/ layout, but routes the
 * text through the native `vernacula-phonemizer` engine instead. The two output
 * trees are then compared by `compare_ipa_engines.py` to decide where the new
 * engine still needs refinement.
 *
 *   old (espeak):     /mnt/data/omnivoice_ipa/work/phonemized/byid/<code>.tsv
 *   new (vernacula):  /mnt/data/omnivoice_ipa/work/phonemized_vernacula/byid/<code>.tsv
 *
 * Uses `phonemizeAsync` — the unified best-output entry. This matters: it is the
 * path that restores unwritten vowels on the unpointed abjads (Arabic) and engages
 * the neural OOV/tagger models (en, fr, bn, fa, ...). Plain `phonemize` would emit
 * an Arabic consonant skeleton and silently downgrade English OOV handling.
 *
 * Usage:
 *   npx tsx omnivoice_fleurs_phonemize_vernacula.ts --all
 *   npx tsx omnivoice_fleurs_phonemize_vernacula.ts en_us ar_eg
 *   npx tsx omnivoice_fleurs_phonemize_vernacula.ts --all --limit 200   # timing probe
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { phonemizeAsync } from "/home/chris/Programming/vernacula-phonemizer/src/index.ts";
import { restoreAbbreviationDots, restoreInitialismCasing, restoreNguniConcordAcronyms } from "./initialism_casing.mts";

const ROOT = "/mnt/data/omnivoice_ipa";
const TSV = `${ROOT}/corpus/fleurs_transcripts/data`;
const OLD = `${ROOT}/work/phonemized/byid`;
const OUT = `${ROOT}/work/phonemized_vernacula/byid`;

/**
 * FLEURS code → vernacula-phonemizer registry code. The default is the first '_'
 * segment; these are the cases where the registry has a closer-matching variety
 * than the bare macro-code, and we take it (that is what production would ship).
 *
 * NOTE for diff reading: espeak was run on the *base* code for these three, so
 * some of their diff will be genuine dialect difference rather than engine
 * disagreement. `compare_ipa_engines.py` flags them.
 */
const VARIETY: Record<string, string> = {
  ar_eg: "arz",      // Egyptian Arabic — dedicated egy diacritizer + variety data
  es_419: "es-419",  // Latin American Spanish (seseo, yeísmo)
  pt_br: "pt-BR",    // Brazilian Portuguese
};

/** Registry code for a FLEURS code. */
function regCode(lang: string): string {
  return VARIETY[lang] ?? lang.split("_")[0];
}

/** The languages actually present in the existing espeak corpus — the diff set. */
function allLangs(): string[] {
  return readdirSync(OLD)
    .filter((f) => f.endsWith(".tsv"))
    .map((f) => f.slice(0, -4))
    .sort();
}

/** Raw tab-split (FLEURS tsv is not quoted CSV). col0=id, col3=normalized text. */
function rows(lang: string): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  const text = readFileSync(join(TSV, lang, "train.tsv"), "utf8");
  for (const line of text.split("\n")) {
    if (!line) continue;
    const c = line.split("\t");
    if (c.length >= 4 && c[3].trim()) out.push([c[0], c[3]]);
  }
  return out;
}

async function run(lang: string, limit: number): Promise<void> {
  const code = regCode(lang);
  const t0 = Date.now();
  let data = rows(lang);
  if (limit > 0) data = data.slice(0, limit);

  const lines: string[] = [];
  let ok = 0;
  const errs: string[] = [];
  for (const [id, txt] of data) {
    try {
      // Restore the casing FLEURS stripped, so the phonemizer's initialism pass can see it. That pass is
      // gated on `\p{Lu}{2,}` — capitals ARE its signal — so on lowercased input `pbs`/`rspca`/`xdr` reached
      // the OOV g2p and were read as words. Repairing the input rather than the shared matcher: see
      // initialism_casing.mts for why, and for the hand-reviewed list this is allowed to touch.
      // Two of the three signals FLEURS stripped, restored before the engine sees the text: the CAPITALS
      // the initialism pass is gated on, and the PERIOD each language's abbreviation table is keyed on.
      // (The third, apostrophes, is not repairable from our side — see initialism_casing.mts.)
      const repaired = restoreNguniConcordAcronyms(
        restoreAbbreviationDots(restoreInitialismCasing(txt), lang), lang);
      const ipa = (await phonemizeAsync(repaired, code)).replace(/[\r\n]+/g, " ").trim();
      if (ipa) { lines.push(`${id}\t${ipa}`); ok++; }
      else errs.push(`${id}\tEMPTY`);
    } catch (e) {
      errs.push(`${id}\t${(e as Error).message.replace(/[\r\n\t]+/g, " ").slice(0, 160)}`);
    }
  }

  if (!existsSync(OUT)) mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, `${lang}.tsv`), lines.join("\n") + "\n", "utf8");
  if (errs.length) writeFileSync(join(OUT, `${lang}.errors.tsv`), errs.join("\n") + "\n", "utf8");

  const dt = ((Date.now() - t0) / 1000).toFixed(0);
  const rate = (ok / Math.max(1, (Date.now() - t0) / 1000)).toFixed(0);
  console.log(
    `${lang} (${code}): ${ok} phonemized, ${errs.length} err, ${dt}s (${rate}/s) -> byid/${lang}.tsv`,
  );
}

const argv = process.argv.slice(2);
const li = argv.indexOf("--limit");
const limit = li >= 0 ? Number(argv[li + 1]) : 0;
// Guard the `li + 1` skip on li >= 0 — otherwise, with no --limit, li is -1 and this drops argv[0],
// so `... sd_in` silently fell through to "no positional args" and re-ran all 28 languages.
const positional = argv.filter((a, i) => !a.startsWith("--") && !(li >= 0 && i === li + 1));
const langs = positional.length === 0 ? allLangs() : positional;

console.log(`# ${langs.length} language(s), limit=${limit || "none"}, out=${OUT}`);
for (const l of langs) await run(l, limit);
