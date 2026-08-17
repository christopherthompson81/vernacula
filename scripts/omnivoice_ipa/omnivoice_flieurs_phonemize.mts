/**
 * OmniVoice IPA corpus — phonemize FLEURS transcripts to id-keyed IPA.
 *
 * Rebuilt after the reorg lost the original `phonemize_fleurs.ts` (the corpus
 * workflow moved to Wikipedia/Leipzig wordlists for the primitive census; this
 * FLEURS-transcript path is a Vernacula consumer, not part of the census).
 *
 * For each FLEURS language it reads train.tsv (col0=utt id, col3=normalized
 * transcription), runs the transcription through `phonemize()` with the default
 * `ipaRendering:"canonical"` (#844 — unified Chao tone letters + canonical
 * codepoints, so tone is consistent across languages), and writes
 *   <OUT>/byid/<fleurs_code>.tsv   as   `${id}\t${ipa}`  (newlines stripped)
 * which `scripts/omnivoice_ipa/ingest_fleurs.py` reads via `id_to_ipa()`.
 *
 * FLEURS code → phonemizer data/ dir is the first '_' segment (en_us→en,
 * cmn_hans_cn→cmn, ff_sn→ff, …), EXCEPT where espeak-ng-portable ships a
 * closer variety (VARIETY below). pt_br must NOT fall through to bare `pt` —
 * that is EUROPEAN Portuguese (ɨ-reduction, coda ʃ: pɐkiʃtɐ̃ᶷ̃), and the first
 * corpus build shipped EP phonemes on Brazilian audio because of exactly that
 * fallthrough. espeak's `pt-br` gives the real BP (nˈɔɾt͡ʃi, ˈĩnd͡ʒjɐ).
 * (es_419/ar_eg have no closer espeak variety — bare es/ar is all it ships.)
 *
 * Usage:
 *   npx tsx tools/omnivoice-fleurs-phonemize.ts cs_cz [more codes...]
 *   npx tsx tools/omnivoice-fleurs-phonemize.ts --all      # the 24 FLEURS-sourced
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadLanguage, phonemize } from "/home/chris/Programming/espeak-ng-portable/src/index.ts";

const ROOT = "/mnt/data/omnivoice_ipa";
const TSV = `${ROOT}/corpus/fleurs_transcripts/data`;
const OUT = `${ROOT}/work/phonemized/byid`;

// The 24 FLEURS-sourced languages of the census-based 25-lang minimal set
// (si is external-audio, no FLEURS code — handled separately).
const ALL_24 = [
  "en_us", "cmn_hans_cn", "hi_in", "es_419", "ar_eg", "fr_fr", "pt_br", "ru_ru",
  "de_de", "ja_jp", "tr_tr", "vi_vn", "ta_in", "ko_kr", "ha_ng", "th_th",
  "ff_sn", "kk_kz", "zu_za", "cs_cz", "sv_se", "ca_es", "ga_ie", "cy_gb",
];

// FLEURS code → espeak data dir, where the default first-'_'-segment rule is wrong.
const VARIETY: Record<string, string> = { pt_br: "pt-br" };

function rows(lang: string): Array<[string, string]> {
  // Raw tab-split (FLEURS tsv is not quoted CSV). col0=id, col3=normalized text.
  const out: Array<[string, string]> = [];
  const text = readFileSync(join(TSV, lang, "train.tsv"), "utf8");
  for (const line of text.split("\n")) {
    if (!line) continue;
    const c = line.split("\t");
    if (c.length >= 4 && c[3].trim()) out.push([c[0], c[3]]);
  }
  return out;
}

async function run(lang: string): Promise<void> {
  const phonCode = VARIETY[lang] ?? lang.split("_")[0];
  const t0 = Date.now();
  const loaded = await loadLanguage(phonCode);
  const data = rows(lang);
  const lines: string[] = [];
  let ok = 0, err = 0;
  for (const [id, txt] of data) {
    try {
      const ipa = phonemize(txt, loaded).replace(/[\r\n]+/g, " ").trim();
      if (ipa) { lines.push(`${id}\t${ipa}`); ok++; }
    } catch {
      err++;
    }
  }
  if (!existsSync(OUT)) mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, `${lang}.tsv`), lines.join("\n") + "\n", "utf8");
  const dt = ((Date.now() - t0) / 1000).toFixed(0);
  console.log(`${lang} (${phonCode}): ${ok} phonemized, ${err} err, ${dt}s -> byid/${lang}.tsv`);
}

const argv = process.argv.slice(2);
const langs = argv.length === 0 || argv[0] === "--all" ? ALL_24 : argv;
for (const l of langs) await run(l);
