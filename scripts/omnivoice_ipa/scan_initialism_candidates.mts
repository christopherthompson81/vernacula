/**
 * Find the letter runs FLEURS' lowercasing hid from the initialism pass — the QC gate's input.
 *
 * `core/initialisms.ts` matches `\p{Lu}{2,}`: CAPITALS ARE THE SIGNAL that a letter run is spelled out
 * rather than read as a word. FLEURS ships every transcript lowercased, so `pbs`, `xdr`, `rspca` and
 * `wned` never enter the pass at all and reach the OOV g2p, which reads them as words — `ɹspkˈɑː`,
 * `wnˈɛd`, unpronounceable onsets both. Run 33's wav2vec2 probe confirmed against the audio that the
 * readers say LETTER NAMES.
 *
 * Rather than widen that matcher (it is shared by ~190 engines — it is what reads French TGV and
 * Russian США — so loosening it is a fleet-wide change), we repair the INPUT: restore the casing the
 * corpus destroyed, and the existing well-tested pass fires by itself.
 *
 * This script only PROPOSES. It emits every candidate with its count, its current reading and the
 * reading it would get uppercased, for hand review — because the predicate cannot make the call alone:
 *
 *   - readability is not convention (initialisms.ts says so): `ong`, `pib`, `us`, `uk` are perfectly
 *     pronounceable and still spelled out, so a phonotactic test under-selects; and
 *   - the predicate OVER-selects on foreign proper nouns, which are full of clusters English does not
 *     license (`nkurunziza`, `srebrenica`) and are words, not initialisms.
 *
 * So the output is a worklist, and `INITIALISM_UPPERCASE` in fix_initialism_casing.mts is the reviewed
 * subset that actually gets applied.
 *
 * Usage:
 *   npx tsx scan_initialism_candidates.mts            # all languages, writes the TSV
 *   npx tsx scan_initialism_candidates.mts en_us ko_kr
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { phonemizeAsync } from "/home/chris/Programming/vernacula-phonemizer/src/index.ts";
import { getPhonemizer } from "/home/chris/Programming/vernacula-phonemizer/src/registry.ts";
import type { EnglishPhonemizer } from "/home/chris/Programming/vernacula-phonemizer/src/languages/english/english.ts";

import { isUnreadableEnglish } from "/home/chris/Programming/vernacula-phonemizer/src/languages/english/normalize.ts";
import { isUnreadableWelsh } from "/home/chris/Programming/vernacula-phonemizer/src/languages/welsh/normalize.ts";
import { isUnreadableCzech } from "/home/chris/Programming/vernacula-phonemizer/src/languages/czech/normalize.ts";
import { isUnreadableIrish } from "/home/chris/Programming/vernacula-phonemizer/src/languages/irish/normalize.ts";
import { isUnreadableGerman } from "/home/chris/Programming/vernacula-phonemizer/src/languages/german/normalize.ts";
import { isUnreadableFrench } from "/home/chris/Programming/vernacula-phonemizer/src/languages/french/normalize.ts";
import { isUnreadableSpanish } from "/home/chris/Programming/vernacula-phonemizer/src/languages/spanish/normalize.ts";
import { isUnreadableCatalan } from "/home/chris/Programming/vernacula-phonemizer/src/languages/catalan/normalize.ts";
import { isUnreadablePortuguese } from "/home/chris/Programming/vernacula-phonemizer/src/languages/portuguese/normalize.ts";
import { isUnreadableSwedish } from "/home/chris/Programming/vernacula-phonemizer/src/languages/swedish/normalize.ts";
import { isUnreadableTurkish } from "/home/chris/Programming/vernacula-phonemizer/src/languages/turkish/normalize.ts";
import { isUnreadableRussian } from "/home/chris/Programming/vernacula-phonemizer/src/languages/russian/normalize.ts";
import { isUnreadableHausa } from "/home/chris/Programming/vernacula-phonemizer/src/languages/hausa/normalize.ts";
import { isUnreadableFula } from "/home/chris/Programming/vernacula-phonemizer/src/languages/fula/normalize.ts";
import { isUnreadableOromo } from "/home/chris/Programming/vernacula-phonemizer/src/languages/oromo/normalize.ts";

/**
 * ⚠ READABILITY IS PER-LANGUAGE, and using English's test everywhere is what produced the first version's
 * 2,164-candidate flood. `makeUnreadableTest` is parameterized by PhonotacticsData — each language declares
 * its OWN vowels, legal onsets and legal codas — and 38 languages ship one. Welsh spells a vowel ⟨w⟩
 * (`bwrdd`, `cwmwl`), Czech has syllabic r/l (`smrt`, `skrz`), Irish has its own cluster inventory
 * (`bhfuil`): every one of those is perfectly readable IN ITS OWN LANGUAGE and was only "unreadable"
 * because an English test was asked a question about Welsh.
 *
 * So each host is judged by its own phonotactics. English is the FALLBACK, and for the non-Latin-script
 * hosts that is not a compromise but the correct test: a Latin run there is foreign by definition and is
 * delegated to English, so English phonotactics is exactly the question being asked.
 *
 * Still imperfect for vi/xh/zu — Latin-script languages with no table yet — where the English fallback
 * still over-selects their native vocabulary. Those are left to the cross-language filter and the judge.
 */
const UNREADABLE: Record<string, (w: string) => boolean> = {
    en_us: isUnreadableEnglish, cy_gb: isUnreadableWelsh, cs_cz: isUnreadableCzech,
    ga_ie: isUnreadableIrish, de_de: isUnreadableGerman, fr_fr: isUnreadableFrench,
    es_419: isUnreadableSpanish, ca_es: isUnreadableCatalan, pt_br: isUnreadablePortuguese,
    sv_se: isUnreadableSwedish, tr_tr: isUnreadableTurkish, ru_ru: isUnreadableRussian,
    ha_ng: isUnreadableHausa, ff_sn: isUnreadableFula, om_et: isUnreadableOromo,
};

/** Languages with no phonotactics table of their own, whose script is LATIN — so the English fallback is
 *  a genuine approximation rather than the right question. Flagged in the output for the reviewer. */
const NO_TABLE_LATIN = new Set(["vi_vn", "xh_za", "zu_za"]);

const unreadableFor = (lang: string): ((w: string) => boolean) => UNREADABLE[lang] ?? isUnreadableEnglish;

const ROOT = "/mnt/data/omnivoice_ipa";
const TSV = `${ROOT}/corpus/fleurs_transcripts/data`;
const BYID = `${ROOT}/work/phonemized_vernacula/byid`;
const OUT = `${ROOT}/work/initialism_gate`;

const VARIETY: Record<string, string> = { ar_eg: "arz", es_419: "es-419", pt_br: "pt-BR" };
const regCode = (l: string): string => VARIETY[l] ?? l.split("_")[0]!;

/**
 * A run of Latin letters, 2+, entirely lowercase — the only shape the pass cannot see.
 *
 * ⚠ The trailing `(?!'\p{L})` excludes a run that is the FIRST HALF OF A CONTRACTION. An apostrophe is
 * neither a letter nor a mark, so without it `didn't` matched as `didn` + `t` and the scan reported
 * `isn`/`didn`/`wouldn`/`wasn`/`hadn`/`doesn`/`couldn` as candidate initialisms. That was this regex, not
 * the corpus: FLEURS keeps the apostrophe (col3 reads `didn't` intact). Recorded because I first wrote it
 * down as a FLEURS normalization defect, which it is not.
 *
 * Note it must stay one-sided. A run PRECEDED by an apostrophe is legitimate and wanted — Catalan `l'adn`
 * is the article plus ADN, and `adn` is a real initialism there.
 */
const RUN = /(?<![\p{L}\p{M}])[a-z]{2,}(?![\p{L}\p{M}]|'\p{L})/gu;

/** Longest sensible initialism. Beyond this a letter run is a word or a name, not an abbreviation. */
const MAX_LEN = 6;

const langs = (): string[] =>
    readdirSync(BYID).filter((f) => f.endsWith(".tsv") && !f.endsWith(".errors.tsv")).map((f) => f.slice(0, -4)).sort();

/** id → col3 (the normalized transcript), first occurrence wins. */
function rows(lang: string): Array<[string, string]> {
    const out: Array<[string, string]> = [];
    const seen = new Set<string>();
    for (const line of readFileSync(join(TSV, lang, "train.tsv"), "utf8").split("\n")) {
        if (!line) continue;
        const c = line.split("\t");
        if (c.length >= 4 && c[3]!.trim() && !seen.has(c[0]!)) {
            seen.add(c[0]!);
            out.push([c[0]!, c[3]!.trim()]);
        }
    }
    return out;
}

const en = (): EnglishPhonemizer => getPhonemizer("en") as EnglishPhonemizer;

const only = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const targets = only.length ? only : langs();

// token → {langs, count, one example sentence}
const cand = new Map<string, { langs: Set<string>; n: number; eg: string }>();

for (const lang of targets) {
    for (const [, txt] of rows(lang)) {
        for (const m of txt.matchAll(RUN)) {
            const w = m[0]!;
            if (w.length > MAX_LEN) continue;
            // The dictionary owns it — `isRecorded` in the pass, and the same call the pass makes.
            if (en().knownWord(w) !== undefined) continue;
            // Judged by THIS HOST's phonotactics, not English's. See UNREADABLE above.
            if (!unreadableFor(lang)(w)) continue;
            const e = cand.get(w) ?? { langs: new Set<string>(), n: 0, eg: txt };
            e.langs.add(lang);
            e.n += 1;
            cand.set(w, e);
        }
    }
}

// Both readings, so the review sees the actual delta rather than guessing at it.
const out: string[] = ["count\tlangs\tno_table_latin\ttoken\tas_lowercase\tas_uppercase\texample"];
const sorted = [...cand.entries()].sort((a, b) => b[1].n - a[1].n);
for (const [w, e] of sorted) {
    const lo = await phonemizeAsync(w, "en");
    const up = await phonemizeAsync(w.toUpperCase(), "en");
    const approx = [...e.langs].some((l) => NO_TABLE_LATIN.has(l)) ? "approx" : "";
    out.push(`${e.n}\t${[...e.langs].sort().join(",")}\t${approx}\t${w}\t${lo}\t${up}\t${e.eg.slice(0, 90)}`);
}

if (!existsSync(OUT)) mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, "candidates.tsv"), out.join("\n") + "\n", "utf8");
console.log(`${sorted.length} distinct candidate runs across ${targets.length} languages -> ${OUT}/candidates.tsv`);
console.log(`total occurrences: ${sorted.reduce((s, [, e]) => s + e.n, 0)}`);
console.log(`differing readings: ${out.slice(1).filter((l) => { const c = l.split("\t"); return c[4] !== c[5]; }).length}`);
void regCode;
