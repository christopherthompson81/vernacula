/**
 * Restore the CASING that FLEURS' normalization destroyed, so the phonemizer's initialism pass can see it.
 *
 * `core/initialisms.ts` matches `\p{Lu}{2,}` — capitals are the signal that a letter run is spelled out
 * rather than read as a word — and FLEURS lowercases every transcript. So `pbs`, `rspca`, `xdr` and `wned`
 * never entered the pass and reached the OOV g2p, which read them as words: `ɹspkˈɑː`, `wnˈɛd`,
 * unpronounceable onsets both. Run 33's wav2vec2 probe confirmed against the AUDIO that readers say letter
 * names here (`acma` → "A-C-M-A", `rspca` → "ar-es-pee-see-ay").
 *
 * We repair the INPUT rather than widen that matcher, because the matcher is shared by ~190 engines (it is
 * what reads French TGV and Russian США) and loosening it is a fleet-wide change. Fixing the corpus text
 * lets the existing, well-tested pass fire by itself, per language, with each language's own letter names.
 *
 * ⚠ THIS LIST IS HAND-REVIEWED AND MUST STAY THAT WAY. A phonotactic test cannot generate it, in both
 * directions at once:
 *
 *   - It OVER-selects. `isUnreadableEnglish` is an ENGLISH test, and every language's native vocabulary is
 *     full of clusters English does not license — Welsh `mewn`/`roedd`, German `nicht`, Czech `jsou`, Irish
 *     `bhfuil`, Vietnamese `khi`, Xhosa `kwaye`. Those are ordinary words their own engine reads correctly.
 *     2,164 candidate runs came out of the scan; ~46 survived cross-language filtering; 29 survived review.
 *   - It UNDER-selects, and `core/initialisms.ts` says so itself: "readability is not convention". `us`,
 *     `uk`, `ong`, `pib` are perfectly pronounceable and still spelled out.
 *
 * The discriminator that actually works is CROSS-LANGUAGE SPREAD: an international abbreviation shows up as
 * a Latin run in twenty-odd different corpora, a native word shows up in one. Everything below appears in
 * ≥4 languages (most in 20+) and was then checked by hand in context — see scan_initialism_candidates.mts
 * for the worklist that produced it, and `EXCLUDED` for what was rejected and why.
 */

/** Reviewed: uppercase these so the initialism pass claims them. Verified in context, one by one. */
export const INITIALISM_UPPERCASE: readonly string[] = [
    // Time / geodesy
    "utc", "gmt", "adt", "bce",
    // Organisations, agencies, networks
    "rspca", "usgs", "nsa", "npws", "nhc", "ptwc", "ndp", "pmo", "hjr", "afcfta", "plc",
    // Broadcast / brand call signs
    "pbs", "wned", "qvc",
    // Technical
    "vpn", "pstn", "dslr", "gp", "xdr", "qc",
    // ⚠ Added by the CASING DIFFERENTIAL (scan_casing_differential.mts), which compares
    // phonemize(token) against phonemize(TOKEN) and reports every disagreement. It found 73 English
    // candidates against the earlier phonotactic scan's 1,464, and each came with its own answer
    // attached. These four were the only genuinely new initialisms in that set — everything else it
    // surfaced was already on this list, already in EXCLUDED, or an ordinary word.
    "gps", "hiv", "usaf", "un",
    // ⚠ Added from the CASING DIFFERENTIAL run across the whole corpus rather than English alone.
    // All three read as impossible onset clusters when lowercased -- `nba` -> nbˈa / n̪ˠˈəbˠə,
    // `fbi` -> fβˈi, `cctv` -> the Fula geminate t͡ʃːtv -- and correctly as letter names when
    // uppercased. None is a word in any of the 28 corpora, which is the test that matters for a
    // GLOBALLY-applied list. `nba` x97 and `fbi` x72 appear across the parallel translations, so
    // the repair pays out in a dozen languages each, not just English.
    "nba", "fbi", "cctv",
    // ⚠ `ceo` is here because ONE OF THE "UNFIXABLE" NGUNI CLICK CASES WAS THE CASING WALL IN
    // DISGUISE. Run 40 measured the xh/zu click classifier and found three foreign words it wrongly
    // claims — china, canada, ceo — and called all three the intrinsic ceiling of a phonotactic test.
    // Two of them are: `canada` is CV.CV.CV and shaped exactly like a native word. But `ceo` is not a
    // word at all, it is an INITIALISM, and uppercasing it makes xh/zu spell it `sˈiː ˈiː ˈɔː` —
    // which is precisely what the recognizer heard (`s i i o`) where we were emitting the click
    // `kǀˈɛːɔ`. Inert in the other seven languages that have it. The ceiling was real for two of the
    // three, not all three.
    "ceo",
    // Places / companies commonly initialised
    "png", "hk",
    // Alphanumeric codes — the pass has an attached-to-digits branch (`CG4684` is its own cited example),
    // which is likewise uppercase-gated, so these need the same repair: cg4684, kv62.
    "cg", "kv",
    // Model designations read as letters (audi tt)
    "tt",
    // Members of Parliament
    "mps",

    // ── Added after the local-model sweep of the 1,408 low-spread candidates (Run 34). The model
    // proposed 52 LETTERS; these are the ones that survived reading them in context. Most are
    // language-local abbreviations, which is exactly the class cross-language spread cannot find:
    // spread 1 is normal for them, so they had to be judged individually rather than filtered.
    "adn",      // ca/es/fr — DNA
    "usd", "jpy", "gbp",          // currency codes
    "tda", "dehb", "adhs",        // ADHD in Catalan / Turkish / German
    "hdp",      // cs — GDP
    "pdg",      // fr — CEO, said "pé-dé-gé"
    "npd", "cpm", "cnh", "cftv", "afb", "ghb", "qvl", "rm", "rpv",
    "lkw",      // de — truck, said "el-ka-ve"
    "vtt",      // fr — mountain bike
    "vd",       // sv — CEO
    "fn", "fns", // sv — the UN, and its genitive FN:s
    "fc",       // football club
    "www",
    "udssr", "sssr", "sscb",      // the USSR in German / Czech / Turkish
    "hsyh",
    "kb",       // cy — a corpus typo for the code KV62; still read as code letters
    // RSPCA, misspelled three different ways in three corpora. Letters is right either way, and it is
    // much closer than the word readings (ɹspˈɑː, ɹskpˈeᶦ, ɹpskˈeᶦ) these currently get.
    "rspa", "rscpa", "rpsca",

    // ── Promoted from the model's UNSURE bucket on a second pass. It hedged; these are unambiguous once
    // you know the language, and each was verified to read as that language's OWN letter names:
    "bnp",      // sv — GDP (bruttonationalprodukt), said be-enn-pe
    "mcs",      // ga — REM (mearchorraí súl) → em-ce-as in Irish letter names
    "rs",       // cs — multiple sclerosis (roztroušená skleróza)
    "rmn",      // ca/es/fr/pt — MRI (ressonància magnètica nuclear) → erra-ema-ena
    "irm",      // ca/es/fr/pt — the other MRI abbreviation; audio has the reader spelling it
    "mrt",      // de/sv — MRI (Magnetresonanztomographie) → em-er-te
    "osn",      // cs — the UN (Organizace spojených národů) → o-es-en
    "bm",       // tr — the UN (Birleşmiş Milletler) → be-me
    "dda",      // pt — ADHD (déficit de atenção) → dê-dê-á
];

/**
 * Rejected candidates, kept as a record so a future pass does not re-litigate them. Each would be made
 * WORSE by uppercasing.
 */
export const EXCLUDED: Readonly<Record<string, string>> = {
    // ⚠ REJECTED FROM THE CASING DIFFERENTIAL and the COLLISION GATE. Both report a token that reads
    // differently when uppercased, or that also occurs in another corpus; that is a strong signal but
    // never a verdict, and these are exactly the cases where acting on it would do damage.
    eu: "WORD in cy (their) ×660, fr (eu, avoir) ×46, pt (I) ×15 — the initialism is the minority",
    wwii: "roman-numeral compound — WWII is 'World War Two', NOT the letter names the uppercase pass gives it",
    led: "ordinary verb ×13 in context ('led to', 'led by'), not the diode. The homograph loses to frequency",
    ll: "TOKENIZER ARTIFACT — the tail of we'll / I'll, split on the apostrophe. Not in the source text at all",

    // UNITS. Letter names are wrong; these want the unit-expansion layer, which reads the ones attached to a
    // number already (`40 km` → kilometers). The standalone residue is a normalization gap, not an initialism.
    km: "unit — kilometre, wants 'kilometers' not K-M",
    cm: "unit — centimetre",
    kg: "unit — kilogram",
    kph: "unit — km/h",
    sq: "unit — 'sq mi' is square miles, wants 'square'",
    mbit: "unit — megabit (802.11n speeds)",
    // NAMES AND WORDS the English phonotactic test mislabels as unreadable.
    zmapp: "drug name, read 'zee-map' — Z-M-A-P-P is wrong",
    jagr: "surname (jaromir jagr)",
    dzong: "Dzongkha word (fortress)",
    angkor: "place name",
    rossby: "surname (Rossby waves)",
    bhog: "Indic word",
    bhutha: "Indic word",
    bhajan: "Indic word",
    lakkha: "Indic word",
    // METALINGUISTIC. The sentence is ABOUT Spanish spelling ("pronounce the r and the rr differently");
    // a reader produces the trill, not English letter names.
    rr: "metalinguistic mention of Spanish orthography",
    // ABBREVIATION WANTING A WORD, not letters.
    rd: "'Rd' is road",
    // CONTRACTION STEMS — and these were a bug in the SCAN, not in the corpus. FLEURS keeps the
    // apostrophe (`didn't` is intact in col3); the scan's run regex treated `'` as a run boundary and so
    // split `didn't` into `didn` + `t`. Fixed there with a `(?!'\p{L})` lookahead. Kept listed because
    // uppercasing them would still be wrong ("I-S-N") if they ever reappear.
    isn: "scan artifact — first half of a contraction, not a stripped apostrophe",
    didn: "scan artifact — contraction half",
    wouldn: "scan artifact — contraction half",
    wasn: "scan artifact — contraction half",
    hadn: "scan artifact — contraction half",
    doesn: "scan artifact — contraction half",
    couldn: "scan artifact — contraction half",
    // Ordinary English words CMUdict happens to miss.
    mown: "English word",
    jousts: "English word",

    // ── Rejected from the local-model sweep's 52 LETTERS proposals (Run 34).
    //
    // BANTU CONCORD + ACRONYM — SOLVED SINCE, and kept here only so the whole run is never uppercased.
    // Xhosa and Zulu agglutinate the noun-class concord onto the borrowed acronym (`ye`+`PBS`,
    // `i`+`XDR`, `ku`+`TV`), so uppercasing the WHOLE run reads the concord as letters ("Y-E-P-B-S").
    // I originally wrote that casing could not fix these even in principle, because the pass required a
    // non-letter before an uppercase run. That is no longer true: the xh/zu initialism rule added since
    // deliberately allows a lowercase letter there, precisely because this is the shape those languages
    // write. So they ARE repaired — by `NGUNI_CONCORD_ACRONYM` below, which uppercases only the acronym
    // half (`yepbs` → `yePBS` → *ye* + P-B-S, click-free).
    ihk: "xh/zu concord: i-HK",
    enpws: "xh concord: e-NPWS",
    iqvc: "zu concord: i-QVC",
    ixdr: "zu concord: i-XDR",
    kutv: "zu concord: ku-TV",
    kwitv: "xh concord: kwi-TV",
    lwepbs: "zu concord: lwe-PBS",
    segps: "xh concord: se-GPS",
    yepbs: "xh concord: ye-PBS",
    zegps: "xh concord: ze-GPS",
    zetv: "xh concord: ze-TV",
    // MIS-TOKENIZED UNITS. `mmx` is not an acronym: it is "56mmx56mm", i.e. mm × mm with the
    // multiplication sign glued on. "M-M-X" would be wrong; this wants the symbol layer.
    mmx: "mis-tokenized '56mm x 56mm' — units plus a multiplication sign",
    // NO EFFECT. Uppercasing changes nothing (the pass still declines, being readable), so including
    // them would only add noise to the allowlist.
    pyo: "uppercase reading identical — pass still declines",
    mip: "uppercase reading identical — pass still declines",
    // WANTS THE FULL PHRASE, not letters. Vietnamese reads TCN as "trước Công nguyên".
    tcn: "vi — read as the full phrase 'trước Công nguyên'",
    // UNCLEAR, single occurrence, no reading is confidently better. Left alone rather than guessed at.
    kmr: "unclear, 1 occurrence",
    fvt: "unclear, 1 occurrence",
    rptc: "unclear scramble, 1 occurrence",
    rctp: "unclear scramble, 1 occurrence",
};

/**
 * One matcher per token: the run, not embedded in a longer letter run. The trailing lookahead excludes
 * letters and combining marks but NOT digits, so `cg4684` and `kv62` are claimed too — which is the point,
 * since the pass's alphanumeric-code branch is uppercase-gated in the same way.
 */
const PATTERNS: ReadonlyArray<readonly [RegExp, string]> = INITIALISM_UPPERCASE.map(
    (t) => [new RegExp(`(?<![\\p{L}\\p{M}])${t}(?![\\p{L}\\p{M}])`, "gu"), t.toUpperCase()] as const,
);

/** Uppercase every reviewed initialism in `text`. Case-preserving for everything else. */
export function restoreInitialismCasing(text: string): string {
    let out = text;
    for (const [re, up] of PATTERNS) out = out.replace(re, up);
    return out;
}

/**
 * The SAME defect one layer over: FLEURS also strips the trailing PERIOD, and each language's
 * abbreviation table is keyed on the dotted form.
 *
 * This is the third stripped signal in the corpus, and naming them together is the useful part:
 *   - CAPITALS  → the initialism pass (`\p{Lu}{2,}`) never fires        → `restoreInitialismCasing`
 *   - PERIODS   → the abbreviation tables never match                  → here
 *   - APOSTROPHES → contractions arrive as stems (`isn't` → `isn`)     → not repairable by us; logged
 *
 * Every token below was verified to expand CORRECTLY once the dot is restored, using the language's own
 * existing table — so this adds no data and no upstream change, it just stops hiding the input from a pass
 * that already works. Verified pairs (bare → dotted):
 *
 *     de bzw   pt͡sf              → bət͡sˈiːʊŋsvaɪ̯zə   (beziehungsweise)
 *     de usw   ʊsf               → ʊnt zoː vˈaɪ̯tɐ      (und so weiter)
 *     de nr    nʁ                → nˈʊmɐ                (Nummer)
 *     sv kl    kl                → klˈɔ̀kːan             (klockan)
 *     cs tzv   tsf               → tˈaɡzvaniː           (takzvaný)
 *     cs atd   ˈatt              → ˈa tˈak dˈaːlɛ       (a tak dále)
 *     cs tzn   dzn               → tˈo znˈamɛnaː        (to znamená)
 *     cs sv    sf                → svˈatiː              (svatý)
 *     fr cf    kf                → kɔ̃fɛʁ                (confer)
 *
 * ⚠ PER-LANGUAGE, unlike the casing list. `sv` is *svatý* in Czech and a language code elsewhere; `nr` and
 * `kl` mean nothing in most languages. Applying these globally would corrupt other corpora, so the repair
 * is keyed on the FLEURS language.
 *
 * Abbreviations whose table entry is MISSING (de chr., cs cca., sv bnp/kr, zu udkt/skwele/kma,
 * xh mnu/ugq, om inchii/dvdtti, cy cdg, es mbps) are NOT here — a dot does not help when there is nothing
 * to match. Those are upstream data work, tracked separately.
 */
export const ABBREVIATION_DOT: Readonly<Record<string, readonly string[]>> = {
    de_de: ["bzw", "usw", "nr"],
    // `kr` is the ERA marker here, not the currency: all three sv occurrences are `f.kr`, and FLEURS
    // dropped the trailing dot the rule needs (`f.kr.`). Checked every occurrence — no currency `kr`.
    sv_se: ["kl", "kr"],
    cs_cz: ["tzv", "atd", "tzn", "sv", "cca"],
    fr_fr: ["cf"],
    // From the UNSURE second pass: `vb.` → *ve benzeri* ("and so on"), already in the Turkish table.
    tr_tr: ["vb"],
};

const DOT_PATTERNS: Readonly<Record<string, ReadonlyArray<RegExp>>> = Object.fromEntries(
    Object.entries(ABBREVIATION_DOT).map(([lang, toks]) => [
        lang,
        // Not already followed by a dot, and not glued to another letter. The trailing lookahead permits a
        // following digit so `nr 5` and `nr5` both repair.
        toks.map((t) => new RegExp(`(?<![\\p{L}\\p{M}])${t}(?![\\p{L}\\p{M}.])`, "gu")),
    ]),
);

/**
 * NGUNI CONCORD + ACRONYM. Xhosa and Zulu glue the noun-class concord straight onto a borrowed acronym, so
 * the lowercased corpus has `yepbs`, `ixdr`, `kwitv` as single runs. Uppercasing the WHOLE run would read
 * the concord as letters ("Y-E-P-B-S"), which is why these were excluded from `INITIALISM_UPPERCASE`.
 *
 * They are recoverable now: the xh/zu initialism pass takes a lowercase letter before the capitals
 * precisely because this is the shape those languages write, so uppercasing only the ACRONYM half works —
 * `yepbs` → `yePBS` → *ye* + P-B-S, click-free. Verified through the engine for every entry below.
 *
 * Split by hand, not by rule: the concord inventory overlaps the acronyms' own letters, so no regex
 * separates `zetv` (ze+TV) from a hypothetical acronym `ZETV` without knowing which is which.
 */
export const NGUNI_CONCORD_ACRONYM: Readonly<Record<string, readonly [string, string]>> = {
    ihk: ["i", "HK"], iqvc: ["i", "QVC"], ixdr: ["i", "XDR"], iptwc: ["i", "PTWC"],
    enpws: ["e", "NPWS"], nenpws: ["ne", "NPWS"], nedslr: ["ne", "DSLR"], wendp: ["we", "NDP"],
    kutv: ["ku", "TV"], kwitv: ["kwi", "TV"], zetv: ["ze", "TV"], zecctv: ["ze", "CCTV"],
    lwepbs: ["lwe", "PBS"], yepbs: ["ye", "PBS"], segps: ["se", "GPS"], zegps: ["ze", "GPS"],
};

const NGUNI_LANGS = new Set(["xh_za", "zu_za"]);
const CONCORD_PATTERNS: ReadonlyArray<readonly [RegExp, string]> = Object.entries(NGUNI_CONCORD_ACRONYM)
    .map(([tok, [c, a]]) => [new RegExp(`(?<![\\p{L}\\p{M}])${tok}(?![\\p{L}\\p{M}])`, "gu"), `${c}${a}`] as const);

/** Uppercase only the acronym half of a concord+acronym run. Nguni languages only. */
export function restoreNguniConcordAcronyms(text: string, lang: string): string {
    if (!NGUNI_LANGS.has(lang)) return text;
    let out = text;
    for (const [re, rep] of CONCORD_PATTERNS) out = out.replace(re, rep);
    return out;
}

/** Restore the stripped period on `lang`'s reviewed abbreviations. A no-op for languages with no list. */
export function restoreAbbreviationDots(text: string, lang: string): string {
    const pats = DOT_PATTERNS[lang];
    if (pats === undefined) return text;
    let out = text;
    const toks = ABBREVIATION_DOT[lang]!;
    pats.forEach((re, i) => {
        out = out.replace(re, `${toks[i]!}.`);
    });
    return out;
}
