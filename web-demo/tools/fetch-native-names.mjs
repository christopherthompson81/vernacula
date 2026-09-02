#!/usr/bin/env node
/**
 * Generate tools/data/native-names.json — each language's name AS ITS OWN SPEAKERS WRITE IT, for the
 * picker to show and to search on.
 *
 * Two sources, in order, both recorded per entry so a wrong name can be traced rather than guessed at:
 *
 *   cldr      `Intl.DisplayNames` in the language's own locale. Authoritative and maintained, and it
 *             covers the 128 languages CLDR actually has locale data for.
 *   wikidata  P1705 "native label", keyed by ISO 639-1 (P218) or 639-3 (P220), for the rest.
 *
 * ⚠ THE RESULT IS BAKED INTO A CHECKED-IN FILE RATHER THAN READ AT RUNTIME. `Intl.DisplayNames`
 * answers from whatever ICU the running Node/browser was built against, so computing this at build
 * time would let the picker's contents change under an unrelated toolchain bump, and computing it in
 * the BROWSER would vary per visitor. Re-run this deliberately; review the diff.
 *
 * ⚠ CLDR FALLS BACK SILENTLY. `Intl.DisplayNames(["an"])` has no Aragonese locale, so it answers from
 * English and returns "Aragonese" — a real-looking string that is not an endonym. The test is
 * therefore "differs from the English display name", not "is non-empty", and everything that fails it
 * drops through to Wikidata.
 *
 *   node tools/fetch-native-names.mjs            # network: query.wikidata.org
 *   node tools/fetch-native-names.mjs --offline  # CLDR + PICKS only, keeps existing wikidata entries
 */
import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

const META = "tools/data/language-meta.json";
const OUT = "tools/data/native-names.json";

/**
 * Where Wikidata offers several labels, or offers one that is wrong for OUR variety, the choice is
 * made here rather than by taking `[0]`. Every entry carries the reason, because "pick the first" is
 * exactly the defect this repo just found in a Punjabi lexicon mined the same way.
 */
const PICKS = {
  // Script must match the variety the demo actually phonemizes — checked against each language's
  // own sample sentence in language-meta.json.
  wuu: "吴语",            // sample is simplified
  hak: "客家話",           // sample is traditional
  gan: "贛語",             // sample is traditional
  nan: "Bân-lâm-gí",      // the engine reads POJ romanisation, not Han
  cdo: "Mìng-dĕ̤ng-ngṳ̄",  // likewise Bàng-uâ-cê
  crh: "Qırımtatar tili", // the demo's Crimean Tatar is Latin, not Cyrillic
  bal: "بلوچی",            // Perso-Arabic, matching the sample; Wikidata also lists a Cyrillic form
  nog: "Ногайша",          // Cyrillic sample
  kaa: "Qaraqalpaq tili",  // Latin sample
  rup: "armãneashti",      // nine spellings listed; this is the one the phonemizer catalogue also uses
  ig: "Asụsụ Igbo",        // prefer the form that differs from the English name
  mi: "te reo Māori",
  syl: "ꠍꠤꠟꠐꠤ ꠝꠣꠔ",
  bpy: "বিষ্ণুপ্রিয়া মণিপুরী",
  ab: "Аԥсуа бызшәа",
  ba: "башҡорт теле",
  ltg: "latgaļu volūda",
  mad: "Bhâsa Madhurâ",
  // ⚠ WIKIDATA IS WRONG FOR THIS VARIETY. Its pnb label is Gurmukhi (ਪੰਜਾਬੀ); Western Punjabi is
  // written in SHAHMUKHI, which is what our reference voice and sample sentence both use.
  pnb: "پنجابی",
  // Not in either source. Each is attested in the language's own sample sentence or the phonemizer's
  // language catalogue, which is the strongest evidence available for these.
  cjy: "晉語",            // appears verbatim in the demo's own Jin sample sentence
  hsn: "湘語",
  mto: "ayöök",           // phonemizer catalogue
  hmn: "Hmoob",
  hne: "छत्तीसगढ़ी",
};

/**
 * Dropped on purpose — recorded so the search is not repeated, and so a later run does not
 * "helpfully" restore them.
 */
const REFUSED = {
  afb: "Wikidata's label (اللهجة الإماراتية العربية) names EMIRATI specifically, which is one Gulf variety, not Gulf Arabic",
  acw: "no sourced endonym found; the Arabic varieties are inconsistently labelled and a guess is worse than a blank",
  ajp: "as acw", apc: "as acw", apd: "as acw", ayl: "as acw",
  rkt: "Wikidata's দেশী is tagged zxx (no linguistic content) rather than a Rangpuri locale — unverified",
  pbt: "پښتو names Pashto, which is already listed as `ps`; a picker must not offer one name twice",
  zsm: "CLDR answers Melayu, which is `ms` — the macrolanguage, not Standard Malay",
};

const meta = JSON.parse(readFileSync(META, "utf8"));

// ⚠ A KEY THAT IS NOT A LANGUAGE CODE IS SILENTLY DEAD, and both tables below are hand-maintained
// while language-meta.json is not. The phonemizer's own numeral register learned this the hard way
// (FLEURS writes `ny_mw`, the registry ships `nya`), so the tables are checked rather than trusted.
const stray = [...Object.keys(PICKS), ...Object.keys(REFUSED)].filter((c) => !(c in meta));
if (stray.length) {
  console.error(`stray key(s) in PICKS/REFUSED that name no language: ${stray.join(" ")}`);
  process.exit(1);
}
const en = new Intl.DisplayNames(["en"], { type: "language", fallback: "none" });
const prev = (() => { try { return JSON.parse(readFileSync(OUT, "utf8")); } catch { return {}; } })();

const out = {};
const gap = [];
/** Why a language has no entry — recorded at the point of decision, never inferred afterwards. */
const skipped = {};
// The refusal reasons live in REFUSED above, which is the readable record; the summary only needs
// to say that a refusal is why, not repeat nine sentences of it.
for (const c of Object.keys(REFUSED)) skipped[c] = "refused on purpose (see REFUSED in this file)";
for (const code of Object.keys(meta)) {
  if (REFUSED[code]) continue;
  if (PICKS[code]) { out[code] = { name: PICKS[code], src: "picked" }; continue; }
  let nat = null, eng = null;
  try { nat = new Intl.DisplayNames([code], { type: "language", fallback: "none" }).of(code) ?? null; } catch { /* unknown code */ }
  try { eng = en.of(code) ?? null; } catch { /* unknown code */ }
  // Equal to the English name means either a silent CLDR fallback or a language whose endonym simply
  // IS the English name. Either way there is nothing to add, so it is not an entry.
  if (nat && eng && nat !== eng) out[code] = { name: nat, src: "cldr" };
  else {
    gap.push(code);
    // Provisional; the Wikidata pass below replaces it on a hit.
    skipped[code] = nat && nat === eng ? "endonym is the English name" : "no source";
  }
}

if (!process.argv.includes("--offline") && gap.length) {
  const two = gap.filter((c) => c.split("-")[0].length === 2).map((c) => c.split("-")[0]);
  const three = gap.filter((c) => c.split("-")[0].length > 2).map((c) => c.split("-")[0]);
  const q = `SELECT ?c ?native WHERE { { VALUES ?c { ${two.map((c) => `"${c}"`).join(" ")} } ?l wdt:P218 ?c ; wdt:P1705 ?native . }`
    + ` UNION { VALUES ?c { ${three.map((c) => `"${c}"`).join(" ")} } ?l wdt:P220 ?c ; wdt:P1705 ?native . } }`;
  const raw = execFileSync("curl", ["-sfL", "--max-time", "120", "-G", "https://query.wikidata.org/sparql",
    "--data-urlencode", "format=json", "--data-urlencode", `query=${q}`,
    "-H", "Accept: application/sparql-results+json", "-A", "vernacula-web-demo/1.0"], { encoding: "utf8", maxBuffer: 1 << 24 });
  const byCode = new Map();
  for (const b of JSON.parse(raw).results.bindings) {
    const c = b.c.value, tag = b.native["xml:lang"] ?? "", v = b.native.value;
    // ⚠ Prefer a label tagged with the language's OWN code over a script- or region-suffixed one
    // (`crh-cyrl`, `crh-ro`, `bal-latn`) or a foreign-tagged one (Wikidata files the Iraqi Arabic
    // label under plain `ar`). A label in another language is a description, not a name.
    if (!byCode.has(c) || (tag === c && byCode.get(c).tag !== c)) byCode.set(c, { tag, v });
  }
  for (const code of gap) {
    // ⚠ A REGION-SUFFIXED CODE MUST NOT INHERIT ITS BASE LANGUAGE'S LABEL. Querying `en` for `en-GB`
    // and `en-IN` gave both of them "English", which names the parent rather than the variety —
    // the same error as taking Wikidata's Gurmukhi label for Shahmukhi `pnb`. CLDR handles the
    // varieties it knows (es-419 → "español latinoamericano"), and where it does not, blank is right.
    if (code.includes("-")) { skipped[code] = "no source for this variety (the base code names the parent)"; continue; }
    const hit = byCode.get(code);
    if (!hit) continue;
    const name = hit.v;
    if (name.toLowerCase() === (meta[code].name ?? "").toLowerCase()) {
      skipped[code] = "endonym is the English name";                             // adds nothing
      continue;
    }
    delete skipped[code];
    out[code] = { name, src: "wikidata" };
  }
} else {
  for (const code of gap) if (prev[code]?.src === "wikidata") out[code] = prev[code];
}

/**
 * ⚠ TWO LANGUAGES SHARING AN ENDONYM MEANS ONE OF THEM INHERITED ITS PARENT'S NAME, and a picker that
 * offers the same native name twice is worse than one that offers it once. CLDR falls back silently
 * from a variety to its macrolanguage — `zsm` (Standard Malay) answers "Melayu", which is `ms` — so
 * the duplicate is dropped from the MORE SPECIFIC code, keeping it on the one it actually names.
 * Detected rather than listed, because the same fallback will produce new pairs as CLDR grows.
 */
const byName = new Map();
for (const code of Object.keys(out)) {
  const k = out[code].name.toLowerCase();
  const held = byName.get(k);
  if (held === undefined) { byName.set(k, code); continue; }
  // The base language is the shorter code; on a tie, the one whose English name carries a
  // parenthetical ("Pashto (Southern)") is the variety and loses to the plain one.
  const heldSpecific = (meta[held].name ?? "").includes("(");
  const codeSpecific = (meta[code].name ?? "").includes("(");
  let loser;
  if (held.length !== code.length) loser = held.length < code.length ? code : held;
  else if (heldSpecific !== codeSpecific) loser = heldSpecific ? held : code;
  else {
    // Neither is more specific by either test — a real ambiguity, not something to resolve by
    // iteration order. Say so and keep both out, so the picker never offers one name twice.
    console.error(`ambiguous endonym "${out[code].name}" shared by ${held} and ${code} with no `
      + `rule to separate them — add one to PICKS or REFUSED`);
    process.exit(1);
  }
  const winner = loser === code ? held : code;
  byName.set(k, winner);
  const why = out[loser].src === "picked"
    ? ` — ⚠ it is a PICKS entry, so that table now has a dead key`
    : "";
  console.log(`  dropped ${loser} "${out[loser].name}" — names ${winner}, not ${loser}${why}`);
  skipped[loser] = `duplicate of ${winner}`;
  delete out[loser];
}

const sorted = Object.fromEntries(Object.keys(out).sort().map((k) => [k, out[k]]));
writeFileSync(OUT, JSON.stringify(sorted, null, 1) + "\n", "utf8");
const by = (s) => Object.values(sorted).filter((v) => v.src === s).length;
// ⚠ THE ABSENT LANGUAGES ARE ABSENT FOR FOUR DIFFERENT REASONS and one "none found" hides the only
// one anybody could act on. "endonym is the English name" is a correct outcome — Afrikaans, Akan,
// Wolof — not a gap.
const reasons = {};
for (const c of Object.keys(meta)) if (!(c in sorted)) (reasons[skipped[c] ?? "no source"] ??= []).push(c);
console.log(`${Object.keys(sorted).length}/${Object.keys(meta).length} languages have a native name`
  + ` (cldr ${by("cldr")}, wikidata ${by("wikidata")}, picked ${by("picked")})`);
for (const [why, codes] of Object.entries(reasons).sort((a, b) => b[1].length - a[1].length))
  console.log(`  ${String(codes.length).padStart(3)} ${why.slice(0, 62)}: ${codes.join(" ")}`);

