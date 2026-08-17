/**
 * THE CASING WALL, measured instead of guessed.
 *
 * ⚠ FLEURS ships lowercased text. Every rule in the fleet that keys on CAPITALS therefore declines
 * silently, and an initialism that should be spelled out as letter NAMES is instead handed to the
 * ordinary word g2p — which does not refuse it, because a letter run is perfectly readable as a word.
 * The output is fluent and wrong, and sometimes it is worse than wrong: letters vanish.
 *
 *     vpn  -> "vpn"      the raw letters, sitting in an IPA stream
 *     vhs  -> "vs"       the h DELETED
 *     hq   -> "k"        two letters, one phone
 *     nhs  -> "ns"       h deleted again
 *     wto  -> "ˈuːt"     unrecognisable
 *     hdmi -> "dmˈɪ"
 *
 * The earlier pass at this (scan_initialism_candidates.mts) asked each language's own phonotactics
 * whether a token was UNREADABLE, and produced 1,464 candidates needing bulk triage. This asks a
 * sharper question that needs no judgement at all:
 *
 *     does phonemize(token) differ from phonemize(TOKEN)?
 *
 * For an ordinary word the two agree — casing is not phonemic. When they disagree, a capital-keyed
 * rule fired for one and not the other, which is the wall itself, directly observed. The uppercase
 * reading is also the ANSWER, so each hit arrives with its own fix already attached.
 *
 *   npx tsx scan_casing_differential.mts en_us en
 */
import { phonemize } from "/home/chris/Programming/vernacula-phonemizer/src/index.ts";
import { readFileSync } from "node:fs";

// Token counts are extracted from the SQLite corpus by a small python step (node:sqlite is not in
// this Node) — `tokens.json` is {lang: {token: count}} over 2-6 letter all-lowercase runs.
const TOKENS = "/tmp/claude-1000/-mnt-data-Programming-vernacula/094be646-3763-4932-9f18-6babb274e16e/scratchpad/tokens.json";
const [lang, code] = [process.argv[2] ?? "en_us", process.argv[3] ?? "en"];

const all = JSON.parse(readFileSync(TOKENS, "utf-8")) as Record<string, Record<string, number>>;
const counts = new Map<string, number>(Object.entries(all[lang] ?? {}));

const hits: { tok: string; n: number; lo: string; up: string }[] = [];
for (const [tok, n] of counts) {
    const lo = (await phonemize(tok, code)).trim();
    const up = (await phonemize(tok.toUpperCase(), code)).trim();
    if (lo !== up) hits.push({ tok, n, lo, up });
}
hits.sort((a, b) => b.n - a.n);

console.log(`# ${lang}: ${counts.size} distinct 2-6 letter lowercase tokens`);
console.log(`# ${hits.length} read DIFFERENTLY when uppercased — each one is a capital-keyed rule that`);
console.log(`# declined on the lowercased corpus. The UPPER column is the reading we should be getting.\n`);
console.log(`${"token".padEnd(8)}${"n".padStart(5)}   ${"as lowercased".padEnd(28)}as UPPERCASE`);
for (const h of hits) console.log(`${h.tok.padEnd(8)}${String(h.n).padStart(5)}   ${h.lo.padEnd(28)}${h.up}`);
