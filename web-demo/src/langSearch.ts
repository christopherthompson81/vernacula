import type { LanguageOption } from "./inference/languages.ts";

/**
 * Rank a language against a type-ahead query. Lower is better; -1 means no match.
 *
 * Matches on the English name, the code, AND the language's own name for itself.
 *
 * ⚠ THE ENDONYM IS SEARCHABLE, NOT DECORATIVE. Someone looking for their own language types it the
 * way they write it — "ελληνικά", "मराठी", "Tiếng Việt" — and against an English-only matcher none of
 * those hits anything. Matching it is most of the point of showing it.
 *
 * ⚠ `\b` IS ASCII-ONLY IN JS REGEX, so a word-boundary anchor never fires inside 中文 or ไทย. The
 * endonym therefore also gets a plain `includes`, which is what actually matches a non-Latin query;
 * relying on the same `\b` path as the English name would have made 128 of the 179 endonyms
 * unsearchable while looking, in a Latin-script spot check, like it worked.
 */
export function score(l: LanguageOption, q: string): number {
  const name = l.name.toLowerCase(), code = l.code.toLowerCase();
  // Case-folding a caseless script is a no-op, so this is safe for Han/Arabic/Devanagari alike; it
  // matters for the Latin-script endonyms (Boarisch, Kreyòl ayisyen, Qaraqalpaq tili).
  const nat = l.native?.toLowerCase() ?? "";
  if (code === q) return 0;
  // ⚠ An EXACT name match outranks a prefix match, or a language whose endonym merely starts with
  // another's loses to it: "Sesotho" ranked Sepedi first, because its endonym is "Sesotho sa Leboa"
  // and the tie then fell to alphabetical order.
  if (name === q || nat === q) return 0;
  if (name.startsWith(q) || nat.startsWith(q)) return 1;
  if (code.startsWith(q)) return 2;
  const word = new RegExp(`\\b${q.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&")}`, "u");
  if (word.test(name)) return 3;
  if (nat && (word.test(nat) || nat.includes(q))) return 3;
  if (name.includes(q)) return 4;
  return -1;
}

/** The picker's match list for a query: scored, filtered, best first, then alphabetical. */
export function search(langs: readonly LanguageOption[], query: string): LanguageOption[] {
  const q = query.trim().toLowerCase();
  if (!q) return [...langs];
  return langs.map((l) => [score(l, q), l] as const)
    .filter(([s]) => s >= 0)
    .sort((a, b) => a[0] - b[0] || a[1].name.localeCompare(b[1].name))
    .map(([, l]) => l);
}
