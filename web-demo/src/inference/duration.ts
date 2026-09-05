/**
 * Rule-based estimate of how many audio tokens a target text needs — the TypeScript port of
 * `Vernacula.Tts.Base.OmniVoiceDuration` (OmniVoice's RuleDurationEstimator). "Duration" here is
 * measured in audio-token counts, at 25 tokens/second.
 *
 * ⚠ FEED IT THE IPA, NOT THE ORTHOGRAPHY. The estimate is a RATIO — target script-weight ÷
 * reference script-weight × reference tokens — so it is only self-consistent when both sides are
 * the same representation, and IPA-on-both is what the fine-tune was listen-accepted with
 * (scripts/omnivoice_ipa/gen_accept_test.py). Mixing the two is silently wrong, worst where IPA
 * length diverges hardest from character count (Han, Kanji).
 */

const WEIGHTS: Record<string, number> = {
  cjk: 3.0, hangul: 2.5, kana: 2.2, ethiopic: 3.0, yi: 3.0,
  indic: 1.8, thai_lao: 1.5, khmer_myanmar: 1.8, arabic: 1.5,
  hebrew: 1.5, latin: 1.0, cyrillic: 1.0, greek: 1.0,
  armenian: 1.0, georgian: 1.0, punctuation: 0.5, space: 0.2,
  digit: 3.5, mark: 0.0, default: 1.0,
};

// Unicode block end-codepoints (ascending) -> script type. Mirrors duration.py's `ranges`.
const RANGES: [number, string][] = [
  [0x02af, "latin"], [0x03ff, "greek"], [0x052f, "cyrillic"], [0x058f, "armenian"],
  [0x05ff, "hebrew"], [0x077f, "arabic"], [0x089f, "arabic"], [0x08ff, "arabic"],
  [0x097f, "indic"], [0x09ff, "indic"], [0x0a7f, "indic"], [0x0aff, "indic"],
  [0x0b7f, "indic"], [0x0bff, "indic"], [0x0c7f, "indic"], [0x0cff, "indic"],
  [0x0d7f, "indic"], [0x0dff, "indic"], [0x0eff, "thai_lao"], [0x0fff, "indic"],
  [0x109f, "khmer_myanmar"], [0x10ff, "georgian"], [0x11ff, "hangul"], [0x137f, "ethiopic"],
  [0x139f, "ethiopic"], [0x13ff, "default"], [0x167f, "default"], [0x169f, "default"],
  [0x16ff, "default"], [0x171f, "default"], [0x173f, "default"], [0x175f, "default"],
  [0x177f, "default"], [0x17ff, "khmer_myanmar"], [0x18af, "default"], [0x18ff, "default"],
  [0x194f, "indic"], [0x19df, "indic"], [0x19ff, "khmer_myanmar"], [0x1a1f, "indic"],
  [0x1aaf, "indic"], [0x1b7f, "indic"], [0x1bbf, "indic"], [0x1bff, "indic"],
  [0x1c4f, "indic"], [0x1c7f, "indic"], [0x1c8f, "cyrillic"], [0x1cbf, "georgian"],
  [0x1ccf, "indic"], [0x1cff, "indic"], [0x1d7f, "latin"], [0x1dbf, "latin"],
  [0x1dff, "default"], [0x1eff, "latin"], [0x309f, "kana"], [0x30ff, "kana"],
  [0x312f, "cjk"], [0x318f, "hangul"], [0x9fff, "cjk"], [0xa4cf, "yi"],
  [0xa4ff, "default"], [0xa63f, "default"], [0xa69f, "cyrillic"], [0xa6ff, "default"],
  [0xa7ff, "latin"], [0xa82f, "indic"], [0xa87f, "default"], [0xa8df, "indic"],
  [0xa8ff, "indic"], [0xa92f, "indic"], [0xa95f, "indic"], [0xa97f, "hangul"],
  [0xa9df, "indic"], [0xa9ff, "khmer_myanmar"], [0xaa5f, "indic"], [0xaa7f, "khmer_myanmar"],
  [0xaadf, "indic"], [0xaaff, "indic"], [0xab2f, "ethiopic"], [0xab6f, "latin"],
  [0xabbf, "default"], [0xabff, "indic"], [0xd7af, "hangul"], [0xfaff, "cjk"],
  [0xfdff, "arabic"], [0xfe6f, "default"], [0xfeff, "arabic"], [0xffef, "latin"],
];

// The C# switches on UnicodeCategory; these are the direct property-escape equivalents.
const RE_MARK = /\p{Mn}|\p{Mc}|\p{Me}/u;
const RE_PUNCT = /\p{Pc}|\p{Pd}|\p{Ps}|\p{Pe}|\p{Pi}|\p{Pf}|\p{Po}|\p{Sm}|\p{Sc}|\p{Sk}|\p{So}/u;
const RE_SPACE = /\p{Zs}|\p{Zl}|\p{Zp}/u;
const RE_DIGIT = /\p{Nd}|\p{Nl}|\p{No}/u;

export function charWeight(code: number): number {
  if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) return WEIGHTS.latin;
  if (code === 32) return WEIGHTS.space;
  if (code === 0x0640) return WEIGHTS.mark; // Arabic Tatweel

  const ch = String.fromCodePoint(code);
  if (RE_MARK.test(ch)) return WEIGHTS.mark;
  if (RE_PUNCT.test(ch)) return WEIGHTS.punctuation;
  if (RE_SPACE.test(ch)) return WEIGHTS.space;
  if (RE_DIGIT.test(ch)) return WEIGHTS.digit;

  // bisect_left over the block end-codepoints: first range whose end >= code.
  let lo = 0, hi = RANGES.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (RANGES[mid][0] < code) lo = mid + 1; else hi = mid;
  }
  if (lo < RANGES.length) return WEIGHTS[RANGES[lo][1]] ?? WEIGHTS.default;
  if (code > 0x20000) return WEIGHTS.cjk;
  return WEIGHTS.default;
}

/** Sum of per-character weights. Iterates CODE POINTS (`for…of`), matching C#'s EnumerateRunes. */
export function totalWeight(text: string): number {
  let sum = 0;
  for (const ch of text) sum += charWeight(ch.codePointAt(0)!);
  return sum;
}

export function estimateDuration(targetText: string, refText: string, refDuration: number,
                                 lowThreshold = 50, boostStrength = 3): number {
  if (refDuration <= 0 || !refText) return 0;
  const refWeight = totalWeight(refText);
  if (refWeight === 0) return 0;
  const speedFactor = refWeight / refDuration;
  const estimated = totalWeight(targetText) / speedFactor;
  if (estimated < lowThreshold) return lowThreshold * Math.pow(estimated / lowThreshold, 1 / boostStrength);
  return estimated;
}

/** Port of `_estimate_target_tokens`: falls back to a fixed reference when none is given, scales
 *  by speed, floors at 1 token. */
export function estimateTargetTokens(text: string, refText: string | null,
                                     numRefAudioTokens: number | null, speed = 1.0): number {
  let rt = refText, n = numRefAudioTokens;
  if (n === null || !rt) { rt = "Nice to meet you."; n = 25; }
  let est = estimateDuration(text, rt, n);
  if (speed > 0 && speed !== 1.0) est /= speed;
  return Math.max(1, Math.trunc(est));
}
