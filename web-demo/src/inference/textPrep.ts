/**
 * Builds the diffusion transformer's conditioning input — the TypeScript port of
 * `Chatterbox.Base.OmniVoiceTextPrep`, itself a port of OmniVoice's `_combine_text` /
 * `_prepare_inference_inputs`.
 *
 * input_ids [8, total] = style ++ text ++ [ref audio codes] ++ masked target. Style and text
 * positions carry the same text-token id across all 8 codebook rows (the graph reads row 0);
 * audio positions carry per-codebook codes (ref) or the mask id (target).
 */
import type { Qwen3Tokenizer } from "./qwen3Tokenizer.ts";

export const NUM_CODEBOOKS = 8;
export const AUDIO_VOCAB_SIZE = 1025;
export const AUDIO_MASK_ID = 1024;

export interface Prepared {
  /** [8, total], row-major. */
  inputIds: BigInt64Array;
  audioMask: Uint8Array;
  textLen: number;
  refLen: number;
  targetLen: number;
  total: number;
}

/** The style prefix: [&lt;|denoise|&gt;] &lt;|lang_start|&gt;LANG&lt;|lang_end|&gt;
 *  &lt;|instruct_start|&gt;INSTRUCT&lt;|instruct_end|&gt;. denoise only when cloning. */
export function buildStyleText(lang: string | null, instruct: string | null,
                               denoise: boolean, hasRef: boolean): string {
  let s = denoise && hasRef ? "<|denoise|>" : "";
  s += `<|lang_start|>${lang ? lang : "None"}<|lang_end|>`;
  s += `<|instruct_start|>${instruct ? instruct : "None"}<|instruct_end|>`;
  return s;
}

/** Port of `_combine_text`: join ref+target, strip newlines, normalise full-width parens and
 *  space runs, and drop spaces adjacent to CJK. */
export function combineText(text: string, refText: string | null): string {
  let full = refText ? `${refText.trim()} ${text.trim()}` : text.trim();
  full = full.replace(/[\r\n]+/g, "");
  full = full.replace(/（/g, "(").replace(/）/g, ")");
  full = full.replace(/[ \t]+/g, " ");
  full = full.replace(/(?<=[一-鿿])\s+|\s+(?=[一-鿿])/gu, "");
  return full;
}

// End-of-sentence punctuation (utils/text.py END_PUNCTUATION).
const END_PUNCT = new Set([
  ";", ":", ",", ".", "!", "?", "…", ")", "]", "}", '"', "'", "“", "”",
  "‘", "’", "；", "：", "，", "。", "！", "？", "、", "）", "】",
]);

/**
 * Port of `add_punctuation`: append a period (or 。 when the text contains CJK) unless it already
 * ends in sentence punctuation. Python applies this to the REFERENCE transcript inside
 * `create_voice_clone_prompt`, so the punctuated form feeds both the duration estimate and the
 * combined text — do not skip it on one side only.
 */
export function addPunctuation(text: string): string {
  const t = text.trim();
  if (!t) return t;
  if (END_PUNCT.has(t[t.length - 1])) return t;
  const isChinese = /[一-鿿]/u.test(t);
  return t + (isChinese ? "。" : ".");
}

export function prepare(
  tok: Qwen3Tokenizer,
  text: string,
  numTargetTokens: number,
  refText: string | null,
  refCodes: Int32Array | null,   // [8, refLen] row-major
  refLen: number,
  lang: string | null,
  instruct: string | null,
  denoise: boolean,
): Prepared {
  const styleIds = tok.encode(buildStyleText(lang, instruct, denoise, refCodes !== null));
  const textIds = tok.encodeWithNonverbalTags(`<|text_start|>${combineText(text, refText)}<|text_end|>`);

  const textLen = styleIds.length + textIds.length;
  const rLen = refCodes ? refLen : 0;
  const total = textLen + rLen + numTargetTokens;

  const ids = new BigInt64Array(NUM_CODEBOOKS * total);
  for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
    let col = cb * total;
    for (const id of styleIds) ids[col++] = BigInt(id);
    for (const id of textIds) ids[col++] = BigInt(id);
  }
  if (refCodes) {
    for (let cb = 0; cb < NUM_CODEBOOKS; cb++)
      for (let t = 0; t < rLen; t++)
        ids[cb * total + textLen + t] = BigInt(refCodes[cb * rLen + t]);
  }
  for (let cb = 0; cb < NUM_CODEBOOKS; cb++)
    for (let t = 0; t < numTargetTokens; t++)
      ids[cb * total + textLen + rLen + t] = BigInt(AUDIO_MASK_ID);

  const audioMask = new Uint8Array(total);
  audioMask.fill(1, textLen);

  return { inputIds: ids, audioMask, textLen, refLen: rLen, targetLen: numTargetTokens, total };
}
