/**
 * Model manifest and the language list the demo offers.
 *
 * ⚠ The list is now EVERY language the phonemizer routes (193), not the 28 the fine-tune trained
 * on. Generated into `languages.ts` by tools/make-language-catalog.mjs — see it for what "trained"
 * and a donor `voice` mean, and edit tools/data/language-meta.json rather than the output.
 */
export { LANGUAGES, LANGUAGE_BY_CODE, type LanguageOption } from "./languages.ts";

/**
 * Where the ONNX bundle lives. Netlify serves /models/* with immutable caching (netlify.toml);
 * the filename carries the precision so a cached older build cannot be mistaken for a newer one.
 *
 * ⚠ PRECISION IS THE OPEN QUESTION, not a preference. The diffusion loop is precision-sensitive:
 * TF32 produced incoherent noise and fp16 a different-but-valid rendering, both measured in
 * docs/omnivoice_onnx_investigation.md. int8 is 617 MB and pending a listening test; fp16 is
 * ~1.2 GB and was previously listen-confirmed good. Do not treat the small one as the default
 * until it has been heard.
 */
/**
 * Model bundle, served from the public HuggingFace repo (the Parakeet demo uses the same pattern:
 * HF sends `access-control-allow-origin: *` and supports range requests, which is what the chunked
 * cache needs). `tokenizer.json` and `voices.json` are small and ship with the site.
 *
 * ⚠ PRECISION IS NOT A PREFERENCE HERE. The diffusion loop is precision-sensitive: naive INT8
 * dynamic quantization produced output that was not recognizable as speech, because it quantizes
 * ACTIVATIONS and a 32-iteration loop compounds that error. This build is WEIGHT-ONLY int4
 * (MatMulNBits, block 32) with an int8 per-row embedding, listen-confirmed indistinguishable from
 * fp32. Do not swap it for something smaller without a listening test.
 */
const HF = "https://huggingface.co/christopherthompson81/omnivoice-ipa-onnx/resolve/main";

export const MODELS = {
  transformerUrl: `${HF}/omnivoice_transformer_ipa.int4.onnx`,
  transformerDataUrl: `${HF}/omnivoice_transformer_ipa.int4.onnx.data`,
  decoderUrl: `${HF}/higgs_decoder.onnx`,
  tokenizerUrl: `${HF}/tokenizer.json`,
  voicesUrl: "/models/voices.jsonc",
  voiceCodesUrl: "/models/voice-codes.json",
} as const;

/**
 * Diffusion steps.
 *
 * ⚠ 32, NOT 16 — and this is a quality setting, not a performance knob. 16 was tried to halve
 * browser latency and the output came back audibly degraded ("quirks"), on the FP32 model as well
 * as the quantized ones, which is how it was told apart from quantization damage. The same
 * sentence, same voice, same model and same execution provider is clean at 32 and quirky at 16.
 * The desktop CLI has always defaulted to 32; matching it is what makes the browser sound the same.
 */
export const NUM_STEPS = 32;
