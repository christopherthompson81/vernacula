/**
 * ONNX session plumbing plus the end-to-end pipeline: IPA -> tokens -> waveform.
 *
 * The TypeScript counterpart of `Vernacula.Tts.Base.OmniVoice` + `OmniVoiceTts`. Only two graphs ship
 * to the browser — the (already IPA-merged, quantized) transformer and the Higgs decoder. The
 * 654 MB Higgs ENCODER does not: it exists only to turn a reference WAV into codec codes, and those
 * codes are a few KB, so they are precomputed offline and fetched as `voices.json`.
 *
 * ⚠ GENERATION IS ALWAYS VOICE-CLONED, and that is a correctness decision rather than a feature
 * choice. With no reference, input under ~5 s is outside the fine-tune's distribution (corpus median
 * 12 s, 0.21% under 3 s) and can emit noise rather than degrade — and a demo receives short phrases
 * almost exclusively. Shipping precomputed reference codes fixes that AND drops 654 MB.
 */
import type * as ort from "onnxruntime-web";
import { getOrt, type Ort, pickExecutionProvider, type Ep } from "./ortInit.ts";
import { Qwen3Tokenizer } from "./qwen3Tokenizer.ts";
import { addPunctuation, prepare, NUM_CODEBOOKS } from "./textPrep.ts";
import { estimateTargetTokens } from "./duration.ts";
import { runDiffusion, DEFAULT_CONFIG, type GenConfig } from "./diffusion.ts";
import { removeSilence, fadeAndPad, peakNormalize } from "./audioPost.ts";
import { attributeWords, type WordTiming } from "./alignment.ts";

export const SAMPLE_RATE = 24000;

export interface VoiceSource {
  dataset: string; lang: string; file: string; split: string | null;
  sentenceId: string | null; gender: string | null; durationS: number;
  candidateIndex: number; text: string | null;
}

export interface Voice {
  id: string;
  label: string;
  /** Demo language code this reference is a NATIVE speaker of. */
  lang: string;
  /** Preferred voice for its language when several are listed. */
  default?: boolean;
  /** Which FLEURS clip this is — so a noisy exemplar can be traced and swapped. */
  source?: VoiceSource;
  /** Reference transcript, ALREADY IPA — it is fed to the model alongside the target IPA. */
  refIpa: string;
  /** Pre-encoded codec codes [8, refLen], row-major. */
  codes: number[];
  refLen: number;
  /** RMS of the reference waveform before encoding, for the output un-boost (Python parity). */
  refRms: number;
  /** Speaker sex, from a median-F0 measurement of the decoded reference (tools/measure-voice-gender.mjs).
   *  Absent where the pitch sat in the ambiguous 155-185 Hz band. A listener's correction overrides it —
   *  edit voices.jsonc directly; nothing re-derives this field. */
  sex?: "M" | "F";
}

export interface Backend { ep: Ep; }

/** Force an execution provider (tests/debugging). Unset = auto. */
// The EP choice lives in ortInit.ts, because ORT's proxy setting depends on it and must be made
// before the first session; re-exported so the smoke tools keep their hooks.
export { forcedEp, setForcedEp, pickExecutionProvider } from "./ortInit.ts";

/**
 * Execution provider for the DECODER only.
 *
 * ⚠ DEFAULTS TO WASM EVEN WHEN THE TRANSFORMER IS ON WebGPU, and that is a correctness fix rather
 * than a preference. Running the Higgs codec decoder on WebGPU garbles the audio — measured, with
 * everything else held fixed: transformer WebGPU + decoder WebGPU gave peak 0.394 / rms 0.055 and
 * was audibly bad, while transformer WebGPU + decoder WASM gave peak 0.530 / rms 0.066, byte-for-
 * byte the same figures as an all-WASM run, which was listen-confirmed perfect.
 *
 * It is NOT the transformer: one forward pass with identical inputs agrees between the two
 * providers to max |Δlogit| 2e-4 with 100.000% argmax agreement. The transformer is attention over
 * a quantized MatMulNBits graph; the decoder is a convolutional DAC, and ORT's WebGPU conv kernels
 * are what diverge.
 *
 * The cost is negligible: the transformer runs 64 times per generation (32 steps x 2 CFG passes)
 * and the decoder once, so this buys correctness for about 1 s out of 20.
 */
export let decoderEp: Ep | undefined = "wasm";
export function setDecoderEp(ep: Ep | undefined) { decoderEp = ep; }

/** Graph optimization level override (tests/debugging). Fusions change numerics, and this loop is
 *  precision-sensitive — the C# CUDA path disables TF32 for the same reason. */
type GraphOpt = NonNullable<ort.InferenceSession.SessionOptions["graphOptimizationLevel"]>;
export let graphOpt: GraphOpt = "all";
export function setGraphOpt(l: GraphOpt) { graphOpt = l; }

export interface LoadOptions {
  transformerUrl: string;
  transformerDataUrl?: string;
  decoderUrl: string;
  tokenizerUrl: string;
  /** JSONC metadata (hand-editable). */
  voicesUrl: string;
  /** The code arrays, keyed by voice id — kept out of the JSONC so it stays scannable. */
  voiceCodesUrl: string;
  /** Fetch a URL as bytes, e.g. through a chunked/resumable cache. */
  fetchBytes: (url: string, label: string) => Promise<ArrayBuffer>;
  onProgress?: (detail: string) => void;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json() as Promise<T>;
}

/**
 * Parse JSONC — JSON plus `//` and block comments.
 *
 * ⚠ String-aware, deliberately. A regex that strips `//` anywhere would cut into any string
 * containing one, and these entries carry free text (FLEURS transcripts) and IPA. Escapes are
 * honoured so a `\"` inside a string does not end it.
 */
export function parseJsonc<T>(text: string): T {
  let out = "", inStr = false, esc = false, line = false, block = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i], n = text[i + 1];
    if (line) { if (c === "\n") { line = false; out += c; } continue; }
    if (block) { if (c === "*" && n === "/") { block = false; i++; } continue; }
    if (inStr) { out += c; if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') { inStr = true; out += c; continue; }
    if (c === "/" && n === "/") { line = true; i++; continue; }
    if (c === "/" && n === "*") { block = true; i++; continue; }
    out += c;
  }
  return JSON.parse(out) as T;
}

async function loadVoicesJsonc(url: string): Promise<Voice[]> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return parseJsonc<Voice[]>(await r.text());
}

export class OmniVoice {
  private constructor(
    private readonly ort: Ort,
    private readonly transformer: ort.InferenceSession,
    private readonly decoder: ort.InferenceSession,
    readonly tokenizer: Qwen3Tokenizer,
    readonly voices: Voice[],
    private readonly codes: Record<string, number[]>,
    readonly backend: Backend,
  ) {}

  static async load(o: LoadOptions): Promise<OmniVoice> {
    const ORT = await getOrt();
    const ep = await pickExecutionProvider();
    if (ep === "webgpu") await useMaxLimitsDevice();
    o.onProgress?.(`execution provider: ${ep}`);

    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: [ep],
      graphOptimizationLevel: graphOpt,
    };

    o.onProgress?.("downloading transformer");
    const tBytes = await o.fetchBytes(o.transformerUrl, "transformer");
    if (o.transformerDataUrl) {
      // The .onnx records its sidecar's filename internally, so `path` must be that exact name.
      const name = o.transformerDataUrl.split("/").pop()!;
      const data = await o.fetchBytes(o.transformerDataUrl, "transformer weights");
      opts.externalData = [{ path: name, data: new Uint8Array(data) }];
    }
    o.onProgress?.("creating transformer session");
    const transformer = await ORT.InferenceSession.create(tBytes, opts);

    o.onProgress?.("downloading decoder");
    const dBytes = await o.fetchBytes(o.decoderUrl, "decoder");
    const dEp = decoderEp ?? ep;
    o.onProgress?.(`decoder provider: ${dEp}`);
    const decoder = await ORT.InferenceSession.create(dBytes, { executionProviders: [dEp], graphOptimizationLevel: graphOpt });

    const tokenizer = await Qwen3Tokenizer.load(o.tokenizerUrl);
    const [voices, codes] = await Promise.all([
      loadVoicesJsonc(o.voicesUrl),
      fetchJson<Record<string, number[]>>(o.voiceCodesUrl),
    ]);
    return new OmniVoice(ORT, transformer, decoder, tokenizer, voices, codes, { ep });
  }

  private codesFor(v: Voice): number[] {
    const c = this.codes[v.id];
    // A voice whose codes are missing would otherwise be silently treated as no-reference, which
    // is the regime that emits noise on short input. Fail loudly instead.
    if (!c) throw new Error(`voice "${v.id}" has no codes in voice-codes.json`);
    return c;
  }

  /** IPA string -> 24 kHz mono waveform. `ipa` and `voice.refIpa` must BOTH be IPA. */
  async synthesize(ipa: string, voice: Voice, cfg: Partial<GenConfig> = {},
                   onStep?: (step: number, total: number) => void) {
    const config: GenConfig = { ...DEFAULT_CONFIG, ...cfg };

    // Parity with create_voice_clone_prompt: the punctuated reference feeds BOTH the duration
    // estimate and the combined text.
    const refIpa = addPunctuation(voice.refIpa);
    // ⚠ Estimated on the IPA, both sides — see duration.ts.
    const target = estimateTargetTokens(ipa, refIpa, voice.refLen);

    const cond = prepare(this.tokenizer, ipa, target, refIpa,
                         Int32Array.from(this.codesFor(voice)), voice.refLen,
                         null,          // language: null is the IPA fine-tune's conditioning
                         null, config.denoise);

    const t0 = performance.now();
    const { tokens, targetLen, transformerMs, hostMs } = await runDiffusion(
      cond, config, (ids, am, attn, seq) => this.runTransformer(ids, am, attn, seq), onStep);

    let audio = await this.decode(tokens, targetLen);
    const generateMs = performance.now() - t0;

    // Python `_post_process_audio` order: remove silence -> volume -> fade + pad. Volume: un-boost
    // a reference that was boosted for being quiet; otherwise peak-normalise.
    // ⚠ DELIBERATE DEVIATION FROM PYTHON'S POST-CHAIN, in both gain AND ORDER. Reasons, measured:
    //
    // 1. Cloning copies the reference's LOUDNESS, and the corpus references span rms 0.0017-0.099,
    //    a 58x spread. Python's volume step only un-boosts a reference that IT boosted at encode
    //    time; ours were never boosted, so applying it undoes something that never happened —
    //    German came out at 17% of level.
    // 2. Normalising must come BEFORE silence removal. Oromo's reference is rms 0.0017, so its
    //    output sat entirely below the -50 dBFS silence threshold and removeSilence deleted the
    //    whole utterance: 0.0 s of audio.
    // 3. And again AFTER the fade, because the fine-tune emits a leading transient inside the first
    //    0.1 s. Normalising before the fade lets that transient take the headroom and the fade then
    //    removes it — German measured peak 0.038 with a single pre-fade normalise.
    //
    // The desktop CLI keeps Python's behaviour exactly; this is a demo-only choice, made because a
    // demo with silent languages is worse than one that is not bit-faithful to the post-chain.
    peakNormalize(audio, 0.5);
    audio = removeSilence(audio, SAMPLE_RATE, 500, 100, 100);
    audio = fadeAndPad(audio, SAMPLE_RATE);
    peakNormalize(audio, 0.5);

    // Word timings in FINAL-audio seconds: word shares placed on the speech-energy envelope of
    // the finished audio — see alignment.ts for what "estimated" means here.
    const words: WordTiming[] = attributeWords(ipa, audio, SAMPLE_RATE);

    return { audio, sampleRate: SAMPLE_RATE, words, targetTokens: target, generateMs, transformerMs, hostMs };
  }

  private async runTransformer(ids: BigInt64Array, audioMask: Uint8Array,
                               attn: Uint8Array, seq: number): Promise<Float32Array> {
    // ⚠ COPY EVERY INPUT, DO NOT HAND OVER A LIVE BUFFER. The diffusion loop mutates input_ids in
    // place every step and would otherwise pass the same backing array each time. The C# port hit
    // exactly this on CUDA: a bound tensor uploads to the device ONCE, and in-place mutation
    // between runs is ignored, so every step silently re-ran on the step-0 all-mask input
    // (docs/omnivoice_onnx_investigation.md, "IO-binding root cause"). A fresh copy per call costs
    // a few hundred KB and removes the whole class of bug.
    //
    // ⚠ `attention_mask` USED TO BE THE EXCEPTION, on the reasoning that it is never mutated. That
    // held until `env.wasm.proxy` moved the session into a worker: ORT TRANSFERS input buffers
    // across, which DETACHES them, so the second step threw "attempting to access detached
    // ArrayBuffer" and every generation died after step 1. Not-mutated was never the property that
    // mattered; not-detached is. seq² bytes per step is nothing beside a forward pass.
    const feeds = {
      input_ids: new this.ort.Tensor("int64", ids.slice(), [1, NUM_CODEBOOKS, seq]),
      audio_mask: new this.ort.Tensor("bool", audioMask.slice(), [1, seq]),
      attention_mask: new this.ort.Tensor("bool", attn.slice(), [1, 1, seq, seq]),
    };
    const out = await this.transformer.run(feeds);
    return out.logits.data as Float32Array;
  }

  private async decode(tokens: Int32Array, tc: number): Promise<Float32Array> {
    const codes = new BigInt64Array(tokens.length);
    for (let i = 0; i < tokens.length; i++) codes[i] = BigInt(tokens[i]);
    const out = await this.decoder.run({
      audio_codes: new this.ort.Tensor("int64", codes, [1, NUM_CODEBOOKS, tc]),
    });
    return out.audio_values.data as Float32Array;
  }
}

/**
 * Fallbacks for languages the fine-tune corpus has no speaker for. Chosen for phonetic proximity
 * rather than alphabetically: an Icelandic sentence read by a Swedish voice is a far better
 * demonstration than one read by an English voice, because voice cloning is ACOUSTIC and the
 * reference carries the speaker's accent along with their timbre.
 */
/** Voices available for a language, preferred first. */
export function voicesFor(voices: Voice[], lang: string): Voice[] {
  const own = voices.filter((v) => v.lang === lang);
  return own.sort((a, b) => Number(b.default ?? false) - Number(a.default ?? false));
}

/** The reference voice for a language. Every offered language now has a NATIVE exemplar — the
 *  earlier phonetic-proximity fallback (is -> sv, it -> es) is gone because FLEURS has both. */
export function voiceFor(voices: Voice[], lang: string, id?: string): Voice {
  const own = voicesFor(voices, lang);
  const byId = id ? own.find((v) => v.id === id) : undefined;
  return byId ?? own[0] ?? voices.find((v) => v.lang === "en") ?? voices[0];
}

/**
 * WebGPU where it works, WASM otherwise — and the difference is not cosmetic. Measured on an
 * RTX 3090 at 16 steps: WebGPU/Chrome 177 ms per forward (~2.8 s per phrase) against 1295 ms on
 * 8-thread WASM (~20.7 s). Firefox's WebGPU was SLOWER than WASM and flat in sequence length, so
 * probing for an adapter is not enough on its own — the UI reports which path it got.
 */
/**
 * Hand ORT a device built with the adapter's MAXIMUM limits.
 *
 * ⚠ WebGPU's `requestDevice()` grants DEFAULT limits — maxBufferSize 268 MB — however large the
 * adapter's maximum, and ORT takes the default. A model with any single tensor above that kills the
 * device with "Out of memory" on a GPU with tens of GB free. This build's largest tensor is 155 MB
 * so it clears the default, but that is luck rather than design, and it forecloses larger models.
 */
interface AdapterLike {
  limits: { maxBufferSize: number; maxStorageBufferBindingSize: number };
  requestDevice(d?: { requiredLimits?: Record<string, number> }): Promise<unknown>;
}

async function useMaxLimitsDevice(): Promise<void> {
  try {
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<AdapterLike | null> } }).gpu;
    const adapter = await gpu?.requestAdapter();
    if (!adapter) return;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    ((await getOrt()).env.webgpu as unknown as { device?: unknown }).device = device;
  } catch { /* fall back to ORT's own default-limits device */ }
}

