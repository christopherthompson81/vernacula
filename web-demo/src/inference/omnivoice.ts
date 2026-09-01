/**
 * ONNX session plumbing plus the end-to-end pipeline: IPA -> tokens -> waveform.
 *
 * The TypeScript counterpart of `Chatterbox.Base.OmniVoice` + `OmniVoiceTts`. Only two graphs ship
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
import { getOrt, type Ort } from "./ortInit.ts";
import { Qwen3Tokenizer } from "./qwen3Tokenizer.ts";
import { addPunctuation, prepare, NUM_CODEBOOKS } from "./textPrep.ts";
import { estimateTargetTokens } from "./duration.ts";
import { runDiffusion, DEFAULT_CONFIG, type GenConfig } from "./diffusion.ts";
import { removeSilence, fadeAndPad, peakNormalize, scale } from "./audioPost.ts";

export const SAMPLE_RATE = 24000;

export interface Voice {
  id: string;
  label: string;
  /** Reference transcript, ALREADY IPA — it is fed to the model alongside the target IPA. */
  refIpa: string;
  /** Pre-encoded codec codes [8, refLen], row-major. */
  codes: number[];
  refLen: number;
  /** RMS of the reference waveform before encoding, for the output un-boost (Python parity). */
  refRms: number;
}

export interface Backend { ep: "webgpu" | "wasm"; }

/** Force an execution provider (tests/debugging). Unset = auto. */
export let forcedEp: "webgpu" | "wasm" | undefined;
export function setForcedEp(ep: "webgpu" | "wasm" | undefined) { forcedEp = ep; }

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
export let decoderEp: "webgpu" | "wasm" | undefined = "wasm";
export function setDecoderEp(ep: "webgpu" | "wasm" | undefined) { decoderEp = ep; }

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
  voicesUrl: string;
  /** Fetch a URL as bytes, e.g. through a chunked/resumable cache. */
  fetchBytes: (url: string, label: string) => Promise<ArrayBuffer>;
  onProgress?: (detail: string) => void;
}

export class OmniVoice {
  private constructor(
    private readonly ort: Ort,
    private readonly transformer: ort.InferenceSession,
    private readonly decoder: ort.InferenceSession,
    readonly tokenizer: Qwen3Tokenizer,
    readonly voices: Voice[],
    readonly backend: Backend,
  ) {}

  static async load(o: LoadOptions): Promise<OmniVoice> {
    const ORT = await getOrt();
    const ep = forcedEp ?? await pickExecutionProvider();
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
    const voices: Voice[] = await (await fetch(o.voicesUrl)).json();
    return new OmniVoice(ORT, transformer, decoder, tokenizer, voices, { ep });
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
                         Int32Array.from(voice.codes), voice.refLen,
                         null,          // language: null is the IPA fine-tune's conditioning
                         null, config.denoise);

    const t0 = performance.now();
    const { tokens, targetLen, transformerMs, hostMs } = await runDiffusion(
      cond, config, (ids, am, attn, seq) => this.runTransformer(ids, am, attn, seq), onStep);

    let audio = await this.decode(tokens, targetLen);
    const generateMs = performance.now() - t0;

    // Python `_post_process_audio` order: remove silence -> volume -> fade + pad. Volume: un-boost
    // a reference that was boosted for being quiet; otherwise peak-normalise.
    audio = removeSilence(audio, SAMPLE_RATE, 500, 100, 100);
    if (voice.refRms > 0 && voice.refRms < 0.1) scale(audio, voice.refRms / 0.1);
    else peakNormalize(audio, 0.5);
    audio = fadeAndPad(audio, SAMPLE_RATE);

    return { audio, sampleRate: SAMPLE_RATE, targetTokens: target, generateMs, transformerMs, hostMs };
  }

  private async runTransformer(ids: BigInt64Array, audioMask: Uint8Array,
                               attn: Uint8Array, seq: number): Promise<Float32Array> {
    // ⚠ COPY THE INPUT, DO NOT HAND OVER THE LIVE BUFFER. The diffusion loop mutates input_ids in
    // place every step and would otherwise pass the same backing array each time. The C# port hit
    // exactly this on CUDA: a bound tensor uploads to the device ONCE, and in-place mutation
    // between runs is ignored, so every step silently re-ran on the step-0 all-mask input
    // (docs/omnivoice_onnx_investigation.md, "IO-binding root cause"). A fresh copy per call costs
    // a few hundred KB and removes the whole class of bug.
    const feeds = {
      input_ids: new this.ort.Tensor("int64", ids.slice(), [1, NUM_CODEBOOKS, seq]),
      audio_mask: new this.ort.Tensor("bool", audioMask.slice(), [1, seq]),
      attention_mask: new this.ort.Tensor("bool", attn, [1, 1, seq, seq]),
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

export async function pickExecutionProvider(): Promise<"webgpu" | "wasm"> {
  try {
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (gpu && (await gpu.requestAdapter())) return "webgpu";
  } catch { /* fall through to wasm */ }
  return "wasm";
}
