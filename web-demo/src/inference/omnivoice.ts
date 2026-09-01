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
import * as ort from "onnxruntime-web";
import { initOrt } from "./ortInit.ts";
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
    private readonly transformer: ort.InferenceSession,
    private readonly decoder: ort.InferenceSession,
    readonly tokenizer: Qwen3Tokenizer,
    readonly voices: Voice[],
    readonly backend: Backend,
  ) {}

  static async load(o: LoadOptions): Promise<OmniVoice> {
    initOrt();
    const ep = await pickExecutionProvider();
    o.onProgress?.(`execution provider: ${ep}`);

    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: [ep],
      graphOptimizationLevel: "all",
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
    const transformer = await ort.InferenceSession.create(tBytes, opts);

    o.onProgress?.("downloading decoder");
    const dBytes = await o.fetchBytes(o.decoderUrl, "decoder");
    const decoder = await ort.InferenceSession.create(dBytes, { executionProviders: [ep], graphOptimizationLevel: "all" });

    const tokenizer = await Qwen3Tokenizer.load(o.tokenizerUrl);
    const voices: Voice[] = await (await fetch(o.voicesUrl)).json();
    return new OmniVoice(transformer, decoder, tokenizer, voices, { ep });
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
    const feeds = {
      input_ids: new ort.Tensor("int64", ids, [1, NUM_CODEBOOKS, seq]),
      audio_mask: new ort.Tensor("bool", audioMask, [1, seq]),
      attention_mask: new ort.Tensor("bool", attn, [1, 1, seq, seq]),
    };
    const out = await this.transformer.run(feeds);
    return out.logits.data as Float32Array;
  }

  private async decode(tokens: Int32Array, tc: number): Promise<Float32Array> {
    const codes = new BigInt64Array(tokens.length);
    for (let i = 0; i < tokens.length; i++) codes[i] = BigInt(tokens[i]);
    const out = await this.decoder.run({
      audio_codes: new ort.Tensor("int64", codes, [1, NUM_CODEBOOKS, tc]),
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
export async function pickExecutionProvider(): Promise<"webgpu" | "wasm"> {
  try {
    const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (gpu && (await gpu.requestAdapter())) return "webgpu";
  } catch { /* fall through to wasm */ }
  return "wasm";
}
