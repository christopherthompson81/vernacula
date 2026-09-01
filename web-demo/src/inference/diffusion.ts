/**
 * The greedy iterative-unmasking loop — the TypeScript port of
 * `Chatterbox.Base.OmniVoiceTts.RunDiffusion`.
 *
 * Each step runs the transformer twice (classifier-free guidance), scores every masked
 * (codebook, position) slot, and commits the top-k most confident. Ported faithfully: the
 * schedule, the tie-break, and the layer penalty all change which tokens get committed, and the
 * loop is chaotic — one flipped token diverges the whole field into a different rendering.
 */
import { AUDIO_MASK_ID, AUDIO_VOCAB_SIZE, NUM_CODEBOOKS, type Prepared } from "./textPrep.ts";

const C = NUM_CODEBOOKS, V = AUDIO_VOCAB_SIZE, MASK = AUDIO_MASK_ID;

export interface GenConfig {
  numStep: number;
  guidanceScale: number;
  tShift: number;
  layerPenaltyFactor: number;
  denoise: boolean;
}

/** Defaults match OmniVoice's upstream config except the temperatures, which are 0 — generation is
 *  greedy and therefore DETERMINISTIC. Re-running a failure reproduces it exactly; it is not a
 *  bad sample to re-roll. */
export const DEFAULT_CONFIG: GenConfig = {
  numStep: 16, guidanceScale: 2.0, tShift: 0.1, layerPenaltyFactor: 5.0, denoise: true,
};

/** Runs the transformer once: input_ids [1,8,S] + audio_mask [1,S] + attention_mask [1,1,S,S]
 *  -> logits [1,8,S,1025] flattened. */
export type TransformerFn = (
  ids: BigInt64Array, audioMask: Uint8Array, attn: Uint8Array, seqLen: number,
) => Promise<Float32Array>;

/** `_get_time_steps`: shifted linspace. */
function timeSteps(numStep: number, tShift: number): number[] {
  const ts: number[] = [];
  for (let i = 0; i <= numStep; i++) {
    const lin = i / numStep;
    ts.push((tShift * lin) / (1 + (tShift - 1) * lin));
  }
  return ts;
}

/** Per-step unmask counts: ceil(total·Δt) clamped to the remaining budget; the last step takes
 *  whatever is left, so every slot is always committed by the end. */
function schedule(totalMask: number, numStep: number, ts: number[]): number[] {
  const out: number[] = [];
  let rem = totalMask;
  for (let s = 0; s < numStep; s++) {
    const n = s === numStep - 1 ? rem : Math.min(Math.ceil(totalMask * (ts[s + 1] - ts[s])), rem);
    out.push(n);
    rem -= n;
  }
  return out;
}

function logSoftmaxInto(src: Float32Array, off: number, dst: Float64Array): void {
  let max = -Infinity;
  for (let v = 0; v < V; v++) { const x = src[off + v]; if (x > max) max = x; }
  let sum = 0;
  for (let v = 0; v < V; v++) sum += Math.exp(src[off + v] - max);
  const lse = max + Math.log(sum);
  for (let v = 0; v < V; v++) dst[v] = src[off + v] - lse;
}

function logSoftmaxInPlace(x: Float64Array): void {
  let max = -Infinity;
  for (let v = 0; v < x.length; v++) if (x[v] > max) max = x[v];
  let sum = 0;
  for (let v = 0; v < x.length; v++) sum += Math.exp(x[v] - max);
  const lse = max + Math.log(sum);
  for (let v = 0; v < x.length; v++) x[v] -= lse;
}

export interface DiffusionResult {
  /** [8, T] row-major. */
  tokens: Int32Array;
  targetLen: number;
  transformerMs: number;
  hostMs: number;
}

export async function runDiffusion(
  cond: Prepared, cfg: GenConfig, runTransformer: TransformerFn,
  onStep?: (step: number, total: number) => void,
): Promise<DiffusionResult> {
  const condLen = cond.total, T = cond.targetLen;
  const targetStart = condLen - T; // = textLen + refLen

  // CFG needs two forwards per step. Upstream batches them as [2,8,condLen] by padding the
  // unconditional row out to condLen with block-diagonal attention — but that pad is discarded,
  // so both engines instead run two B=1 passes at their natural lengths (condLen and T) and skip
  // the wasted uncond compute.
  const condIds = cond.inputIds.slice();
  const condAmask = cond.audioMask.slice();
  const condAttn = new Uint8Array(condLen * condLen).fill(1);

  const uncondIds = new BigInt64Array(C * T).fill(BigInt(MASK));
  const uncondAmask = new Uint8Array(T).fill(1);
  const uncondAttn = new Uint8Array(T * T).fill(1);

  const tokens = new Int32Array(C * T).fill(MASK);
  const sched = schedule(T * C, cfg.numStep, timeSteps(cfg.numStep, cfg.tShift));

  const pred = new Int32Array(C * T);
  const score = new Float64Array(C * T);
  const cl = new Float64Array(V), ul = new Float64Array(V), comb = new Float64Array(V);
  const order = new Int32Array(C * T);

  let transformerMs = 0, hostMs = 0;

  for (let step = 0; step < cfg.numStep; step++) {
    let t0 = performance.now();
    const cLogits = await runTransformer(condIds, condAmask, condAttn, condLen);
    const uLogits = await runTransformer(uncondIds, uncondAmask, uncondAttn, T);
    transformerMs += performance.now() - t0;

    t0 = performance.now();
    // ⚠ The C# parallelises this across (codebook, position) — it is the host hot path, three
    // 1025-way softmaxes per slot. JS is single-threaded here, so it runs sequentially; the
    // buffers above are reused to keep it off the allocator.
    for (let cb = 0; cb < C; cb++) {
      for (let t = 0; t < T; t++) {
        const cOff = (cb * condLen + (targetStart + t)) * V;
        const uOff = (cb * T + t) * V;
        logSoftmaxInto(cLogits, cOff, cl);
        logSoftmaxInto(uLogits, uOff, ul);
        // combined = log_softmax(cl + g·(cl − ul)) = log_softmax((1+g)·cl − g·ul)
        for (let v = 0; v < V; v++) comb[v] = (1 + cfg.guidanceScale) * cl[v] - cfg.guidanceScale * ul[v];
        logSoftmaxInPlace(comb);
        comb[MASK] = -Infinity;
        let best = 0, bestVal = -Infinity;
        for (let v = 0; v < V; v++) if (comb[v] > bestVal) { bestVal = comb[v]; best = v; }
        const i = cb * T + t;
        pred[i] = best;
        // Already-committed slots are excluded; later codebooks are penalised so the coarse
        // layers commit first.
        score[i] = tokens[i] !== MASK ? -Infinity : bestVal - cb * cfg.layerPenaltyFactor;
      }
    }

    const k = sched[step];
    if (k > 0) {
      // Top-k by score, ties broken by flat index cb*T+t ASCENDING — matching the row-major
      // flatten Python's topk operates on. A different tie-break commits different tokens.
      for (let i = 0; i < C * T; i++) order[i] = i;
      // ⚠ COMPARE, DO NOT SUBTRACT. Committed slots carry -Infinity, and `score[b] - score[a]`
      // is NaN when both are -Infinity — a comparator returning NaN leaves the sort order
      // undefined, so the top-k picks arbitrary slots and the token field degrades. The C#
      // uses CompareTo, which orders infinities correctly.
      const ord = Array.from(order).sort((a, b) => {
        if (score[a] !== score[b]) return score[a] > score[b] ? -1 : 1;   // descending by score
        return a - b;                                                     // ties: flat index ascending
      });
      for (let i = 0; i < k && i < ord.length; i++) tokens[ord[i]] = pred[ord[i]];

      // Write the whole field back into both passes for the next step.
      for (let cb = 0; cb < C; cb++)
        for (let t = 0; t < T; t++) {
          const tk = BigInt(tokens[cb * T + t]);
          condIds[cb * condLen + (targetStart + t)] = tk;
          uncondIds[cb * T + t] = tk;
        }
    }
    hostMs += performance.now() - t0;
    onStep?.(step + 1, cfg.numStep);
  }

  return { tokens, targetLen: T, transformerMs, hostMs };
}
