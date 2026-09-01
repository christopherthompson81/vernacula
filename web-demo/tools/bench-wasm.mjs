// Measure one transformer forward on the WASM backend, then extrapolate to a generation.
// ORT-web runs in Node on the same wasm build the browser uses, so this is a fair proxy for
// browser CPU throughput (browsers add no per-op overhead; only thread count differs).
import * as ort from "onnxruntime-web/wasm";
import fs from "node:fs";

const MODEL = process.argv[2];
const S = Number(process.argv[3] ?? 200);   // sequence length: style+text+ref+target
const STEPS = Number(process.argv[4] ?? 32);

ort.env.wasm.numThreads = Number(process.env.THREADS ?? 4);
ort.env.logLevel = "error";
// In Node, ORT-web resolves its .wasm/.mjs sidecars with fetch(), which has no base URL here.
// Point it at the local dist directory instead; the browser gets these from a CDN (see ort-init).
const DIST = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url).pathname;
ort.env.wasm.wasmPaths = DIST;

const model = fs.readFileSync(MODEL);
const dataPath = MODEL + ".data";
const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
if (fs.existsSync(dataPath)) {
  opts.externalData = [{ path: dataPath.split("/").pop(), data: fs.readFileSync(dataPath) }];
}

let t = Date.now();
const sess = await ort.InferenceSession.create(model, opts);
console.log(`load: ${((Date.now() - t) / 1000).toFixed(1)}s   threads=${ort.env.wasm.numThreads}`);

const C = 8, B = 2;
const ids = BigInt64Array.from({ length: B * C * S }, (_, i) => BigInt((i * 7 + 3) % 1024));
const amask = Uint8Array.from({ length: B * S }, (_, i) => (i % 2 === 0 ? 1 : 0));
const attn = new Uint8Array(B * 1 * S * S).fill(1);
const feeds = {
  input_ids: new ort.Tensor("int64", ids, [B, C, S]),
  audio_mask: new ort.Tensor("bool", amask, [B, S]),
  attention_mask: new ort.Tensor("bool", attn, [B, 1, S, S]),
};

await sess.run(feeds);                       // warm-up (kernel selection, allocs)
const N = 3, times = [];
for (let i = 0; i < N; i++) { t = Date.now(); await sess.run(feeds); times.push(Date.now() - t); }
const per = times.reduce((a, b) => a + b, 0) / N;
console.log(`forward @S=${S}: ${per.toFixed(0)} ms  (runs: ${times.join(", ")})`);
console.log(`=> ${STEPS} steps ≈ ${((per * STEPS) / 1000).toFixed(1)}s per generation`);
