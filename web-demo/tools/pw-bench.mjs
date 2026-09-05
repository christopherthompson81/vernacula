#!/usr/bin/env node
/**
 * One transformer forward, N times, in a real Chrome under Playwright — the per-forward instrument
 * for kernel and overhead experiments. Session options come from SESSION_OPTS (JSON) so a knob can
 * be A/B'd without editing anything:
 *
 *   MODEL=<.onnx> S=350 B=2 ITERS=6 SESSION_OPTS='{"enableGraphCapture":true}' node tools/pw-bench.mjs
 *
 * ⚠ bench-webgpu.mjs measured 3× slower than the demo on what should have been the same GPU; its
 * adapter reported different limits. This one launches Chrome exactly as pw-tts.mjs does, which
 * matches the demo's numbers.
 */
import http from "node:http"; import fs from "node:fs"; import path from "node:path";
import { chromium } from "playwright";
const PORT = 8798;
const MODEL = process.env.MODEL ?? "/mnt/data/omnivoice_ipa/onnx_web/omnivoice_transformer_ipa.int4.onnx";
const NAME = MODEL.split("/").pop();
const S = Number(process.env.S ?? 350), B = Number(process.env.B ?? 2);
const WARM = Number(process.env.WARM ?? 2), ITERS = Number(process.env.ITERS ?? 6);
const OPTS = process.env.SESSION_OPTS ?? "{}", EP = process.env.EP ?? "webgpu";
const DEADLINE = Number(process.env.DEADLINE ?? 240) * 1000;
const FEATURES_ONLY = process.env.FEATURES_ONLY === "1";     // print the adapter's features and stop
const EXTRA_FLAGS = (process.env.EXTRA_FLAGS ?? "").split(" ").filter(Boolean);
const VERBOSE = process.env.VERBOSE === "1";                 // echo every console line (ORT placement logs)

const PAGE = `<!doctype html><meta charset=utf-8><body><script type="module">
const p = (m) => console.log("progress: " + m);
try {
  { const ad = await navigator.gpu?.requestAdapter(); p("adapter features: " + (ad ? [...ad.features].sort().join(" ") : "none"));
    if (${FEATURES_ONLY}) { window.__result = { ok: true, warm: [0], times: [0] }; throw new Error("__stop"); } }
  const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.all.bundle.min.mjs");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
  ort.env.logLevel = "error"; ort.env.wasm.numThreads = 8;
  const model = await (await fetch("/m/${NAME}")).arrayBuffer();
  const data = new Uint8Array(await (await fetch("/m/${NAME}.data")).arrayBuffer());
  const extra = ${OPTS};
  let t = performance.now();
  const sess = await ort.InferenceSession.create(model, { executionProviders: [${JSON.stringify(EP)}], graphOptimizationLevel: "all",
    externalData: [{ path: "${NAME}.data", data }], ...extra });
  p("session " + ((performance.now() - t) / 1000).toFixed(1) + "s  opts=" + JSON.stringify(extra));
  const B = ${B}, C = 8, S = ${S};
  const ids = BigInt64Array.from({ length: B * C * S }, (_, i) => BigInt((i * 7 + 3) % 1024));
  const am = Uint8Array.from({ length: B * S }, (_, i) => i % 2 === 0 ? 1 : 0);
  const at = new Uint8Array(B * S * S).fill(1);
  const feeds = () => ({ input_ids: new ort.Tensor("int64", ids, [B, C, S]), audio_mask: new ort.Tensor("bool", am, [B, S]),
    attention_mask: new ort.Tensor("bool", at, [B, 1, S, S]) });
  const warm = [], times = [];
  for (let i = 0; i < ${WARM}; i++) { t = performance.now(); const r = await sess.run(feeds()); warm.push(performance.now() - t); r.logits.dispose?.(); }
  for (let i = 0; i < ${ITERS}; i++) { t = performance.now(); const r = await sess.run(feeds()); times.push(performance.now() - t); r.logits.dispose?.(); }
  window.__result = { ok: true, warm, times };
} catch (e) { if (!String(e).includes("__stop")) window.__result = { ok: false, error: String(e && e.stack || e) }; }
</script>`;
const iso = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
const server = http.createServer((req, res) => {
  if (req.url === "/") { res.writeHead(200, { ...iso, "Content-Type": "text/html" }); return res.end(PAGE); }
  const f = req.url === `/m/${NAME}` ? MODEL : req.url === `/m/${NAME}.data` ? MODEL + ".data" : null;
  if (f && fs.existsSync(f)) { res.writeHead(200, { ...iso, "Content-Type": "application/octet-stream", "Content-Length": fs.statSync(f).size }); return fs.createReadStream(f).pipe(res); }
  res.writeHead(404, iso); res.end();
}).listen(PORT);
const T0 = Date.now(); const at = () => `[${((Date.now() - T0) / 1000).toFixed(1)}s]`;
const browser = await chromium.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: false,
  args: ["--no-sandbox", "--disable-gpu-sandbox", "--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check",
    "--disable-search-engine-choice-screen", "--noerrdialogs", "--ozone-platform=x11", "--enable-gpu", "--ignore-gpu-blocklist",
    "--enable-features=Vulkan", "--use-angle=vulkan", ...EXTRA_FLAGS], env: { ...process.env, DISPLAY: process.env.DISPLAY || ":0" } });
const page = await browser.newPage();
page.on("console", (m) => { const t = m.text(); if (t.startsWith("progress: ")) console.log(`${at()} ${t.slice(10)}`); else if (VERBOSE) console.log(`${at()} ${m.type()}: ${t.slice(0, 300)}`); else if (m.type() === "error" && !/VerifyEachNode|404/.test(t)) console.log(`${at()} console.error: ${t.slice(0, 240)}`); });
page.on("pageerror", (e) => console.log(`${at()} pageerror: ${String(e).slice(0, 240)}`));
async function finish(code) { try { await browser.close(); } catch {} server.close(); process.exit(code); }
await page.goto(`http://localhost:${PORT}/`);
let r = null;
while (Date.now() - T0 < DEADLINE) { r = await page.evaluate(() => window.__result ?? null).catch(() => null); if (r) break; await new Promise((f) => setTimeout(f, 500)); }
if (!r) { console.log(`${at()} DEADLINE — hung`); await finish(4); }
if (!r.ok) { console.log(`${at()} FAILED: ${r.error.slice(0, 400)}`); await finish(3); }
const mean = r.times.reduce((a, x) => a + x, 0) / r.times.length;
console.log(`${at()} S=${S} B=${B}  warm ${r.warm.map((x) => x.toFixed(0)).join(",")}  forward mean ${mean.toFixed(0)} ms  min ${Math.min(...r.times).toFixed(0)}  (${r.times.map((x) => x.toFixed(0)).join(", ")})`);
await finish(0);
