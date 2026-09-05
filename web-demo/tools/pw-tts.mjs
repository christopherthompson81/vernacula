#!/usr/bin/env node
/**
 * Drive the demo's engine in a REAL Chrome under Playwright and report as it goes.
 *
 * Replaces the http+spawn harness pattern (smoke-tts.mjs) for anything that can hang: this one
 * sees the page's console and errors, prints progress the moment it happens, and gives up on a
 * deadline of its own instead of waiting for a timeout to expire on a page that will never post.
 *
 *   MODEL=<int4.onnx> EP=webgpu|wasm STEPS=32 TEXT="..." LANG_CODE=en DEADLINE=300 \
 *     node tools/pw-tts.mjs
 */
import http from "node:http"; import fs from "node:fs"; import path from "node:path";
import { chromium } from "playwright";

const PORT = 8797, PUB = path.resolve("public");
const MODEL = process.env.MODEL ?? "/mnt/data/omnivoice_ipa/onnx_web/omnivoice_transformer_ipa.int4.onnx";
const NAME = MODEL.split("/").pop();
const FILES = {
  [`/models/${NAME}`]: MODEL, [`/models/${NAME}.data`]: MODEL + ".data",
  "/models/higgs_decoder.onnx": "/mnt/data/Programming/vernacula/scripts/omnivoice_export/onnx/higgs_decoder.onnx",
  "/models/tokenizer.json": "/mnt/data/models/omnivoice/k2-fsa-OmniVoice/tokenizer.json",
};
const TEXT = process.env.TEXT ?? "The quick brown fox jumps over the lazy dog, and the weather today is remarkably pleasant.";
const LANG = process.env.LANG_CODE ?? "en", STEPS = Number(process.env.STEPS ?? 32);
const EP = process.env.EP ?? null, DEADLINE = Number(process.env.DEADLINE ?? 300) * 1000;
const PROFILE = process.env.PROFILE === "1", OUT = process.env.OUT ?? "/tmp/pw_tts.wav";

const PAGE = `<!doctype html><meta charset=utf-8><body><script type="module">
const p = (m) => console.log("progress: " + m);
try {
  const m = await (await fetch("/vphon-data/_keys.json")).json();
  const entry = m.languages[${JSON.stringify(LANG)}] ?? { dirs: [], core: [], exclude: [] };
  const skip = new Set(entry.exclude ?? []);
  const keys = [...m.engine, ...entry.dirs.flatMap((d) => m.dirs[d] ?? []), ...(entry.core ?? [])].filter((k) => !skip.has(k));
  const bytes = new Map();
  await Promise.all(keys.map(async k => { const r = await fetch("/vphon-data/"+k); bytes.set(k, new Uint8Array(await r.arrayBuffer())); }));
  const vp = await import("/vphon/src/browser.js");
  vp.setDataSource({ read:(k)=>{ const b=bytes.get(k); if(!b) throw new Error("missing key "+k); return b; } });
  vp.setOrtLoader(()=>import("/ort/ort.wasm.bundle.min.mjs"));
  const { phonemizeAsync } = await vp.loadEngine();
  const ipa = (await phonemizeAsync(${JSON.stringify(TEXT)}, ${JSON.stringify(LANG)})).trim();
  p("ipa: " + ipa);
  const mod = await import("/app/omnivoice.js");
  if (${JSON.stringify(EP)}) mod.setForcedEp(${JSON.stringify(EP)});
  const prof = new Map();
  if (${PROFILE}) {
    const ortMod = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.all.bundle.min.mjs");
    ortMod.env.webgpu.profiling = { mode: "default", ondata: (d) => {
      const e = prof.get(d.kernelType) ?? { n: 0, ms: 0 }; e.n++; e.ms += (d.endTime - d.startTime) / 1e6; prof.set(d.kernelType, e); } };
  }
  { // diagnostics: what the adapter offers, and which device ORT ends up on
    const ad = await navigator.gpu?.requestAdapter();
    p("adapter features: " + (ad ? [...ad.features].join(" ") : "none"));
  }
  const t0 = performance.now();
  const ov = await mod.OmniVoice.load({
    transformerUrl: "/models/${NAME}", transformerDataUrl: "/models/${NAME}.data",
    decoderUrl: "/models/higgs_decoder.onnx", tokenizerUrl: "/models/tokenizer.json",
    voicesUrl: "/models/voices.jsonc", voiceCodesUrl: "/models/voice-codes.json",
    fetchBytes: async (u) => (await fetch(u)).arrayBuffer(), onProgress: p,
  });
  p("load " + ((performance.now()-t0)/1000).toFixed(1) + "s ep=" + ov.backend.ep);
  { const ortMod = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.all.bundle.min.mjs");
    const d = ortMod.env.webgpu.device; p("ORT device features: " + (d ? [...d.features].join(" ") : "(no device exposed)")); }
  const voice = mod.voiceFor(ov.voices, ${JSON.stringify(LANG)});
  const r = await ov.synthesize(ipa, voice, { numStep: ${STEPS} }, (s, t) => { if (s === 1 || s % 8 === 0 || s === t) p("step " + s + "/" + t); });
  const a = r.audio; let peak = 0, sum = 0; for (const v of a) { peak = Math.max(peak, Math.abs(v)); sum += v*v; }
  window.__result = { ok: true, ep: ov.backend.ep, targetTokens: r.targetTokens, seconds: a.length / r.sampleRate,
    generateMs: r.generateMs, transformerMs: r.transformerMs, hostMs: r.hostMs, peak, rms: Math.sqrt(sum / a.length),
    profile: prof.size ? [...prof.entries()].map(([k, v]) => ({ k, n: v.n, ms: v.ms })) : null,
    wav: Array.from(new Uint8Array(new Float32Array(a).buffer)) };
} catch (e) { window.__result = { ok: false, error: String(e && e.stack || e) }; }
</script>`;

const iso = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
const server = http.createServer((req, res) => {
  if (req.url === "/") { res.writeHead(200, { ...iso, "Content-Type": "text/html" }); return res.end(PAGE); }
  const f = FILES[req.url]
    ?? (req.url.startsWith("/ort/") ? path.join("node_modules/onnxruntime-web/dist", req.url.slice(5))
    :  req.url.startsWith("/app/") ? path.join("build-smoke", req.url.slice(5))
    :  path.join(PUB, decodeURIComponent(req.url)));
  if (fs.existsSync(f) && fs.statSync(f).isFile()) {
    const ct = /\.(js|mjs)$/.test(f) ? "text/javascript" : /\.wasm$/.test(f) ? "application/wasm" : /\.json$/.test(f) ? "application/json" : "application/octet-stream";
    res.writeHead(200, { ...iso, "Content-Type": ct, "Content-Length": fs.statSync(f).size });
    return fs.createReadStream(f).pipe(res);
  }
  res.writeHead(404, iso); res.end();
}).listen(PORT);

const T0 = Date.now(); const at = () => `[${((Date.now() - T0) / 1000).toFixed(1)}s]`;
const browser = await chromium.launch({
  executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: false,
  args: ["--no-sandbox", "--disable-gpu-sandbox", "--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check",
    "--disable-search-engine-choice-screen", "--noerrdialogs", "--ozone-platform=x11", "--enable-gpu",
    "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan"],
  env: { ...process.env, DISPLAY: process.env.DISPLAY || ":0" },
});
const page = await browser.newPage();
page.on("console", (m) => { const t = m.text(); if (t.startsWith("progress: ")) console.log(`${at()} ${t.slice(10)}`); else if (m.type() === "error" || m.type() === "warning") console.log(`${at()} console.${m.type()}: ${t.slice(0, 300)}`); });
page.on("pageerror", (e) => console.log(`${at()} pageerror: ${String(e).slice(0, 300)}`));
page.on("crash", () => { console.log(`${at()} PAGE CRASHED`); finish(5); });
async function finish(code) { try { await browser.close(); } catch {} server.close(); process.exit(code); }
await page.goto(`http://localhost:${PORT}/`);
let r = null;
while (Date.now() - T0 < DEADLINE) {
  r = await page.evaluate(() => window.__result ?? null).catch(() => null);
  if (r) break;
  await new Promise((f) => setTimeout(f, 1000));
}
if (!r) { console.log(`${at()} DEADLINE (${DEADLINE / 1000}s) with no result — hung`); await finish(4); }
if (!r.ok) { console.log(`${at()} FAILED: ${r.error}`); await finish(3); }
const f32 = new Float32Array(new Uint8Array(r.wav).buffer);
const wav = (s, sr) => { const b = Buffer.alloc(44 + s.length * 4); b.write("RIFF", 0); b.writeUInt32LE(36 + s.length * 4, 4); b.write("WAVEfmt ", 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(3, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 4, 28); b.writeUInt16LE(4, 32); b.writeUInt16LE(32, 34); b.write("data", 36); b.writeUInt32LE(s.length * 4, 40); Buffer.from(s.buffer).copy(b, 44); return b; };
fs.writeFileSync(OUT, wav(f32, 24000));
console.log(`${at()} ep=${r.ep} targetTokens=${r.targetTokens} audio=${r.seconds.toFixed(2)}s  generate ${(r.generateMs / 1000).toFixed(1)}s (transformer ${(r.transformerMs / 1000).toFixed(1)}s, host ${(r.hostMs / 1000).toFixed(1)}s)  peak=${r.peak.toFixed(3)} rms=${r.rms.toFixed(4)} -> ${OUT}`);
if (r.profile) {
  const ks = r.profile.sort((a, b) => b.ms - a.ms), total = ks.reduce((a, k) => a + k.ms, 0), count = ks.reduce((a, k) => a + k.n, 0);
  console.log(`  profile per forward: ${Math.round(count / STEPS)} kernels, ${(total / STEPS).toFixed(0)} ms GPU (wall ${(r.transformerMs / STEPS).toFixed(0)} ms)`);
  for (const k of ks.slice(0, 8)) console.log(`    ${(k.ms / STEPS).toFixed(1).padStart(8)} ms  ${String(Math.round(k.n / STEPS)).padStart(5)}×  ${k.k}`);
}
await finish(0);
