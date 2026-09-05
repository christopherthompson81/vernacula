// Measure a transformer forward on the WebGPU EP, in headless Chrome.
//
// ORT's WebGPU backend cannot be driven from Node (no navigator.gpu), so this serves a bench page
// over localhost with the cross-origin-isolation headers ORT needs, launches headless Chrome at it,
// and collects the timing the page posts back. Same model file the WASM bench uses, so the two
// numbers are directly comparable.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

let CHILD = null;
/** Never exit without reaping the browser: a leaked headless process keeps its GPU
 *  allocation, and a handful of them starve the next run at session creation. */
function done(code) { try { CHILD?.kill("SIGKILL"); } catch {} process.exit(code); }
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => done(130));
process.on("exit", () => { try { CHILD?.kill("SIGKILL"); } catch {} });

const MODEL = process.argv[2];
const S = Number(process.argv[3] ?? 100);
const STEPS = Number(process.argv[4] ?? 16);
const WARM = Number(process.env.WARM ?? 3);
const ITERS = Number(process.env.ITERS ?? 10);
const RAISE = process.env.RAISE === "1" ? "true" : "false";
// PROFILE=1: ORT's WebGPU kernel profiler — per-kernel GPU time from timestamp queries — so a
// forward's wall time can be split into GPU kernel time and everything else (dispatch, readback,
// host). That split is the difference between "fuse the graph" and "make the kernels faster".
const PROFILE = process.env.PROFILE === "1" ? "true" : "false";
const DIST = path.resolve("node_modules/onnxruntime-web/dist");
const PORT = 8791;
const DATA_NAME = MODEL.split("/").pop() + ".data";

const PAGE = `<!doctype html><meta charset=utf-8><body><script type="module">
import * as ort from "/dist/ort.webgpu.bundle.min.mjs";
const say = (o) => fetch("/result", {method:"POST", body: JSON.stringify(o)});
try {
  if (!navigator.gpu) throw new Error("navigator.gpu absent");
  const ad = await navigator.gpu.requestAdapter();
  const ai = ad ? (ad.info ?? (ad.requestAdapterInfo ? await ad.requestAdapterInfo() : null)) : null;
  const info = ad ? (ai ? \`\${ai.vendor} \${ai.architecture} \${ai.device||""} \${ai.description||""}\` : "adapter ok")
    + (ad.isFallbackAdapter ? " [FALLBACK/software]" : "") + " ts=" + ad.features.has("timestamp-query") : "no adapter";
  const lim = ad.limits;
  await say({note:"limits", maxBuffer: lim.maxBufferSize, maxStorageBinding: lim.maxStorageBufferBindingSize});
  // ⚠ WebGPU grants DEFAULT limits (maxBufferSize 268 MB) unless you ask for more, whatever the
  // adapter advertises. ORT takes the default, so any single tensor above that -- the 621 MB fp32
  // embedding, or 310 MB at fp16 -- kills the device with "Out of memory" on a card with 23 GB
  // free. Build the device ourselves with the adapter's maximum and hand it to ORT.
  const raise = ${RAISE};
  // The profiler needs the device created WITH timestamp-query, or it reports nothing at all.
  const feats = (${PROFILE} && ad.features.has("timestamp-query")) ? ["timestamp-query"] : [];
  const dev0 = await ad.requestDevice(raise ? { requiredFeatures: feats, requiredLimits: {
      maxBufferSize: ad.limits.maxBufferSize,
      maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } } : { requiredFeatures: feats });
  dev0.lost.then(i => say({ok:false, error: "GPU DEVICE LOST: reason=" + i.reason + " msg=" + i.message}));
  if (raise) ort.env.webgpu.device = dev0;
  await say({note:"device", maxBuffer: dev0.limits.maxBufferSize, raised: raise});
  ort.env.wasm.wasmPaths = "/dist/";
  ort.env.logLevel = "error";
  // Profiling is configured BEFORE the session exists: the backend reads it at initialisation.
  const prof = new Map();
  if (${PROFILE}) {
    ort.env.webgpu.profiling = { mode: "default", ondata: (d) => {
      const e = prof.get(d.kernelType) ?? { n: 0, ms: 0 };
      e.n++; e.ms += (d.endTime - d.startTime) / 1e6; prof.set(d.kernelType, e);
    } };
  }
  let t = performance.now();
  const sess = await ort.InferenceSession.create("/model.onnx", {
    executionProviders: ["webgpu"], graphOptimizationLevel: "all",
    externalData: [{ path: "${DATA_NAME}", data: "/${DATA_NAME}" }],
  });
  const load = performance.now() - t;
  const B=2, C=8;
  const ids = BigInt64Array.from({length:B*C*${S}}, (_,i)=>BigInt((i*7+3)%1024));
  const am = Uint8Array.from({length:B*${S}}, (_,i)=> i%2===0?1:0);
  const at = new Uint8Array(B*${S}*${S}).fill(1);
  const feeds = {
    input_ids: new ort.Tensor("int64", ids, [B,C,${S}]),
    audio_mask: new ort.Tensor("bool", am, [B,${S}]),
    attention_mask: new ort.Tensor("bool", at, [B,1,${S},${S}]),
  };
  // Report warm-up runs separately: the first includes shader compilation and pipeline creation,
  // and any downward drift across a longer series is the fixed overhead amortizing.
  const warm=[];
  for (let i=0;i<${WARM};i++){ t=performance.now(); await sess.run(feeds); warm.push(performance.now()-t); }
  prof.clear(); let profRuns = 0;   // the warm-ups are not counted
  const times=[];
  for (let i=0;i<${ITERS};i++){ t=performance.now(); await sess.run(feeds); times.push(performance.now()-t); profRuns++; }
  if (${PROFILE}) ort.env.webgpu.profiling = { mode: "" };
  say({ok:true, info, load, warm, times, S:${S}, steps:${STEPS},
       profile: ${PROFILE} ? { runs: profRuns, kernels: [...prof.entries()].map(([k,v]) => ({ k, n: v.n, ms: v.ms })) } : null});
} catch (e) {
  let d;
  try { d = JSON.stringify(e, Object.getOwnPropertyNames(Object(e))); } catch { d = ""; }
  say({ok:false, error: [String(e), e && e.name, e && e.message, d, e && e.stack].filter(Boolean).join(" | ") || "(empty error object)"});
}
</script>`;

const server = http.createServer((req, res) => {
  const iso = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
  if (req.method === "POST" && req.url === "/result") {
    let b = ""; req.on("data", c => b += c);
    req.on("end", () => {
      res.writeHead(200, iso); res.end("ok");
      const r = JSON.parse(b);
      if (r.note === "device") {
        console.log(`device in use: maxBufferSize=${(r.maxBuffer/1e6).toFixed(0)} MB  (raised=${r.raised})`);
        return;
      }
      if (r.note === "limits") {
        console.log(`adapter limits: maxBufferSize=${(r.maxBuffer/1e6).toFixed(0)} MB  maxStorageBufferBindingSize=${(r.maxStorageBinding/1e6).toFixed(0)} MB`);
        return;
      }
      if (!r.ok) { console.error("WebGPU bench FAILED:\n" + r.error); done(3); }
      const per = r.times.reduce((a, x) => a + x, 0) / r.times.length;
      const best = Math.min(...r.times);
      console.log(`warm-up: ${r.warm.map(x => x.toFixed(0)).join(", ")} ms`);
      console.log(`steady (${r.times.length}): min ${best.toFixed(0)}  mean ${per.toFixed(0)} ms`);
      console.log(`adapter: ${r.info}`);
      console.log(`load: ${(r.load / 1000).toFixed(1)}s`);
      console.log(`forward @S=${r.S}: ${per.toFixed(0)} ms  (runs: ${r.times.map(x => x.toFixed(0)).join(", ")})`);
      console.log(`=> ${r.steps} steps ≈ ${(per * r.steps / 1000).toFixed(1)}s per generation`);
      if (r.profile) {
        const ks = r.profile.kernels.map(k => ({ ...k, n: k.n / r.profile.runs, ms: k.ms / r.profile.runs }))
          .sort((a, b) => b.ms - a.ms);
        const total = ks.reduce((a, k) => a + k.ms, 0), count = ks.reduce((a, k) => a + k.n, 0);
        console.log(`profile: ${count.toFixed(0)} kernels/forward, ${total.toFixed(0)} ms GPU time/forward `
          + `(wall ${per.toFixed(0)} ms → ${(per - total).toFixed(0)} ms not in kernels)`);
        for (const k of ks.slice(0, 14)) console.log(`  ${k.ms.toFixed(1).padStart(8)} ms  ${String(k.n.toFixed(0)).padStart(5)}×  ${k.k}`);
      }
      done(0);
    });
    return;
  }
  const send = (p, type) => {
    res.writeHead(200, { ...iso, "Content-Type": type, "Content-Length": fs.statSync(p).size });
    fs.createReadStream(p).pipe(res);
  };
  if (req.url === "/") { res.writeHead(200, { ...iso, "Content-Type": "text/html" }); return res.end(PAGE); }
  if (req.url === "/model.onnx") return send(MODEL, "application/octet-stream");
  if (req.url === "/" + DATA_NAME) return send(MODEL + ".data", "application/octet-stream");
  if (req.url.startsWith("/dist/")) {
    const f = path.join(DIST, req.url.slice(6));
    if (fs.existsSync(f)) return send(f, f.endsWith(".wasm") ? "application/wasm" : "text/javascript");
  }
  res.writeHead(404, iso); res.end("nope");
});

server.listen(PORT, () => {
  // Firefox is the one that gives a real adapter headlessly on this box; Chrome headless returns
  // NULL from requestAdapter (its GPU process segfaults) or falls back to SwiftShader.
  let browser;
  if ((process.env.BROWSER || "firefox") === "firefox") {
    // Fresh profile per run; mkdtemp rather than reusing a path, because Firefox leaves IndexedDB
    // dirs that rmSync trips over (ENOTEMPTY) on the next run.
    const prof = fs.mkdtempSync("/tmp/ff-wgpu-bench-");
    fs.writeFileSync(path.join(prof, "user.js"),
      ['dom.webgpu.enabled', 'gfx.webgpu.force-enabled', 'dom.webgpu.workers.enabled']
        .map(k => `user_pref("${k}", true);`).join("\n"));
    browser = CHILD = spawn(process.env.FIREFOX || "firefox",
      ["--headless", "--profile", prof, `http://localhost:${PORT}/`], { stdio: ["ignore", "ignore", "pipe"] });
  } else {
    const EXTRA = (process.env.CHROME_FLAGS || "--enable-features=Vulkan").split(" ").filter(Boolean);
    // ⚠ HEADED + X11 + Vulkan is what gets the real GPU on Linux/NVIDIA. Headless Chrome (any
    // version, 131 and 152 both tested) returns a SwiftShader adapter or none at all; the same
    // flags with a window on DISPLAY=:0 return vendor "nvidia". Suppress the first-run UI or the
    // page never navigates.
    const head = process.env.HEADLESS === "0" ? [] : ["--headless=new"];
    browser = CHILD = spawn(process.env.CHROME || "google-chrome", [
      ...head, "--no-sandbox", "--disable-gpu-sandbox", "--enable-unsafe-webgpu",
      "--no-first-run", "--no-default-browser-check", "--disable-search-engine-choice-screen",
      "--noerrdialogs", ...EXTRA,
      `--user-data-dir=${process.env.CHROME_PROFILE || "/tmp/chrome-webgpu-bench"}`,
      `http://localhost:${PORT}/`,
    ], { stdio: ["ignore", "ignore", "pipe"] });
  }
  browser.stderr.on("data", d => { const s = String(d); if (/error|fail/i.test(s)) process.stderr.write(s); });
  setTimeout(() => { console.error("timed out waiting for the page"); done(4); }, Number(process.env.BENCH_TIMEOUT||600000));
});
