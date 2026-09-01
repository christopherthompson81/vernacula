// Fail-fast WebGPU probe: does headless Chrome give us an adapter at all, and does ORT's WebGPU EP
// run a trivial graph? No model download, no 472 MB — seconds, not minutes.
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { spawn } from "node:child_process";

let CHILD = null;
/** Never exit without reaping the browser: a leaked headless process keeps its GPU
 *  allocation, and a handful of them starve the next run at session creation. */
function done(code) { try { CHILD?.kill("SIGKILL"); } catch {} process.exit(code); }
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => done(130));
process.on("exit", () => { try { CHILD?.kill("SIGKILL"); } catch {} });
const DIST = path.resolve("node_modules/onnxruntime-web/dist"); const PORT = 8792;
const TINY = process.argv[2];   // optional: path to a tiny .onnx to try on the webgpu EP
const PAGE = `<!doctype html><meta charset=utf-8><body><script type="module">
const say = (o) => fetch("/r",{method:"POST",body:JSON.stringify(o)});
const steps = [];
try {
  steps.push("navigator.gpu: " + (navigator.gpu ? "present" : "ABSENT"));
  if (!navigator.gpu) throw new Error("no navigator.gpu");
  const ad = await navigator.gpu.requestAdapter();
  steps.push("requestAdapter: " + (ad ? "ok" : "NULL"));
  if (!ad) throw new Error("requestAdapter returned null");
  try { const i = await ad.info; steps.push("adapter.info: " + JSON.stringify({vendor:i?.vendor,arch:i?.architecture,desc:i?.description})); } catch {}
  const dev = await ad.requestDevice(); steps.push("requestDevice: " + (dev ? "ok" : "NULL"));
  // WebGPU grants DEFAULT limits unless you ask for more; the default maxBufferSize is 256 MB
  // however large the adapter maximum. (Concatenation, not template literals: this source is
  // itself inside a Node template literal and inner placeholders would be substituted there.)
  const MB = (x) => (x/1e6).toFixed(0) + " MB";
  steps.push("default device: maxBufferSize=" + MB(dev.limits.maxBufferSize) +
             "  maxStorageBufferBindingSize=" + MB(dev.limits.maxStorageBufferBindingSize));
  steps.push("adapter max:    maxBufferSize=" + MB(ad.limits.maxBufferSize) +
             "  maxStorageBufferBindingSize=" + MB(ad.limits.maxStorageBufferBindingSize));
  try {
    // A GPUAdapter is consumed by requestDevice(); ask for a fresh one before requesting again.
    const ad2 = await navigator.gpu.requestAdapter();
    const big = await ad2.requestDevice({ requiredLimits: {
      maxBufferSize: ad2.limits.maxBufferSize,
      maxStorageBufferBindingSize: ad2.limits.maxStorageBufferBindingSize } });
    steps.push("RAISED device:  maxBufferSize=" + MB(big.limits.maxBufferSize) +
               "  maxStorageBufferBindingSize=" + MB(big.limits.maxStorageBufferBindingSize) +
               "  => cap IS raisable");
  } catch (e) { steps.push("raising limits FAILED: " + e.message); }
  steps.push("f16 supported: " + ad.features.has("shader-f16"));
  ${TINY ? `
  const ort = await import("/dist/ort.webgpu.bundle.min.mjs");
  ort.env.wasm.wasmPaths = "/dist/"; ort.env.logLevel = "error";
  const t = performance.now();
  const s = await ort.InferenceSession.create("/tiny.onnx", {executionProviders:["webgpu"]});
  steps.push("tiny session on webgpu: ok in " + (performance.now()-t).toFixed(0) + " ms");
  const a = new ort.Tensor("float32", Float32Array.from({length:8*256},(_,i)=>(i%7)/7), [8,256]);
  const ids = new ort.Tensor("int64", BigInt64Array.from([1n,2n,3n,4n]), [4]);
  let o = await s.run({A:a, ids:ids});
  const mm = o.mm.data, emb = o.emb.data;
  const fin = (x)=>Array.from(x).every(v=>Number.isFinite(v));
  steps.push("tiny run: ok  mm[0]="+mm[0].toFixed(5)+" finite="+fin(mm)+"  emb[0]="+emb[0].toFixed(5)+" finite="+fin(emb));
  // A 76 KB model has essentially no compute. If a run still costs ~3 s, the cost is a FIXED
  // per-run stall in the WebGPU path, not GPU work — which is what the flat-vs-S curve implies.
  const ts=[]; for (let i=0;i<5;i++){ const t0=performance.now(); await s.run({A:a, ids:ids}); ts.push(performance.now()-t0); }
  steps.push("tiny run times (ms): " + ts.map(x=>x.toFixed(1)).join(", "));
  ` : ""}
  say({ok:true, steps});
} catch(e){ say({ok:false, steps, error:String(e&&e.stack||e)}); }
</script>`;
const iso = {"Cross-Origin-Opener-Policy":"same-origin","Cross-Origin-Embedder-Policy":"require-corp"};
http.createServer((req,res)=>{
  if(req.method==="POST"&&req.url==="/r"){let b="";req.on("data",c=>b+=c);req.on("end",()=>{res.writeHead(200,iso);res.end("ok");
    const r=JSON.parse(b); r.steps.forEach(s=>console.log("  "+s));
    if(!r.ok){console.error("FAILED: "+r.error);done(3);} console.log("WebGPU OK");done(0);});return;}
  if(req.url==="/"){res.writeHead(200,{...iso,"Content-Type":"text/html"});return res.end(PAGE);}
  if(req.url==="/tiny.onnx"&&TINY){res.writeHead(200,{...iso,"Content-Type":"application/octet-stream"});return fs.createReadStream(TINY).pipe(res);}
  if(req.url.startsWith("/dist/")){const f=path.join(DIST,req.url.slice(6));
    if(fs.existsSync(f)){res.writeHead(200,{...iso,"Content-Type":f.endsWith(".wasm")?"application/wasm":"text/javascript"});return fs.createReadStream(f).pipe(res);}}
  res.writeHead(404,iso);res.end();
}).listen(PORT,()=>{
  let c;
  if ((process.env.BROWSER||"chrome") === "firefox") {
    // Firefox needs the pref set in a profile; there is no command-line switch for it.
    const prof = "/tmp/ff-wgpu-probe";
    fs.rmSync(prof, {recursive:true, force:true}); fs.mkdirSync(prof, {recursive:true});
    fs.writeFileSync(path.join(prof,"user.js"),
      ['dom.webgpu.enabled','gfx.webgpu.force-enabled','dom.webgpu.workers.enabled']
        .map(k=>`user_pref("${k}", true);`).join("\n"));
    c = CHILD = spawn(process.env.FIREFOX||"firefox",
      ["--headless","--profile",prof,`http://localhost:${PORT}/`],{stdio:["ignore","ignore","pipe"]});
  } else {
    const EXTRA=(process.env.CHROME_FLAGS||"").split(" ").filter(Boolean);
    // HEADLESS=0 runs on the real display: headless Chrome on Linux+NVIDIA initialises Vulkan
    // differently and falls back to SwiftShader, so a headed run is the control.
    const head = process.env.HEADLESS === "0" ? [] : ["--headless=new"];
    c = CHILD = spawn(process.env.CHROME||"google-chrome",[...head,"--no-sandbox","--disable-gpu-sandbox",
      "--enable-unsafe-webgpu","--no-first-run","--no-default-browser-check","--disable-search-engine-choice-screen","--noerrdialogs",...EXTRA,`--user-data-dir=${process.env.CHROME_PROFILE||"/tmp/chrome-wgpu-probe"}`,`http://localhost:${PORT}/`],
      {stdio:["ignore","ignore","pipe"]});
  }
  c.stderr.on("data",d=>{const s=String(d); if(/gpu|vulkan|webgpu|error/i.test(s)) process.stderr.write("[chrome] "+s);});
  setTimeout(()=>{console.error("probe timed out");done(4);},60000);
});
