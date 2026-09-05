#!/usr/bin/env node
/**
 * End-to-end browser test of the ported pipeline: IPA -> tokenizer -> textPrep -> diffusion ->
 * decoder -> post-processing, and out to a WAV written back to disk for listening.
 *
 * The deterministic halves are checked against the C# CLI by number: `targetTokens` and the
 * conditioning length depend only on the tokenizer, text prep and duration estimator, so if those
 * agree the ports are exact. The diffusion field itself will differ slightly between execution
 * providers, which is why the audio is written out rather than diffed.
 */
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { spawn } from "node:child_process";
let CHILD = null;
const done = (c) => { try { CHILD?.kill("SIGKILL"); } catch {} process.exit(c); };
process.on("exit", () => { try { CHILD?.kill("SIGKILL"); } catch {} });

const PORT = 8794, PUB = path.resolve("public");
const MODEL = process.env.MODEL ?? "/mnt/data/omnivoice_ipa/onnx_web/omnivoice_transformer_ipa.int4.onnx";
const MODEL_NAME = MODEL.split("/").pop();
const MODELS = {
  [`/models/${MODEL_NAME}`]: MODEL,
  [`/models/${MODEL_NAME}.data`]: MODEL + ".data",
  "/models/higgs_decoder.onnx": "/mnt/data/Programming/vernacula/scripts/omnivoice_export/onnx/higgs_decoder.onnx",
  "/models/tokenizer.json": "/mnt/data/models/omnivoice/k2-fsa-OmniVoice/tokenizer.json",
};
const TEXT = process.env.TEXT ?? "The quick brown fox jumps over the lazy dog, and the weather today is remarkably pleasant along the coast.";
const LANG = process.env.LANG_CODE ?? "en";
const STEPS = Number(process.env.STEPS ?? 16);

const PAGE = `<!doctype html><meta charset=utf-8><body><script type="module">
const say=(o)=>fetch("/r",{method:"POST",body:JSON.stringify(o)});
const log=[];
try{
  // 1. phonemize, through the upstream seams
  // The manifest is the demo's: the engine set plus, per language, its dirs (looked up in
  // dirs) and core files, minus its excludes (src/inference/phonemizer.ts). Fetch the engine
  // and this run's language.
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
  log.push("IPA: " + ipa);

  // 2. synthesize
  const mod = await import("/app/omnivoice.js");
  if (${JSON.stringify(process.env.EP ?? null)}) mod.setForcedEp(${JSON.stringify(process.env.EP ?? null)});
  if (${JSON.stringify(process.env.GRAPH_OPT ?? null)}) mod.setGraphOpt(${JSON.stringify(process.env.GRAPH_OPT ?? null)});
  if (${JSON.stringify(process.env.DECODER_EP ?? null)}) mod.setDecoderEp(${JSON.stringify(process.env.DECODER_EP ?? null)});
  const t0 = performance.now();
  const ov = await mod.OmniVoice.load({
    transformerUrl: "/models/${MODEL_NAME}",
    transformerDataUrl: "/models/${MODEL_NAME}.data",
    decoderUrl: "/models/higgs_decoder.onnx",
    tokenizerUrl: "/models/tokenizer.json",
    voicesUrl: "/models/voices.jsonc",
    voiceCodesUrl: "/models/voice-codes.json",
    fetchBytes: async (u) => (await fetch(u)).arrayBuffer(),
    onProgress: (d) => log.push("  " + d),
  });
  log.push("load: " + ((performance.now()-t0)/1000).toFixed(1) + "s   ep=" + ov.backend.ep);

  const voice = ov.voices[0];
  const r = await ov.synthesize(ipa, voice, { numStep: ${STEPS} });
  const a = r.audio;
  let peak=0, sum=0; for (const v of a) { peak=Math.max(peak,Math.abs(v)); sum+=v*v; }
  say({ok:true, log, ipa, ep: ov.backend.ep,
       targetTokens: r.targetTokens, seconds: a.length/r.sampleRate,
       generateMs: r.generateMs, transformerMs: r.transformerMs, hostMs: r.hostMs,
       peak, rms: Math.sqrt(sum/a.length),
       wav: Array.from(new Uint8Array(new Float32Array(a).buffer))});
}catch(e){ say({ok:false, log, error:[String(e), e&&e.stack].join(" | ")}); }
</script>`;

const iso = {"Cross-Origin-Opener-Policy":"same-origin","Cross-Origin-Embedder-Policy":"require-corp"};
http.createServer((req,res)=>{
  if (req.method==="POST" && req.url==="/r") {
    let b=""; req.on("data",c=>b+=c);
    req.on("end",()=>{ res.writeHead(200,iso); res.end("ok");
      const r=JSON.parse(b); r.log.forEach(l=>console.log("  "+l));
      if(!r.ok){ console.error("FAILED: "+r.error); done(3); }
      const f32 = new Float32Array(new Uint8Array(r.wav).buffer);
      const out = process.env.OUT ?? "/tmp/ts_pipeline.wav";
      fs.writeFileSync(out, wav(f32, 24000));
      console.log(`  ep=${r.ep}  targetTokens=${r.targetTokens}  audio=${r.seconds.toFixed(2)}s`);
      console.log(`  generate ${(r.generateMs/1000).toFixed(1)}s  (transformer ${(r.transformerMs/1000).toFixed(1)}s, host ${(r.hostMs/1000).toFixed(1)}s)`);
      console.log(`  peak=${r.peak.toFixed(4)} rms=${r.rms.toFixed(4)}  -> ${out}`);
      done(0); });
    return;
  }
  if (req.url==="/") { res.writeHead(200,{...iso,"Content-Type":"text/html"}); return res.end(PAGE); }
  let f = MODELS[req.url]
    ?? (req.url.startsWith("/ort/") ? path.join("node_modules/onnxruntime-web/dist", req.url.slice(5))
    :  req.url.startsWith("/app/") ? path.join("build-smoke", req.url.slice(5))
    :  path.join(PUB, decodeURIComponent(req.url)));
  if (fs.existsSync(f) && fs.statSync(f).isFile()) {
    const ct = /\.(js|mjs)$/.test(f) ? "text/javascript" : /\.json$/.test(f) ? "application/json"
             : /\.wasm$/.test(f) ? "application/wasm" : "application/octet-stream";
    res.writeHead(200,{...iso,"Content-Type":ct,"Content-Length":fs.statSync(f).size});
    return fs.createReadStream(f).pipe(res);
  }
  res.writeHead(404,iso); res.end();
}).listen(PORT, () => {
  const useChrome = process.env.BROWSER !== "firefox";
  if (useChrome) {
    CHILD = spawn(process.env.CHROME || "google-chrome",
      ["--no-sandbox","--disable-gpu-sandbox","--enable-unsafe-webgpu","--no-first-run",
       "--no-default-browser-check","--disable-search-engine-choice-screen","--noerrdialogs",
       "--ozone-platform=x11","--enable-gpu","--ignore-gpu-blocklist",
       "--enable-features=Vulkan","--use-angle=vulkan",
       "--user-data-dir=/tmp/chrome-wgpu-probe", `http://localhost:${PORT}/`],
      {stdio:["ignore","ignore","pipe"]});
  } else {
    const prof = fs.mkdtempSync("/tmp/ff-tts-");
    fs.writeFileSync(path.join(prof,"user.js"), 'user_pref("dom.webgpu.enabled", true);');
    CHILD = spawn("firefox", ["--headless","--profile",prof,`http://localhost:${PORT}/`], {stdio:["ignore","ignore","pipe"]});
  }
  setTimeout(()=>{ console.error("timed out"); done(4); }, Number(process.env.SMOKE_TIMEOUT ?? 600000));
});

function wav(s, sr) {
  const b = Buffer.alloc(44 + s.length*4);
  b.write("RIFF",0); b.writeUInt32LE(36+s.length*4,4); b.write("WAVE",8);
  b.write("fmt ",12); b.writeUInt32LE(16,16); b.writeUInt16LE(3,20); b.writeUInt16LE(1,22);
  b.writeUInt32LE(sr,24); b.writeUInt32LE(sr*4,28); b.writeUInt16LE(4,32); b.writeUInt16LE(32,34);
  b.write("data",36); b.writeUInt32LE(s.length*4,40);
  Buffer.from(s.buffer, s.byteOffset, s.length*4).copy(b, 44);
  return b;
}
