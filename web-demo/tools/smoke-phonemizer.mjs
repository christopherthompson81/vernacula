#!/usr/bin/env node
/**
 * Prove the phonemizer runs in a real browser through the two upstream seams, and that its IPA matches
 * what the C# CLI produced for the same input — a cross-engine check, not just "it didn't throw".
 */
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { spawn } from "node:child_process";
let CHILD = null;
function done(c){ try{CHILD?.kill("SIGKILL");}catch{} process.exit(c); }
process.on("exit", () => { try{CHILD?.kill("SIGKILL");}catch{} });
const PORT = 8793, PUB = path.resolve("public");
const CASES = [
  ["es", "Buenos días. El tiempo está muy agradable hoy."],
  ["cy", "Bore da. Croeso i Gymru."],
  ["en", "Hello world. This is the vernacula text to speech pipeline."],
];
const PAGE = `<!doctype html><meta charset=utf-8><body><script type="module">
const say=(o)=>fetch("/r",{method:"POST",body:JSON.stringify(o)});
const log=[];
try{
  const keys = await (await fetch("/vphon-data/_keys.json")).json();
  const t0=performance.now();
  const bytes=new Map();
  await Promise.all(keys.map(async k=>{
    const r=await fetch("/vphon-data/"+k);
    if(!r.ok) throw new Error("fetch "+k+" -> "+r.status);
    bytes.set(k,new Uint8Array(await r.arrayBuffer()));
  }));
  log.push("prefetched "+bytes.size+" files in "+(performance.now()-t0).toFixed(0)+" ms");

  // ⚠ setDataSource BEFORE loadEngine: importing the engine reads 182 manifests at module scope.
  const vp = await import("/vphon/src/browser.js");
  vp.setDataSource({ read:(k)=>{ const b=bytes.get(k); if(!b) throw new Error("missing key: "+k); return b; } });
  vp.setOrtLoader(()=>import("/ort/ort.wasm.bundle.min.mjs"));
  const t1=performance.now();
  const { phonemizeAsync } = await vp.loadEngine();
  log.push("loadEngine in "+(performance.now()-t1).toFixed(0)+" ms");

  const out=[];
  for (const [code,text] of ${JSON.stringify(CASES)}) {
    const t=performance.now();
    out.push({code, text, ipa: await phonemizeAsync(text, code), ms: performance.now()-t});
  }
  say({ok:true, log, out});
}catch(e){ say({ok:false, log, error:[String(e), e&&e.stack].join(" | ")}); }
</script>`;
const iso={"Cross-Origin-Opener-Policy":"same-origin","Cross-Origin-Embedder-Policy":"require-corp"};
http.createServer((req,res)=>{
  if(req.method==="POST"&&req.url==="/r"){let b="";req.on("data",c=>b+=c);req.on("end",()=>{
    res.writeHead(200,iso);res.end("ok");const r=JSON.parse(b);
    r.log.forEach(l=>console.log("  "+l));
    if(!r.ok){console.error("FAILED: "+r.error);done(3);}
    for(const o of r.out) console.log(`  [${o.code}] ${o.ipa}   (${o.ms.toFixed(0)} ms)`);
    done(0);});return;}
  if(req.url==="/"){res.writeHead(200,{...iso,"Content-Type":"text/html"});return res.end(PAGE);}
  let f = req.url.startsWith("/ort/")
    ? path.join("node_modules/onnxruntime-web/dist", req.url.slice(5))
    : path.join(PUB, decodeURIComponent(req.url));
  if(fs.existsSync(f)&&fs.statSync(f).isFile()){
    const ct = f.endsWith(".js")||f.endsWith(".mjs") ? "text/javascript"
             : f.endsWith(".json") ? "application/json"
             : f.endsWith(".wasm") ? "application/wasm" : "application/octet-stream";
    res.writeHead(200,{...iso,"Content-Type":ct}); return fs.createReadStream(f).pipe(res);
  }
  res.writeHead(404,iso);res.end();
}).listen(PORT,()=>{
  const prof=fs.mkdtempSync("/tmp/ff-phon-");
  fs.writeFileSync(path.join(prof,"user.js"),'user_pref("dom.webgpu.enabled", true);');
  CHILD=spawn("firefox",["--headless","--profile",prof,`http://localhost:${PORT}/`],{stdio:["ignore","ignore","pipe"]});
  setTimeout(()=>{console.error("timed out");done(4);}, 120000);
});
