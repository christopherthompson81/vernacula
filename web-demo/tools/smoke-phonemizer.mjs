#!/usr/bin/env node
/**
 * Prove the phonemizer runs in a real browser through the two upstream seams, and that its IPA matches
 * what the C# CLI produced for the same input — a cross-engine check, not just "it didn't throw".
 */
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { spawn, execFileSync } from "node:child_process";
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
  // ⚠ _keys.json IS AN OBJECT, NOT AN ARRAY: {engine, dirs, languages, foreign}. It became one when
  // the demo moved to 193 languages fetched per language, and this test kept calling keys.map() —
  // so it failed with "keys.map is not a function" before it ever reached the engine, and the
  // cross-engine IPA check it exists for had not run since. Take the engine keys plus the dirs the
  // CASES actually need, which is what src/inference/phonemizer.ts does.
  const manifest = await (await fetch("/vphon-data/_keys.json")).json();
  const want = new Set(manifest.engine);
  for (const [code] of ${JSON.stringify(CASES)})
    for (const d of (manifest.languages[code]?.dirs ?? [])) for (const k of (manifest.dirs[d] ?? [])) want.add(k);
  for (const [code] of ${JSON.stringify(CASES)})
    for (const k of (manifest.languages[code]?.core ?? [])) want.add(k);
  const keys = [...want];
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
    // ⚠ THE CROSS-ENGINE COMPARISON, which this file's header promised and did not do. The C# port
    // and the TS engine are separate implementations of the same rules, and they DID silently
    // diverge: the submodule pin sat six commits behind while the demo shipped the C# build, so
    // en-GB lost a word-initial ɹ and kept the pre-#1252 digraphs, and nothing failed. Printing the
    // browser's IPA cannot catch that — only comparing the two can.
    const CLI = "../src/Vernacula.Tts.CLI/bin/Release/net10.0/vernacula-tts";
    if(!fs.existsSync(CLI)){ console.log("  (C# CLI not built — cross-engine check skipped)"); done(0); }
    let bad = 0;
    for(const o of r.out){
      let cli;
      try {
        cli = execFileSync(CLI, ["--lang", o.code, "--text", o.text, "--print-ipa"], {encoding:"utf8"})
          .split("\n").find(l=>l.startsWith("IPA:"))?.slice(4).trim();
      } catch(e){ console.error(`  [${o.code}] CLI failed: ${e.message.split("\n")[0]}`); bad++; continue; }
      if(cli !== o.ipa.trim()){
        console.error(`  [${o.code}] ENGINES DISAGREE\n      browser: ${o.ipa.trim()}\n      C#     : ${cli}`);
        bad++;
      }
    }
    console.log(bad ? `  cross-engine: ${bad} of ${r.out.length} DISAGREE` : `  cross-engine: ${r.out.length}/${r.out.length} identical (browser TS vs C#)`);
    done(bad ? 5 : 0);});return;}
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
