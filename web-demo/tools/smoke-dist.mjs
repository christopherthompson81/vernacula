#!/usr/bin/env node
/** Verify the BUILT site: the phonemizer still loads from the bundled output (its keys come from
 *  import.meta.url, which a bundler can silently break), and the HuggingFace model URLs are
 *  fetchable cross-origin under the COEP headers netlify.toml sets. */
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { spawn } from "node:child_process";
let CHILD=null; const done=(c)=>{try{CHILD?.kill("SIGKILL");}catch{} process.exit(c);};
process.on("exit",()=>{try{CHILD?.kill("SIGKILL");}catch{}});
const PORT=8795, DIST=path.resolve("dist");
const HF="https://huggingface.co/christopherthompson81/omnivoice-ipa-onnx/resolve/main";
const PAGE=`<!doctype html><meta charset=utf-8><body><script type="module">
const say=(o)=>fetch("/r",{method:"POST",body:JSON.stringify(o)});
const steps=[];
try{
  const keys=await (await fetch("/vphon-data/_keys.json")).json();
  const bytes=new Map();
  await Promise.all(keys.map(async k=>{const r=await fetch("/vphon-data/"+k); bytes.set(k,new Uint8Array(await r.arrayBuffer()));}));
  const vp=await import("/vphon/src/browser.js");
  vp.setDataSource({read:(k)=>{const b=bytes.get(k); if(!b) throw new Error("missing "+k); return b;}});
  const {phonemizeAsync}=await vp.loadEngine();
  steps.push("phonemizer from built site: "+await phonemizeAsync("Hello world.","en"));
  steps.push("crossOriginIsolated: "+self.crossOriginIsolated);
  // Range request, which is what the chunked cache relies on.
  const r=await fetch("${HF}/omnivoice_transformer_ipa.int4.onnx",{headers:{Range:"bytes=0-1023"}});
  steps.push("HF ranged fetch: HTTP "+r.status+", got "+(await r.arrayBuffer()).byteLength+" bytes");
  const v=await fetch("${HF}/voices.json").catch(()=>null);
  say({ok:true,steps});
}catch(e){ say({ok:false,steps,error:[String(e),e&&e.stack].join(" | ")}); }
</script>`;
const iso={"Cross-Origin-Opener-Policy":"same-origin","Cross-Origin-Embedder-Policy":"require-corp"};
http.createServer((req,res)=>{
  if(req.method==="POST"&&req.url==="/r"){let b="";req.on("data",c=>b+=c);req.on("end",()=>{
    res.writeHead(200,iso);res.end("ok");const r=JSON.parse(b);
    r.steps.forEach(s=>console.log("  "+s));
    if(!r.ok){console.error("FAILED: "+r.error);done(3);} console.log("  dist OK");done(0);});return;}
  if(req.url==="/"){res.writeHead(200,{...iso,"Content-Type":"text/html"});return res.end(PAGE);}
  const f=path.join(DIST,decodeURIComponent(req.url.split("?")[0]));
  if(fs.existsSync(f)&&fs.statSync(f).isFile()){
    const ct=/\.(js|mjs)$/.test(f)?"text/javascript":/\.json$/.test(f)?"application/json":/\.wasm$/.test(f)?"application/wasm":"application/octet-stream";
    res.writeHead(200,{...iso,"Content-Type":ct});return fs.createReadStream(f).pipe(res);}
  res.writeHead(404,iso);res.end();
}).listen(PORT,()=>{
  const prof=fs.mkdtempSync("/tmp/ff-dist-");
  fs.writeFileSync(path.join(prof,"user.js"),'user_pref("dom.webgpu.enabled", true);');
  CHILD=spawn("firefox",["--headless","--profile",prof,`http://localhost:${PORT}/`],{stdio:["ignore","ignore","pipe"]});
  setTimeout(()=>{console.error("timed out");done(4);},180000);
});
