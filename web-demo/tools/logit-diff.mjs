#!/usr/bin/env node
/**
 * Run ONE transformer forward with identical inputs on WASM and on WebGPU, and compare the logits.
 *
 * The diffusion loop is chaotic, so "the audio differs" cannot say WHERE the divergence starts.
 * A single forward pass can: if the logits already differ materially, the kernels disagree and no
 * amount of loop tuning will fix it.
 */
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { spawn } from "node:child_process";
let CHILD=null; const done=(c)=>{try{CHILD?.kill("SIGKILL");}catch{} process.exit(c);};
process.on("exit",()=>{try{CHILD?.kill("SIGKILL");}catch{}});
const PORT=8796;
const MODEL = process.env.MODEL ?? "/mnt/data/omnivoice_ipa/onnx_web/omnivoice_transformer_ipa.int4.onnx";
const NAME = MODEL.split("/").pop();
const S = Number(process.env.S ?? 120);
const PAGE=`<!doctype html><meta charset=utf-8><body><script type="module">
const say=(o)=>fetch("/r",{method:"POST",body:JSON.stringify(o)});
try{
  const ort=await import("/ort/ort.all.bundle.min.mjs");
  ort.env.wasm.wasmPaths="/ort/"; ort.env.logLevel="error";
  const model=await (await fetch("/m/${NAME}")).arrayBuffer();
  const data=new Uint8Array(await (await fetch("/m/${NAME}.data")).arrayBuffer());
  const B=1,C=8,S=${S};
  const ids=BigInt64Array.from({length:B*C*S},(_,i)=>BigInt((i*7+3)%1024));
  const am=Uint8Array.from({length:B*S},(_,i)=>i%2===0?1:0);
  const at=new Uint8Array(B*S*S).fill(1);
  const feeds=()=>({input_ids:new ort.Tensor("int64",ids,[B,C,S]),
                    audio_mask:new ort.Tensor("bool",am,[B,S]),
                    attention_mask:new ort.Tensor("bool",at,[B,1,S,S])});
  const out={};
  for (const ep of ["wasm","webgpu"]) {
    const s=await ort.InferenceSession.create(model,{executionProviders:[ep],graphOptimizationLevel:"all",
      externalData:[{path:"${NAME}.data",data}]});
    const r=await s.run(feeds());
    out[ep]=Array.from(r.logits.data);
  }
  const a=out.wasm,b=out.webgpu;
  let maxAbs=0,sum=0,agree=0,n=0;
  const V=1025;
  for(let i=0;i<a.length;i++){const d=Math.abs(a[i]-b[i]); if(d>maxAbs)maxAbs=d; sum+=d;}
  for(let p=0;p<a.length/V;p++){let am2=0,bm=0,av=-1e30,bv=-1e30;
    for(let v=0;v<V;v++){const i=p*V+v; if(a[i]>av){av=a[i];am2=v;} if(b[i]>bv){bv=b[i];bm=v;}}
    if(am2===bm)agree++; n++;}
  say({ok:true,len:a.length,maxAbs,meanAbs:sum/a.length,argmaxAgree:agree/n});
}catch(e){ say({ok:false,error:[String(e),e&&e.stack].join(" | ")}); }
</script>`;
const iso={"Cross-Origin-Opener-Policy":"same-origin","Cross-Origin-Embedder-Policy":"require-corp"};
http.createServer((q,res)=>{
  if(q.method==="POST"&&q.url==="/r"){let b="";q.on("data",c=>b+=c);q.on("end",()=>{
    res.writeHead(200,iso);res.end("ok");const r=JSON.parse(b);
    if(!r.ok){console.error("FAILED: "+r.error);done(3);}
    console.log(`  logits: ${r.len} values`);
    console.log(`  max |wasm-webgpu| = ${r.maxAbs.toFixed(4)}   mean = ${r.meanAbs.toFixed(5)}`);
    console.log(`  argmax agreement  = ${(r.argmaxAgree*100).toFixed(3)}%`);
    done(0);});return;}
  if(q.url==="/"){res.writeHead(200,{...iso,"Content-Type":"text/html"});return res.end(PAGE);}
  let f = q.url===`/m/${NAME}` ? MODEL : q.url===`/m/${NAME}.data` ? MODEL+".data"
        : q.url.startsWith("/ort/") ? path.join("node_modules/onnxruntime-web/dist",q.url.slice(5)) : null;
  if(f&&fs.existsSync(f)){const ct=/\.(js|mjs)$/.test(f)?"text/javascript":/\.wasm$/.test(f)?"application/wasm":"application/octet-stream";
    res.writeHead(200,{...iso,"Content-Type":ct});return fs.createReadStream(f).pipe(res);}
  res.writeHead(404,iso);res.end();
}).listen(PORT,()=>{
  CHILD=spawn(process.env.CHROME||"google-chrome",
    ["--no-sandbox","--disable-gpu-sandbox","--enable-unsafe-webgpu","--no-first-run",
     "--no-default-browser-check","--disable-search-engine-choice-screen","--noerrdialogs",
     "--ozone-platform=x11","--enable-gpu","--ignore-gpu-blocklist",
     "--enable-features=Vulkan","--use-angle=vulkan",
     "--user-data-dir=/tmp/chrome-wgpu-probe",`http://localhost:${PORT}/`],{stdio:["ignore","ignore","pipe"]});
  setTimeout(()=>{console.error("timed out");done(4);},600000);
});
