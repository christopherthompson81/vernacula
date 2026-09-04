/**
 * Load onnxruntime-web at RUNTIME from a CDN, and hand out the one module instance.
 *
 * ⚠ ORT MUST NOT BE BUNDLED INTO THE APP CHUNK. It spawns a Web Worker for its WASM backend, and
 * the worker loads whatever module ORT itself came from. Bundled by Vite, that is the app chunk —
 * which also contains React and DOM code — so the worker throws
 * `ReferenceError: document is not defined`, dies, and ORT's session promise NEVER SETTLES. The
 * symptom is a silent hang with no error and no network traffic, which is exactly what it looks
 * like: an application that stopped, not a library that failed.
 *
 * Loading it from a CDN keeps ORT in its own module, whose worker has no DOM references. It also
 * removes ~27 MB of wasm from dist/ and keeps ORT's own dynamic import of its threaded `.mjs`
 * sidecar out of Vite's pipeline, which cannot resolve a dynamic import of a file in public/.
 *
 * ⚠ The import specifier is assembled at runtime for the same reason as the phonemizer's: a literal
 * would be statically analyzed, and Vite would bundle the very thing this file exists to keep out.
 */
import type * as OrtNS from "onnxruntime-web";

export type Ort = typeof OrtNS;

const VERSION = "1.29.0";   // keep in step with package.json's onnxruntime-web
const BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSION}/dist/`;

/**
 * How many WASM threads to give ORT.
 *
 * ⚠ NOT `navigator.hardwareConcurrency`, AND NOT THE DEFAULT. Left unset, ORT ran the transformer
 * single-threaded: 14.1 s per forward, ~450 s for a 32-step generation. Measured on an i7-10700K
 * (8 physical cores, 16 logical) against the int4 web model, one forward at S=200:
 *
 *     threads   1      2      4      8      12     16
 *     ms     14078   7120   3729   2279   2568   2553
 *
 * Scaling is near-linear to 8 (6.2x) and then REVERSES. 8 is one thread per PHYSICAL core; beyond
 * that two compute-bound threads share a core's execution units and contend, costing ~12%. But
 * `hardwareConcurrency` reports 16 on this machine — the logical count — so using it directly picks
 * the slowest of the three and takes the whole CPU while doing it. Half, capped at 8, approximates
 * the physical count on an SMT machine and leaves the rest of the system usable.
 *
 * ⚠ And threads need SharedArrayBuffer, which needs CROSS-ORIGIN ISOLATION (the COOP/COEP pair in
 * vite.config.ts and netlify.toml) AND a secure context. Over plain HTTP to a LAN address neither
 * holds, so this correctly returns 1 rather than requesting threads that cannot be created.
 */
function wasmThreads(): number {
  if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === "undefined") return 1;
  return Math.max(1, Math.min(8, Math.floor((navigator.hardwareConcurrency ?? 2) / 2)));
}

/** Why inference is single-threaded, for the UI. Null when threads are available. */
export function threadingUnavailableReason(): string | null {
  if (globalThis.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined") return null;
  return `single-threaded on ${location.origin} — WASM threads need cross-origin isolation and a `
    + "secure context, so generation is several times slower here. Use localhost or https.";
}

let cached: Promise<Ort> | undefined;

/** The ORT module, loaded once. Await this instead of importing onnxruntime-web directly. */
export function getOrt(): Promise<Ort> {
  cached ??= (async () => {
    const url = BASE + "ort.all.bundle.min.mjs";
    const ort = (await import(/* @vite-ignore */ url)) as unknown as Ort;
    ort.env.wasm.wasmPaths = BASE;
    ort.env.logLevel = "warning";
    ort.env.wasm.numThreads = wasmThreads();
    // ⚠ RUN THE WASM IN A WORKER. Without this the session executes on the MAIN thread, so a
    // generation freezes the page for its whole duration and Firefox raises "this tab is slowing
    // Firefox down" — which is true, and which no amount of threading fixes, because the threads
    // ORT spawns still join back on a blocked main thread. `proxy` moves the session itself off it.
    ort.env.wasm.proxy = true;
    return ort;
  })();
  return cached;
}
