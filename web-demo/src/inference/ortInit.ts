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

let cached: Promise<Ort> | undefined;

/** The ORT module, loaded once. Await this instead of importing onnxruntime-web directly. */
export function getOrt(): Promise<Ort> {
  cached ??= (async () => {
    const url = BASE + "ort.all.bundle.min.mjs";
    const ort = (await import(/* @vite-ignore */ url)) as unknown as Ort;
    ort.env.wasm.wasmPaths = BASE;
    ort.env.logLevel = "warning";
    return ort;
  })();
  return cached;
}
