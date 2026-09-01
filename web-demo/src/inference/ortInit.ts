/**
 * Point onnxruntime-web at its wasm sidecars before any session is created.
 *
 * They are served from a CDN rather than bundled: shipping all four variants put ~100 MB into
 * dist/ for a site whose actual payload is a 472 MB model fetched from HuggingFace anyway. ORT
 * also dynamically imports a threaded `.mjs` sidecar, and an absolute CDN URL keeps that out of
 * Vite's module pipeline, which cannot resolve a dynamic import of a file in public/.
 *
 * ⚠ Cross-origin under COEP `require-corp` is fine because ORT fetches these in CORS mode and
 * jsDelivr sends `access-control-allow-origin: *` — the same arrangement the Parakeet demo uses.
 */
import * as ort from "onnxruntime-web";

const VERSION = "1.29.0";  // keep in step with package.json's onnxruntime-web
const CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSION}/dist/`;

let done = false;
export function initOrt(): void {
  if (done) return;
  done = true;
  ort.env.wasm.wasmPaths = CDN;
  ort.env.logLevel = "warning";
}
