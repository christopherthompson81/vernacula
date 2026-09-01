import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // onnxruntime-web ships its own wasm loader; pre-bundling breaks its worker resolution.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: {
    headers: {
      // SharedArrayBuffer, required by onnxruntime-web's threaded WASM build.
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
