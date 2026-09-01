import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

/**
 * Serve the transpiled phonemizer verbatim at /vphon/, ahead of Vite's own middleware.
 *
 * ⚠ THE ENGINE MUST NOT GO THROUGH VITE, and public/ is not a way to achieve that. Its data keys
 * come from `import.meta.url` (dataPath.ts slices after the last "/src/"), so a bundler that
 * rewrites module URLs erases the only thing naming the data — the engine throws rather than guess.
 * But a dynamic import of a file in public/ ALSO fails in dev: Vite refuses outright for a literal
 * specifier ("should not be imported from source code"), and for a computed one it still intercepts
 * the request, appends `?import`, and 500s. Neither shows up in a test that serves the files from a
 * plain Node server, which is why this survived two green smokes and failed on the first real click.
 *
 * So: claim the path before Vite sees it, strip the query it adds, and send the file as-is. In a
 * production build public/ is copied verbatim and none of this applies.
 */
function serveEngineVerbatim(): Plugin {
  const root = path.resolve("public");
  return {
    name: "serve-phonemizer-verbatim",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        if (!url.startsWith("/vphon/")) return next();
        const file = path.join(root, decodeURIComponent(url.split("?")[0]));
        if (!file.startsWith(root) || !fs.existsSync(file) || !fs.statSync(file).isFile()) return next();
        res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : "application/octet-stream");
        res.setHeader("Cache-Control", "no-cache");
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

const ISOLATION = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react(), serveEngineVerbatim()],
  // onnxruntime-web ships its own wasm loader; pre-bundling breaks its worker resolution.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  // SharedArrayBuffer, required by onnxruntime-web's threaded WASM build. netlify.toml sets the
  // same pair for the deployed site; `preview` needs its own copy or the production build cannot be
  // checked under the headers it will actually run with.
  server: { headers: ISOLATION },
  preview: { headers: ISOLATION },
});
