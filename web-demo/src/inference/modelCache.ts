/**
 * Chunked, resumable download cache for the large ONNX files.
 *
 * The transformer is 472 MB, so a plain fetch that dies at 80% has to start over, and a monolithic
 * Cache entry needs a contiguous allocation the size of the model. Both are avoided by storing
 * 32 MiB chunks and resuming with a Range request. Adapted from the Parakeet browser demo, which
 * already carried this.
 */
const CACHE = "vernacula-tts-models";
const CHUNK = 32 * 1024 * 1024;

/**
 * Is the Cache API available at all?
 *
 * ⚠ `caches` EXISTS ONLY IN A SECURE CONTEXT — https, or a localhost origin. Serving the dev server
 * to another machine (`npm run dev -- --host`, then http://192.168.x.x:5173) is NOT one, so the
 * global is simply undefined and every function here threw `caches is not defined` — surfaced in the
 * UI as those three words, which say nothing about the cause or the fix. The page otherwise works
 * fine over plain HTTP: phonemizing, inference and playback need no secure context, and only the
 * 472 MB model cache does. So degrade instead of failing — download it, hold it in memory, and say
 * why it will not persist.
 */
const CACHE_OK = typeof caches !== "undefined";

if (navigator.storage?.persist) void navigator.storage.persist();

/** Why persistence is unavailable, for the UI to show once. Null when the cache works. */
export const cacheUnavailableReason: string | null = CACHE_OK ? null
  : `no model cache on ${location.origin} — the Cache API needs https or localhost, `
    + "so the model is re-downloaded each load. Open the demo on localhost, or serve it over https.";

export interface DownloadProgress { url: string; loaded: number; total: number; cached: boolean; }
/**
 * ⚠ `sizes` holds each chunk's ACTUAL byte length, and that is load-bearing. `flush()` fires once
 * `partBytes >= CHUNK` and writes everything accumulated, so a stored chunk is CHUNK plus whatever
 * the last read added — essentially never exactly CHUNK. Deriving the resume offset as
 * `chunks * CHUNK` therefore UNDER-counts what is cached: the Range request re-fetches bytes already
 * stored, and reassembly then writes more bytes than `total`, throwing RangeError or — worse —
 * silently producing a duplicated-byte, corrupt model. A 472 MB download interrupted at 200 MB
 * would fail on every subsequent load.
 */
interface Meta { total: number; chunks: number; sizes: number[]; done: boolean; }

const cachedBytes = (m: Meta) => m.sizes.reduce((a, b) => a + b, 0);

const metaKey = (u: string) => `${u}\x00meta`;
const chunkKey = (u: string, i: number) => `${u}\x00c:${i}`;

async function getMeta(u: string): Promise<Meta | null> {
  const r = await (await caches.open(CACHE)).match(metaKey(u));
  if (!r) return null;
  try { return await r.json() as Meta; } catch { return null; }
}
async function putMeta(u: string, m: Meta) {
  await (await caches.open(CACHE)).put(metaKey(u), new Response(JSON.stringify(m), {
    headers: { "Content-Type": "application/json" },
  }));
}

async function ensureDownloaded(url: string, onProgress?: (p: DownloadProgress) => void): Promise<Meta> {
  let meta = await getMeta(url);
  if (meta?.done) return meta;
  // A meta written before `sizes` existed cannot be resumed correctly — its offset is unknowable.
  // Discard it rather than resume from a wrong offset.
  if (meta && !Array.isArray(meta.sizes)) {
    const c = await caches.open(CACHE);
    await c.delete(metaKey(url));
    await Promise.all(Array.from({ length: meta.chunks }, (_, i) => c.delete(chunkKey(url, i))));
    meta = null;
  }

  const resumeFrom = meta ? cachedBytes(meta) : 0;
  let res: Response, start = resumeFrom;
  if (resumeFrom > 0) {
    res = await fetch(url, { headers: { Range: `bytes=${resumeFrom}-` } });
    if (res.status === 200) {
      // Server ignored the Range header; the partial cache is unusable, so start over.
      const c = await caches.open(CACHE);
      await c.delete(metaKey(url));
      await Promise.all(Array.from({ length: meta!.chunks }, (_, i) => c.delete(chunkKey(url, i))));
      meta = null; start = 0;
    } else if (!res.ok) throw new Error(`resume ${url} -> ${res.status}`);
  } else {
    res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  }

  const total = start + Number(res.headers.get("Content-Length") ?? 0);
  meta ??= { total, chunks: 0, sizes: [], done: false };
  meta.total = total || meta.total;

  const reader = res.body!.getReader();
  const cache = await caches.open(CACHE);
  let parts: Uint8Array[] = [], partBytes = 0, loaded = start, idx = meta.chunks;
  const flush = async () => {
    const buf = new Uint8Array(partBytes);
    let o = 0; for (const p of parts) { buf.set(p, o); o += p.byteLength; }
    await cache.put(chunkKey(url, idx), new Response(new Blob([buf.buffer as ArrayBuffer])));
    idx++; meta!.chunks = idx; meta!.sizes.push(buf.byteLength); await putMeta(url, meta!);
    parts = []; partBytes = 0;
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); partBytes += value.byteLength; loaded += value.byteLength;
    onProgress?.({ url, loaded, total: meta.total, cached: false });
    if (partBytes >= CHUNK) await flush();
  }
  if (partBytes > 0) await flush();
  meta.done = true; meta.total = cachedBytes(meta);
  await putMeta(url, meta);
  return meta;
}

/** Stream a model straight into memory, for origins with no Cache API. No resume, no persistence. */
async function fetchUncached(url: string, onProgress?: (p: DownloadProgress) => void): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  const total = Number(res.headers.get("Content-Length") ?? 0);
  const reader = res.body!.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.byteLength;
    onProgress?.({ url, loaded, total: total || loaded, cached: false });
  }
  const out = new Uint8Array(loaded);
  let o = 0; for (const p of parts) { out.set(p, o); o += p.byteLength; }
  return out.buffer;
}

/** Fetch a model file, resuming and caching, and return it whole. */
export async function fetchModel(url: string, onProgress?: (p: DownloadProgress) => void): Promise<ArrayBuffer> {
  if (!CACHE_OK) return fetchUncached(url, onProgress);
  const meta = await ensureDownloaded(url, onProgress);
  const cache = await caches.open(CACHE);
  // `total`, not `cachedBytes`: a COMPLETED meta written before `sizes` existed is still a valid
  // cache (every chunk is present and total was set from Content-Length), and it returns from
  // ensureDownloaded before the legacy check — summing its absent `sizes` threw here and made a
  // fully-cached model unloadable until the cache was cleared by hand.
  const out = new Uint8Array(meta.total);
  let off = 0;
  for (let i = 0; i < meta.chunks; i++) {
    const r = await cache.match(chunkKey(url, i));
    if (!r) throw new Error(`missing cached chunk ${i} of ${url}`);
    const b = new Uint8Array(await r.arrayBuffer());
    out.set(b, off); off += b.byteLength;
    onProgress?.({ url, loaded: off, total: meta.total, cached: true });
  }
  return out.buffer;
}

export async function clearModelCache(): Promise<void> {
  if (CACHE_OK) await caches.delete(CACHE);
}
