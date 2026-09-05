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

/**
 * Fired once, with a reason string, when the cache REFUSED the model mid-download and the load
 * carried on in memory. The page subscribes and shows it as a notice.
 *
 * ⚠ THIS IS THE "Quota exceeded" FAILURE. Chrome's `cache.put` throws `QuotaExceededError`
 * ("Failed to execute 'put' on 'Cache': Quota exceeded.") when the site's storage allowance cannot
 * hold the next 32 MiB chunk of a 472 MB model — a small disk, an incognito window, or another
 * site's data in the same allowance — and that DOMException used to surface verbatim as the
 * demo's error, with the download abandoned at whatever point it had reached. Nothing about the
 * model needs to be PERSISTED to run it; the cache only saves the next visit's download. So a
 * full cache is the uncached path with a better message, not a failure.
 */
export const cacheEvents = new EventTarget();
export type CacheFallbackEvent = CustomEvent<string>;

const isQuotaError = (e: unknown) =>
  (e instanceof DOMException && e.name === "QuotaExceededError")
  || /quota/i.test(e instanceof Error ? e.message : String(e));

async function quotaReason(needBytes: number): Promise<string> {
  let numbers = "";
  try {
    const est = await navigator.storage?.estimate?.();
    if (est?.quota) numbers = ` (this site's storage allowance is ${(est.quota / 1e6).toFixed(0)} MB, ${((est.usage ?? 0) / 1e6).toFixed(0)} MB used)`;
  } catch { /* the message is still right without the figures */ }
  return `model cache full — the browser refused to store the ${(needBytes / 1e6).toFixed(0)} MB model${numbers}. `
    + "It is held in memory for this session and will be downloaded again next load. "
    + "Free disk space, close other tabs of this site, or leave a private window to let it persist.";
}

/** Drop every cached piece of one url — after a quota failure the partial set is only dead weight. */
async function forget(url: string, chunks: number) {
  const c = await caches.open(CACHE);
  await c.delete(metaKey(url));
  await Promise.all(Array.from({ length: chunks }, (_, i) => c.delete(chunkKey(url, i))));
}

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

/** The completed cache entry, or — when the cache ran out of room part-way — the bytes themselves. */
async function ensureDownloaded(url: string, onProgress?: (p: DownloadProgress) => void): Promise<Meta | ArrayBuffer> {
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
  // Set when the cache refuses a chunk: from then on every byte — those already cached, those
  // accumulated, and those still to come — is kept here instead, and the caller gets the buffer.
  // (An object, not a `let`: it is assigned inside `flush`, and TypeScript's flow analysis does
  // not see closure writes, so a plain variable reads as `never` after the loop.)
  const fallback: { memory: Uint8Array[] | null } = { memory: null };
  const flush = async () => {
    if (fallback.memory) { fallback.memory.push(...parts); parts = []; partBytes = 0; return; }
    const buf = new Uint8Array(partBytes);
    let o = 0; for (const p of parts) { buf.set(p, o); o += p.byteLength; }
    try {
      await cache.put(chunkKey(url, idx), new Response(new Blob([buf.buffer as ArrayBuffer])));
      idx++; meta!.chunks = idx; meta!.sizes.push(buf.byteLength); await putMeta(url, meta!);
    } catch (e) {
      if (!isQuotaError(e)) throw e;
      // Pull what was cached back into memory BEFORE forgetting it: the download resumes from
      // wherever it is, so the earlier chunks exist nowhere else.
      const memory: Uint8Array[] = [];
      for (let i = 0; i < idx; i++) {
        const r = await cache.match(chunkKey(url, i));
        if (!r) throw new Error(`missing cached chunk ${i} of ${url} while falling back to memory`);
        memory.push(new Uint8Array(await r.arrayBuffer()));
      }
      memory.push(buf);
      fallback.memory = memory;
      await forget(url, idx);
      cacheEvents.dispatchEvent(new CustomEvent("fallback", { detail: await quotaReason(meta!.total) }));
    }
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
  if (fallback.memory) {
    const size = fallback.memory.reduce((a, b) => a + b.byteLength, 0);
    const out = new Uint8Array(size);
    let o = 0; for (const p of fallback.memory) { out.set(p, o); o += p.byteLength; }
    return out.buffer;
  }
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
  // ⚠ WRITE STRAIGHT INTO THE FINAL BUFFER when the length is known. Collecting chunks and copying
  // them afterwards holds the model TWICE — ~940 MB at the join for a 470 MB transformer, on the
  // path taken by machines that already lack the cache. Falling back to chunk-collection only when
  // the server sends no Content-Length keeps that cost for the case that cannot avoid it.
  if (total > 0) {
    const out = new Uint8Array(total);
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      out.set(value, loaded); loaded += value.byteLength;
      onProgress?.({ url, loaded, total, cached: false });
    }
    return out.buffer;
  }
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.byteLength;
    onProgress?.({ url, loaded, total: loaded, cached: false });
  }
  const out = new Uint8Array(loaded);
  let o = 0; for (const p of parts) { out.set(p, o); o += p.byteLength; }
  return out.buffer;
}

/** Fetch a model file, resuming and caching, and return it whole. */
export async function fetchModel(url: string, onProgress?: (p: DownloadProgress) => void): Promise<ArrayBuffer> {
  if (!CACHE_OK) return fetchUncached(url, onProgress);
  const meta = await ensureDownloaded(url, onProgress);
  if (meta instanceof ArrayBuffer) return meta;   // the cache ran out of room; see cacheEvents
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
