#!/usr/bin/env node
/**
 * Precompute reference voice codes so the browser never downloads the 654 MB Higgs ENCODER.
 *
 * The encoder exists only to turn a reference WAV into codec codes; the codes themselves are a few
 * KB. Running it once here and shipping voices.json drops 654 MB from the demo AND is what lets
 * every generation be voice-cloned — which matters because with no reference, input under ~5 s is
 * out of the fine-tune's distribution and can emit noise rather than degrade.
 *
 * Mirrors Chatterbox.Base.OmniVoiceTts.EncodeReference exactly: RMS-boost a quiet clip, remove
 * silence (mid 200 / lead 100 / trail 200 — NOT the output chain's 500/100/100), clip to a hop
 * multiple, then encode.
 *
 *   node tools/make-voices.mjs <higgs_encoder.onnx> <ref.wav> <ref-ipa-file> <id> <label>
 */
import * as ort from "onnxruntime-node";
import fs from "node:fs";

const [encPath, wavPath, ipaPath, id, label] = process.argv.slice(2);
const HOP = 960, SR = 24000;

function readWavMonoF32(p) {
    const b = fs.readFileSync(p);
    let off = 12, fmt = null, data = null;
    while (off + 8 <= b.length) {
        const cid = b.toString("ascii", off, off + 4), sz = b.readUInt32LE(off + 4);
        if (cid === "fmt ") fmt = { tag: b.readUInt16LE(off + 8), ch: b.readUInt16LE(off + 10), sr: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
        if (cid === "data") data = b.subarray(off + 8, off + 8 + sz);
        off += 8 + sz + (sz & 1);
    }
    if (!fmt || !data) throw new Error("not a WAV: " + p);
    if (fmt.sr !== SR) throw new Error(`expected ${SR} Hz, got ${fmt.sr}`);
    let s;
    if (fmt.tag === 3 && fmt.bits === 32) s = new Float32Array(data.buffer, data.byteOffset, data.length / 4);
    else if (fmt.tag === 1 && fmt.bits === 16) {
        s = new Float32Array(data.length / 2);
        for (let i = 0; i < s.length; i++) s[i] = data.readInt16LE(i * 2) / 32768;
    } else throw new Error(`unsupported WAV format tag=${fmt.tag} bits=${fmt.bits}`);
    if (fmt.ch === 1) return Float32Array.from(s);
    const mono = new Float32Array(s.length / fmt.ch);
    for (let i = 0; i < mono.length; i++) { let a = 0; for (let c = 0; c < fmt.ch; c++) a += s[i * fmt.ch + c]; mono[i] = a / fmt.ch; }
    return mono;
}

const rms = (x) => { let s = 0; for (const v of x) s += v * v; return x.length ? Math.sqrt(s / x.length) : 0; };

// Same silence logic as audioPost.ts / OmniVoiceAudioPost, at the REFERENCE parameters.
const TH = Math.pow(10, -50 / 20);
const chunkRms = (a, st, len) => { const e = Math.min(st + len, a.length); if (e <= st) return 0; let s = 0; for (let i = st; i < e; i++) s += a[i] * a[i]; return Math.sqrt(s / (e - st)); };
function leadSil(a, chunkMs = 10) { const c = Math.max(1, (chunkMs * SR) / 1000 | 0); let t = 0; while (t < a.length && chunkRms(a, t, c) < TH) t += c; return Math.min(t, a.length); }
function detectSil(a, minSilMs, seekMs = 10) {
    const minSil = (minSilMs * SR / 1000) | 0, seek = Math.max(1, (seekMs * SR / 1000) | 0), out = [];
    if (a.length < minSil) return out;
    const st = []; for (let i = 0; i <= a.length - minSil; i += seek) if (chunkRms(a, i, minSil) <= TH) st.push(i);
    if (!st.length) return out;
    let rs = st[0], prev = st[0];
    for (let j = 1; j < st.length; j++) { const s = st[j]; if (!(s === prev + seek) && s > prev + minSil) { out.push([rs, prev + minSil]); rs = s; } prev = s; }
    out.push([rs, prev + minSil]); return out;
}
function removeSilence(a, midSilMs, leadMs, trailMs) {
    if (!a.length) return a;
    let cur = a;
    if (midSilMs > 0) {
        const sil = detectSil(a, midSilMs), non = [];
        if (!sil.length) non.push([0, a.length]);
        else { let c = 0; for (const [s, e] of sil) { if (s > c) non.push([c, s]); c = e; } if (c < a.length) non.push([c, a.length]); }
        const keep = (midSilMs * SR / 1000) | 0;
        const r = non.map(([s, e]) => [s - keep, e + keep]);
        for (let i = 0; i + 1 < r.length; i++) if (r[i + 1][0] < r[i][1]) { const m = ((r[i][1] + r[i + 1][0]) / 2) | 0; r[i][1] = m; r[i + 1][0] = m; }
        const buf = []; for (const [s0, e0] of r) for (let i = Math.max(0, s0); i < Math.min(a.length, e0); i++) buf.push(a[i]);
        cur = Float32Array.from(buf);
    }
    if (!cur.length) return cur;
    const start = Math.max(0, leadSil(cur) - ((leadMs * SR / 1000) | 0));
    const rev = Float32Array.from(cur).reverse();
    const end = cur.length - Math.max(0, leadSil(rev) - ((trailMs * SR / 1000) | 0));
    return end <= start ? new Float32Array(0) : cur.slice(start, end);
}

let wav = readWavMonoF32(wavPath);
const refRms = rms(wav);
if (refRms > 0 && refRms < 0.1) { const g = 0.1 / refRms; wav = Float32Array.from(wav, (v) => v * g); }
wav = removeSilence(wav, 200, 100, 200);
const clip = wav.length % HOP;
if (clip > 0) wav = wav.slice(0, wav.length - clip);

const sess = await ort.InferenceSession.create(encPath);
const out = await sess.run({ input_values: new ort.Tensor("float32", wav, [1, 1, wav.length]) });
const codes = out.audio_codes;                    // [1, 8, Tc] int64
const [, cb, tc] = codes.dims;
const flat = Array.from(codes.data, Number);      // already row-major [8, Tc]

const voice = {
    id, label,
    refIpa: fs.readFileSync(ipaPath, "utf8").trim(),
    codes: flat, refLen: tc, refRms,
};
console.log(JSON.stringify([voice]));
console.error(`${id}: ${wav.length} samples -> ${cb}x${tc} codes, refRms=${refRms.toFixed(4)}`);
