#!/usr/bin/env node
/**
 * One reference voice per language, taken straight from the fine-tune corpus.
 *
 * ⚠ WHY THIS MATTERS: generation is always voice-cloned, and cloning is ACOUSTIC — the reference
 * carries the speaker, including their accent. With a single English reference, every language came
 * out in an English speaker's voice, which is what a listener notices immediately when switching
 * languages. A native reference per language fixes that.
 *
 * No audio and no encoder are needed: the corpus already stores encoded codes per utterance
 * (`tokens/codes_<lang>.npz`) alongside the exact IPA they were trained with
 * (`tokens/manifest_<lang>.jsonl`). The reference IPA MUST be that stored text, not a fresh
 * phonemization — it is what the model saw with those codes.
 *
 * The one thing not stored is the reference waveform's RMS, which the output volume step needs, so
 * the codes are decoded once through the Higgs decoder to measure it.
 */
import * as ort from "onnxruntime-node";
import { execFileSync } from "node:child_process";
import fs from "node:fs";

const CORPUS = "/mnt/data/omnivoice_ipa/corpus/tokens";
const DECODER = "/mnt/data/Programming/vernacula/scripts/omnivoice_export/onnx/higgs_decoder.onnx";
// demo code -> FLEURS corpus code
const MAP = {
  en: "en_us", es: "es_419", de: "de_de", fr: "fr_fr", pt: "pt_br", ca: "ca_es", cs: "cs_cz",
  sv: "sv_se", ru: "ru_ru", tr: "tr_tr", cy: "cy_gb", ga: "ga_ie", cmn: "cmn_hans_cn",
  ja: "ja_jp", ko: "ko_kr", hi: "hi_in", ta: "ta_in", th: "th_th", vi: "vi_vn", ar: "ar_eg",
  am: "am_et", om: "om_et", ha: "ha_ng", ff: "ff_sn", zu: "zu_za", xh: "xh_za", kk: "kk_kz",
  sd: "sd_in",
};

const rms = (x) => { let s = 0; for (const v of x) s += v * v; return x.length ? Math.sqrt(s / x.length) : 0; };
const dec = await ort.InferenceSession.create(DECODER);

// npz is a zip of .npy; unzip one member and parse its header rather than adding a dependency.
function loadCodes(npz, id) {
  const raw = execFileSync("unzip", ["-p", npz, `${id}.npy`], { maxBuffer: 1 << 28 });
  const hlen = raw.readUInt16LE(8);
  const hdr = raw.subarray(10, 10 + hlen).toString("latin1");
  const shape = /\(([\d,\s]+)\)/.exec(hdr)[1].split(",").map((x) => parseInt(x)).filter((n) => !isNaN(n));
  const dtype = /'descr':\s*'([^']+)'/.exec(hdr)[1];
  if (!dtype.includes("i2")) throw new Error(`unexpected dtype ${dtype}`);
  const body = raw.subarray(10 + hlen);
  const n = shape.reduce((a, b) => a * b, 1);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = body.readInt16LE(i * 2);
  return { data: out, shape };
}

const voices = [];
for (const [code, fl] of Object.entries(MAP)) {
  const mf = `${CORPUS}/manifest_${fl}.jsonl`, npz = `${CORPUS}/codes_${fl}.npz`;
  if (!fs.existsSync(mf) || !fs.existsSync(npz)) { console.error(`  skip ${code}: no corpus`); continue; }
  const rows = fs.readFileSync(mf, "utf8").trim().split("\n").map((l) => JSON.parse(l))
      .filter((r) => r.status === "verified" && r.dur_s >= 4 && r.dur_s <= 8 && r.ipa);
  if (!rows.length) { console.error(`  skip ${code}: no verified 4-8s utterance`); continue; }
  // Deterministic pick: shortest IPA among the longest-duration candidates reads as unhurried.
  rows.sort((a, b) => b.dur_s - a.dur_s || a.ipa.length - b.ipa.length);
  let picked = null;
  for (const r of rows.slice(0, 12)) {
    try { picked = { r, c: loadCodes(npz, r.id) }; break; } catch { /* id absent from the npz */ }
  }
  if (!picked) { console.error(`  skip ${code}: no codes for any candidate`); continue; }
  const { r, c } = picked;
  const [cb, tc] = c.shape;
  const big = BigInt64Array.from(c.data, BigInt);
  const wav = (await dec.run({ audio_codes: new ort.Tensor("int64", big, [1, cb, tc]) })).audio_values.data;
  voices.push({ id: code, label: `${code} (${r.gender ?? "?"}, ${r.dur_s.toFixed(1)}s)`,
                lang: code, refIpa: r.ipa, codes: Array.from(c.data), refLen: tc, refRms: rms(wav) });
  console.error(`  ${code}: ${cb}x${tc} codes, ${r.dur_s.toFixed(1)}s, rms=${rms(wav).toFixed(4)}, ${r.gender}`);
}
console.log(JSON.stringify(voices));
