#!/usr/bin/env node
/**
 * One reference voice per language, taken straight from the FLEURS-derived fine-tune corpus.
 *
 * ⚠ WHY CLONING AT ALL: generation is always voice-cloned, and cloning is ACOUSTIC — the reference
 * carries the speaker's accent as well as their timbre. A single English reference made every
 * language come out in an English speaker's voice. (Not cloning was considered and rejected: it
 * produced noise on 2-3 s input, which is exactly the demo's sample length. See
 * docs/web_demo_investigation.md Run 7.)
 *
 * No audio and no encoder are needed. The corpus already stores encoded codes per utterance
 * (`tokens/codes_<lang>.npz`) alongside the exact IPA they were trained with
 * (`tokens/manifest_<lang>.jsonl`). ⚠ The reference IPA MUST be that stored text rather than a
 * fresh phonemization — it is what the model saw with those codes.
 *
 *   node tools/make-voices-from-corpus.mjs --out public/models --previews /tmp/voice-previews
 *   node tools/make-voices-from-corpus.mjs --out public/models --alt de=3 --alt th=1
 *   node tools/make-voices-from-corpus.mjs --out public/models --extra de=1,2   # 3 German voices
 *
 * `--alt <lang>=<n>` replaces a language's exemplar with the n-th candidate — how a noisy one gets
 * swapped without changing the selection rule. `--extra <lang>=<a,b>` adds further candidates as
 * ALTERNATIVE voices for that language.
 *
 * ⚠ TWO FILES, ON PURPOSE. `voices.jsonc` is metadata only — commented, one short block per voice,
 * meant to be read and edited by hand. The 1600 integers per voice live in `voice-codes.json`,
 * because inlining them is what makes a voices file unscannable.
 */
import * as ort from "onnxruntime-node";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const CORPUS = "/mnt/data/omnivoice_ipa/corpus/tokens";
const TRANSCRIPTS = "/mnt/data/omnivoice_ipa/corpus/fleurs_transcripts/data";
const DECODER = "/mnt/data/Programming/vernacula/scripts/omnivoice_export/onnx/higgs_decoder.onnx";
const SR = 24000;

/** demo language code -> FLEURS config code. */
const MAP = {
  en: "en_us", es: "es_419", de: "de_de", fr: "fr_fr", pt: "pt_br", ca: "ca_es", cs: "cs_cz",
  sv: "sv_se", ru: "ru_ru", tr: "tr_tr", cy: "cy_gb", ga: "ga_ie", cmn: "cmn_hans_cn",
  ja: "ja_jp", ko: "ko_kr", hi: "hi_in", ta: "ta_in", th: "th_th", vi: "vi_vn", ar: "ar_eg",
  am: "am_et", om: "om_et", ha: "ha_ng", ff: "ff_sn", zu: "zu_za", xh: "xh_za", kk: "kk_kz",
  sd: "sd_in",
  // Not in the v6 fine-tune's 28-language coverage set, but FLEURS has them and the corpus
  // ingested all 102 — so these get a NATIVE reference rather than a phonetic-proximity stand-in.
  is: "is_is", it: "it_it",
};

const args = process.argv.slice(2);
const previewDir = args.includes("--previews") ? args[args.indexOf("--previews") + 1] : null;
const outDir = args.includes("--out") ? args[args.indexOf("--out") + 1] : null;
const extra = {};
args.forEach((a, i) => { if (a === "--extra") { const [l, n] = args[i + 1].split("="); extra[l] = n.split(",").map(Number); } });
const alt = {};
args.forEach((a, i) => { if (a === "--alt") { const [l, n] = args[i + 1].split("="); alt[l] = Number(n); } });

const rms = (x) => { let s = 0; for (const v of x) s += v * v; return x.length ? Math.sqrt(s / x.length) : 0; };

/** wav-stem -> {split, sentenceId, text} for one FLEURS language. */
function transcriptIndex(fl) {
  const idx = new Map();
  for (const split of ["train", "dev", "test"]) {
    const p = path.join(TRANSCRIPTS, fl, `${split}.tsv`);
    if (!fs.existsSync(p)) continue;
    for (const line of fs.readFileSync(p, "utf8").split("\n")) {
      const c = line.split("\t");
      if (c.length >= 3) idx.set(c[1].replace(/\.wav$/, ""), { split, sentenceId: c[0], text: c[2] });
    }
  }
  return idx;
}

/** npz is a zip of .npy members; read one without adding a dependency. */
function loadCodes(npz, id) {
  const raw = execFileSync("unzip", ["-p", npz, `${id}.npy`], { maxBuffer: 1 << 28 });
  const hlen = raw.readUInt16LE(8);
  const hdr = raw.subarray(10, 10 + hlen).toString("latin1");
  const shape = /\(([\d,\s]+)\)/.exec(hdr)[1].split(",").map(Number).filter((n) => !isNaN(n));
  if (!/'descr':\s*'[<|]i2'/.test(hdr)) throw new Error(`unexpected dtype in ${id}.npy`);
  const body = raw.subarray(10 + hlen);
  const n = shape.reduce((a, b) => a * b, 1);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = body.readInt16LE(i * 2);
  return { data: out, shape };
}

function wav(samples) {
  const b = Buffer.alloc(44 + samples.length * 4);
  b.write("RIFF", 0); b.writeUInt32LE(36 + samples.length * 4, 4); b.write("WAVE", 8);
  b.write("fmt ", 12); b.writeUInt32LE(16, 16); b.writeUInt16LE(3, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(SR, 24); b.writeUInt32LE(SR * 4, 28); b.writeUInt16LE(4, 32); b.writeUInt16LE(32, 34);
  b.write("data", 36); b.writeUInt32LE(samples.length * 4, 40);
  Buffer.from(samples.buffer, samples.byteOffset, samples.length * 4).copy(b, 44);
  return b;
}

const dec = await ort.InferenceSession.create(DECODER);
if (previewDir) fs.mkdirSync(previewDir, { recursive: true });

const voices = [];
const codesOut = {};

for (const [code, fl] of Object.entries(MAP)) {
  const mf = `${CORPUS}/manifest_${fl}.jsonl`, npz = `${CORPUS}/codes_${fl}.npz`;
  if (!fs.existsSync(mf) || !fs.existsSync(npz)) { console.error(`  skip ${code}: no corpus for ${fl}`); continue; }
  const idx = transcriptIndex(fl);
  const rows = fs.readFileSync(mf, "utf8").trim().split("\n").map((l) => JSON.parse(l))
      .filter((r) => r.status === "verified" && r.dur_s >= 4 && r.dur_s <= 8 && r.ipa);
  // Deterministic order so `--alt n` and `--extra n` mean the same thing on every run.
  rows.sort((a, b) => b.dur_s - a.dur_s || a.ipa.length - b.ipa.length || a.id.localeCompare(b.id));

  const wanted = [alt[code] ?? 0, ...(extra[code] ?? [])];
  let first = true;
  for (const want of wanted) {
    let picked = null, tried = 0;
    for (const r of rows.slice(want)) {
      try { picked = { r, c: loadCodes(npz, r.id) }; break; } catch { tried++; }
    }
    if (!picked) { console.error(`  skip ${code}[${want}]: no codes for any candidate`); continue; }
    const { r, c } = picked;
    const [cb, tc] = c.shape;
    const big = BigInt64Array.from(c.data, BigInt);
    const audio = (await dec.run({ audio_codes: new ort.Tensor("int64", big, [1, cb, tc]) })).audio_values.data;
    const meta = idx.get(r.id) ?? {};
    const id = first ? code : `${code}-${want}`;

    voices.push({
      id, lang: code, default: first || undefined,
      label: `${code} · ${r.gender ?? "?"} · ${r.dur_s.toFixed(1)}s`,
      refIpa: r.ipa, refLen: tc, refRms: Number(rms(audio).toFixed(5)),
      source: {
        dataset: "google/fleurs", lang: fl,
        file: `${r.id}.wav`, split: meta.split ?? null,
        sentenceId: meta.sentenceId ?? r.sentence_id ?? null,
        gender: r.gender ?? null, durationS: r.dur_s, candidateIndex: want,
        text: meta.text ?? null,
      },
    });
    codesOut[id] = Array.from(c.data);
    if (previewDir) fs.writeFileSync(path.join(previewDir, `${id}.wav`), wav(Float32Array.from(audio)));
    console.error(`  ${id.padEnd(7)} ${fl.padEnd(12)} ${r.id}.wav  ${(meta.split ?? "?").padEnd(5)} `
                + `${(r.gender ?? "?").padEnd(6)} ${r.dur_s.toFixed(1)}s  rms=${rms(audio).toFixed(4)}`
                + `  [${want}/${rows.length}]${tried ? ` (+${tried} skipped)` : ""}`);
    first = false;
  }
}

/** Hand-editable metadata: comments, one block per voice, no code arrays. */
function toJsonc(vs) {
  const L = [
    "// Reference voices for vernacula-tts, one or more per language.",
    "//",
    "// Generated by tools/make-voices-from-corpus.mjs from the FLEURS-derived fine-tune corpus.",
    "// Editable by hand: reorder, delete, or flip `default` to choose a language's voice. The",
    "// matching code arrays live in voice-codes.json, keyed by `id` — change an `id` here and you",
    "// must change it there too.",
    "//",
    "// ⚠ refIpa is the IPA the model was TRAINED with for this clip. Do not re-phonemize it; it is",
    "//   what the model saw alongside these codes.",
    "// ⚠ `source` identifies the exact FLEURS clip, so a noisy exemplar can be traced and replaced:",
    "//   re-run with --alt <lang>=<n> (or --extra <lang>=<a,b> for additional voices).",
    "[",
  ];
  vs.forEach((v, i) => {
    const s = v.source;
    L.push(`  // ${v.lang} — ${s.lang} ${s.split ?? "?"} · ${s.gender ?? "?"} · ${s.durationS.toFixed(1)}s · candidate ${s.candidateIndex}`);
    if (s.text) L.push(`  // "${String(s.text).replace(/\s+/g, " ").slice(0, 100)}"`);
    L.push("  " + JSON.stringify(v) + (i < vs.length - 1 ? "," : ""));
  });
  L.push("]", "");
  return L.join("\n");
}

if (outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "voices.jsonc"), toJsonc(voices));
  fs.writeFileSync(path.join(outDir, "voice-codes.json"), JSON.stringify(codesOut));
  const a = fs.statSync(path.join(outDir, "voices.jsonc")).size;
  const b = fs.statSync(path.join(outDir, "voice-codes.json")).size;
  console.error(`\n${voices.length} voices -> ${outDir}/voices.jsonc (${(a/1024).toFixed(0)} KB, hand-editable)`
              + ` + voice-codes.json (${(b/1024).toFixed(0)} KB)`);
} else {
  console.log(JSON.stringify(voices));
}
if (previewDir) console.error(`previews -> ${previewDir}`);
