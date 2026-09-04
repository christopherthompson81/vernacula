#!/usr/bin/env node
/**
 * Render EVERY (language, voice) pair through the real page into ONE folder, with an index that
 * remembers which ones have been listened to.
 *
 * `preview-langs.mjs` renders one clip per language, which is the right tool while sourcing a single
 * voice. It is the wrong tool for a sweep: 66 languages carry more than one voice, so a per-language
 * pass hears 193 of the 318 pairs a visitor can actually reach and silently never plays the other
 * 125. This enumerates all of them.
 *
 *   node tools/preview-all.mjs /tmp/listen-all                   # resumable; re-run to continue
 *   node tools/preview-all.mjs /tmp/listen-all --only an,bal,fr-CA
 *   node tools/preview-all.mjs /tmp/listen-all --index-only      # rebuild index.html, render nothing
 *
 * ⚠ IT DRIVES THE C# CLI, NOT THE BROWSER. `vernacula-tts --voice-id` renders from the SAME stored
 * codes and the same IPA the page uses, so what is judged is still what a visitor gets — but a 318
 * clip sweep is a scriptable loop rather than 318 puppeteer round-trips, and it needs no dev server,
 * no tab, and no reload-after-editing-voices.jsonc discipline.
 *
 * ⚠ THE RENDERED VOICE IS ASSERTED, NOT ASSUMED. The CLI echoes the voice it loaded, and a pair whose
 * echo disagrees with the request is recorded as a MISMATCH rather than saved under the name that was
 * asked for. A sweep exists to be trusted; one silently mislabelled file undoes that.
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { LANGUAGES, voiceLangOf, DONOR_NAMES } from "../src/inference/languages.ts";

const args = process.argv.slice(2);
const outDir = args.find((a) => !a.startsWith("--"));
const opt = (k) => { const i = args.indexOf(`--${k}`); return i < 0 ? null : args[i + 1]; };
const has = (k) => args.includes(`--${k}`);
if (!outDir) { console.error("usage: preview-all.mjs <outDir> [--only a,b] [--force] [--index-only]"); process.exit(2); }
mkdirSync(outDir, { recursive: true });

/** voices.jsonc is JSONC; the demo's own loader strips comments the same way. */
const strip = (t) => {
  let out = "", inStr = false, esc = false, line = false, block = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i], n = t[i + 1];
    if (line) { if (c === "\n") { line = false; out += c; } continue; }
    if (block) { if (c === "*" && n === "/") { block = false; i++; } continue; }
    if (inStr) { out += c; if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
    if (c === '"') { inStr = true; out += c; continue; }
    if (c === "/" && n === "/") { line = true; i++; continue; }
    if (c === "/" && n === "*") { block = true; i++; continue; }
    out += c;
  }
  return out;
};
const voices = JSON.parse(strip(readFileSync("public/models/voices.jsonc", "utf8")));
const byLang = new Map();
for (const v of voices) { if (!byLang.has(v.lang)) byLang.set(v.lang, []); byLang.get(v.lang).push(v); }

const only = opt("only")?.split(",").map((s) => s.trim()).filter(Boolean) ?? null;

/** Every pair a visitor can reach, in a stable order so the numbering never shifts under a re-run. */
const pairs = [];
for (const l of [...LANGUAGES].sort((a, b) => a.code.localeCompare(b.code))) {
  const vl = voiceLangOf(l.code);
  for (const v of (byLang.get(vl) ?? []).slice().sort((a, b) => a.id.localeCompare(b.id)))
    pairs.push({ lang: l, voice: v, donor: vl === l.code ? null : vl });
}
// ⚠ THE FILENAME MUST NOT CARRY THE SEQUENCE NUMBER. The number is a position in a sorted list, so
// adding or removing ONE voice renumbers everything after it and orphans every file downstream —
// giving Hawaiian two more voices turned 194 already-rendered clips into "missing". The identity of
// a clip is its language and its voice; the number is display order and lives in the manifest only.
const safe = (s) => s.replace(/[^\w.-]+/gu, "_");
pairs.forEach((p, i) => { p.n = i + 1; p.file = `${safe(p.lang.code)}__${safe(p.voice.id)}.wav`; });

/** Duration and sample rate from a RIFF header — the file itself, not the page's own report. */
function wavSeconds(path) {
  const b = readFileSync(path);
  if (b.length < 44 || b.toString("ascii", 0, 4) !== "RIFF") return null;
  const sr = b.readUInt32LE(24), bits = b.readUInt16LE(34), ch = b.readUInt16LE(22);
  let off = 12;
  while (off + 8 <= b.length) {
    const id = b.toString("ascii", off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === "data") return sz / (sr * ch * (bits / 8));
    off += 8 + sz + (sz & 1);
  }
  return null;
}

/**
 * ⚠ "SHORT, POSSIBLY BROKEN" IS THE COMMONEST FAILURE AND IT IS VISIBLE WITHOUT LISTENING. A clip
 * far shorter than its text can be spoken in is truncated, whatever it sounds like. The band is the
 * one `trim-to-sentences.mjs` calibrated on the shipped references — 4 to 30 SOURCE characters per
 * second — so this reuses a measured constant rather than inventing a threshold.
 */
const RATE_MAX = 30, RATE_MIN = 4;
/**
 * ⚠ CHARACTERS PER SECOND IS A PROPERTY OF THE SCRIPT, NOT ONLY OF THE SPEECH, and a single band
 * condemns languages that write compactly. One Han character is a whole syllable and often a
 * morpheme, so a 12-character Mandarin sentence legitimately runs 3.5 s — six of the first seven
 * clips this flagged were Chinese, Japanese, Cantonese and Cherokee, all of them fine. The corpus
 * investigation reached the same conclusion measuring FLEURS (Run 51: 5th percentiles run
 * cmn_hans_cn 4.4 against en_us 7.8), so the floor is relaxed where one character carries a syllable.
 */
const SYLLABIC = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}\p{Script=Cherokee}\p{Script=Ethiopic}\p{Script=Yi}\p{Script=Vai}]/u;
function verdict(sec, text) {
  if (sec === null) return { flag: "unreadable", note: "not a readable WAV" };
  const chars = [...(text ?? "")].filter((c) => !/\s/u.test(c)).length;
  if (!chars) return { flag: "", note: "" };
  const dense = [...(text ?? "")].filter((c) => SYLLABIC.test(c)).length / Math.max(1, chars) > 0.3;
  const floor = chars / RATE_MAX, ceil = chars / (dense ? 1.2 : RATE_MIN);
  if (sec < 0.6) return { flag: "short", note: `${sec.toFixed(1)}s — near silence` };
  if (sec < floor) return { flag: "short", note: `${sec.toFixed(1)}s for ${chars} chars — needs ≥ ${floor.toFixed(1)}s at 30 ch/s` };
  if (sec > ceil) return { flag: "long", note: `${sec.toFixed(1)}s for ${chars} chars — over ${ceil.toFixed(1)}s at 4 ch/s` };
  return { flag: "", note: `${sec.toFixed(1)}s` };
}

const BIN = opt("bin") ?? "../src/Vernacula.Tts.CLI/bin/Release/net10.0/vernacula-tts";
// ⚠ onnx_base/, NOT onnx/. Those two directories hold DIFFERENT 2.45 GB .onnx.data files under
// byte-identical graph protos, and only onnx_base/ matches the upstream checkpoint (publish_hf.py
// verifies it by sha256; the onnx/ copy differs in all 151,676 embedding rows). The diff carries
// ABSOLUTE REPLACEMENT embed rows, so applying it to the wrong base yields a plausible-looking
// model that is quietly wrong -- and nothing in the diff or the audio can flag it. This defaulted
// to onnx/, so the whole 318-clip listening sweep was judged on the wrong base.
const ONNX = opt("onnx-dir") ?? process.env.OMNIVOICE_ONNX_DIR ?? "/mnt/data/omnivoice_ipa/onnx_base";
const MODEL = opt("model-dir") ?? process.env.OMNIVOICE_MODEL_DIR ?? "/mnt/data/models/omnivoice/k2-fsa-OmniVoice";
// ⚠ NAME THE DIFF. Leaving it to the CLI's default meant the sweep silently rendered whatever
// version happened to be the fallback; a sweep that cannot say which model it heard is not evidence.
const DIFF = opt("diff") ?? process.env.OMNIVOICE_DIFF ?? "/mnt/data/omnivoice_ipa/onnx/ipa_diff_v7.onnx";
console.log(`model: ${ONNX.split("/").pop()} + ${DIFF.split("/").pop()}`);

if (!has("index-only")) {
  let done = 0, skipped = 0, failed = 0;
  const t0 = Date.now();
  const todo = pairs.filter((p) => (!only || only.includes(p.lang.code))
    && (has("force") || !existsSync(join(outDir, p.file))));
  // ⚠ "already present" means ON DISK, not "everything else". With --only, pairs.length - todo.length
  // counts every filtered-out language too, so a 3-pair run reported "609 already present" when 609
  // had never been rendered — a resumable sweep whose progress line lies about what it has is worse
  // than one with no progress line.
  const inScope = pairs.filter((p) => !only || only.includes(p.lang.code));
  console.log(`${todo.length} to render (${inScope.length - todo.length} already present`
    + (only ? `, ${pairs.length - inScope.length} filtered out by --only` : "") + ")");
  for (const p of todo) {
    const out = join(outDir, p.file);
    let log;
    try {
      log = execFileSync(BIN, [
        "--lang", p.lang.code, "--text", p.lang.sample,
        "--onnx-dir", ONNX, "--model-dir", MODEL, "--diff", DIFF,
        "--voice-lib", "public/models", "--voice-id", p.voice.id,
        // ⚠ --post demo, ALWAYS. The CLI's default is Python's post-chain, which un-boosts the
        // output by the reference's RMS; the browser deliberately does not. Rendering the demo's
        // voices under the Python chain reported 91 of 318 clips as quiet, inaudible or empty when
        // the deployed page plays them fine — a whole listening pass spent on an artifact of the
        // instrument. What is being audited is what a visitor hears.
        "--post", "demo",
        "--ep", opt("ep") ?? "cuda", "--out", out,
      ], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], maxBuffer: 1 << 24 });
    } catch (e) {
      const msg = String(e.stderr ?? e.message).split("\n").filter((l) => l && !/^\s*\x1b?\[?\d*;?\d*m?2026-/.test(l)).slice(-2).join(" ");
      console.log(`  ${p.n}/${pairs.length} ${p.lang.code}/${p.voice.id}: FAILED — ${msg.slice(0, 140)}`);
      failed++; continue;
    }
    // "reference: <id> (<label>) -> N codes" — the voice the CLI actually loaded.
    const echoed = /reference:\s+(\S+)\s/u.exec(log)?.[1];
    if (echoed && echoed !== p.voice.id) {
      console.log(`  ${p.n}/${pairs.length} ${p.lang.code}: MISMATCH — asked ${p.voice.id}, CLI loaded ${echoed}`);
      failed++; continue;
    }
    const secs = wavSeconds(out);
    const v = verdict(secs, p.lang.sample);
    done++;
    const rate = (Date.now() - t0) / 1000 / done;
    console.log(`  ${p.n}/${pairs.length} ${p.lang.code}/${p.voice.id}: ${v.note}${v.flag ? `  <-- ${v.flag}` : ""}`
      + `   [${done}/${todo.length}, ~${Math.round(rate * (todo.length - done) / 60)} min left]`);
  }
  console.log(`\nrendered ${done}, skipped ${skipped}, ${failed} failed`);
}

// ---- prior verdicts ------------------------------------------------------
// ⚠ A RE-RENDER MUST NOT COST THE LISTENING THAT IS ALREADY DONE. `listen_samples.out` is the
// reviewer's own file — "<file>; <verdict>" per line — and the index carries each verdict onto the
// new clip so only what actually needs re-hearing has to be re-heard.
// ⚠ KEYED ON THE VOICE, NOT THE NUMBERED FILENAME. The number is a position in a sorted list, so
// replacing one voice renumbers its neighbours — swapping `acw-omnili1` moved `acw-omnili2` from
// 0009 to 0008, which would have orphaned a verdict that is still perfectly valid. `<lang>/<voice>`
// survives renumbering; the filename is kept as a fallback for a manifest written before this.
const prior = new Map();
const addPrior = (key, verdict) => { if (key && verdict) prior.set(key, verdict); };
// ⚠ AND KEYED ON THE SOURCE CLIP TOO. Voice ids are positional in their own way — re-sourcing a
// language reuses `<lang>-omnili0` for a DIFFERENT recording — so lang+voice alone carried a stale
// "a lot of noise in the middle" onto a clean replacement. The source file is what identifies the
// audio a verdict was actually about.
const keyOf = (code, voice, src) => `${code}/${voice}/${src ?? ""}`;
for (const name of ["manifest.tsv", "listen_samples.out"]) {
  const path = join(outDir, name);
  if (!existsSync(path)) continue;
  const lines = readFileSync(path, "utf8").split("\n");
  if (name === "manifest.tsv") {
    for (const [i, line] of lines.entries()) {
      if (i === 0 || !line.trim()) continue;
      const c = line.split("\t");
      addPrior(keyOf(c[2], c[4], c[10]), (c[10] !== undefined ? c[9] : "").trim());
      // ⚠ The filename fallback is for LEGACY numbered names only. Once filenames became
      // `<lang>__<voice>.wav` the fallback matched exactly what the voice key matches, which put the
      // stale-verdict path straight back: a re-sourced `haw-omnili0` inherited "a lot of noise in the
      // middle" from the recording it replaced.
      if (/^\d{4}_/u.test(c[1] ?? "")) addPrior(c[1], (c[9] ?? "").trim());
    }
  } else {
    for (const line of lines) {
      const i = line.indexOf(";");
      if (i > 0) addPrior(line.slice(0, i).trim(), line.slice(i + 1).trim());
    }
  }
}

/** Was this verdict about the LEVEL — i.e. produced by the Python post-chain rather than the voice? */
const levelVerdict = (v) => /quiet|inaudible|silence|58 bytes|empty/iu.test(v ?? "")
  && !/short|cut|noise|pronunc/iu.test(v ?? "");

// ---- index ----------------------------------------------------------------
const present = new Set(readdirSync(outDir).filter((f) => f.endsWith(".wav")));
const rows = pairs.map((p) => {
  const there = present.has(p.file);
  const sec = there ? wavSeconds(join(outDir, p.file)) : null;
  const v = there ? verdict(sec, p.lang.sample) : { flag: "missing", note: "not rendered" };
  const src = (p.voice.source ?? {}).file ?? "";
  const said = prior.get(`${p.lang.code}/${p.voice.id}/${src}`) ?? prior.get(p.file) ?? "";
  return { ...p, there, sec, ...v, said, stale: levelVerdict(said) };
});
writeFileSync(join(outDir, "manifest.tsv"),
  "n\tfile\tcode\tlanguage\tvoice\tdonor\tseconds\tflag\tnote\tprior_verdict\tsource\n"
  + rows.map((r) => [r.n, r.file, r.lang.code, r.lang.name, r.voice.id,
      r.donor ?? "", r.sec?.toFixed(2) ?? "", r.flag, r.note, r.said,
      (r.voice.source ?? {}).file ?? ""].join("\t")).join("\n") + "\n");

const esc = (s) => String(s ?? "").replace(/[&<>"]/gu, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const flagged = rows.filter((r) => r.flag).length;
writeFileSync(join(outDir, "index.html"), `<!doctype html><meta charset="utf-8">
<title>Vernacula voice sweep — ${rows.length} pairs</title>
<style>
 body{font:14px/1.45 system-ui,sans-serif;margin:0;padding:1rem 1.2rem;background:#12141a;color:#e8e8ea}
 h1{font-size:1.1rem;margin:0 0 .2rem} .sub{color:#9aa0aa;margin-bottom:.8rem}
 .bar{position:sticky;top:0;background:#12141a;padding:.6rem 0;border-bottom:1px solid #2a2e37;z-index:5;
      display:flex;gap:.8rem;align-items:center;flex-wrap:wrap}
 table{border-collapse:collapse;width:100%} td,th{padding:.3rem .5rem;border-bottom:1px solid #23262e;text-align:left;vertical-align:middle}
 th{color:#9aa0aa;font-weight:600;font-size:.8rem;position:sticky;top:3rem;background:#12141a}
 tr.heard{opacity:.42} .stale{color:#f0c070} tr.bad{background:rgba(220,70,70,.13)}
 .n{color:#6b7280;font-variant-numeric:tabular-nums} .cd{color:#9aa0aa;font-family:ui-monospace,monospace;font-size:.8rem}
 .nat{color:#9aa0aa} .flag{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:#f0a0a0}
 audio{height:30px;width:15rem;vertical-align:middle}
 button{font:inherit;background:#232733;color:#e8e8ea;border:1px solid #343a46;border-radius:6px;padding:.25rem .6rem;cursor:pointer}
 .note{color:#9aa0aa;font-size:.8rem}
</style>
<h1>Vernacula voice sweep</h1>
<div class="sub">${rows.length} language-voice pairs · ${present.size} rendered · ${flagged} flagged by duration.
 Tick a row once you have listened to it; the ticks live in this browser, so keep using the same one.</div>
<div class="bar">
 <strong id="prog"></strong>
 <label><input type="checkbox" id="fUnheard"> unheard only</label>
 <label><input type="checkbox" id="fFlag"> flagged only</label>
 <label><input type="checkbox" id="fStale"> needs re-listening only</label>
 <label><input type="checkbox" id="fBad"> prior verdict not "good"</label>
 <button id="reset">clear ticks</button>
</div>
<table><thead><tr><th>✓</th><th>#</th><th>language</th><th>code</th><th>voice</th><th>audio</th><th>duration</th><th>previously</th></tr></thead><tbody>
${rows.map((r) => `<tr data-k="${esc(r.file)}" data-flag="${r.flag ? 1 : 0}" data-stale="${r.stale ? 1 : 0}" data-good="${/^good|^ok/iu.test(r.said) ? 1 : 0}" data-said="${r.said ? 1 : 0}"${r.flag ? ' class="bad"' : ""}>
 <td><input type="checkbox" class="hk"></td>
 <td class="n">${r.n}</td>
 <td>${esc(r.lang.name)}${r.lang.native ? ` <span class="nat">${esc(r.lang.native)}</span>` : ""}</td>
 <td class="cd">${esc(r.lang.code)}</td>
 <td class="cd">${esc(r.voice.id)}${r.donor ? ` <span class="flag">donor ${esc(DONOR_NAMES[r.donor] ?? r.donor)}</span>` : ""}</td>
 <td>${r.there ? `<audio controls preload="none" src="${esc(r.file)}"></audio>` : "<span class=note>not rendered</span>"}</td>
 <td>${r.flag ? `<span class="flag">${esc(r.flag)}</span> ` : ""}<span class="note">${esc(r.note)}</span></td>
 <td class="note${r.stale ? " stale" : ""}">${esc(r.said)}${r.stale ? " <em>(level artefact — re-listen)</em>" : ""}</td>
</tr>`).join("\n")}
</tbody></table>
<script>
// ⚠ Ticks are per-browser, and localStorage can throw outright (private windows, blocked site data),
// so every read and write is guarded and the page still renders with nothing stored.
const KEY = "vernacula-sweep-heard";
let heard = new Set();
try { heard = new Set(JSON.parse(localStorage.getItem(KEY) ?? "[]")); } catch {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify([...heard])); } catch {} };
const rows = [...document.querySelectorAll("tbody tr")];
const prog = document.getElementById("prog");
function paint() {
  const un = document.getElementById("fUnheard").checked, fl = document.getElementById("fFlag").checked;
  const st = document.getElementById("fStale").checked, bd = document.getElementById("fBad").checked;
  for (const tr of rows) {
    const on = heard.has(tr.dataset.k);
    tr.querySelector(".hk").checked = on;
    tr.classList.toggle("heard", on);
    tr.hidden = (un && on) || (fl && tr.dataset.flag !== "1")
      || (st && tr.dataset.stale !== "1")
      || (bd && (tr.dataset.good === "1" || tr.dataset.said !== "1"));
  }
  prog.textContent = heard.size + " / " + rows.length + " listened";
}
for (const tr of rows) tr.querySelector(".hk").addEventListener("change", (e) => {
  e.target.checked ? heard.add(tr.dataset.k) : heard.delete(tr.dataset.k); save(); paint();
});
// Ticking on play is what makes the sweep survive being interrupted: you never have to remember.
for (const a of document.querySelectorAll("audio")) a.addEventListener("ended", () => {
  const tr = a.closest("tr"); heard.add(tr.dataset.k); save(); paint();
});
for (const id of ["fUnheard", "fFlag", "fStale", "fBad"]) document.getElementById(id).addEventListener("change", paint);
document.getElementById("reset").addEventListener("click", () => { heard = new Set(); save(); paint(); });
paint();
</script>
`);
console.log(`index: ${join(outDir, "index.html")}  (${present.size}/${rows.length} rendered, ${flagged} flagged)`);
