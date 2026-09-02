#!/usr/bin/env node
/**
 * Generate one clip per language through the REAL page, and save it for a listening test.
 *
 * ⚠ Listening is the only test that decides whether a sourced voice is usable. Automated scores
 * (noise floor, speech fraction) rank candidates; they cannot hear a bad read, a second speaker, a
 * cough, or a reference whose accent is wrong for the language. This drives the deployed page so
 * what is judged is what a visitor gets.
 *
 *   node tools/preview-langs.mjs /tmp/listen ab sq eu …
 *
 * ⚠ RELOAD THE PAGE FIRST after changing voices.jsonc — `node tools/browser-repl.mjs goto <url>`.
 * The engine loads the voice list once per page load and keeps it, so a preview run against a stale
 * tab renders with the OLD voice and looks like the new one failed: Akan came back read by the
 * Yoruba donor it had just been given a native replacement for.
 */
import puppeteer from "puppeteer-core";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [outDir, ...langs] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const browser = await puppeteer.connect({ browserURL: "http://127.0.0.1:9222", defaultViewport: null });
const pages = await browser.pages();
const page = pages.find((p) => p.url().includes("localhost:4188")) ?? pages[0];

for (const code of langs) {
  const stats = await page.evaluate(async (code) => {
    const setV = (el, v) => {
      const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, "value").set.call(el, v);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    };
    const inp = document.querySelector(".picker input");
    inp.focus(); setV(inp, code);
    await new Promise((r) => setTimeout(r, 200));
    const li = [...document.querySelectorAll(".options li")]
      .find((l) => l.querySelector(".cd")?.textContent === code);
    if (!li) return { err: `no picker row for ${code}` };
    li.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 300));
    if (!document.querySelector("textarea").value.trim()) return { err: `${code} has no sample text` };
    // ⚠ Wait for the stats line to CHANGE, not merely to exist. The previous language's stats and
    // audio are still on the page when Generate is clicked, and the progress bar has not rendered
    // on the first poll — so "stats present and not busy" was already true, and the run saved the
    // PREVIOUS language's audio under this language's name.
    const before = document.querySelector(".stats")?.textContent ?? "";
    document.querySelector("button").click();
    for (let i = 0; i < 600; i++) {
      await new Promise((r) => setTimeout(r, 500));
      const err = document.querySelector(".error")?.textContent;
      if (err) return { err };
      const st = document.querySelector(".stats")?.textContent;
      if (st && st !== before && !document.querySelector(".progress")) return { stats: st };
    }
    return { err: "timed out" };
  }, code);

  if (stats.err) { console.log(`  ${code}: FAILED — ${stats.err}`); continue; }
  const b64 = await page.evaluate(async () => {
    const buf = new Uint8Array(await (await fetch(document.querySelector("audio").src)).arrayBuffer());
    let s = ""; for (let i = 0; i < buf.length; i += 8192) s += String.fromCharCode(...buf.subarray(i, i + 8192));
    return btoa(s);
  });
  writeFileSync(join(outDir, `${code}.wav`), Buffer.from(b64, "base64"));
  console.log(`  ${code}: ${stats.stats}`);
}
browser.disconnect();
