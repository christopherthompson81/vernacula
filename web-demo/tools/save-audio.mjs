#!/usr/bin/env node
/** Pull the page's current audio blob out of the long-lived REPL browser and write it to a file.
 *  Listening is the only test that decides anything here, and a blob: URL cannot be opened from
 *  outside the page. */
import puppeteer from "puppeteer-core";
import { writeFileSync } from "node:fs";

const out = process.argv[2] ?? "/tmp/audio.wav";
const browser = await puppeteer.connect({ browserURL: "http://127.0.0.1:9222", defaultViewport: null });
const pages = await browser.pages();
const page = pages.find((p) => p.url().includes(process.env.PAGE_MATCH ?? "localhost:4188")) ?? pages[0];
const b64 = await page.evaluate(async () => {
  const a = document.querySelector("audio");
  if (!a) return null;
  const buf = new Uint8Array(await (await fetch(a.src)).arrayBuffer());
  let s = "";
  for (let i = 0; i < buf.length; i += 8192) s += String.fromCharCode(...buf.subarray(i, i + 8192));
  return btoa(s);
});
browser.disconnect();
if (!b64) { console.error("no audio on the page"); process.exit(1); }
writeFileSync(out, Buffer.from(b64, "base64"));
console.log(`${out} (${(Buffer.from(b64, "base64").length / 1e6).toFixed(2)} MB)`);
