#!/usr/bin/env node
/**
 * Drive the running dev server like a user: load the page, click Generate, report what happens.
 *
 * ⚠ THIS EXISTS BECAUSE THE FILE-SERVER SMOKES COULD NOT SEE A WHOLE CLASS OF BUG. They serve
 * public/ and build-smoke/ from a plain Node server, so Vite never participates — and Vite refusing
 * to serve a dynamic import of a file in public/ is invisible to them. Only the real dev server,
 * driven through the real UI, exercises that path.
 */
import puppeteer from "puppeteer-core";

const URL_ = process.env.URL ?? "http://localhost:5174/";
const WAIT = Number(process.env.WAIT_MS ?? 600000);

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || "/usr/bin/google-chrome",
  headless: false,                       // headless Chrome yields no GPU adapter on this box
  args: ["--no-sandbox", "--disable-gpu-sandbox", "--enable-unsafe-webgpu",
         "--no-first-run", "--no-default-browser-check", "--disable-search-engine-choice-screen",
         "--ozone-platform=x11", "--enable-gpu", "--ignore-gpu-blocklist",
         "--enable-features=Vulkan", "--use-angle=vulkan"],
  userDataDir: "/tmp/chrome-wgpu-probe",
});
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text().slice(0, 300)); });
page.on("pageerror", (e) => errors.push("pageerror: " + String(e).slice(0, 300)));
// Name the failing URLs — "a 404" is not actionable, the path is.
page.on("response", (r) => { if (r.status() >= 400) errors.push(`HTTP ${r.status()}  ${r.url()}`); });
page.on("requestfailed", (r) => errors.push(`FAILED  ${r.url()}  ${r.failure()?.errorText ?? ""}`));

await page.goto(URL_, { waitUntil: "networkidle2", timeout: 60000 });
console.log("  loaded:", await page.title());
await page.click("button");
console.log("  clicked Generate");

const t0 = Date.now();
let last = "";
while (Date.now() - t0 < WAIT) {
  const st = await page.evaluate(() => ({
    err: document.querySelector(".error")?.textContent ?? "",
    prog: document.querySelector(".progress span")?.textContent ?? "",
    ipa: document.querySelector(".pairs, .ipa-flat")?.textContent?.slice(0, 90) ?? "",
    audio: !!document.querySelector("audio"),
    stats: document.querySelector(".stats")?.textContent ?? "",
  }));
  if (st.err) { console.error("  UI ERROR: " + st.err); errors.forEach(e => console.error("  " + e)); await browser.close(); process.exit(3); }
  if (st.prog && st.prog !== last) { console.log("  " + st.prog); last = st.prog; }
  if (st.audio) {
    console.log("  IPA: " + st.ipa);
    console.log("  audio element present · " + st.stats);
    await browser.close(); process.exit(0);
  }
  await new Promise(r => setTimeout(r, 1500));
}
console.error("  timed out"); errors.forEach(e => console.error("  " + e));
await browser.close(); process.exit(4);
