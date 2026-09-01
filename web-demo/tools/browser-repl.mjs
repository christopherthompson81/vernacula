#!/usr/bin/env node
/**
 * Interactive console against a LONG-LIVED browser.
 *
 * `start` launches Chrome once with remote debugging and leaves it running; `eval` connects to it,
 * runs an expression in the page, prints the result, and disconnects WITHOUT closing the browser.
 * So each probe costs about a second instead of a full page load, and page state (a loaded model, a
 * half-finished run) survives between probes — which is the whole point when debugging a hang.
 *
 *   node tools/browser-repl.mjs start http://localhost:4188/
 *   node tools/browser-repl.mjs eval  "document.querySelector('.progress span')?.textContent"
 *   node tools/browser-repl.mjs logs                 # console + errors seen so far
 *   node tools/browser-repl.mjs stop
 */
import puppeteer from "puppeteer-core";
import fs from "node:fs";
import { spawn } from "node:child_process";

const PORT = 9222, LOGS = "/tmp/browser-repl.log";
const cmd = process.argv[2];

async function connect() {
  return puppeteer.connect({ browserURL: `http://127.0.0.1:${PORT}`, defaultViewport: null });
}

if (cmd === "start") {
  fs.writeFileSync(LOGS, "");
  const child = spawn("/usr/bin/google-chrome", [
    `--remote-debugging-port=${PORT}`, "--no-sandbox", "--disable-gpu-sandbox",
    "--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check",
    "--disable-search-engine-choice-screen", "--ozone-platform=x11", "--enable-gpu",
    "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-angle=vulkan",
    "--user-data-dir=/tmp/chrome-repl", process.argv[3] ?? "about:blank",
  ], { detached: true, stdio: "ignore" });
  child.unref();
  // ⚠ Listeners must live in a PERSISTENT process. Registering them here and exiting takes them
  // with it — which is why an instrumented build appeared to log nothing at all.
  for (let i = 0; i < 60; i++) {
    try { const b = await connect(); b.disconnect(); break; }
    catch { await new Promise(r => setTimeout(r, 500)); }
  }
  const watcher = spawn(process.execPath, [process.argv[1], "watch"], { detached: true, stdio: "ignore" });
  watcher.unref();
  await new Promise(r => setTimeout(r, 1200));
  console.log("browser ready on", PORT, "(log collector attached)"); process.exit(0);
}

const browser = await connect();
// ⚠ Pick the page by URL, not pages()[0]. A tab switch silently retargets every probe at whatever
// happens to be first — which produced a run of measurements from the wrong page that looked like
// a real defect and cost a round of debugging.
const MATCH = process.env.PAGE_MATCH ?? "localhost:4188";
const all = await browser.pages();
const page = all.find((p) => p.url().includes(MATCH)) ?? all[0];

if (cmd === "watch") {
  // Stays connected for the session, re-attaching on navigation, appending everything to LOGS.
  const attach = (p) => {
    p.on("console", (m) => fs.appendFileSync(LOGS, `[${m.type()}] ${m.text()}\n`));
    p.on("pageerror", (e) => fs.appendFileSync(LOGS, `[pageerror] ${String(e.stack ?? e).slice(0, 400)}\n`));
    p.on("response", (r) => { if (r.status() >= 400) fs.appendFileSync(LOGS, `[http ${r.status()}] ${r.url()}\n`); });
    p.on("requestfailed", (r) => fs.appendFileSync(LOGS, `[failed] ${r.url()} ${r.failure()?.errorText ?? ""}\n`));
  };
  for (const p of await browser.pages()) attach(p);
  browser.on("targetcreated", async (t) => { const p = await t.page(); if (p) attach(p); });
  await new Promise(() => {});   // run until the browser closes
} else if (cmd === "eval") {
  const expr = process.argv.slice(3).join(" ");
  try {
    const v = await page.evaluate(`(async () => { return (${expr}); })()`);
    console.log(typeof v === "string" ? v : JSON.stringify(v, null, 1));
  } catch (e) { console.error("EVAL ERROR: " + String(e).slice(0, 600)); }
} else if (cmd === "pages") {
  const ps = await browser.pages();
  ps.forEach((p, i) => console.log(`  [${i}] ${p.url()}`));
} else if (cmd === "goto") {
  // ⚠ Close every other tab first. `start` reuses the Chrome profile, so tabs accumulate across
  // runs and probes silently read an OLD one — which produced a set of measurements matching a
  // previous build and read as "the change did not apply". Twice.
  const ps = await browser.pages();
  for (const p of ps) if (p !== page) await p.close().catch(() => {});
  await page.goto(process.argv[3], { waitUntil: "domcontentloaded", timeout: 30000 });
  console.log("at", page.url(), `(closed ${ps.length - 1} other tab(s))`);
} else if (cmd === "logs") {
  console.log(fs.readFileSync(LOGS, "utf8").split("\n").slice(-Number(process.argv[3] ?? 25)).join("\n"));
} else if (cmd === "stop") {
  await browser.close(); console.log("closed");
}
browser.disconnect();
