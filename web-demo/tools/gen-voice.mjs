import puppeteer from "puppeteer-core";
const idx = Number(process.argv[2]);
const b = await puppeteer.connect({ browserURL: "http://127.0.0.1:9222", defaultViewport: null });
const pages = await b.pages();
const page = pages.find(p => p.url().includes("localhost:4188")) ?? pages[0];
await page.evaluate((idx) => {
  const sel = document.querySelectorAll(".controls select")[0];
  const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set;
  set.call(sel, sel.options[idx].value);
  sel.dispatchEvent(new Event("change", { bubbles: true }));
  setTimeout(() => document.querySelector("button").click(), 200);
}, idx);
b.disconnect();
console.log("voice", idx);
