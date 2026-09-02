import puppeteer from "puppeteer-core";
const [code, text] = [process.argv[2], process.argv[3]];
const b = await puppeteer.connect({ browserURL: "http://127.0.0.1:9222", defaultViewport: null });
const pages = await b.pages();
const page = pages.find((p) => p.url().includes("localhost:4188")) ?? pages[0];
await page.evaluate(async (code, text) => {
  const setV = (el, v) => {
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(el, v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  };
  const inp = document.querySelector(".picker input");
  inp.focus(); setV(inp, code);
  await new Promise(r => setTimeout(r, 150));
  const li = [...document.querySelectorAll(".options li")].find(l => l.querySelector(".cd")?.textContent === code)
           ?? document.querySelector(".options li");
  li.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
  await new Promise(r => setTimeout(r, 200));
  if (text) setV(document.querySelector("textarea"), text);
  await new Promise(r => setTimeout(r, 100));
  document.querySelector("button").click();
}, code, text ?? "");
b.disconnect();
console.log("generating", code);
