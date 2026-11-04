// Headless Chrome driver over CDP: node shot.mjs <url> '<steps json>'
// steps: [{"eval": "js"}, {"wait": ms}, {"until": "js boolean", "timeout": ms}, {"shot": "file.png"}]
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url, stepsJson, outDir = "."] = process.argv.slice(2);
const steps = JSON.parse(stepsJson);
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "cfshot-"))}`,
  "--window-size=1920,1080", "--hide-scrollbars", "--no-first-run", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let list;
for (let i = 0; i < 50; i++) { try { list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); if (list.length) break; } catch {} await sleep(200); }
const page = list.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
  else if (d.method === "Runtime.exceptionThrown") console.log("PAGE EXCEPTION:", d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text);
  else if (d.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(d.params.type)) console.log("CONSOLE", d.params.type, d.params.args.map((a) => a.value ?? a.description).join(" "));
};
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result?.exceptionDetails) console.log("EVAL ERROR:", r.result.exceptionDetails.exception?.description); return r.result?.result?.value; };
await send("Runtime.enable"); await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });
await sleep(2500);
for (const s of steps) {
  if (s.evalFile) { const v = await ev((await import("node:fs")).readFileSync(s.evalFile, "utf8")); console.log(JSON.stringify(v, null, 1)); }
  if (s.eval) { const v = await ev(s.eval); if (v !== undefined) console.log("eval ->", JSON.stringify(v).slice(0, 400)); }
  if (s.wait) await sleep(s.wait);
  if (s.until) { const t0 = Date.now(); while (!(await ev(s.until)) && Date.now() - t0 < (s.timeout || 60000)) await sleep(500); console.log("until", s.until.slice(0, 60), "after", Date.now() - t0, "ms"); }
  if (s.shot) { const r = await send("Page.captureScreenshot", { format: "png" }); writeFileSync(join(outDir, s.shot), Buffer.from(r.result.data, "base64")); console.log("shot", s.shot); }
}
ws.close(); chrome.kill();
process.exit(0);
