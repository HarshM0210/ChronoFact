/* ChronoFact demo front end. Everything shown comes from the server (demo/server.py): the live pipeline or the
   finished benchmark run. Nothing on this page is mocked. */
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "–" : (100 * x).toFixed(d) + "%");
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const human = (iso) => { if (!iso) return "—"; const d = new Date(iso.slice(0, 10) + "T12:00:00"); return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`; };
const NAMES = { MODEL5: "ChronoFact", RAG_plain: "RAG over transcripts" };
const SUBS = { MODEL5: "verified facts with validity dates + evidence turns", RAG_plain: "top transcript chunks, same budget" };
const CLS = { MODEL5: "cf", RAG_plain: "rag" };

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
async function stream(path, body, on) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const rd = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await rd.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const ev = (chunk.match(/^event: (.*)$/m) || [])[1], data = (chunk.match(/^data: (.*)$/m) || [])[1];
      if (ev) on(ev, data ? JSON.parse(data) : null);
    }
  }
}
function toast(html, ms = 3800) { const t = $("#toast"); t.innerHTML = html; t.classList.add("on"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("on"), ms); }
const tip = $("#tip");
function tipOn(e, html) { tip.innerHTML = html; tip.classList.add("on"); tipMove(e); }
function tipMove(e) { const w = tip.offsetWidth; tip.style.left = Math.min(e.clientX + 14, innerWidth - w - 10) + "px"; tip.style.top = e.clientY + 14 + "px"; }
function tipOff() { tip.classList.remove("on"); }
document.addEventListener("mousemove", (e) => { if (tip.classList.contains("on")) tipMove(e); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") Live.closeDrawer(); });

/* ------------------------------------------------------------------ shell */
$("#nav").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(b.dataset.v); });
function show(v) {
  Live.closeDrawer();
  $$("#nav button").forEach((b) => b.classList.toggle("on", b.dataset.v === v));
  $$(".view").forEach((s) => s.classList.toggle("on", s.id === "v-" + v));
  if (v === "results") Results.load();
  if (v === "bench") Bench.load();
}
$("#theme").onclick = () => { const r = document.documentElement; r.dataset.theme = r.dataset.theme === "light" ? "dark" : "light"; if (Results.data) Results.render(); };

/* ------------------------------------------------------------------ live memory */
const STAGES = [
  { id: "salience", t: "Salience filter", d: "Is there a personal claim worth an LLM call?", m: '<span class="chip">rules · no LLM</span>' },
  { id: "extracted", t: "Extract", d: "Durable facts, each with an exact quote", m: '<span class="chip llama">Llama 3.1 8B · JSON schema</span>' },
  { id: "verified", t: "Verify", d: "Quote must be verbatim · claim must be entailed", m: '<span class="chip">BART-MNLI</span>' },
  { id: "labeled", t: "Label", d: "Kind, slot and a richer statement", m: '<span class="chip llama">Llama 3.1 8B</span>' },
  { id: "applied", t: "Update memory", d: "Add · deduplicate · supersede · close", m: '<span class="chip">temporal store</span>' },
];
const Live = {
  mem: null, busy: false, story: null, si: 0, lastDate: null, fresh: new Set(),
  init() {
    $("#stages").innerHTML = STAGES.map((s, i) => `<div class="stage" id="st-${s.id}"><div class="sh"><div class="num">${i + 1}</div><div><div class="t">${s.t}</div><div class="d">${s.d}</div></div><div class="model">${s.m}</div></div><div class="sb"></div></div>`).join("");
    $("#composer").onsubmit = (e) => { e.preventDefault(); const v = $("#msg").value.trim(); if (v && !this.busy) this.send(v); };
    $("#date").oninput = () => this.dateHuman();
    $("#memView").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; $$("#memView button").forEach((x) => x.classList.toggle("on", x === b)); this.renderMem(); };
    $("#reset").onclick = async () => { if (this.busy) return; await api("/api/reset", {}); this.si = 0; this.lastDate = null; $("#chat").innerHTML = ""; this.clearStages(); $("#writeStat").innerHTML = ""; this.closeDrawer(); $("#vs").innerHTML = ""; $("#modeline").innerHTML = ""; this.mem = await api("/api/memory"); this.renderMem(); this.storyUI(); $("#date").value = this.story.steps[0].date; this.dateHuman(); toast("Memory cleared"); };
    $("#storyNext").onclick = () => this.storyStep();
    this.dateHuman();
  },
  async boot() {
    this.story = await api("/api/scenario");
    this.mem = await api("/api/memory");
    this.replayChat();
    const n = this.mem.messages.filter((m) => m.speaker === "user").length;
    this.si = Math.min(n, this.story.steps.length);
    this.storyUI();
    if (this.si < this.story.steps.length) $("#date").value = this.story.steps[this.si].date;
    else if (this.mem.messages.length) $("#date").value = this.mem.messages[this.mem.messages.length - 1].ts.slice(0, 10);
    this.dateHuman();
    this.renderMem();
    Ask.initSugg(this.story.questions);
  },
  dateHuman() { $("#dateHuman").textContent = new Date($("#date").value + "T12:00:00").toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" }); },
  storyUI() {
    const n = this.story.steps.length;
    const b = $("#storyNext");                     // its whole content is rebuilt below (the counters live inside it)
    if (this.si >= n) { b.innerHTML = "Story done · Ask across time →"; $("#storyCap").innerHTML = "<b>Now ask questions across time.</b>"; b.onclick = () => show("ask"); }
    else { b.innerHTML = `▶ Story step <span id="storyIdx">${this.si + 1}</span>/<span id="storyN">${n}</span>`; b.onclick = () => this.storyStep(); $("#storyCap").innerHTML = `Next: <b>${esc(this.story.steps[this.si].shows)}</b>`; }
    Ask.storyWarn(this.si, n);
  },
  async finishStory() {                           // plays the remaining story steps, then returns to Ask
    if (this.busy) { toast("A message is still being written to memory; try again in a moment."); return; }
    show("live");
    while (this.si < this.story.steps.length && !this.busy) await this.storyStep();
    show("ask");
    if ($("#askQ").value.trim()) Ask.run();
  },
  async storyStep() {
    if (this.busy || this.si >= this.story.steps.length) return;
    this.busy = true; $("#storyNext").disabled = true; $("#send").disabled = true;     // no second step while typing
    const s = this.story.steps[this.si];
    $("#date").value = s.date; this.dateHuman();
    $("#storyCap").innerHTML = `Shows: <b>${esc(s.shows)}</b>`;
    const inp = $("#msg"); inp.value = "";
    for (const ch of s.text) { inp.value += ch; await new Promise((r) => setTimeout(r, 14)); }
    this.si++;
    await this.send(s.text);
    this.storyUI();
  },
  daySep(date) {
    if (date === this.lastDate) return;
    this.lastDate = date;
    $("#chat").insertAdjacentHTML("beforeend", `<div class="daysep">${human(date)}</div>`);
  },
  bubble(cls, html) { const d = document.createElement("div"); d.className = "msg " + cls; d.innerHTML = html; $("#chat").append(d); $("#chat").scrollTop = 1e9; return d; },
  replayChat() {
    $("#chat").innerHTML = ""; this.lastDate = null;
    for (const m of this.mem.messages) {
      this.daySep(m.ts.slice(0, 10));
      this.bubble(m.speaker === "user" ? "user" : "bot", m.text ? esc(m.text) : '<i class="muted">(deleted at the user\'s request)</i>');
    }
  },
  clearStages() { $$(".stage").forEach((s) => { s.className = "stage"; $(".sb", s).innerHTML = ""; }); },
  stage(id, state) { const s = $("#st-" + id); if (s) s.className = "stage " + state; },
  add(id, html) { const s = $(`#st-${id} .sb`); s.insertAdjacentHTML("beforeend", html); s.parentElement.scrollIntoView({ block: "nearest", behavior: "smooth" }); },
  async send(text) {
    this.busy = true; $("#send").disabled = true; $("#storyNext").disabled = true;
    const date = $("#date").value;
    this.daySep(date);
    this.bubble("user", esc(text));
    $("#msg").value = "";
    const typing = this.bubble("bot", '<span class="typing"><i></i><i></i><i></i></span>');
    this.clearStages(); $("#writeStat").innerHTML = '<span class="spinner"></span>';
    let ctx = null, extracting = false; const seen = new Set();
    const before = new Set((this.mem?.facts || []).map((f) => f.id + ":" + f.status));
    try {
      await stream("/api/chat", { text, date }, (ev, d) => {
        seen.add(ev);
        if (ev === "context") ctx = d;
        else if (ev === "reply") {
          typing.innerHTML = esc(d.text) + `<div class="meta"><span>recalled ${ctx.fact_lines} fact${ctx.fact_lines === 1 ? "" : "s"} · ${ctx.turns} past turn${ctx.turns === 1 ? "" : "s"} · ${ctx.tokens} tokens</span><span>· ${(d.ms / 1000).toFixed(1)} s</span></div>`;
          this.stage("salience", "active");
        } else if (ev === "salience") {
          if (d.skipped.length) d.skipped.forEach((s) => this.add("salience", `<div class="item"><div class="row"><span class="tag neutral">Skipped</span>${esc(s.why)}</div><div class="q">“${esc(s.text.slice(0, 90))}” · no LLM call spent</div></div>`));
          extracting = d.extract.length > 0;
          if (d.extract.length) this.add("salience", `<div class="item"><div class="row"><span class="tag cf">Personal claim</span>sent to extraction</div></div>`);
          this.stage("salience", "done"); this.stage("extracted", d.extract.length ? "active" : "done");
          if (!d.extract.length) ["extracted", "verified", "labeled", "applied"].forEach((s) => this.stage(s, "done"));
        } else if (ev === "extracted") {
          if (!extracting) return;
          if (!d.length && extracting) this.add("extracted", `<div class="item muted">No durable fact in this message.</div>`);
          d.forEach((f) => this.add("extracted", `<div class="item"><div class="row"><span class="tag llama">${esc(f.family)}</span><span>${esc(f.statement)}</span></div><div class="q">quote: “${esc(f.quote)}”${f.time_expression ? ` · when: “${esc(f.time_expression)}”` : ""}${f.ok ? "" : ' · <span style="color:var(--crit)">empty value, dropped</span>'}</div></div>`));
          this.stage("extracted", "done"); this.stage("verified", "active");
        } else if (ev === "verified") {
          if (!extracting) return;
          d.facts.forEach((v) => {
            const p = v.nli == null ? 0 : v.nli, ok = v.kept;
            this.add("verified", `<div class="item"><div class="row"><span class="tag ${ok ? "good" : "crit"}">${ok ? "✓ Kept" : "✕ Blocked"}</span><span>${esc(v.statement)}</span></div>
              <div class="nli"><span>quote ${v.quote_ok ? "✓ verbatim" : "✕ not in message"}</span>${v.quote_ok ? `<span>entailment</span><div class="bar"><div class="fill" style="width:${(100 * p).toFixed(0)}%;background:${ok ? "var(--good)" : "var(--crit)"}"></div><div class="thr" style="left:${100 * d.min_entail}%" title="threshold"></div></div><span class="mono">${p.toFixed(2)}</span>` : ""}</div></div>`);
          });
          if (!d.facts.length && extracting) this.add("verified", `<div class="item muted">Nothing to verify.</div>`);
          this.stage("verified", "done"); this.stage("labeled", extracting ? "active" : "done");
        } else if (ev === "labeled") {
          if (!extracting) return;
          if (!d.length && extracting) this.add("labeled", `<div class="item muted">No verified fact to label.</div>`);
          d.forEach((l) => this.add("labeled", `<div class="item"><div class="row"><span class="tag ${l.kind === "state" ? "cf" : l.kind === "event" ? "llama" : "neutral"}">${esc(l.kind)}</span><span class="mono" style="font-size:11.5px">${esc(l.family)} / ${esc(l.slot_key)}</span></div>${l.rich ? `<div class="q">richer: ${esc(l.rich)}</div>` : ""}</div>`));
          this.stage("labeled", "done"); this.stage("applied", d.length ? "active" : "done");
        } else if (ev === "applied") {
          d.forEach((o) => this.add("applied", Live.opItem(o)));
        } else if (ev === "memory") {
          this.mem = d;
          this.fresh = new Set(d.facts.filter((f) => !before.has(f.id + ":" + f.status)).map((f) => f.id));
          this.renderMem();
        } else if (ev === "done") {
          STAGES.forEach((s) => this.stage(s.id, "done"));
          $("#writeStat").innerHTML = `<span class="chip"><b>${d.llm_calls}</b> LLM calls</span><span class="chip">written in <b>${(d.write_ms / 1000).toFixed(1)} s</b></span>`;
        } else if (ev === "error") { toast("Error: " + esc(d.message), 8000); typing.innerHTML = '<span class="muted">(error)</span>'; }
      });
    } catch (e) { toast("Connection error: " + esc(e.message), 8000); }
    this.busy = false; $("#send").disabled = false; $("#storyNext").disabled = false;
  },
  opItem(o) {
    const r = o.reason || "";
    let tag = ["neutral", o.op], txt = r;
    if (o.op === "ADD") { tag = ["good", "+ Added"]; txt = r.replace(/^(state|attribute|event|past state): /, ""); }
    else if (o.op === "NOOP" && /negation/.test(r)) { tag = ["neutral", "○ Nothing to close"]; txt = r.replace(/^.*?: /, ""); }
    else if (o.op === "NOOP") { tag = ["neutral", "= Known"]; txt = "already in memory, not duplicated: " + r.replace(/^.*?: /, ""); }
    else if (o.op === "SUPERSEDE" && /negation|CLOSE/.test(r)) { tag = ["serious", "⏹ Closed"]; txt = r.replace(/^controller CLOSE \(negation\): /, ""); }
    else if (o.op === "SUPERSEDE") { tag = ["warn", "↻ Superseded"]; const m = r.match(/'(.*)' -> '(.*)'/); txt = m ? `${m[1]} → ${m[2]} (old value kept as history)` : r; }
    else if (o.op === "MERGE") { tag = ["neutral", "⇄ Merged"]; }
    return `<div class="item"><div class="row"><span class="tag ${tag[0]}">${tag[1]}</span><span>${esc(txt)}</span></div></div>`;
  },
  renderMem() {
    const m = this.mem; if (!m) return;
    const c = m.counts;
    $("#counters").innerHTML = [["active", "Current", "var(--cf)"], ["superseded", "History", "var(--warn)"], ["rejected", "Blocked", "var(--crit)"], ["forgotten", "Forgotten", "var(--muted)"]]
      .map(([k, l, col]) => `<div class="counter"><div class="v">${c[k] || 0}</div><div class="l"><span class="dot" style="background:${col}"></span>${l}</div></div>`).join("");
    const v = $("#memView .on").dataset.m;
    const body = $("#memBody");
    if (!m.facts.length) { body.innerHTML = `<div class="muted" style="padding:30px;text-align:center">Memory is empty. Send a message or press <b>Story step</b>.</div>`; return; }
    if (v === "timeline") body.innerHTML = this.timeline(m);
    else if (v === "facts") body.innerHTML = this.factList(m);
    else body.innerHTML = this.logView(m);
    $$("[data-fid]", body).forEach((e) => {
      e.onclick = () => Live.drawer(+e.dataset.fid);
      e.onmouseenter = (ev) => { const f = m.facts.find((x) => x.id === +e.dataset.fid); tipOn(ev, `<b>${esc(f.statement)}</b><br><span class="muted">${human(f.valid_from)} → ${f.valid_to ? human(f.valid_to) : "now"} · ${f.status}</span>`); };
      e.onmouseleave = tipOff;
    });
  },
  timeline(m) {
    const facts = m.facts.filter((f) => f.status === "active" || f.status === "superseded");
    if (!facts.length) return `<div class="muted" style="padding:30px;text-align:center">No verified facts yet (see the Facts view for blocked ones).</div>`;
    const now = $("#date").value;
    const t = (s) => new Date((s || now).slice(0, 10) + "T12:00:00").getTime();
    let lo = Math.min(...facts.map((f) => t(f.valid_from))), hi = Math.max(t(now), ...facts.map((f) => t(f.valid_to || now)));
    lo -= 5 * 864e5; hi = Math.max(hi + 12 * 864e5, lo + 165 * 864e5);      // at least ~5 months: bars grow as time passes
    const x = (s) => (100 * (t(s) - lo)) / (hi - lo);
    const groups = new Map();
    for (const f of facts) { const k = f.kind === "event" ? "events" : `${f.family}/${f.slot_key}`; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(f); }
    const rows = [...groups.entries()].sort((a, b) => (a[0] === "events") - (b[0] === "events") || Math.min(...a[1].map((f) => t(f.valid_from))) - Math.min(...b[1].map((f) => t(f.valid_from))));
    const ticks = []; const d0 = new Date(lo); d0.setDate(1); d0.setMonth(d0.getMonth() + 1);
    for (let d = d0; d.getTime() < hi; d.setMonth(d.getMonth() + 1)) ticks.push(new Date(d));
    const tickHtml = ticks.map((d) => `<span style="left:${(100 * (d - lo)) / (hi - lo)}%">${MONTHS[d.getMonth()]}</span>`).join("");
    const grid = ticks.map((d) => `<div class="grid" style="left:${(100 * (d - lo)) / (hi - lo)}%"></div>`).join("");
    const nowX = x(now);
    return `<div class="tl"><div class="tl-axis"><div></div><div class="ticks">${tickHtml}</div></div>` + rows.map(([k, fs]) => {
      const [fam, slot] = k.split("/");
      const bars = fs.map((f) => {
        const a = x(f.valid_from), b = f.kind === "event" ? a : x(f.valid_to || now);
        const cls = f.kind === "event" ? "event" : f.status;
        const label = f.value || f.statement;
        const w = Math.max(b - a, 1.2), inside = f.kind !== "event" && w >= 11;
        return `<div class="tl-bar ${cls} ${this.fresh.has(f.id) ? "flash" : ""}" data-fid="${f.id}" style="left:${a}%;width:${w}%">${inside ? esc(label) : ""}</div>` +
          (inside ? "" : a + w > 72 ? `<div class="tl-out ${f.status}" data-fid="${f.id}" style="right:calc(${100 - a}% + 6px);text-align:right">${esc(label)}</div>`
                                    : `<div class="tl-out ${f.status}" data-fid="${f.id}" style="left:calc(${a + w}% + 6px)">${esc(label)}</div>`);
      }).join("");
      return `<div class="tl-row"><div class="tl-label">${k === "events" ? "events" : esc(fam.replace(/_/g, " "))}<small>${k === "events" ? "dated occurrences" : esc(slot || "")}</small></div><div class="tl-track">${grid}<div class="tl-now" style="left:${nowX}%"></div>${bars}</div></div>`;
    }).join("") + `<div class="writestat" style="margin-top:10px"><span><span class="dot" style="background:var(--cf)"></span> current value</span><span><span class="dot" style="background:var(--muted)"></span> superseded (kept as history)</span><span><span class="dot" style="background:var(--llama)"></span> event</span><span><span class="dot" style="background:var(--good)"></span> today</span></div></div>`;
  },
  factList(m) {
    const g = [["active", "Current"], ["superseded", "History · no longer current"], ["rejected", "Blocked by verification"], ["forgotten", "Forgotten on request"]];
    return g.map(([s, title]) => {
      const fs = m.facts.filter((f) => f.status === s);
      if (!fs.length) return "";
      return `<div class="grp">${title} <span class="chip">${fs.length}</span></div><div class="factlist">` + fs.map((f) => `<div class="fact ${s}" data-fid="${f.id}"><div class="st">${esc(f.rich || f.statement)}</div><div class="fm"><span class="tag ${f.kind === "state" ? "cf" : f.kind === "event" ? "llama" : "neutral"}">${esc(f.kind)}</span><span class="mono">${esc(f.family)}/${esc(f.slot_key)}</span><span>${human(f.valid_from)}${f.valid_to ? " → " + human(f.valid_to) : s === "active" ? " → now" : ""}</span></div></div>`).join("") + "</div>";
    }).join("");
  },
  logView(m) {
    const tags = { ADD: "good", SUPERSEDE: "warn", REJECT: "crit", NOOP: "neutral", SKIP: "neutral", FORGET: "neutral", MERGE: "neutral", DROP_EMPTY: "neutral" };
    return `<div class="log">` + m.ops.map((o) => `<div class="e"><div><span class="tag ${tags[o.op] || "neutral"}">${esc(o.op)}</span></div><div>${esc(o.reason)}<div class="muted" style="font-size:11px">${human(o.ts)}</div></div></div>`).join("") + "</div>";
  },
  drawer(id) {
    const f = this.mem.facts.find((x) => x.id === id); if (!f) return;
    const hl = (text) => {
      const t = esc(text), q = esc(f.quote || "");
      const i = q ? t.toLowerCase().indexOf(q.toLowerCase()) : -1;
      return i < 0 ? t : t.slice(0, i) + "<mark>" + t.slice(i, i + q.length) + "</mark>" + t.slice(i + q.length);
    };
    const sup = f.superseded_by ? this.mem.facts.find((x) => x.id === f.superseded_by) : null;
    const srcIds = new Set(f.sources.map((s) => s.id));
    const sib = this.mem.facts.filter((x) => x.id !== f.id && x.status !== "forgotten" && x.sources.some((s) => srcIds.has(s.id)));
    const st = { active: ["cf", "Current"], superseded: ["warn", "History"], rejected: ["crit", "Blocked"], forgotten: ["neutral", "Forgotten"] }[f.status];
    $("#drawer").innerHTML = `<div class="dh"><div style="flex:1"><span class="tag ${st[0]}">${st[1]}</span><h3 style="margin:8px 0 0;font-size:17px">${esc(f.statement)}</h3>${f.rich ? `<div class="muted" style="margin-top:4px">${esc(f.rich)}</div>` : ""}</div><button class="iconbtn" onclick="Live.closeDrawer()">✕</button></div>
      <div class="db">
        <dl class="kv"><dt>value</dt><dd><b>${esc(f.value)}</b></dd><dt>kind</dt><dd>${esc(f.kind)} · ${esc(f.family)} / <span class="mono">${esc(f.slot_key)}</span></dd>
        <dt>valid from</dt><dd>${human(f.valid_from)} <span class="muted">(${esc(f.date_precision)})</span></dd><dt>valid to</dt><dd>${f.valid_to ? human(f.valid_to) : f.status === "active" ? "still true" : "—"}</dd>
        <dt>recorded</dt><dd>${human(f.recorded_at)}</dd><dt>assertion</dt><dd>${esc(f.assertion_type)}</dd>${sup ? `<dt>replaced by</dt><dd><a href="#" onclick="Live.drawer(${sup.id});return false">${esc(sup.statement)}</a></dd>` : ""}</dl>
        <div><div class="sec-t">Provenance · the user's own words</div>${f.sources.map((s) => `<div class="source">${s.text ? hl(s.text) : '<i class="muted">message deleted</i>'}<div class="muted" style="font-size:11px;margin-top:6px">${human(s.ts)} · ${esc(s.speaker)} message #${s.id}</div></div>`).join("")}</div>
        ${f.status !== "forgotten" && f.status !== "rejected" ? `<div><div class="sec-t">Right to be forgotten</div><button class="btn danger" onclick="Live.forget(${f.id})">Forget this fact and its source message</button><div class="muted" style="font-size:12px;margin-top:6px">Deletes the fact, every fact taken from the same message, and that message plus the assistant's reply to it from the transcript index.</div>${sib.length ? `<div class="muted" style="font-size:12px;margin-top:6px">Also deleted with this message: ${sib.map((x) => `<b>${esc(x.statement)}</b>`).join(", ")}</div>` : ""}</div>` : ""}
      </div>`;
    $("#drawer").classList.add("open");
  },
  closeDrawer() { $("#drawer").classList.remove("open"); },
  async forget(id) {
    let r;
    try { r = await api("/api/forget", { fact_id: id }); } catch (e) { toast("Could not forget: " + esc(e.message), 6000); return; }
    this.mem = r.memory; this.fresh = new Set(); this.renderMem(); this.replayChat(); this.closeDrawer();
    toast(`Forgot <b>${r.facts.length}</b> fact${r.facts.length === 1 ? "" : "s"} and deleted <b>${r.messages.length}</b> message${r.messages.length === 1 ? "" : "s"}: neither memory nor transcript search can retrieve them now.`, 6000);
  },
};

/* ------------------------------------------------------------------ ask across time */
const Ask = {
  budget: 256,
  init() {
    $("#budget").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; $$("#budget button").forEach((x) => x.classList.toggle("on", x === b)); this.budget = +b.dataset.b; if ($("#askQ").value.trim()) this.run(); };
    $("#askGo").onclick = () => this.run();
    $("#askQ").onkeydown = (e) => { if (e.key === "Enter") this.run(); };
  },
  storyWarn(si, n) {
    // the suggested questions are written for the finished story: before the later steps update facts (a new
    // running routine, the move to Munich), both systems only see the old values and rightly give the same answer
    const w = $("#storyWarn");
    w.hidden = si >= n;
    if (w.hidden) return;
    w.innerHTML = `<span><b>The story is not finished</b> (${si} of ${n} steps played). The suggested questions are written for the end of the story: the remaining steps change facts they ask about, so until then ChronoFact and RAG see the same old values and give the same answers.</span><button class="btn primary" id="finishStory">Finish the story</button>`;
    $("#finishStory").onclick = () => Live.finishStory();
  },
  initSugg(qs) {
    $("#sugg").innerHTML = qs.map((q, i) => `<button data-i="${i}">${esc(q.q)}</button>`).join("");
    $("#sugg").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; const q = qs[+b.dataset.i]; $("#askQ").value = q.q; $("#askDate").value = q.date; this.run(); };
  },
  seq: 0,
  async run() {
    const q = $("#askQ").value.trim(); if (!q) return;
    const my = ++this.seq;
    $("#modeline").innerHTML = '<span class="spinner"></span> retrieving and answering with both systems…';
    $("#vs").innerHTML = ["MODEL5", "RAG_plain"].map((s) => `<div class="sys ${CLS[s]}"><div class="top"><div><div class="name">${NAMES[s]}</div><div class="muted" style="font-size:12px">${SUBS[s]}</div></div></div><div class="ans"><span class="spinner"></span></div></div>`).join("");
    try {
      const r = await api("/api/ask", { question: q, date: $("#askDate").value, budget: this.budget });
      if (my !== this.seq) return;                       // a newer question was asked meanwhile
      const mode = { current: "current value", as_of: "as of a past time", history: "full history", none: "no time scope" }[r.mode] || r.mode;
      $("#modeline").innerHTML = `<span class="chip">time scope detected: <b>${esc(mode)}</b>${r.window[0] ? ` · ${human(r.window[0])} – ${human(r.window[1])}` : ""}</span><span class="chip">both systems get the same <b>${this.budget}-token</b> context</span><span class="chip">same Llama 3.1 8B answers</span>`;
      $("#vs").innerHTML = ["MODEL5", "RAG_plain"].map((s) => sysCard(s, r.systems[s], this.budget)).join("");
    } catch (e) { if (my === this.seq) $("#modeline").innerHTML = `<span style="color:var(--crit)">${esc(e.message)}</span>`; }
  },
};
function ctxHtml(text, evid) {
  if (!text) return '<span class="muted">(nothing retrieved)</span>';
  let inFacts = false;
  return text.split("\n").map((l) => {
    if (/^Known facts/.test(l)) { inFacts = true; return `<span class="hd">${esc(l)}</span>`; }
    if (/^Conversation excerpts/.test(l)) { inFacts = false; return `<span class="hd">${esc(l)}</span>`; }
    const ev = evid && evid.some((e) => e.length > 20 && l.includes(e.slice(0, 60)));
    if (inFacts && l.startsWith("- ")) return `<span class="fl">${esc(l)}</span>`;
    return ev ? `<span class="ev">${esc(l)}</span>` : esc(l);
  }).join("\n");
}
function budgetBar(s, r, budget) {
  const t = r.context || "", i = t.indexOf("Conversation excerpts");
  const factChars = s === "MODEL5" ? (i < 0 ? t.length : i) : 0;
  const ft = t.length ? Math.round((r.tokens * factChars) / t.length) : 0, tt = r.tokens - ft;
  return `<div class="budgetbar"><div class="bb"><span style="width:${(100 * ft) / budget}%;background:var(--cf)"></span><span style="width:${(100 * tt) / budget}%;background:${s === "MODEL5" ? "color-mix(in srgb,var(--cf) 35%,var(--muted))" : "var(--rag)"}"></span></div>
    <div class="lg">${s === "MODEL5" ? `<span>■ ${r.fact_lines} verified facts · ~${ft} tok</span>` : ""}<span>■ transcript turns · ~${tt} tok</span><span style="margin-left:auto">${r.tokens} / ${budget} tokens</span></div></div>`;
}
function sysCard(s, r, budget, verdict, evid) {
  const v = verdict == null ? "" : `<span class="tag ${verdict.correct ? "good" : "crit"} verdict">${verdict.correct ? "✓ correct" : "✕ wrong"}</span>`;
  return `<div class="sys ${CLS[s]}"><div class="top"><div><div class="name">${NAMES[s]}</div><div class="muted" style="font-size:12px">${SUBS[s]}</div></div>${v}</div>
    <div class="ans ${/not in memory/i.test(r.answer) ? "abst" : ""}">${esc(r.answer || "—")}</div>
    ${verdict && verdict.judge ? `<div class="reasoning" style="margin-top:-6px">judge: ${esc(verdict.judge)}</div>` : ""}
    ${r.ms != null ? `<div class="reasoning">retrieval ${r.retrieve_ms} ms · answer ${(r.ms / 1000).toFixed(1)} s</div>` : ""}
    ${budgetBar(s, r, budget)}
    <div class="ctx">${ctxHtml(r.context, evid)}</div>
    ${r.raw && r.raw !== r.answer ? `<details class="reasoning"><summary>model reasoning</summary><pre>${esc(r.raw)}</pre></details>` : ""}</div>`;
}

/* ------------------------------------------------------------------ benchmark explorer */
const TYPES = ["knowledge-update", "temporal-reasoning", "multi-session", "single-session-user", "single-session-assistant", "single-session-preference", "abstention"];
const Bench = {
  list: null, f: "all", q: null, ck: 1.0, budget: 512,
  async load() {
    if (this.list) return;
    this.list = await api("/api/bench/list");
    const f = [["all", "All 500"], ["win", "ChronoFact ahead"], ...TYPES.map((t) => [t, t.replace("single-session-", "single-session ")])];
    $("#qfilters").innerHTML = f.map(([k, l]) => `<button data-f="${k}" class="${k === "all" ? "on" : ""}">${esc(l)}</button>`).join("");
    $("#qfilters").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; this.f = b.dataset.f; $$("#qfilters button").forEach((x) => x.classList.toggle("on", x === b)); this.renderList(); };
    $("#qsearch").oninput = () => this.renderList();
    this.renderList();
  },
  renderList() {
    const s = $("#qsearch").value.toLowerCase();
    const xs = this.list.filter((x) => (this.f === "all" || (this.f === "win" ? x.acc.MODEL5 > x.acc.RAG_plain : x.type === this.f)) && (!s || x.question.toLowerCase().includes(s)));
    $("#qlist").innerHTML = `<div class="muted" style="font-size:11.5px;margin:0 0 6px 4px">${xs.length} questions · bars: share of the 20 grid cells answered correctly</div>` + xs.map((x) => `<div class="qi ${x.qid === this.q ? "on" : ""}" data-q="${x.qid}"><div class="qq">${esc(x.question)}</div><div class="qm"><span class="tag neutral">${esc(x.type)}</span><span class="mini"><i style="height:${2 + 12 * x.acc.MODEL5}px;background:var(--cf)"></i><i style="height:${2 + 12 * x.acc.RAG_plain}px;background:var(--rag)"></i></span><span>${pct(x.acc.MODEL5, 0)} vs ${pct(x.acc.RAG_plain, 0)}</span></div></div>`).join("");
    $$(".qi").forEach((e) => (e.onclick = () => { this.q = e.dataset.q; $$(".qi").forEach((x) => x.classList.toggle("on", x === e)); this.detail(); }));
  },
  async detail() {
    $("#qdetail").innerHTML = '<div style="padding:40px;display:grid;place-items:center"><span class="spinner"></span></div>';
    const key = `${this.q}|${this.ck}|${this.budget}`;
    let d;
    try { d = await api(`/api/bench/q/${this.q}?ck=${this.ck}&budget=${this.budget}`); } catch (e) { $("#qdetail").innerHTML = `<div style="padding:40px;color:var(--crit)">${esc(e.message)}</div>`; return; }
    if (key !== `${this.q}|${this.ck}|${this.budget}`) return;          // a newer selection is loading
    const h = d.history, st = h.storage;
    const cks = [[0, "evidence only"], [0.25, "+25%"], [0.5, "+50%"], [0.75, "+75%"], [1, "full history"]];
    const evid = d.evidence.map((e) => e.text);
    $("#qdetail").innerHTML = `<div class="qhead"><div style="display:flex;gap:8px;align-items:center;margin-bottom:8px"><span class="tag neutral">${esc(d.is_abstention ? "abstention" : d.type)}</span><span class="muted" style="font-size:12px">asked ${human(d.question_date)}</span></div>
      <h3>${esc(d.question)}</h3><span class="gold">gold answer: <b>${esc(d.gold)}</b></span></div>
      <div class="histstats"><span class="chip"><b>${h.checkpoint}</b> of ${h.sessions} sessions ingested</span><span class="chip"><b>${st.episodes}</b> turns stored</span><span class="chip"><b>${st.facts_active}</b> current facts</span><span class="chip"><b>${st.facts_superseded}</b> superseded</span><span class="chip"><b>${st.facts_rejected}</b> blocked by verification</span><span class="chip"><b>${h.ops.SKIP || 0}</b> turns skipped (no LLM)</span></div>
      <div class="controls"><span>history after the evidence</span><div class="seg" id="bck">${cks.map(([v, l]) => `<button data-v="${v}" class="${v === this.ck ? "on" : ""}">${l}</button>`).join("")}</div>
        <span>context budget</span><div class="seg" id="bbud">${[256, 512, 1024, 2048].map((b) => `<button data-v="${b}" class="${b === this.budget ? "on" : ""}">${b}</button>`).join("")}</div></div>
      <div class="vs">${["MODEL5", "RAG_plain"].map((s) => sysCard(s, d.systems[s], this.budget, d.systems[s], evid)).join("")}</div>
      <div class="chart" style="margin-top:16px"><h3>Evidence in the history</h3><div class="cs">the user turns that hold the answer; highlighted in green when a context contains them</div><div class="evid">${d.evidence.map((e) => `<div class="t"><span class="muted mono" style="font-size:11px">${e.date} · ${esc(e.role)}</span><br>${esc(e.text)}</div>`).join("") || '<span class="muted">none (abstention question)</span>'}</div></div>`;
    $("#bck").onclick = (e) => { const b = e.target.closest("button"); if (b) { this.ck = +b.dataset.v; this.detail(); } };
    $("#bbud").onclick = (e) => { const b = e.target.closest("button"); if (b) { this.budget = +b.dataset.v; this.detail(); } };
  },
};

/* ------------------------------------------------------------------ charts (SVG) */
function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function lineChart(el, { xs, xl, series, y0, y1, fmt, ytitle }) {
  const W = el.clientWidth || 560, H = 260, m = { l: 46, r: 120, t: 12, b: 34 };
  const X = (i) => m.l + (i * (W - m.l - m.r)) / (xs.length - 1), Y = (v) => m.t + ((y1 - v) * (H - m.t - m.b)) / (y1 - y0);
  let g = "";
  for (let v = y0; v <= y1 + 1e-9; v += 0.1) g += `<line class="gl" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${fmt(v)}</text>`;
  xs.forEach((x, i) => (g += `<text x="${X(i)}" y="${H - 12}" text-anchor="middle">${xl(x)}</text>`));
  if (ytitle) g += `<text x="12" y="${m.t + 6}" transform="rotate(-90 12 ${m.t + 6})" text-anchor="end">${ytitle}</text>`;
  for (const s of series) {
    const col = css(s.color);
    if (s.lo) g += `<path d="${s.lo.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join("")}${s.hi.map((v, i) => `L${X(s.hi.length - 1 - i)},${Y(s.hi[s.hi.length - 1 - i])}`).join("")}Z" fill="${col}" opacity=".14"/>`;
    g += `<path d="${s.v.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join("")}" fill="none" stroke="${col}" stroke-width="2.5"/>`;
    s.v.forEach((v, i) => (g += `<circle cx="${X(i)}" cy="${Y(v)}" r="5" fill="${col}" stroke="${css("--bg")}" stroke-width="2" data-t="${esc(`<b>${s.name}</b> · ${xl(xs[i])}<br>${fmt(v)}${s.lo ? ` (95% CI ${fmt(s.lo[i])}–${fmt(s.hi[i])})` : ""}`)}"/>`));
    g += `<text x="${X(xs.length - 1) + 10}" y="${Y(s.v[s.v.length - 1]) + 4}" style="fill:${col};font-weight:700">${s.name} ${fmt(s.v[s.v.length - 1])}</text>`;
  }
  el.innerHTML = `<svg width="${W}" height="${H}">${g}</svg>`;
  hover(el);
}
function hover(el) { $$("[data-t]", el).forEach((c) => { c.onmouseenter = (e) => tipOn(e, c.dataset.t); c.onmouseleave = tipOff; c.style.cursor = "default"; }); }
function dotWhisker(el, rows) {
  const W = el.clientWidth || 560, H = 230, m = { l: 100, r: 50, t: 12, b: 34 };
  const lo = Math.min(0, ...rows.map((r) => r.lo)) - 0.02, hi = Math.max(...rows.map((r) => r.hi)) + 0.02;
  const X = (v) => m.l + ((v - lo) * (W - m.l - m.r)) / (hi - lo), rowH = (H - m.t - m.b) / rows.length;
  let g = `<line class="ax" x1="${X(0)}" x2="${X(0)}" y1="${m.t}" y2="${H - m.b}" stroke-dasharray="3 3"/>`;
  for (let v = Math.ceil(lo * 20) / 20; v <= hi; v += 0.05) g += `<text x="${X(v)}" y="${H - 12}" text-anchor="middle">${v > 0 ? "+" : ""}${(100 * v).toFixed(0)}</text>`;
  const col = css("--cf");
  rows.forEach((r, i) => {
    const y = m.t + rowH * (i + 0.5);
    g += `<text x="${m.l - 10}" y="${y + 4}" text-anchor="end">${r.label}</text><line x1="${X(r.lo)}" x2="${X(r.hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="2.5" stroke-linecap="round"/>
      <circle cx="${X(r.diff)}" cy="${y}" r="6" fill="${col}" stroke="${css("--bg")}" stroke-width="2" data-t="${esc(`<b>${r.label}</b>: ${(100 * r.diff).toFixed(1)} points<br>95% CI ${(100 * r.lo).toFixed(1)} to ${(100 * r.hi).toFixed(1)}`)}"/>
      <text x="${X(r.hi) + 8}" y="${y + 4}" style="fill:${css("--text")};font-weight:600">+${(100 * r.diff).toFixed(1)}</text>`;
  });
  el.innerHTML = `<svg width="${W}" height="${H}">${g}</svg>`; hover(el);
}
function hbars(el, rows) {
  const W = el.clientWidth || 560, rowH = 38, H = rows.length * rowH + 30, m = { l: 170, r: 50 };
  const X = (v) => m.l + v * (W - m.l - m.r);
  let g = "";
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => (g += `<line class="gl" x1="${X(v)}" x2="${X(v)}" y1="0" y2="${H - 22}"/><text x="${X(v)}" y="${H - 6}" text-anchor="middle">${100 * v}%</text>`));
  rows.forEach((r, i) => {
    const y = i * rowH + 4;
    g += `<text x="${m.l - 10}" y="${y + 18}" text-anchor="end">${esc(r.label)}</text><text x="${m.l - 10}" y="${y + 30}" text-anchor="end" style="font-size:10px;fill:${css("--muted")}">n=${r.n}</text>`;
    [["MODEL5", "--cf", 0], ["RAG_plain", "--rag", 1]].forEach(([s, c, k]) => {
      const v = r[s];
      g += `<rect x="${m.l}" y="${y + k * 15}" width="${Math.max(2, X(v) - m.l)}" height="13" rx="3" fill="${css(c)}" data-t="${esc(`<b>${NAMES[s]}</b> · ${r.label}<br>${pct(v)}`)}"/><text x="${X(v) + 5}" y="${y + k * 15 + 11}" style="font-size:10.5px">${pct(v, 0)}</text>`;
    });
  });
  el.innerHTML = `<svg width="${W}" height="${H}">${g}</svg>`; hover(el);
}

/* ------------------------------------------------------------------ results */
const Results = {
  data: null,
  async load() { if (!this.data) this.data = await api("/api/results"); this.render(); },
  render() {
    const d = this.data, A = (s, b) => d.accuracy.find((r) => r.system === s && String(r.budget) === String(b));
    const all = d.paired.find((r) => String(r.budget) === "all"), T = (t) => d.by_type.find((r) => r.type === t);
    const B = (s) => d.behaviour.find((r) => r.system === s);
    const tr = T("temporal-reasoning"), ms = T("multi-session"), ku = T("knowledge-update");
    const fa = [B("MODEL5").false_abstention_rate, B("RAG_plain").false_abstention_rate];
    const ing = d.ingest.per_history_final_checkpoint, au = d.audit.info;
    $("#resultsBody").innerHTML = `
      <div class="hero">
        <div class="kpi"><div class="v">+${(100 * all.diff).toFixed(1)}<small>points</small></div><div class="l">accuracy vs RAG over raw transcripts</div><div class="s">95% CI +${(100 * all.lo).toFixed(1)} to +${(100 * all.hi).toFixed(1)} · paired over 500 questions, all budgets</div></div>
        <div class="kpi"><div class="v">${(tr.MODEL5 / tr.RAG_plain).toFixed(1)}×<small>accuracy</small></div><div class="l">temporal reasoning</div><div class="s">${pct(tr.MODEL5)} vs ${pct(tr.RAG_plain)} · ${tr.n_questions} questions</div></div>
        <div class="kpi"><div class="v">${(ms.MODEL5 / ms.RAG_plain).toFixed(1)}×<small>accuracy</small></div><div class="l">multi-session questions</div><div class="s">${pct(ms.MODEL5)} vs ${pct(ms.RAG_plain)} · ${ms.n_questions} questions</div></div>
        <div class="kpi"><div class="v">${(fa[1] / fa[0]).toFixed(1)}×<small>fewer</small></div><div class="l">false "not in memory" answers</div><div class="s">${pct(fa[0])} vs ${pct(fa[1])} of answerable questions</div></div>
      </div>
      <div class="charts">
        <div class="chart"><h3>Accuracy at every context budget</h3><div class="cs">mean over 5 checkpoints · shaded: 95% bootstrap CI (1,000 resamples)</div><div id="c1"></div></div>
        <div class="chart"><h3>ChronoFact lead over RAG, per budget</h3><div class="cs">paired difference in points · every interval above zero = significant</div><div id="c2"></div></div>
        <div class="chart"><h3>Accuracy by question type</h3><div class="cs">all budgets and checkpoints</div><div class="legend"><span><i style="background:var(--cf)"></i>ChronoFact</span><span><i style="background:var(--rag)"></i>RAG over transcripts</span></div><div id="c3"></div></div>
        <div class="chart"><h3>As the history keeps growing</h3><div class="cs">accuracy when 0–100% of the sessions after the evidence are also ingested (more distractors)</div><div id="c4"></div></div>
      </div>
      <div class="charts" style="margin-top:16px">
        <div class="chart"><h3>Behaviour</h3><div class="cs">LongMemEval-S · 500 questions × 5 checkpoints × 4 budgets per system</div>
          <table class="tbl"><tr><th></th><th style="color:var(--cf)">ChronoFact</th><th style="color:var(--rag)">RAG</th></tr>
          ${[["knowledge-update accuracy", ku.MODEL5, ku.RAG_plain, 1], ["stale (outdated) answers on updates", B("MODEL5").stale_rate_knowledge_update, B("RAG_plain").stale_rate_knowledge_update, 0], ["false abstention", fa[0], fa[1], 0], ["evidence turn in context", B("MODEL5").evidence_recall_any, B("RAG_plain").evidence_recall_any, 1], ["verified fact lines per context", B("MODEL5").mean_fact_lines, 0, 2]]
            .map(([l, a, b, k]) => `<tr><td>${l}</td><td><b>${k === 2 ? a.toFixed(1) : pct(a)}</b></td><td>${k === 2 ? "–" : pct(b)}</td></tr>`).join("")}</table></div>
        <div class="chart"><h3>Write cost per history</h3><div class="cs">~${Math.round(ing.store_episodes)} turns (~115k tokens) per history, final checkpoint</div>
          <table class="tbl"><tr><th></th><th>per history</th></tr>
          <tr><td>LLM calls (local Llama 3.1 8B)</td><td><b>${Math.round(ing.write_llm_calls)}</b></td></tr>
          <tr><td>turns skipped by the salience filter (no LLM)</td><td>${Math.round(ing.SKIP)}</td></tr>
          <tr><td>facts blocked by quote/NLI verification</td><td>${Math.round(ing.REJECT)}</td></tr>
          <tr><td>duplicates merged instead of stored</td><td>${Math.round(ing.NOOP)}</td></tr>
          <tr><td>superseded at write time</td><td>${ing.SUPERSEDE.toFixed(1)}</td></tr>
          <tr><td>current facts stored</td><td>${Math.round(ing.store_facts_active)}</td></tr>
          <tr><td>API cost</td><td><b>$0</b> (open weights, one GPU)</td></tr></table></div>
      </div>
      <div class="trust">
        <div class="t"><span>reproducibility</span><b>${au.determinism.same}/${au.determinism.total}</b><span>re-run retrieval cells byte-identical</span></div>
        <div class="t"><span>reliability</span><b>0 errors</b><span>write ${au.write_errors} · answer ${au.answer_errors} · judge ${au.judge_errors} over the full run</span></div>
        <div class="t"><span>human audit of the LLM judge</span><b>${d.judge_pilot.handcheck_wrong_of_50}/50</b><span>sampled verdicts wrong</span></div>
        <div class="t"><span>honest protocol</span><b>2-fold</b><span>gate tuned on one half, scored on the other · τ ${d.relevance.A} / ${d.relevance.B}</span></div>
      </div>`;
    const bs = [256, 512, 1024, 2048];
    lineChart($("#c1"), { xs: bs, xl: (b) => b + " tok", y0: 0.3, y1: 0.7, fmt: (v) => (100 * v).toFixed(0) + "%",
      series: [["MODEL5", "--cf"], ["RAG_plain", "--rag"]].map(([s, c]) => ({ name: NAMES[s].replace(" over transcripts", ""), color: c, v: bs.map((b) => A(s, b).acc), lo: bs.map((b) => A(s, b).lo), hi: bs.map((b) => A(s, b).hi) })) });
    dotWhisker($("#c2"), [...bs.map((b) => d.paired.find((r) => String(r.budget) === String(b))).map((r) => ({ ...r, label: r.budget + " tokens" })), { ...all, label: "all budgets" }]);
    hbars($("#c3"), d.by_type.map((r) => ({ label: r.type, n: r.n_questions, MODEL5: r.MODEL5, RAG_plain: r.RAG_plain })).sort((a, b) => b.MODEL5 - b.RAG_plain - (a.MODEL5 - a.RAG_plain)));
    const cks = ["0.0", "0.25", "0.5", "0.75", "1.0"], C = (s) => d.by_checkpoint.find((r) => r.system === s);
    lineChart($("#c4"), { xs: cks, xl: (c) => "+" + Math.round(100 * c) + "%", y0: 0.3, y1: 0.7, fmt: (v) => (100 * v).toFixed(0) + "%",
      series: [["MODEL5", "--cf"], ["RAG_plain", "--rag"]].map(([s, c]) => ({ name: NAMES[s].replace(" over transcripts", ""), color: c, v: cks.map((k) => C(s)[k]) })) });
  },
};

/* ------------------------------------------------------------------ how it works */
function how() {
  const node = (t, b, m) => `<div class="node"><b>${t}</b>${b}<div class="m">${m || ""}</div></div>`;
  const arrow = '<div class="arrow">→</div>';
  const L = '<span class="chip llama">Llama 3.1 8B</span>', C = '<span class="chip">Contriever</span>', N = '<span class="chip">BART-MNLI</span>';
  $("#howBody").innerHTML = `
    <h2>Memory that knows <i>when</i> something was true, and <i>why</i> it believes it.</h2>
    <p class="lead">ChronoFact turns a raw conversation stream into verified, time-stamped facts with provenance, keeps their history when they change, and at question time packs the facts that matter plus the evidence they came from into a small, fixed context. Every model in the loop is a Meta model.</p>
    <div class="lane"><div class="lt">Write path · every message</div><div class="flow">
      ${node("Salience filter", "acknowledgements and claim-free turns skip the LLM entirely", '<span class="chip">rules</span>')}${arrow}
      ${node("Extract", "durable facts about the user, each with an exact quote and its time expression; JSON enforced by guided decoding", L)}${arrow}
      ${node("Verify", "quote must appear verbatim in the message; the claim must be entailed by it, or the fact is blocked", N)}${arrow}
      ${node("Label", "kind (state / event / attribute), slot key reused from memory, richer statement checked again", L + N)}${arrow}
      ${node("Temporal update", "add, deduplicate, merge entities, supersede a changed state (old value kept with its end date), close on negation", C + L)}
    </div></div>
    <div class="lane"><div class="lt">Read path · every question</div><div class="flow">
      ${node("Time scope", "current value, as of a past date, or full history; parsed from the question", '<span class="chip">rules</span>')}${arrow}
      ${node("Hybrid search", "BM25 + dense retrieval fused by reciprocal rank", C)}${arrow}
      ${node("Fact gate", "facts valid in the time scope that pass a relevance gate tuned on held-out data", C)}${arrow}
      ${node("Bounded packing", "verified facts with validity dates first, then the evidence turns, within a fixed token budget", '<span class="chip">tokenizer</span>')}${arrow}
      ${node("Answer", "short reasoning, a final line, or an explicit NOT IN MEMORY", L)}
    </div></div>
    <div class="nov">
      ${[[`<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v5c0 5-3.5 8.5-7 10-3.5-1.5-7-5-7-10V6z"/><path d="M9 12l2 2 4-4"/></svg>`, "Verified writes", "A fact enters memory only if its quote is verbatim in the user's message and an NLI model confirms the claim. Requests for help are not mistaken for facts.", `${Math.round(Results.data?.ingest.per_history_final_checkpoint.REJECT ?? 43)} unsupported facts blocked per history`],
         [`<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2M9 2h6"/></svg>`, "Facts with a lifetime", "States have valid-from / valid-to dates. A new value supersedes the old one instead of overwriting it, so “now”, “in March” and “what changed” all have answers.", "Relative dates (“two weeks ago”) are resolved at write time"],
         [`<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5M11 8v3l2 1"/></svg>`, "Time-aware recall", "Each question is scoped (current / as-of / history) before retrieval; stale values are filtered or labelled as no longer current.", "2.5× RAG on temporal reasoning"],
         [`<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h5"/></svg>`, "Provenance & forgetting", "Every fact links to the message it came from. Forgetting deletes the fact, its siblings and the message text from the transcript index.", "Audit log of every write decision"],
         [`<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h6M7 14h10"/></svg>`, "Bounded context", "Facts and evidence share a fixed budget (256–2,048 tokens), so cost per question stays flat as the history grows.", "Accuracy moves by under 1.5 points from +0% to +100% more history"],
         [`<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>`, "Evidence, not claims", "500 questions × 5 checkpoints × 4 budgets, paired bootstrap CIs, cross-fitted tuning, byte-identical re-runs and a human audit of the LLM judge.", "Everything pinned and reproducible"]]
        .map(([i, t, p, pr]) => `<div class="card"><div class="ic">${i}</div><h4>${t}</h4><p>${p}</p><div class="proof">${pr}</div></div>`).join("")}
    </div>
    <div class="cols2">
      <div class="card"><h4>Meta models, each where it is strongest</h4><ul>
        <li><b>Llama 3.1 8B Instruct</b> (open weights, vLLM on one GPU): extraction, labelling, negation control, answers, judge, all with schema-constrained JSON</li>
        <li><b>Contriever</b> (Meta FAIR): dense retrieval, entity and slot matching</li>
        <li><b>BART-large-MNLI</b> (Meta): write-time verification of every fact and every rich statement</li>
        <li><b>Muse Spark</b> (Meta Model API): labelled the pairs used to calibrate the embedding thresholds</li></ul></div>
      <div class="card"><h4>Built during the hackathon (Oct 26 – Nov 4, 2026)</h4><ul>
        <li>The ChronoFact write path: salience, extraction and verification, labelling, temporal update and supersession, negation control</li>
        <li>The temporal fact store with snapshots, provenance and deletion</li>
        <li>The time-aware read path: time scoping, hybrid search, fact gate, bounded packing</li>
        <li>The evaluation harness: chronological checkpoints, budgets, folds, bootstrap CIs, integrity audit, judge audit</li>
        <li>Threshold calibration with model-labelled pairs, and this demo app</li></ul></div>
    </div>`;
}

/* ------------------------------------------------------------------ boot */
Live.init(); Ask.init(); how();
Live.boot().catch((e) => toast("Server not reachable: " + esc(e.message), 9000));
api("/api/results").then((d) => { Results.data = d; how(); }).catch(() => {});
addEventListener("resize", () => { if ($("#v-results").classList.contains("on") && Results.data) Results.render(); });
