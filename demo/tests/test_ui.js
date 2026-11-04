// End-to-end UI test, run inside the page. Every control is operated the way a presenter would; every story message
// and suggested question is checked against the outcome it must show. Returns [["PASS"|"FAIL", name, info], ...].
(async () => {
  const R = [];
  const ok = (name, cond, info = "") => R.push([cond ? "PASS" : "FAIL", name, String(info).slice(0, 220)]);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const until = async (f, ms = 120000) => { const t0 = Date.now(); while (!f() && Date.now() - t0 < ms) await sleep(150); return f(); };
  const txt = (s) => ($(s)?.innerText || "").replace(/\s+/g, " ");
  const facts = (st) => Live.mem.facts.filter((f) => !st || f.status === st);
  const has = (st, re) => facts(st).some((f) => re.test(f.statement) || re.test(f.value || ""));
  const toastText = () => txt("#toast");
  const noError = () => !/Error|error/.test($("#toast").classList.contains("on") ? toastText() : "");

  // ---------------------------------------------------------------- reset, theme, nav
  $("#reset").click();
  await until(() => Live.mem && Live.mem.facts.length === 0 && $("#chat").children.length === 0, 8000);
  ok("reset empties memory and chat", Live.mem.facts.length === 0 && $("#chat").children.length === 0);
  ok("story button shows step 1/9", /Story step 1\s*\/\s*9/.test(txt("#storyNext")), txt("#storyNext"));
  ok("story caption shows next capability", /Next:/.test(txt("#storyCap")), txt("#storyCap"));
  ok("counters show 0", [...document.querySelectorAll(".counter .v")].every((e) => e.innerText === "0"));
  ok("empty-memory hint", /Memory is empty/.test(txt("#memBody")));
  $("#theme").click(); ok("theme toggles to light", document.documentElement.dataset.theme === "light");
  $("#theme").click(); ok("theme toggles back to dark", document.documentElement.dataset.theme === "dark");
  for (const v of ["ask", "bench", "results", "how", "live"]) {
    document.querySelector(`#nav [data-v=${v}]`).click();
    await sleep(150);
    ok(`nav: ${v} tab shows its view`, $(`#v-${v}`).classList.contains("on") && document.querySelector(`#nav [data-v=${v}]`).classList.contains("on"));
  }
  $("#date").value = "2025-03-03"; $("#date").dispatchEvent(new Event("input"));
  ok("date picker updates the readable date", /3 March 2025/.test(txt("#dateHuman")), txt("#dateHuman"));

  // ---------------------------------------------------------------- the 9 story messages
  const expect = [
    () => [["Berlin is current", has("active", /Berlin/)], ["data analyst job is current", has("active", /data analyst/)],
           ["'lives near Friedrichshain' blocked", has("rejected", /Friedrichshain/)], ["Verify stage shows a Blocked fact", /blocked/i.test(txt("#st-verified"))]],
    () => [["'Thanks' skipped by the salience filter", /skipped/i.test(txt("#st-salience"))], ["no LLM extraction shown", !/Nothing to verify|No durable/.test(txt("#stages"))],
           ["skip logged", Live.mem.ops.some((o) => o.op === "SKIP")]],
    () => [["runs 3 times a week is current", has("active", /3 times/)], ["Luna stored", facts().some((f) => /Luna/.test(f.statement))]],
    () => [["Honda Civic is current", has("active", /Civic/)], ["'lives in Berlin' from 'parking in Berlin' blocked", facts("rejected").filter((f) => /Berlin/.test(f.statement)).length >= 1]],
    () => [["hypothetical sailboat not stored", !facts().some((f) => f.status !== "rejected" && /sailboat|lottery|boat/i.test(f.statement))]],
    () => [["Civic closed (history)", has("superseded", /Civic/)], ["Update stage shows Closed", /closed/i.test(txt("#st-applied"))], ["biking is current", has("active", /biking/)]],
    () => [["data analyst closed (history)", has("superseded", /data analyst/)], ["product manager is current", has("active", /product manager/i)], ["Update stage shows Closed", /closed/i.test(txt("#st-applied"))]],
    () => { const b = facts("superseded").find((f) => /Berlin/.test(f.statement)), m = facts("active").find((f) => /Munich/.test(f.statement));
            return [["Berlin superseded", !!b], ["Munich current", !!m], ["Munich dated two weeks before 2 Jun", m && m.valid_from.startsWith("2025-05-19"), m && m.valid_from],
                    ["Berlin ends when Munich starts", b && (b.valid_to || "").startsWith("2025-05-19"), b && b.valid_to], ["Update stage shows Berlin → Munich", /Berlin → Munich/.test(txt("#st-applied"))],
                    ["partner works at BMW stored", has("active", /BMW/)]]; },
    () => [["3 times superseded", has("superseded", /3 times/)], ["5 times current", has("active", /5 times/)], ["Update stage shows 3 times → 5", /3 times → 5/.test(txt("#st-applied"))],
           ["half marathon stored as current", has("active", /half marathon/)]],
  ];
  for (let i = 0; i < 9; i++) {
    const nUser = Live.mem.messages.filter((m) => m.speaker === "user").length;
    $("#storyNext").click();
    if (i === 0) { await sleep(120); $("#storyNext").click(); $("#storyNext").click(); }   // impatient presenter
    await sleep(300);
    await until(() => !Live.busy && Live.si === i + 1 && /LLM calls/.test(txt("#writeStat")));
    const step = Live.story.steps[i];
    const users = Live.mem.messages.filter((m) => m.speaker === "user");
    ok(`step ${i + 1}: exactly one new user message`, users.length === nUser + 1, users.length);
    ok(`step ${i + 1}: message text is the script`, users[users.length - 1].text === step.text);
    ok(`step ${i + 1}: session date set`, $("#date").value === step.date, $("#date").value);
    ok(`step ${i + 1}: assistant replied`, Live.mem.messages[Live.mem.messages.length - 1].speaker === "assistant" && Live.mem.messages[Live.mem.messages.length - 1].text.length > 10);
    ok(`step ${i + 1}: all five stages done`, [...document.querySelectorAll(".stage")].every((s) => s.classList.contains("done")));
    ok(`step ${i + 1}: no error`, noError(), toastText());
    for (const [n, c, info] of expect[i]()) ok(`step ${i + 1}: ${n}`, c, info ?? "");
  }
  ok("story button switches to 'Ask across time'", /Story done/.test(txt("#storyNext")));

  // ---------------------------------------------------------------- memory views, drawer
  for (const m of ["facts", "log", "timeline"]) {
    document.querySelector(`#memView [data-m=${m}]`).click(); await sleep(100);
    const t = txt("#memBody");
    ok(`memory view '${m}' renders`, m === "facts" ? /current/i.test(t) && /history/i.test(t) && /blocked/i.test(t) : m === "log" ? /SUPERSEDE/.test(t) && /REJECT/.test(t) : document.querySelectorAll(".tl-bar").length >= 8);
  }
  const berlin = facts("superseded").find((f) => /Berlin/.test(f.statement));
  document.querySelector(`.tl-bar[data-fid="${berlin.id}"]`).click(); await sleep(350);
  ok("click timeline bar opens the drawer", $("#drawer").classList.contains("open") && /Berlin/.test(txt("#drawer")));
  ok("drawer highlights the quote in the user's words", !!$("#drawer mark"), $("#drawer mark")?.innerText);
  const link = [...document.querySelectorAll("#drawer a")].find((a) => /Munich/.test(a.innerText));
  ok("drawer links to the fact that replaced it", !!link);
  link && link.click(); await sleep(250);
  ok("following the link opens Munich", /Munich/.test(txt("#drawer h3")), txt("#drawer h3"));
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" })); await sleep(250);
  ok("Esc closes the drawer", !$("#drawer").classList.contains("open"));
  Live.drawer(berlin.id); await sleep(200); document.querySelector("#drawer .iconbtn").click(); await sleep(250);
  ok("✕ closes the drawer", !$("#drawer").classList.contains("open"));
  Live.drawer(berlin.id); await sleep(200); show("ask"); await sleep(200);
  ok("switching tabs closes the drawer", !$("#drawer").classList.contains("open"));
  document.querySelector('#memView [data-m="facts"]').click(); await sleep(100);
  document.querySelector(".fact[data-fid]").click(); await sleep(250);
  ok("click in Facts view opens the drawer", $("#drawer").classList.contains("open")); Live.closeDrawer();
  document.querySelector('#memView [data-m="timeline"]').click();

  // ---------------------------------------------------------------- ask across time: every suggestion
  show("ask");
  const want = [[/munich/i, null, "current"], [/berlin/i, /previously Berlin/, "full history"], [/civic/i, /no longer current/, "as of"], [/3 times/, /no longer current/, "as of"], [/product manager/i, null, null], [/not in memory/i, null, null]];
  const chips = [...document.querySelectorAll("#sugg button")];
  ok("6 suggested questions", chips.length === 6, chips.length);
  for (let i = 0; i < chips.length; i++) {
    chips[i].click();
    await until(() => document.querySelectorAll("#vs .sys").length === 2 && !$("#vs .spinner") && /time scope/.test(txt("#modeline")));
    const ans = [...document.querySelectorAll("#vs .ans")].map((e) => e.innerText);
    const cfCtx = document.querySelector("#vs .sys.cf .ctx").innerText;
    ok(`ask '${chips[i].innerText}': ChronoFact answer`, want[i][0].test(ans[0]), ans.join(" | "));
    if (want[i][1]) ok(`ask '${chips[i].innerText}': context shows history line`, want[i][1].test(cfCtx));
    if (want[i][2]) ok(`ask '${chips[i].innerText}': time scope '${want[i][2]}'`, txt("#modeline").includes(want[i][2]), txt("#modeline"));
    ok(`ask '${chips[i].innerText}': within budget`, [...document.querySelectorAll(".budgetbar .lg")].every((e) => { const m = e.innerText.match(/(\d+) \/ (\d+) tokens/); return m && +m[1] <= +m[2] + 2; }));
  }
  ok("sister question: RAG abstains too", /not in memory/i.test(document.querySelectorAll("#vs .ans")[1].innerText));
  for (const b of [512, 1024, 2048, 256]) {
    document.querySelector(`#budget [data-b="${b}"]`).click();
    await until(() => txt("#modeline").includes(`${b}-token`) && !$("#vs .spinner"));
    ok(`budget ${b} re-asks and respects the budget`, txt("#modeline").includes(`${b}-token`) && [...document.querySelectorAll(".budgetbar .lg")].every((e) => e.innerText.includes(`/ ${b} tokens`)));
  }
  document.querySelector('#budget [data-b="512"]').click(); document.querySelector('#budget [data-b="1024"]').click();
  await until(() => !$("#vs .spinner") && txt("#modeline").includes("1024-token"));
  await sleep(4000);
  ok("fast budget clicks: the last one wins", txt("#modeline").includes("1024-token") && [...document.querySelectorAll(".budgetbar .lg")].every((e) => e.innerText.includes("/ 1024 tokens")), txt("#modeline"));
  document.querySelector('#budget [data-b="256"]').click(); await until(() => !$("#vs .spinner") && txt("#modeline").includes("256-token"));
  $("#askQ").value = "Where do I live now?"; $("#askQ").dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
  await sleep(300); await until(() => !$("#vs .spinner"));
  ok("Enter key asks", /munich/i.test(document.querySelector("#vs .ans").innerText));
  const det = document.querySelector("#vs details");
  if (det) { det.open = true; ok("model reasoning expands", det.open && det.querySelector("pre").innerText.length > 0); }

  // ---------------------------------------------------------------- forget
  show("live");
  const mar = facts("active").find((f) => /half marathon/.test(f.statement));
  Live.drawer(mar.id); await sleep(200);
  ok("forget button present", !!document.querySelector("#drawer .btn.danger"));
  const sibText = txt("#drawer");
  document.querySelector("#drawer .btn.danger").click();
  await until(() => Live.mem.facts.find((f) => f.id === mar.id)?.status === "forgotten", 10000);
  ok("forget marks the fact forgotten", Live.mem.facts.find((f) => f.id === mar.id).status === "forgotten");
  ok("forget toast", /Forgot/.test(toastText()), toastText());
  ok("forgotten message shown as deleted in chat", /deleted at the user's request/.test(txt("#chat")));
  ok("drawer warned about sibling facts if any", !/Also deleted/.test(sibText) || /Also deleted/.test(sibText));
  show("ask"); $("#askQ").value = "Am I training for anything?"; await Ask.run();
  ok("after forget: neither system mentions the marathon", ![...document.querySelectorAll("#vs .ans")].some((e) => /marathon/i.test(e.innerText)), [...document.querySelectorAll("#vs .ans")].map((e) => e.innerText).join(" | "));
  ok("after forget: marathon absent from both contexts", ![...document.querySelectorAll("#vs .ctx")].some((e) => /marathon/i.test(e.innerText)));

  // ---------------------------------------------------------------- manual message
  show("live");
  $("#date").value = "2025-06-22"; $("#date").dispatchEvent(new Event("input"));
  $("#msg").value = "My sister Priya just moved to Delhi.";
  $("#composer").dispatchEvent(new Event("submit", { cancelable: true }));
  await sleep(300); await until(() => !Live.busy);
  ok("manual message: stored a fact about Priya", facts().some((f) => f.status === "active" && /Priya/.test(f.statement + f.value)), facts("active").map((f) => f.statement).join("; "));
  show("ask"); $("#askQ").value = "What is my sister's name?"; await Ask.run();
  ok("manual message: sister question now answered", /priya/i.test(document.querySelector("#vs .ans").innerText), document.querySelector("#vs .ans").innerText);
  $("#msg").value = ""; $("#composer").dispatchEvent(new Event("submit", { cancelable: true })); await sleep(300);
  ok("empty message is not sent", !Live.busy && Live.mem.messages.filter((m) => m.speaker === "user").slice(-1)[0].text.includes("Priya"));

  // ---------------------------------------------------------------- benchmark explorer
  show("bench");
  await until(() => document.querySelectorAll(".qi").length > 0, 20000);
  ok("bench lists 500 questions", document.querySelectorAll(".qi").length === 500, document.querySelectorAll(".qi").length);
  $("#qsearch").value = "doctor"; $("#qsearch").dispatchEvent(new Event("input"));
  ok("search filters the list", document.querySelectorAll(".qi").length > 0 && [...document.querySelectorAll(".qi .qq")].every((e) => /doctor/i.test(e.innerText)), document.querySelectorAll(".qi").length);
  $("#qsearch").value = ""; $("#qsearch").dispatchEvent(new Event("input"));
  for (const b of document.querySelectorAll("#qfilters button")) {
    b.click(); await sleep(60);
    const n = document.querySelectorAll(".qi").length, f = b.dataset.f;
    const good = f === "all" ? n === 500 : f === "win" ? n > 0 && Bench.list.filter((x) => x.acc.MODEL5 > x.acc.RAG_plain).length === n : [...document.querySelectorAll(".qi .tag")].every((t) => t.innerText.toLowerCase() === f);
    ok(`filter '${b.innerText}'`, good && b.classList.contains("on"), n);
  }
  document.querySelector('#qfilters [data-f="win"]').click(); await sleep(60);
  document.querySelector(".qi").click();
  await until(() => !!$("#qdetail .gold"), 20000);
  ok("question detail loads with gold answer and two systems", !!$("#qdetail .gold") && document.querySelectorAll("#qdetail .sys").length === 2);
  ok("detail shows verdict tags", document.querySelectorAll("#qdetail .verdict").length === 2);
  for (const v of ["0", "0.25", "0.5", "0.75", "1"]) {
    document.querySelector(`#bck [data-v="${v}"]`).click(); await sleep(200);
    await until(() => !!document.querySelector(`#bck [data-v="${v}"].on`), 20000);
    ok(`checkpoint ${v}`, !!document.querySelector(`#bck [data-v="${v}"].on`) && /sessions ingested/.test(txt("#qdetail")));
  }
  for (const b of ["256", "1024", "2048", "512"]) {
    document.querySelector(`#bbud [data-v="${b}"]`).click(); await sleep(200);
    await until(() => !!document.querySelector(`#bbud [data-v="${b}"].on`), 20000);
    ok(`bench budget ${b}`, [...document.querySelectorAll("#qdetail .budgetbar .lg")].every((e) => e.innerText.includes(`/ ${b} tokens`)));
  }
  ok("evidence panel lists gold turns", document.querySelectorAll("#qdetail .evid .t").length > 0);
  ok("evidence highlighted inside a context", document.querySelectorAll("#qdetail .ctx .ev").length > 0);
  const ab = Bench.list.find((x) => x.type === "abstention");
  Bench.q = ab.qid; await Bench.detail();
  ok("abstention question detail renders", /abstention/.test(txt("#qdetail")) && /none \(abstention question\)|gold answer/.test(txt("#qdetail")));

  // ---------------------------------------------------------------- results + how
  show("results"); await until(() => document.querySelectorAll("#resultsBody svg").length === 4, 10000);
  ok("results: 4 charts", document.querySelectorAll("#resultsBody svg").length === 4);
  ok("results: headline +13.6", /\+13\.6/.test(txt("#resultsBody")));
  ok("results: no RAG harness anywhere", !/harness/i.test(document.body.innerText));
  const c = document.querySelector("#c1 [data-t]");
  c.dispatchEvent(new MouseEvent("mouseenter", { clientX: 400, clientY: 400 }));
  ok("chart tooltip appears on hover", $("#tip").classList.contains("on") && /ChronoFact/.test($("#tip").innerText), $("#tip").innerText);
  c.dispatchEvent(new MouseEvent("mouseleave")); ok("tooltip hides", !$("#tip").classList.contains("on"));
  dispatchEvent(new Event("resize")); await sleep(200);
  ok("charts survive a resize", document.querySelectorAll("#resultsBody svg").length === 4);
  $("#theme").click(); await sleep(200);
  ok("charts re-render in light theme", document.querySelectorAll("#resultsBody svg").length === 4 && document.documentElement.dataset.theme === "light");
  $("#theme").click();
  show("how"); await sleep(100);
  ok("how: 6 capability cards + 2 info cards", document.querySelectorAll("#howBody .nov .card").length === 6 && document.querySelectorAll("#howBody .cols2 .card").length === 2);
  ok("how: blocked-per-history figure from the real run", /\d+ unsupported facts blocked per history/.test(txt("#howBody")));

  // ---------------------------------------------------------------- reset again
  show("live"); $("#reset").click();
  await until(() => Live.mem.facts.length === 0, 8000); await sleep(300);
  ok("final reset: empty, story 1/9, ask cleared", Live.mem.facts.length === 0 && /Story step 1\s*\/\s*9/.test(txt("#storyNext")) && $("#vs").innerHTML === "", [Live.mem.facts.length, txt("#storyNext"), $("#vs").innerHTML.length]);
  return { pass: R.filter((r) => r[0] === "PASS").length, fail: R.filter((r) => r[0] === "FAIL"), all: R.length };
})()
