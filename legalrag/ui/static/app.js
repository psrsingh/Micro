"use strict";

/* ---------- helpers ----------------------------------------------------------------------------
   Everything that reaches the page (evidence text, model output, claim text) is untrusted, so the
   DOM is built with text nodes only: there is no innerHTML anywhere in this file. */
const PROP_KEYS = new Set(["value", "checked", "disabled", "hidden", "open", "selected"]);

function h(spec, props, ...kids) {
  const [tag, ...cls] = spec.split(".");
  const el = document.createElement(tag || "div");
  if (cls.length) el.className = cls.join(" ");
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (PROP_KEYS.has(k)) el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  const add = (k) => {
    if (k == null || k === false) return;
    if (Array.isArray(k)) k.forEach(add);
    else el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  };
  kids.forEach(add);
  return el;
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  let data = {};
  try { data = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

const pct = (v) => (v == null ? "n/a" : `${Math.round(v * 100)}%`);
const num = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));
const ms = (v) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
const when = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
const show = (el, ...kids) => { el.replaceChildren(...kids.flat(Infinity).filter(Boolean)); return el; };

let toastTimer;
function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 9000);
}

/* ---------- vocabulary ------------------------------------------------------------------------- */
const ARM_INFO = {
  baseline: { title: "Baseline LLM", blurb: "The model alone, no retrieval. Anything it cites, it cites from memory." },
  standard_rag: { title: "Standard RAG", blurb: "Retrieves passages and answers from them. Nothing is verified." },
  verified_rag: { title: "Verified RAG", blurb: "Verifies every claim against its evidence, then answers only from what survived." },
};
const STATUS = { VERIFIED: ["✓", "Verified"], FLAGGED: ["⚑", "Flagged"], REJECTED: ["✕", "Rejected"] };
const OUTCOME = { answered: ["✓", "Answered"], abstained: ["–", "Abstained"], error: ["✕", "Error"] };
const CHECK_ICON = { PASS: "✓", WARN: "!", FAIL: "✕", NOT_APPLICABLE: "–" };
const CHECK_LABEL = {
  authority: "Authority exists", citation_metadata: "Citation details", statute: "Statute section",
  claim_support: "Claim supported", temporal: "In force",
};
const CHECK_ORDER = Object.keys(CHECK_LABEL);
const STAGES = [
  ["query_analysis", "Understand question"], ["retrieval", "Retrieve"], ["rerank", "Rerank"],
  ["draft", "Draft (never shown)"], ["decompose", "Split into claims"], ["verify", "Verify"], ["final", "Write answer"],
];
const EXAMPLES = [
  ["statute", "Statute lookup"], ["temporal", "Superseded statute"], ["fact_pattern", "Fact pattern"],
  ["named_case", "Named case"], ["fabricated_trap", "Fabricated case"], ["outside_corpus", "Case not in corpus"],
  ["out_of_scope_law", "Law the corpus lacks"], ["out_of_scope", "Not a legal question"],
];

const badge = (cls, icon, text, title) =>
  h(`span.badge.b-${cls}`, { title }, h("span", { "aria-hidden": "true" }, icon), text);
const statusBadge = (s) => (STATUS[s] ? badge(s.toLowerCase(), STATUS[s][0], STATUS[s][1]) : badge("none", "–", "Not checked"));
const outcomeBadge = (o) => badge(o, ...OUTCOME[o] || ["–", o]);
const checkBadge = (name, c) =>
  badge(c.outcome, CHECK_ICON[c.outcome] || "–", CHECK_LABEL[name] || name, c.explanation || undefined);

const state = { meta: null, running: false, questions: {} };
const views = {};

/* ---------- shared result rendering ------------------------------------------------------------ */
function confidenceBlock(r) {
  if (r.outcome !== "answered") return null;
  if (r.confidence == null) return h("div.conf", {}, h("span.muted.small", {}, "No confidence reported."));
  const verified = r.confidence_source === "verification";
  return h("div.conf", {},
    h("div.row.spread", {}, h("span", {}, "Confidence ", h("span.val", {}, pct(r.confidence)))),
    h("div.meter", { role: "meter", "aria-label": "Confidence", "aria-valuemin": 0, "aria-valuemax": 1, "aria-valuenow": r.confidence },
      h("span", { style: `width:${Math.round(r.confidence * 100)}%` })),
    h("span.muted.small", {}, verified
      ? "Derived from verification: the weakest claim behind the answer."
      : "Stated by the model itself. It was not checked against anything."));
}

function statementsBlock(r) {
  return h("ul.stmts", {}, r.statements.map((s) => h(`li.stmt${s.status === "flagged" ? ".flagged" : ""}`, {},
    statusBadge(s.status.toUpperCase()),
    h("p", {}, s.text),
    (s.source_details || []).length
      ? h("div.sources", {}, "Verified sources:", s.source_details.map((d) => h("span.chip.src", {title: `${d.kind}${d.verification_status ? " · " + d.verification_status : ""}`},
          d.label + (d.pinpoint ? ` — ${d.pinpoint}` : "") + ` [${d.kind.replace(/_/g, " ")}]`)))
      : (s.sources || []).length ? h("div.sources", {}, "Verified sources:", s.sources.map((x) => h("span.chip.src", {}, x))) : null,
    (s.caveats || []).map((c) => h("div.caveat", {}, "⚑ ", c)))));
}

function freeTextBlock(r) {
  const paras = (r.answer_text || "").split(/\n{2,}/).filter((p) => p.trim());
  return h("div", {},
    h("div.answer-text", {}, paras.length ? paras.map((p) => h("p", {}, p)) : h("p.muted", {}, "(empty answer)")),
    r.note ? h("div.callout.warn", {}, "The model did not follow the output format, so this is its raw text: ", r.note) : null,
    (r.citations || []).length ? h("div.sources", {}, "Cited by the model, not verified:", r.citations.map((x) => h("span.chip.src", {}, x))) : null);
}

function abstainBlock(r) {
  let why = r.note;
  if (!why && r.claims) {
    why = r.claims.length ? `The draft made ${r.claims.length} claim(s) and none survived verification, so nothing was stated.`
      : "Nothing retrieved could support an answer.";
  }
  why ||= r.arm === "baseline" ? "The model declined to answer." : "The model declined to answer from the passages it was given.";
  return h("div.callout.warn", {}, h("strong", {}, "No answer given. "), why);
}

function claimCounts(r) {
  const cs = r.claims || [];
  return { total: cs.length, kept: cs.filter((c) => c.survives).length,
    flagged: cs.filter((c) => c.survives && c.support === "flagged").length, dropped: cs.filter((c) => c.survives === false).length };
}

function passagesDetail(r) {
  const kept = (r.retrieved || []).filter((p) => p.kept);
  if (!kept.length) return null;
  return h("details", {}, h("summary", {}, `Passages it was given (${kept.length})`),
    kept.map((p) => h("div.chunk", {}, h("strong", {}, p.title), " ", h("span.muted.small", {}, [p.citation, p.section_label].filter(Boolean).join(" · ")),
      h("p.small", {}, p.snippet))));
}

/* One arm's card. `onInspect` is given only for the verified arm, which has a trace to open. */
function resultCard(r, onInspect) {
  const info = ARM_INFO[r.arm];
  const c = r.arm === "verified_rag" ? claimCounts(r) : null;
  const body = r.outcome === "error" ? h("div.alert", {}, "The run failed: ", r.error || "unknown error")
    : r.outcome === "abstained" ? abstainBlock(r)
    : r.arm === "verified_rag" ? statementsBlock(r) : freeTextBlock(r);
  return h(`article.card.arm${r.arm === "verified_rag" ? ".verified" : ""}`, {},
    h("header", {}, h("div.row.spread", {}, h("h3", {}, info.title), outcomeBadge(r.outcome)), h("p", {}, info.blurb)),
    confidenceBlock(r), body,
    r.arm === "standard_rag" ? passagesDetail(r) : null,
    c && c.total ? h("div.summaryline", {}, `${c.total} claims checked · ${c.kept} kept${c.flagged ? ` (${c.flagged} with caveats)` : ""} · ${c.dropped} dropped`) : null,
    onInspect ? h("button.btn", { type: "button", "aria-expanded": "false", onclick: (e) => {
      const open = onInspect(); e.currentTarget.setAttribute("aria-expanded", String(open));
      e.currentTarget.textContent = open ? "Hide verification detail" : "Inspect verification detail";
    } }, "Inspect verification detail") : null,
    h("div.row.small.muted", {}, `Total ${ms(r.latency_ms && r.latency_ms.total)}`,
      r.run_id ? h("a", { href: `#history/${r.run_id}` }, `run #${r.run_id}`) : null));
}

function pendingCard(arm) {
  return h("article.card.arm", { "aria-busy": "true" }, h("header", {}, h("h3", {}, ARM_INFO[arm].title), h("p", {}, ARM_INFO[arm].blurb)),
    h("div", {}, h("span.spinner"), " Working…"));
}

/* ---------- the verification trace ------------------------------------------------------------- */
function fmtVal(v) {
  if (Array.isArray(v)) return v.join(", ");
  if (v && typeof v === "object") return Object.entries(v).map(([k, x]) => `${k}: ${x}`).join("; ");
  return String(v);
}

function timeline(latency) {
  const parts = STAGES.map(([k, label], i) => ({ k, label, v: latency[k], color: `var(--s${i + 1})` })).filter((p) => p.v != null);
  const sum = parts.reduce((a, p) => a + p.v, 0) || 1;
  return h("div", {},
    h("div.timeline", { role: "img", "aria-label": "Time spent per stage" },
      parts.map((p) => h("span", { title: `${p.label}: ${ms(p.v)}`, style: `width:${(p.v / sum) * 100}%;background:${p.color}` }))),
    h("div.legend", {}, parts.map((p) => h("span", {}, h("i", { style: `background:${p.color}` }), `${p.label} ${ms(p.v)}`))));
}

function questionInfo(stages) {
  const qa = stages.query_analysis, rt = stages.retrieval, rows = [];
  if (qa) {
    rows.push(["Issue", qa.issue], ["Acts", qa.act_names], ["Sections", qa.section_numbers], ["Cases named", qa.case_names],
      ["Court", qa.court], ["From", qa.date_from], ["To", qa.date_to], ["Fact pattern", qa.is_fact_pattern ? "yes" : null],
      ["Search query", qa.rewritten_query]);
  }
  if (rt) {
    rows.push(["Candidates", rt.n_candidates], ["Metadata filters", rt.filters_applied && Object.keys(rt.filters_applied).length ? rt.filters_applied : null],
      ["Filters dropped", rt.filters_dropped ? "the filtered search found nothing, so it was retried without filters" : null]);
  }
  const shown = rows.filter(([, v]) => v != null && v !== "" && !(Array.isArray(v) && !v.length));
  return h("div", {},
    stages.unavailable_authorities ? h("div.callout.bad", {}, h("strong", {}, "Coverage guard stopped this question. "),
      "It names an authority the corpus cannot ground: ", stages.unavailable_authorities.join("; "), ". The model was never asked to answer.") : null,
    qa && qa.fallback ? h("div.callout.warn", {}, "Query analysis fell back to the raw question: ", qa.fallback_reason || "unknown reason") : null,
    h("dl.kv", {}, shown.map(([k, v]) => [h("dt", {}, k), h("dd", {}, fmtVal(v))])));
}

function evidenceRow(e) {
  const rank = (n) => (CHECK_ORDER.includes(n) ? CHECK_ORDER.indexOf(n) : CHECK_ORDER.length);
  const names = Object.keys(e.checks || {}).sort((a, b) => rank(a) - rank(b));
  return h("div.ev", {},
    h("div.who", {}, statusBadge(e.status), h("strong", {}, e.source), h("span.muted.small", {}, [e.section_label, e.court, e.decided_on].filter(Boolean).join(" · "))),
    e.status_reason ? h("p.reason", {}, e.status_reason) : null,
    h("div.checks", {}, names.map((n) => checkBadge(n, e.checks[n]))),
    names.length ? h("details", {}, h("summary", {}, "Why each check ruled this way"),
      h("div.table-wrap", {}, h("table", {}, h("thead", {}, h("tr", {}, ["Check", "Result", "Score", "Explanation"].map((x) => h("th", {}, x)))),
        h("tbody", {}, names.map((n) => { const c = e.checks[n]; return h("tr", {},
          h("td", {}, CHECK_LABEL[n] || n), h("td", {}, checkBadge(n, { ...c, explanation: "" })), h("td.num", {}, num(c.score, 2)),
          h("td", {}, c.explanation || "–", c.evidence_pointer ? h("div.mono.muted", {}, c.evidence_pointer) : null)); }))))) : null,
    h("details", {}, h("summary", {}, "Passage"), h("blockquote", {}, e.text)));
}

function claimCard(c) {
  const verdict = c.survives ? (c.support === "flagged" ? ["flagged", "⚑", "Kept, with caveat"] : ["verified", "✓", "Kept"]) : ["dropped", "✕", "Dropped"];
  return h(`div.claim${c.survives ? "" : ".dropped"}`, {},
    h("div.head", {}, h("strong", {}, `Claim ${c.idx}`), badge(...verdict)),
    h("p.text", {}, c.text),
    c.unresolved_citations && c.unresolved_citations.length ? h("div.callout.bad", {}, "The draft cited something that is not in the retrieved evidence: ", c.unresolved_citations.join("; ")) : null,
    c.caveats && c.caveats.map((x) => h("div.caveat", {}, "⚑ ", x)),
    c.evidence.length ? c.evidence.map(evidenceRow) : h("p.reason", {}, "No evidence was attached to this claim."));
}

function candidatesTable(list) {
  const wrap = h("div.table-wrap"), draw = (all) => show(wrap, h("table", {},
    h("thead", {}, h("tr", {}, ["#", "Source", "Section", "Fused", "Rerank", "Used"].map((x, i) => h(`th${i > 2 ? ".num" : ""}`, {}, x)))),
    h("tbody", {}, (all ? list : list.slice(0, 10)).map((p, i) => h("tr", {}, h("td", {}, i + 1),
      h("td", {}, h("strong", {}, p.title), p.citation ? h("div.muted.small", {}, p.citation) : null), h("td", {}, p.section_label || "–"),
      h("td.num", {}, num(p.scores.fused, 4)), h("td.num", {}, num(p.scores.rerank)), h("td", {}, p.kept ? "✓ kept" : "–"))))),
    list.length > 10 ? h("button.btn.link", { type: "button", onclick: () => draw(!all) }, all ? "Show top 10" : `Show all ${list.length}`) : null);
  draw(false);
  return wrap;
}

function renderTrace(r) {
  const claims = r.claims || [];
  return h("section.card.trace", { "aria-label": "Verification detail" },
    h("h2", {}, "Inside the verified pipeline"),
    h("p.muted", {}, "Each claim from the hidden draft is checked against the passages it cites. Only claims resting on verified evidence (or explicitly flagged evidence, with the caveat attached) reach the final answer."),
    r.latency_ms ? [h("h3", {}, "Where the time went"), timeline(r.latency_ms)] : null,
    h("h3", {}, "How the question was read"), questionInfo(r.stages || {}),
    h("h3", {}, `Claims and their checks (${claims.length})`),
    claims.length ? h("div.stack", {}, claims.map(claimCard)) : h("p.muted", {}, "No claims were drafted, so there was nothing to verify."),
    r.retrieved && r.retrieved.length ? [h("h3", {}, `Retrieved passages (${r.retrieved.length})`), candidatesTable(r.retrieved)] : null);
}

/* ---------- Ask -------------------------------------------------------------------------------- */
function buildAsk(el) {
  const ta = h("textarea", { id: "q", rows: 3, maxlength: 2000, "aria-label": "Your legal question",
    placeholder: "Ask a legal question, for example: how much notice must a landlord give before seeking eviction?" });
  const boxes = state.meta.arms.map((a) => h("label", {}, h("input", { type: "checkbox", value: a, checked: true }), ARM_INFO[a].title));
  const go = h("button.btn.primary", { type: "submit" }, "Ask");
  const results = h("div.stack", { "aria-live": "polite" });
  const fill = (q) => { ta.value = q; ta.focus(); };

  const byTag = (t) => state.meta.samples.find((s) => s.tags.includes(t));
  const chips = EXAMPLES.map(([t, label]) => [byTag(t), label]).filter(([s]) => s)
    .map(([s, label]) => h("button.chip", { type: "button", title: s.question, onclick: () => fill(s.question) }, label));
  const all = h("details.all-q", {}, h("summary", {}, `All ${state.meta.samples.length} benchmark questions`),
    h("ul", {}, state.meta.samples.map((s) => h("li", {}, h("button", { type: "button", onclick: () => fill(s.question) }, s.question)))));

  const form = h("form.card.composer.stack", { onsubmit: (e) => { e.preventDefault(); run(); } },
    h("div", {}, h("label.small.muted", { for: "q" }, "Question"), ta),
    h("div.row.spread", {}, h("div.arm-pick", { role: "group", "aria-label": "Systems to run" }, boxes), go),
    chips.length ? h("div.examples", {}, h("span.small.muted", {}, "Try:"), chips) : null,
    state.meta.samples.length ? all : null);
  ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); } });

  async function run() {
    const query = ta.value.trim();
    const arms = boxes.map((b) => b.firstChild).filter((i) => i.checked).map((i) => i.value);
    if (!query || state.running) return;
    if (!arms.length) return toast("Pick at least one system to run.");
    state.running = true; go.disabled = true; go.textContent = "Asking…";
    document.getElementById("cfgSel").disabled = true;
    const cards = Object.fromEntries(arms.map((a) => [a, pendingCard(a)]));
    const grid = h("div.grid-arms", {}, Object.values(cards));
    const inspector = h("div");
    show(results, grid, inspector);
    for (const arm of arms) {
      try {
        const r = await api("/api/ask", { query, arm });
        const next = resultCard(r, arm === "verified_rag" && r.claims ? () => {
          if (inspector.firstChild) { inspector.replaceChildren(); return false; }
          inspector.append(renderTrace(r)); inspector.scrollIntoView({ behavior: "smooth", block: "nearest" }); return true;
        } : null);
        cards[arm].replaceWith(next); cards[arm] = next;
      } catch (err) {
        const bad = h("article.card.arm", {}, h("h3", {}, ARM_INFO[arm].title), h("div.alert", {}, err.message));
        cards[arm].replaceWith(bad); cards[arm] = bad;
      }
    }
    state.running = false; go.disabled = false; go.textContent = "Ask";
    document.getElementById("cfgSel").disabled = false;
  }

  show(el, h("h2", {}, "Ask a legal question"),
    h("p.lede", {}, "Run the same question through three systems side by side. The difference to look for: the baseline and standard RAG state citations nobody checked, while the verified system shows what it checked and drops what it could not support."),
    form, h("div", { style: "margin-top:18px" }, results));
  return { fill, run, ta, boxes };
}

/* ---------- History ---------------------------------------------------------------------------- */
function buildHistory(el) {
  let scope = "ui";
  const list = h("div.list"), detail = h("div.stack");
  const scopeSel = h("select", { "aria-label": "Which runs to list", onchange: (e) => { scope = e.target.value; load(); } },
    h("option", { value: "ui" }, "Asked in this UI"), h("option", { value: "all" }, "All runs, including benchmarks"));

  async function load(openId) {
    show(list, h("span.spinner"));
    const runs = await api(`/api/runs?scope=${scope}`);
    show(list, runs.length ? runs.map((r) => h("button", { type: "button", "data-id": r.id, "aria-current": String(String(r.id) === String(openId)),
      onclick: () => { location.hash = `#history/${r.id}`; } },
      h("span", {}, r.query.length > 90 ? r.query.slice(0, 89) + "…" : r.query),
      h("span.row.small.muted", {}, outcomeBadge(r.outcome), ARM_INFO[r.arm] ? ARM_INFO[r.arm].title : r.arm,
        r.confidence != null ? `· ${pct(r.confidence)}` : null, `· ${when(r.created_at)}`)))
      : h("p.muted", {}, scope === "ui" ? "Nothing asked in the UI yet." : "No runs recorded."));
  }

  async function open(id) {
    show(detail, h("span.spinner"));
    try {
      const r = await api(`/api/runs/${id}`);
      const inspector = h("div");
      show(detail,
        h("div.card.stack", {}, h("div", {}, h("h3", {}, r.query), h("p.small.muted", {}, `${when(r.created_at)} · config ${r.config_name}`)),
          h("div.row", {}, h("button.btn", { type: "button", onclick: () => { views.ask.api.fill(r.query); location.hash = "#ask"; } }, "Ask this again"))),
        resultCard(r, r.claims ? () => {
          if (inspector.firstChild) { inspector.replaceChildren(); return false; }
          inspector.append(renderTrace(r)); return true;
        } : null), inspector);
    } catch (err) { show(detail, h("div.alert", {}, err.message)); }
  }

  show(el, h("h2", {}, "History"),
    h("p.lede", {}, "Every question is logged with its full evidence trail. Open one to see exactly what was retrieved, checked and answered."),
    h("div.split", {}, h("div.stack", {}, h("div.row", {}, scopeSel), list), detail));
  return { enter: async (id) => { await load(id); if (id) await open(id); else show(detail, h("p.muted", {}, "Select a run to inspect it.")); } };
}

/* ---------- Benchmark -------------------------------------------------------------------------- */
const METRICS = [
  ["answer_accuracy", "Answer accuracy", "up", "n_queries", "Answers containing the required phrases and none forbidden. For out-of-scope questions, abstaining is the correct answer."],
  ["abstention_accuracy_unanswerable", "Abstained when it should", "up", null, "Of the questions the corpus cannot answer, how many the system declined."],
  ["cvr", "Citation validity (CVR)", "up", "n_cited", "Cited authorities that resolve to a real authority."],
  ["csr", "Citation support (CSR)", "up", "n_csr", "Valid, in-corpus cited authorities whose text supports at least one statement of the answer."],
  ["ega", "Evidence grounding (EGA)", "up", "n_statements", "Statements supported by the system's own evidence. The baseline retrieves nothing, so it is 0 by construction."],
  ["ucr", "Unsupported claims (UCR)", "down", "n_statements", "The complement of EGA."],
  ["hcer", "High-confidence errors (HCER)", "down", "n_queries", "Wrong answers stated with high confidence, over all queries. Confidence is verification-derived for Verified RAG and model-stated for the others."],
  ["wrong_answers_disclosed", "Wrong answers disclosed", "up", "n_wrong_answers", "Wrong answers that at least carry a flagged caveat."],
];

function buildBenchmark(el) {
  const out = h("div.stack");
  const sel = h("select", { "aria-label": "Results file", onchange: (e) => { location.hash = `#benchmark/${e.target.value}`; } });

  function metricsTable(s, arms) {
    return h("div.table-wrap", {}, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Metric"), arms.map((a) => h("th", {}, ARM_INFO[a] ? ARM_INFO[a].title : a)))),
      h("tbody", {}, METRICS.map(([key, label, dir, nkey, help]) => {
        const vals = arms.map((a) => s.arms[a][key]), real = vals.filter((v) => v != null);
        const best = real.length ? (dir === "up" ? Math.max(...real) : Math.min(...real)) : null;
        const highlight = real.length > 1 && new Set(real).size > 1;
        return h("tr", {}, h("td", { title: help }, label, h("div.small.muted", {}, dir === "up" ? "higher is better" : "lower is better")),
          arms.map((a, i) => h(`td${highlight && vals[i] === best ? ".best" : ""}`, {}, pct(vals[i]),
            h("div.bar", {}, h("span", { style: `width:${Math.round((vals[i] || 0) * 100)}%` })),
            nkey && s.arms[a][nkey] != null ? h("div.small.muted", {}, `n=${s.arms[a][nkey]}`) : null)));
      }))));
  }

  function questionGrid(d, arms) {
    const ids = [...new Set(d.records.map((r) => r.id))];
    const cell = (r) => !r ? "–" : h("span", {}, r.outcome === "error" ? badge("error", "✕", "error")
      : r.correct ? badge("verified", "✓", r.outcome === "abstained" ? "correctly abstained" : "correct")
      : badge("rejected", "✕", r.outcome === "abstained" ? "abstained, but answerable" : "wrong"), r.confidence != null ? ` ${pct(r.confidence)}` : "");
    return h("div.table-wrap", {}, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Question"), arms.map((a) => h("th", {}, ARM_INFO[a].title)), h("th", {}, ""))),
      h("tbody", {}, ids.map((id) => h("tr", {}, h("td", {}, h("span.muted.small.mono", {}, id), " ", state.questions[id] || ""),
        arms.map((a) => h("td", {}, cell(d.records.find((r) => r.id === id && r.arm === a)))),
        h("td", {}, state.questions[id] ? h("button.btn.link", { type: "button", onclick: () => { views.ask.api.fill(state.questions[id]); location.hash = "#ask"; } }, "Ask it") : null))))));
  }

  function extras(s, arms) {
    const ks = [...new Set(Object.keys(s.arms[arms[0]]).filter((k) => /^recall_doc@/.test(k)).map((k) => k.split("@")[1]))];
    const cats = [...new Set(arms.flatMap((a) => Object.keys(s.arms[a].error_categories || {})))].sort();
    return [
      ks.length ? h("details.card", {}, h("summary", {}, "Retrieval quality (Recall@K, over answerable questions)"), h("div.table-wrap", {}, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Measure"), arms.map((a) => h("th", {}, ARM_INFO[a].title)))),
        h("tbody", {}, ks.flatMap((k) => [["recall_doc", "Right document"], ["recall_span", "Right passage"]].map(([m, label]) =>
          h("tr", {}, h("td", {}, `${label} in top ${k}`), arms.map((a) => h("td", {}, pct(s.arms[a][`${m}@${k}`])))))))))) : null,
      cats.length ? h("details.card", {}, h("summary", {}, "Error taxonomy (questions affected)"), h("div.table-wrap", {}, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Category"), arms.map((a) => h("th.num", {}, ARM_INFO[a].title)))),
        h("tbody", {}, cats.map((c) => h("tr", {}, h("td", {}, c), arms.map((a) => h("td.num", {}, s.arms[a].error_categories[c] || 0)))))))) : null,
    ];
  }

  async function showResult(name) {
    show(out, h("span.spinner"));
    const d = await api(`/api/results/${name}`), s = d.summary, arms = Object.keys(s.arms), ava = s.ava;
    const plumbing = d.notes.find((n) => n.startsWith("PLUMBING RUN"));
    show(out,
      plumbing ? h("div.callout.warn", {}, h("strong", {}, "Plumbing run. "), plumbing.replace(/^PLUMBING RUN\.\s*/, ""), " Read the table as a check that the harness works, not as a ranking.") : null,
      h("div.card", {}, h("h3", {}, "Comparison"), metricsTable(s, arms), h("p.small.muted", { style: "margin-top:10px" }, "Bold marks the best arm per metric on this run. Hover a metric name for its definition.")),
      ava ? h("div.card", {}, h("h3", {}, "Authority verification accuracy (AVA)"),
        h("p", {}, h("strong", {}, pct(ava.ava)), ` over ${ava.n} constructed probes. `, h("span.muted", {}, "The probes were written alongside the checks: a regression suite, not a held-out estimate.")),
        h("div.table-wrap", {}, h("table", {}, h("thead", {}, h("tr", {}, ["Probe kind", "n", "Accuracy"].map((x) => h("th", {}, x)))),
          h("tbody", {}, Object.entries(ava.by_kind).map(([k, v]) => h("tr", {}, h("td", {}, k.replaceAll("_", " ")), h("td", {}, v.n), h("td", {}, pct(v.accuracy)))))))) : null,
      extras(s, arms),
      h("div.card", {}, h("h3", {}, "Per question"), questionGrid(d, arms)),
      h("div.card", {}, h("h3", {}, "How to read this"), h("ul", {}, d.notes.filter((n) => n !== plumbing).map((n) => h("li", {}, n)))));
  }

  show(el, h("h2", {}, "Benchmark"),
    h("p.lede", {}, "Saved comparisons of the three systems on the benchmark questions (python -m legalrag.eval.compare)."),
    h("div.row", {}, h("label.small.muted", {}, "Run "), sel), h("div", { style: "margin-top:16px" }, out));
  return { enter: async (name) => {
    const files = await api("/api/results");
    if (!files.length) return show(out, h("p.muted", {}, "No saved comparisons yet. Run `python -m legalrag.eval.compare --config offline_demo`."));
    show(sel, files.map((f) => h("option", { value: f.name }, `${f.timestamp} · ${f.config_name}${f.plumbing ? " · plumbing run" : ""}`)));
    const pick = files.some((f) => f.name === name) ? name : files[0].name;
    sel.value = pick; await showResult(pick);
  } };
}

/* ---------- Corpus ----------------------------------------------------------------------------- */
function buildCorpus(el) {
  const list = h("div.list"), detail = h("div.stack"), registry = h("div");
  const filter = h("input", { type: "search", placeholder: "Filter documents", "aria-label": "Filter documents", oninput: () => draw() });
  let data = null, current = null;

  function draw() {
    const q = filter.value.toLowerCase(), groups = { judgment: "Judgments", statute: "Statutes" };
    const docs = data.documents.filter((d) => (d.title + (d.neutral_citation || "")).toLowerCase().includes(q));
    show(list, Object.entries(groups).flatMap(([t, label]) => {
      const g = docs.filter((d) => d.doc_type === t);
      return g.length ? [h("div.grp", {}, label), g.map((d) => h("button", { type: "button", "aria-current": String(String(d.id) === String(current)),
        onclick: () => { location.hash = `#corpus/${d.id}`; } }, h("strong", {}, d.title),
        h("span.small.muted", {}, [d.neutral_citation, d.court, d.decided_on].filter(Boolean).join(" · ") || `${d.n_chunks} passages`)))] : [];
    }));
  }

  async function open(id) {
    current = id; draw();
    show(detail, h("span.spinner"));
    const d = await api(`/api/corpus/${id}`);
    show(detail, h("div.card", {}, h("h3", {}, d.title),
      h("p.small.muted", {}, [d.doc_type, d.neutral_citation, d.court, d.decided_on].filter(Boolean).join(" · ")),
      d.versions.length ? [h("h4", { style: "margin-top:14px" }, "Version history"),
        h("p.small.muted", {}, "The temporal check uses these date ranges to catch answers that rest on superseded text."),
        h("div.table-wrap", {}, h("table", {}, h("thead", {}, h("tr", {}, ["Section", "In force", "Note"].map((x) => h("th", {}, x)))),
          h("tbody", {}, d.versions.map((v) => h("tr", {}, h("td", {}, v.section_number),
            h("td", {}, `${v.valid_from} → ${v.valid_until || "today"}`), h("td", {}, v.amendment_note || "–"))))))] : null,
      h("h4", { style: "margin-top:14px" }, `Passages (${d.chunks.length})`),
      d.chunks.map((c) => h("div.chunk", {}, h("span.chip.tag", {}, c.section_label || "unlabelled"), h("pre", {}, c.text)))));
  }

  show(el, h("h2", {}, "Corpus"),
    h("p.lede", {}, "What the system can ground answers in. The corpus is a synthetic fixture. The citation registry is larger: it lets the authority check tell a fabricated case from a real one whose text is not held."),
    h("div.split", {}, h("div.stack", {}, filter, list), h("div.stack", {}, detail, registry)));
  return { enter: async (id) => {
    if (!data) {
      data = await api("/api/corpus");
      const held = data.registry.filter((r) => r.in_corpus).length;
      show(registry, h("details.card", {}, h("summary", {}, `Citation registry (${data.registry.length}; ${held} with full text held)`),
        h("div.table-wrap", {}, h("table", {}, h("thead", {}, h("tr", {}, ["Case", "Citation", "Court", "Text held"].map((x) => h("th", {}, x)))),
          h("tbody", {}, data.registry.map((r) => h("tr", {}, h("td", {}, r.case_name), h("td.mono", {}, r.citation), h("td", {}, r.court || "–"), h("td", {}, r.in_corpus ? "✓" : "no"))))))));
    }
    current = id || null; draw();
    if (id) await open(id); else show(detail, h("p.muted", {}, "Select a document to read it."));
  } };
}

/* ---------- shell ------------------------------------------------------------------------------ */
function renderBanner() {
  const c = state.meta.config;
  show(document.getElementById("banner"),
    c.real_corpus
      ? h("p", {}, "Research assistant, not legal advice. The corpus is real Indian legal text, but judgments are unofficial copies and statute text may not reflect later amendments: verify against the official source.")
      : h("p", {}, "The corpus is synthetic sample data, not real Indian law. Nothing here is legal advice."),
    c.demo ? h("p", {}, "This config uses a deterministic fake generator: it demonstrates the verification pipeline, not the quality of any model.") : null);
}

function renderConfigs() {
  const sel = document.getElementById("cfgSel");
  show(sel, state.meta.configs.map((n) => h("option", { value: n }, n)));
  sel.value = state.meta.config.name;
  sel.onchange = async () => {
    const prev = state.meta.config.name;
    sel.disabled = true;
    try { state.meta = await api("/api/config", { name: sel.value }); renderBanner(); }
    catch (err) { toast(err.message); sel.value = prev; }
    sel.disabled = false;
  };
}

async function route() {
  const [path, qs] = (location.hash.slice(1) || "ask").split("?");
  const [tab, ...rest] = path.split("/");
  const name = views[tab] ? tab : "ask";
  for (const [n, v] of Object.entries(views)) v.el.hidden = n !== name;
  document.querySelectorAll("#tabs a").forEach((a) => {
    if (a.dataset.tab === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  const v = views[name];
  if (name === "ask" && qs) {  // #ask?q=...&run=1 makes a question linkable
    const p = new URLSearchParams(qs);
    if (p.get("q")) v.api.fill(p.get("q"));
    if (p.get("arms")) v.api.boxes.forEach((b) => { b.firstChild.checked = p.get("arms").split(",").includes(b.firstChild.value); });
    if (p.get("run")) { history.replaceState(null, "", "#ask"); v.api.run(); }
  }
  if (v.enter) {
    try { await v.enter(rest.join("/")); }
    catch (err) { show(v.el, h("div.alert", {}, err.message)); }
  }
}

async function init() {
  try { state.meta = await api("/api/meta"); }
  catch (err) { document.getElementById("main").prepend(h("div.alert", {}, `Could not reach the server: ${err.message}`)); return; }
  state.questions = Object.fromEntries(state.meta.samples.map((s) => [s.id, s.question]));
  renderBanner(); renderConfigs();
  const mk = (name, build) => { const el = document.getElementById(`view-${name}`); const api = build(el); views[name] = { el, api, enter: api.enter }; };
  mk("ask", buildAsk); mk("history", buildHistory); mk("benchmark", buildBenchmark); mk("corpus", buildCorpus);
  addEventListener("hashchange", route);
  route();
}
init();
