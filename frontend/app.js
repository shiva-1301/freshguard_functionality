"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
let lastHistoryId = null;
let chosenRating = 0;

const EXAMPLES = [
  "what is the speed of light",
  "who won the 2022 world cup",
  "what is the capital of Australia",
  "what is the current federal reserve interest rate in 2025",
  "which city hosted the 2024 summer olympics",
  "state the pythagorean theorem",
];

// ---- tabs -------------------------------------------------------------------
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.remove("active"));
  $$(".tab-panel").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  $("#tab-" + t.dataset.tab).classList.add("active");
  if (t.dataset.tab === "history") loadHistory();
  if (t.dataset.tab === "admin") { loadConfig(); loadKB(); }
}));

// ---- examples ---------------------------------------------------------------
EXAMPLES.forEach((ex) => {
  const el = document.createElement("span");
  el.className = "ex"; el.textContent = ex;
  el.addEventListener("click", () => { $("#q").value = ex; $("#ask-form").requestSubmit(); });
  $("#examples").appendChild(el);
});

// ---- ask --------------------------------------------------------------------
$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#q").value.trim();
  if (!q) return;
  const btn = $("#ask-btn");
  btn.disabled = true; btn.textContent = "…";
  try {
    const r = await fetch("/api/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });
    const data = await r.json();
    renderResult(data);
  } catch (err) {
    alert("Request failed: " + err);
  } finally {
    btn.disabled = false; btn.textContent = "Ask";
  }
});

function renderResult(d) {
  $("#result").classList.remove("hidden");
  lastHistoryId = d.history_id; chosenRating = 0;

  const conf = Math.round(d.confidence);
  $("#conf-fill").style.width = conf + "%";
  $("#conf-val").textContent = conf + "% confidence";

  const fb = $("#fresh-badge");
  fb.textContent = d.freshness_label === "stale"
    ? `Knowledge likely stale (${d.freshness_score.toFixed(2)})`
    : `Knowledge fresh (${d.freshness_score.toFixed(2)})`;
  fb.className = "badge " + (d.freshness_label === "stale" ? "badge-stale" : "badge-fresh");

  const rb = $("#retrieve-badge");
  rb.textContent = d.retrieved ? "Retrieved evidence" : "Answered directly (no retrieval)";
  rb.className = "badge " + (d.retrieved ? "badge-retrieved" : "badge-skip");
  rb.title = d.skipped_reason || "";

  $("#cat-badge").textContent = d.query_category;

  $("#answer").textContent = d.answer;

  // warnings
  const w = $("#warnings");
  if (d.warnings && d.warnings.length) {
    w.innerHTML = "<strong>Notices</strong><ul>" +
      d.warnings.map((x) => `<li>${esc(x)}</li>`).join("") + "</ul>";
    w.classList.remove("hidden");
  } else w.classList.add("hidden");

  // statements (FR14)
  const sw = $("#statements-wrap"), sl = $("#statements");
  sl.innerHTML = "";
  if (d.statements && d.statements.length) {
    d.statements.forEach((s) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="tag ${s.label}">${s.label}</span>` +
        `<span>${esc(s.text)}</span>` +
        `<span class="st-sup">${s.evidence_doc_id ? "src " + esc(s.evidence_doc_id) + " · " : ""}` +
        `support ${s.support.toFixed(2)}</span>`;
      sl.appendChild(li);
    });
    sw.classList.remove("hidden");
  } else sw.classList.add("hidden");

  // sources (FR15)
  const ow = $("#sources-wrap"), ol = $("#sources");
  ol.innerHTML = "";
  if (d.sources && d.sources.length) {
    d.sources.forEach((s) => {
      const li = document.createElement("li");
      const link = s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a>`
                         : esc(s.title);
      li.innerHTML = `${link}<span class="meta">${esc(s.date)} · relevance ${s.relevance.toFixed(2)} · recency ${s.recency.toFixed(2)}</span>`;
      ol.appendChild(li);
    });
    ow.classList.remove("hidden");
  } else ow.classList.add("hidden");

  buildStars();
  $("#feedback").classList.toggle("hidden", lastHistoryId == null);
  $("#fb-msg").textContent = "";
}

// ---- feedback (FR18) --------------------------------------------------------
function buildStars() {
  const box = $("#stars"); box.innerHTML = "";
  for (let i = 1; i <= 5; i++) {
    const s = document.createElement("span");
    s.className = "star"; s.textContent = "★"; s.dataset.v = i;
    s.addEventListener("click", () => { chosenRating = i; paintStars(); sendFeedback({ rating: i }); });
    box.appendChild(s);
  }
}
function paintStars() {
  $$("#stars .star").forEach((s) => s.classList.toggle("on", Number(s.dataset.v) <= chosenRating));
}
$("#report-btn").addEventListener("click", () => sendFeedback({ report_incorrect: true }));

async function sendFeedback(extra) {
  if (lastHistoryId == null) return;
  const body = Object.assign({ history_id: lastHistoryId }, extra);
  const r = await fetch("/api/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (r.ok) $("#fb-msg").textContent = "Thanks for the feedback!";
}

// ---- history (FR17) ---------------------------------------------------------
$("#refresh-history").addEventListener("click", loadHistory);
async function loadHistory() {
  const r = await fetch("/api/history?limit=50");
  const { items } = await r.json();
  const tb = $("#history-body"); tb.innerHTML = "";
  items.forEach((it) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${it.id}</td><td>${esc(it.query)}</td>` +
      `<td>${Math.round(it.confidence)}%</td><td>${it.freshness.toFixed(2)}</td>` +
      `<td>${it.retrieved ? "yes" : "no"}</td><td>${esc((it.created_at || "").replace("T", " "))}</td>`;
    tb.appendChild(tr);
  });
}

// ---- admin: config (FR19) ---------------------------------------------------
async function loadConfig() {
  const c = await (await fetch("/api/config")).json();
  const f = $("#config-form");
  f.freshness_threshold.value = c.freshness_threshold;
  f.confidence_threshold.value = c.confidence_threshold;
  f.top_k.value = c.top_k;
  f.llm_provider.value = c.llm_provider;
  f.web_search_enabled.checked = c.web_search_enabled;
}
$("#config-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = {
    freshness_threshold: parseFloat(f.freshness_threshold.value),
    confidence_threshold: parseFloat(f.confidence_threshold.value),
    top_k: parseInt(f.top_k.value, 10),
    llm_provider: f.llm_provider.value,
    web_search_enabled: f.web_search_enabled.checked,
  };
  await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  $("#config-msg").textContent = "Saved.";
});

// ---- admin: KB (FR20) -------------------------------------------------------
async function loadKB() {
  const { items } = await (await fetch("/api/kb")).json();
  const ul = $("#kb-list"); ul.innerHTML = "";
  items.forEach((d) => {
    const li = document.createElement("li");
    li.innerHTML = `<span><strong>${esc(d.doc_id)}</strong> — ${esc(d.title)} <em style="color:var(--muted)">(${esc(d.date)})</em></span>`;
    const b = document.createElement("button"); b.textContent = "remove";
    b.addEventListener("click", async () => {
      await fetch("/api/kb/" + encodeURIComponent(d.doc_id), { method: "DELETE" });
      loadKB();
    });
    li.appendChild(b); ul.appendChild(li);
  });
}
$("#kb-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = { title: f.title.value, url: f.url.value, date: f.date.value, text: f.text.value };
  await fetch("/api/kb", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  f.reset(); f.date.value = "2025-01-01"; loadKB();
});

// ---- admin: eval (FR21) -----------------------------------------------------
$("#run-eval").addEventListener("click", async () => {
  $("#eval-out").textContent = "Running…";
  const r = await (await fetch("/api/eval")).json();
  $("#eval-out").textContent = JSON.stringify(r, null, 2);
});

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
