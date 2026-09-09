// --- admin auth -------------------------------------------------------------
// Mutating endpoints require X-Admin-Token and return 503 when the deployment
// has no ADMIN_TOKEN configured (the read-only default). The token is kept in
// localStorage so it is never baked into the served bundle.
const ADMIN_TOKEN_KEY = "twin.adminToken";

function adminToken() {
  try { return localStorage.getItem(ADMIN_TOKEN_KEY) || ""; } catch { return ""; }
}

function setAdminToken(v) {
  try { localStorage.setItem(ADMIN_TOKEN_KEY, v); } catch { /* private mode */ }
}

/** POST to an admin endpoint, surfacing the two auth failures in plain words. */
async function adminPost(url) {
  const res = await fetch(url, {
    method: "POST",
    headers: adminToken() ? { "X-Admin-Token": adminToken() } : {},
  });
  if (res.status === 503) {
    throw new Error("Admin controls are disabled on this deployment "
                    + "(ADMIN_TOKEN is not set on the server).");
  }
  if (res.status === 401) {
    throw new Error("Admin token rejected. Set a valid token to run this.");
  }
  if (!res.ok) throw new Error(`Request failed (${res.status}).`);
  return res.json();
}

/* Digital Twin — ML Ops console controller. */
const CYAN = "#5fe3c0", AMBER = "#f2b441",
      GRID = "rgba(236,231,222,0.07)", INK = "#64716f";

/* ---------- clock ---------- */
setInterval(() => {
  document.getElementById("clock").textContent = new Date().toLocaleTimeString("en-GB");
}, 1000);

/* ---------- canvas plotting ---------- */
function prep(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

function plot(id, series, opts = {}) {
  const cv = document.getElementById(id);
  const { ctx, w, h } = prep(cv);
  const padL = 38, padR = 10, padT = 10, padB = 26;
  const pw = w - padL - padR, ph = h - padT - padB;
  const xd = opts.xdomain || [0, 1], yd = opts.ydomain || [0, 1];
  const X = x => padL + ((x - xd[0]) / (xd[1] - xd[0])) * pw;
  const Y = y => padT + (1 - (y - yd[0]) / (yd[1] - yd[0])) * ph;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = GRID; ctx.fillStyle = INK; ctx.lineWidth = 1;
  ctx.font = "9px 'JetBrains Mono', monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const yy = yd[0] + (yd[1] - yd[0]) * i / 4;
    ctx.beginPath(); ctx.moveTo(padL, Y(yy)); ctx.lineTo(w - padR, Y(yy)); ctx.stroke();
    ctx.fillText(yy.toFixed(2), padL - 5, Y(yy));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const xx = xd[0] + (xd[1] - xd[0]) * i / 4;
    ctx.fillText(xx.toFixed(2), X(xx), h - padB + 5);
  }
  if (opts.diagonal) {
    ctx.strokeStyle = "rgba(236,231,222,0.14)"; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(X(xd[0]), Y(yd[0])); ctx.lineTo(X(xd[1]), Y(yd[1])); ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const s of series) {
    if (!s.x || !s.x.length) continue;
    ctx.beginPath();
    for (let i = 0; i < s.x.length; i++) { const px = X(s.x[i]), py = Y(s.y[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.shadowColor = s.color; ctx.shadowBlur = 6; ctx.stroke(); ctx.shadowBlur = 0;
    if (s.dots) for (let i = 0; i < s.x.length; i++) { ctx.beginPath(); ctx.arc(X(s.x[i]), Y(s.y[i]), 2, 0, 7); ctx.fillStyle = s.color; ctx.fill(); }
  }
}

/* ---------- status ---------- */
async function loadStatus() {
  const d = await (await fetch("/api/admin/status")).json();
  const tiles = [
    ["Cached cases", String(d.cached_cases), ""],
    ["Windows", d.n_windows != null ? d.n_windows.toLocaleString() : "—", ""],
    ["IOH prevalence", d.prevalence != null ? (d.prevalence * 100).toFixed(1) + "%" : "—", ""],
    ["Predictor", d.model_trained ? "TRAINED" : "NONE", d.model_trained ? "ok" : "off"],
    ["Unbiased AUROC", String(d.model_meta?.unbiased?.auroc ?? "—"), ""],
    ["Demo cases", String(d.demo_cases), ""],
  ];
  const g = document.getElementById("statGrid");
  g.replaceChildren();
  for (const [label, val, cls] of tiles) {
    const el = document.createElement("div");
    el.className = "stat";
    const l = document.createElement("div"); l.className = "k"; l.textContent = label;
    const v = document.createElement("div"); v.className = "v " + cls; v.textContent = val;
    el.append(l, v); g.appendChild(el);
  }
  // SYSTEM card
  const sys = {
    sysRuntime: d.runtime ?? "—",
    sysSeed: String(d.seed ?? "—"),
    sysHorizon: d.horizon_min ? `${d.horizon_min} min` : "—",
    sysThresh: d.map_threshold ? `${d.map_threshold} mmHg` : "—",
    sysLastTrain: d.last_train ?? "—",
    sysArtifact: d.artifact ?? "—",
  };
  for (const [id, val] of Object.entries(sys)) {
    const el = document.getElementById(id); if (el) el.textContent = val;
  }
}

/* ---------- metrics table ---------- */
const COLS = ["model", "regime", "auroc", "auprc", "ppv", "ece", "brier", "n", "prevalence"];
async function loadMetrics() {
  const d = await (await fetch("/api/admin/metrics")).json();
  const t = document.getElementById("metricsTable");
  t.replaceChildren();
  if (!d.available) { t.textContent = "No results yet — build a dataset."; return; }
  const thead = document.createElement("tr");
  for (const c of COLS) { const th = document.createElement("th"); th.textContent = c.toUpperCase(); thead.appendChild(th); }
  t.appendChild(thead);
  for (const row of d.rows) {
    const tr = document.createElement("tr"); tr.className = row.regime;
    for (const c of COLS) {
      const td = document.createElement("td");
      let v = row[c];
      if (typeof v === "number" && !Number.isInteger(v)) v = v.toFixed(3);
      td.textContent = v == null ? "—" : String(v);
      tr.appendChild(td);
    }
    t.appendChild(tr);
  }
}

/* ---------- curves ---------- */
async function loadCurves() {
  const d = await (await fetch("/api/admin/curves")).json();
  if (!d.available) return;
  const mm = d.models.map_only, gb = d.models.gbdt;
  plot("rocC", [
    { x: mm.roc.fpr, y: mm.roc.tpr, color: CYAN },
    { x: gb.roc.fpr, y: gb.roc.tpr, color: AMBER },
  ], { diagonal: true });
  plot("prC", [
    { x: mm.pr.recall, y: mm.pr.precision, color: CYAN },
    { x: gb.pr.recall, y: gb.pr.precision, color: AMBER },
  ], {});
  plot("calC", [
    { x: mm.reliability.conf, y: mm.reliability.frac, color: CYAN, dots: true },
    { x: gb.reliability.conf, y: gb.reliability.frac, color: AMBER, dots: true },
  ], { diagonal: true });
  if (d.decision_curve) {
    const by = {};
    for (const r of d.decision_curve) { (by[r.model] ??= { x: [], y: [] }); by[r.model].x.push(r.threshold); by[r.model].y.push(r.net_benefit); }
    const ymin = Math.min(0, ...d.decision_curve.map(r => r.net_benefit));
    const ymax = Math.max(...d.decision_curve.map(r => r.net_benefit), 0.02);
    plot("dcaC", [
      { x: by.map_only?.x, y: by.map_only?.y, color: CYAN },
      { x: by.gbdt?.x, y: by.gbdt?.y, color: AMBER },
    ], { xdomain: [0, 0.5], ydomain: [ymin, ymax] });
  }
}

async function refreshAll() { await Promise.all([loadStatus(), loadMetrics(), loadCurves()]); }

/* ---------- training controls ---------- */
const trainMsg = document.getElementById("trainMsg");
const trainLog = document.getElementById("trainLog");
const trainStage = document.getElementById("trainStage");
const trainPct = document.getElementById("trainPct");
const trainFill = document.getElementById("trainFill");

/* smooth percentage while a fast sync train runs (real % isn't available for a
   few-second sklearn fit, so we ease toward 90% then snap to 100% on completion) */
function startProgress(stage) {
  trainLog.hidden = false; trainStage.textContent = stage;
  let p = 0; trainPct.textContent = "0%"; trainFill.style.width = "0%";
  const iv = setInterval(() => {
    p = Math.min(90, p + Math.max(1, (90 - p) * 0.12));
    trainPct.textContent = `${Math.round(p)}%`; trainFill.style.width = `${p}%`;
  }, 220);
  return () => {
    clearInterval(iv);
    trainPct.textContent = "100%"; trainFill.style.width = "100%"; trainStage.textContent = "complete";
    setTimeout(() => { trainLog.hidden = true; }, 1200);
  };
}

// A full retrain runs for many minutes, so it is a polled background job rather
// than one long request that a hosting proxy would time out.
document.getElementById("btnRetrain").addEventListener("click", async (e) => {
  e.target.disabled = true; trainMsg.textContent = "";
  const done = startProgress("training MAP-only + GBDT…");
  let job_id;
  try {
    ({ job_id } = await adminPost("/api/admin/train-predictor"));
  } catch (err) { done(); trainMsg.textContent = "Could not start training."; e.target.disabled = false; return; }

  const tick = async () => {
    let j;
    try { j = await (await fetch(`/api/admin/job/${job_id}`)).json(); }
    catch { return setTimeout(tick, 3000); }
    if (j.status === "running") {
      trainMsg.textContent = `Training… ${Math.round(j.elapsed)}s elapsed (takes several minutes).`;
      return setTimeout(tick, 3000);
    }
    done();
    if (j.status === "done") {
      const u = j.predictor_meta?.unbiased;
      trainMsg.textContent = u ? `Done. Predictor unbiased AUROC ${u.auroc}, PPV ${u.ppv}.` : "Done.";
      await refreshAll();
    } else {
      trainMsg.textContent = `Training failed. ${j.error || ""}`;
    }
    e.target.disabled = false;
  };
  tick();
});

document.getElementById("btnRescan").addEventListener("click", async (e) => {
  e.target.disabled = true;
  const done = startProgress("rescanning demo cases…");
  try { const d = await adminPost("/api/admin/rescan-cases"); trainMsg.textContent = `Rescanned: ${d.n} demo cases.`; }
  catch (err) { trainMsg.textContent = "Rescan failed."; }
  done(); await loadStatus();
  e.target.disabled = false;
});

const nCases = document.getElementById("nCases");
nCases.addEventListener("input", () => { document.getElementById("nCasesOut").textContent = nCases.value; });

document.getElementById("btnBuild").addEventListener("click", async (e) => {
  e.target.disabled = true;
  const log = document.getElementById("jobLog"); log.hidden = false;
  try {
    const { job_id } = await adminPost(`/api/admin/build?n_cases=${nCases.value}`);
    poll(job_id, e.target);
  } catch (err) { document.getElementById("jobOut").textContent = "Failed to start build."; e.target.disabled = false; }
});

async function poll(id, btn) {
  const stageEl = document.getElementById("jobStage");
  const pctEl = document.getElementById("jobPct");
  const fillEl = document.getElementById("jobFill");
  const elEl = document.getElementById("jobElapsed");
  const out = document.getElementById("jobOut");
  const tick = async () => {
    let j;
    try { j = await (await fetch(`/api/admin/job/${id}`)).json(); }
    catch { return setTimeout(tick, 1500); }
    stageEl.textContent = j.stage;
    pctEl.textContent = `${j.percent ?? 0}%`;
    fillEl.style.width = `${j.percent ?? 0}%`;
    elEl.textContent = `${j.elapsed}s`;
    out.textContent = j.log || "(starting…)"; out.scrollTop = out.scrollHeight;
    if (j.status === "running") return setTimeout(tick, 1500);
    if (j.status === "done") await refreshAll();
    btn.disabled = false; btn.textContent = "Start build ▶";
  };
  tick();
}

/* boot */
refreshAll();

// --- admin token field ------------------------------------------------------
(function wireAdminToken() {
  const input = document.getElementById("adminToken");
  const msg = document.getElementById("adminTokenMsg");
  if (!input) return;
  input.value = adminToken();
  const render = () => {
    msg.textContent = input.value
      ? "Token set — admin actions enabled."
      : "No token — admin actions will be refused.";
  };
  input.addEventListener("input", () => { setAdminToken(input.value.trim()); render(); });
  render();
})();
