/* Perioperative Digital Twin — frontend controller.
   Reads controls, calls /api/simulate, renders the MAP monitor + Ce strip on canvas. */

const C = {
  cyan: "#35e0c8", amber: "#ffb020", red: "#ff3b47",
  ne: "#c98bff", band: "rgba(90,150,175,0.16)", grid: "#0e1a22",
  gridMaj: "#162833", ink: "#7d94a3", danger: "rgba(255,59,71,0.06)",
};

const PRESETS = {
  healthy: { age: 35, weight: 75, height: 175, sex: "M", map0: 95, propofol: 25, norepi: 0 },
  frail:   { age: 82, weight: 54, height: 162, sex: "F", map0: 82, propofol: 45, norepi: 0 },
};

const state = {
  age: 60, weight: 70, height: 170, sex: "M", map0: 90, duration: 15,
  propofol: 30, norepi: 0, norepi_start: 0,
  personalize: false, delta_ec50: 0, delta_ke0: 0, show_band: true,
};

const OUT_FMT = {
  age: v => `${v}`, weight: v => `${v} kg`, height: v => `${v} cm`,
  map0: v => `${v} mmHg`, duration: v => `${v} min`,
  propofol: v => `${v}`, norepi: v => `${v}`, norepi_start: v => `${v} min`,
  delta_ec50: v => Number(v).toFixed(2), delta_ke0: v => Number(v).toFixed(2),
};

/* ---------- wire controls ---------- */
function bindRange(id) {
  const el = document.getElementById(id);
  el.addEventListener("input", () => {
    state[id] = parseFloat(el.value);
    setOut(id);
    schedule();
  });
  setOut(id);
}
function setOut(id) {
  const out = document.querySelector(`[data-out="${id}"]`);
  if (out && OUT_FMT[id]) out.textContent = OUT_FMT[id](state[id]);
}
["age","weight","height","map0","duration","propofol","norepi","norepi_start","delta_ec50","delta_ke0"]
  .forEach(bindRange);

document.getElementById("sex").addEventListener("click", e => {
  const b = e.target.closest("button[data-sex]"); if (!b) return;
  document.querySelectorAll("#sex button").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); state.sex = b.dataset.sex; schedule();
});

const persEl = document.getElementById("personalize");
persEl.addEventListener("change", () => {
  state.personalize = persEl.checked;
  ["delta_ec50","delta_ke0"].forEach(id => {
    document.getElementById(id).disabled = !persEl.checked;
    document.getElementById("wrap_" + id.split("_")[1]).classList.toggle("disabled", !persEl.checked);
  });
  document.querySelector(".lg-pers").hidden = !persEl.checked;
  schedule();
});

document.getElementById("show_band").addEventListener("change", e => {
  state.show_band = e.target.checked; schedule();
});

document.querySelectorAll(".preset").forEach(btn => {
  btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
});
function applyPreset(name) {
  const p = PRESETS[name]; if (!p) return;
  Object.assign(state, p);
  for (const k of ["age","weight","height","map0","propofol","norepi"]) {
    const el = document.getElementById(k); if (el) { el.value = state[k]; setOut(k); }
  }
  document.querySelectorAll("#sex button").forEach(x =>
    x.classList.toggle("active", x.dataset.sex === state.sex));
  schedule(true);
}

/* ---------- API ---------- */
let timer = null, inFlight = false, pending = false;
function schedule(immediate = false) {
  if (immediate) { fire(); return; }
  clearTimeout(timer); timer = setTimeout(fire, 110);
}
async function fire() {
  if (inFlight) { pending = true; return; }
  inFlight = true;
  const body = {
    age: state.age, weight: state.weight, height: state.height, sex: state.sex,
    map0: state.map0, duration_min: state.duration,
    propofol: state.propofol, norepi: state.norepi, norepi_start_min: state.norepi_start,
    personalize: state.personalize, delta_ec50: state.delta_ec50, delta_ke0: state.delta_ke0,
    show_band: state.show_band,
  };
  try {
    const r = await fetch("/api/simulate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await r.json();
    render(data);
  } catch (e) { console.error(e); }
  inFlight = false;
  if (pending) { pending = false; fire(); }
}

/* ---------- readouts ---------- */
function render(d) {
  document.getElementById("riskValue").textContent = d.risk.toUpperCase();
  const tile = document.getElementById("riskTile");
  tile.className = "vital risk " + d.risk.toLowerCase();
  const col = d.risk === "High" ? C.red : d.risk === "Medium" ? C.amber : "#46e07a";
  const led = document.getElementById("statusLed");
  led.style.background = col; led.style.boxShadow = `0 0 22px ${col}`;
  setVal("minMap", d.min_map, "mmHg");
  setVal("minsBelow", d.minutes_below_65, "min");
  setVal("ceEnd", d.ce_end, "µg/mL");
  drawChart(d);
  drawCe(d);
}
function setVal(id, num, unit) {
  const el = document.getElementById(id);
  el.textContent = num + " ";
  const em = document.createElement("em");
  em.textContent = unit;
  el.appendChild(em);
}

/* ---------- canvas helpers ---------- */
function prep(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

const MAP_MIN = 40, MAP_MAX = 120;
let lastData = null;
let animT = 0, animing = false;

function drawChart(d, progress = 1) {
  lastData = d;
  const cv = document.getElementById("chart");
  const { ctx, w, h } = prep(cv);
  const padL = 46, padR = 16, padT = 16, padB = 26;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const n = d.t_min.length;
  const tMax = d.t_min[n - 1] || 1;
  const X = t => padL + (t / tMax) * plotW;
  const Y = m => padT + (1 - (m - MAP_MIN) / (MAP_MAX - MAP_MIN)) * plotH;

  ctx.clearRect(0, 0, w, h);

  // danger zone below threshold
  ctx.fillStyle = C.danger;
  ctx.fillRect(padL, Y(d.threshold), plotW, Y(MAP_MIN) - Y(d.threshold));

  // grid
  ctx.lineWidth = 1; ctx.font = "10px 'IBM Plex Mono', monospace"; ctx.fillStyle = C.ink;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let m = MAP_MIN; m <= MAP_MAX; m += 10) {
    ctx.strokeStyle = (m % 20 === 0) ? C.gridMaj : C.grid;
    ctx.beginPath(); ctx.moveTo(padL, Y(m)); ctx.lineTo(w - padR, Y(m)); ctx.stroke();
    ctx.fillText(m, padL - 8, Y(m));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const step = tMax <= 15 ? 3 : 5;
  for (let t = 0; t <= tMax + 0.01; t += step) {
    ctx.strokeStyle = C.grid;
    ctx.beginPath(); ctx.moveTo(X(t), padT); ctx.lineTo(X(t), h - padB); ctx.stroke();
    ctx.fillText(t + "m", X(t), h - padB + 6);
  }

  // confidence band
  if (d.band_lo && d.band_hi) {
    ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = X(d.t_min[i]); const y = Y(d.band_hi[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    for (let i = n - 1; i >= 0; i--) ctx.lineTo(X(d.t_min[i]), Y(d.band_lo[i]));
    ctx.closePath(); ctx.fillStyle = C.band; ctx.fill();
  }

  // threshold line
  ctx.strokeStyle = C.red; ctx.lineWidth = 1.4; ctx.setLineDash([7, 6]);
  ctx.beginPath(); ctx.moveTo(padL, Y(d.threshold)); ctx.lineTo(w - padR, Y(d.threshold)); ctx.stroke();
  ctx.setLineDash([]);

  const upto = Math.max(1, Math.floor(n * progress));

  // population trace
  trace(ctx, d.t_min, d.map_pop, X, Y, upto, C.cyan, 2.2);
  // personalized trace
  if (d.personalized) trace(ctx, d.t_min, d.map_shown, X, Y, upto, C.amber, 2.2);

  // moving head dot
  const hi = upto - 1;
  const hy = Y((d.personalized ? d.map_shown : d.map_pop)[hi]);
  const hx = X(d.t_min[hi]);
  ctx.beginPath(); ctx.arc(hx, hy, 3.2, 0, 7); ctx.fillStyle = d.personalized ? C.amber : C.cyan;
  ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 14; ctx.fill(); ctx.shadowBlur = 0;
}

function trace(ctx, xs, ys, X, Y, upto, color, lw) {
  ctx.beginPath();
  for (let i = 0; i < upto; i++) { const x = X(xs[i]), y = Y(ys[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineJoin = "round";
  ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.stroke(); ctx.shadowBlur = 0;
}

function drawCe(d) {
  const cv = document.getElementById("ceChart");
  const { ctx, w, h } = prep(cv);
  const padL = 46, padR = 16, padT = 20, padB = 16;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const n = d.ce.length, tMax = d.t_min[n - 1] || 1;
  const ceMax = Math.max(1, Math.ceil(Math.max(...d.ce) * 1.15));
  const X = t => padL + (t / tMax) * plotW;
  const Y = c => padT + (1 - c / ceMax) * plotH;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  ctx.font = "9px 'IBM Plex Mono', monospace"; ctx.fillStyle = C.ink;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 2; i++) {
    const c = (ceMax / 2) * i;
    ctx.beginPath(); ctx.moveTo(padL, Y(c)); ctx.lineTo(w - padR, Y(c)); ctx.stroke();
    ctx.fillText(c.toFixed(1), padL - 6, Y(c));
  }
  // area fill
  ctx.beginPath(); ctx.moveTo(X(0), Y(0));
  for (let i = 0; i < n; i++) ctx.lineTo(X(d.t_min[i]), Y(d.ce[i]));
  ctx.lineTo(X(tMax), Y(0)); ctx.closePath();
  ctx.fillStyle = "rgba(201,139,255,0.12)"; ctx.fill();
  trace(ctx, d.t_min, d.ce, X, Y, n, C.ne, 1.8);
}

/* animate the trace sweep on load / preset */
function sweep() {
  if (!lastData) return;
  animing = true; animT = 0;
  const tick = () => {
    animT += 0.045;
    drawChart(lastData, Math.min(1, animT));
    if (animT < 1) requestAnimationFrame(tick); else animing = false;
  };
  requestAnimationFrame(tick);
}

/* clock */
setInterval(() => {
  document.getElementById("clock").textContent = new Date().toLocaleTimeString("en-GB");
}, 1000);

/* redraw on resize */
let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => lastData && (drawChart(lastData), drawCe(lastData)), 120); });

/* boot */
(async function boot() {
  await fire();
  sweep();
})();
