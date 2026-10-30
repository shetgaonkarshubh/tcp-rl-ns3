"use strict";

/* ================================================================
   Helpers
   ================================================================ */
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

const ALGOS = {
  rl:      { name: "PPO agent", short: "PPO",     note: "Our trained policy", shape: "circle" },
  newreno: { name: "NewReno",   short: "NewReno", note: "Classic, loss-based", shape: "triangle" },
  cubic:   { name: "CUBIC",     short: "CUBIC",   note: "Linux default", shape: "rect" },
  bbr:     { name: "BBR",       short: "BBR",     note: "Google, model-based", shape: "rectRot" },
};
const ORDER = ["rl", "newreno", "cubic", "bbr"];

const METRICS = [
  { key: "throughput",  label: "Throughput",  unit: "Mbps", digits: 2, better: "high", note: "Average data delivered to the receiver." },
  { key: "utilization", label: "Utilisation", unit: "%",    digits: 1, better: "high", note: "Share of the bottleneck's capacity that was used." },
  { key: "rtt",         label: "Delay",       unit: "ms",   digits: 1, better: "low",  note: "Average round-trip time, including time spent waiting in the queue." },
  { key: "rtt_p95",     label: "Delay p95",   unit: "ms",   digits: 1, better: "low",  note: "95th-percentile round-trip time: how bad the slow moments get." },
  { key: "drops",       label: "Drops",       unit: "pkts", digits: 0, better: "low",  note: "Packets lost because the bottleneck queue was full." },
  { key: "cwnd_error",  label: "Window error", unit: "pkts", digits: 1, better: "low", note: "Average gap between the congestion window and the ideal window (link capacity × base delay)." },
];

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const algoColor = (a) => cssVar(`--algo-${a}`);
const fmt = (v, d = 1) => (v == null || isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const icons = () => window.lucide && lucide.createIcons({ attrs: { "aria-hidden": "true" } });

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body ? { "content-type": "application/json" } : {},
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

let toastTimer;
function toast(text, icon = "check") {
  const t = $("#toast");
  t.innerHTML = `<i data-lucide="${icon}"></i><span></span>`;
  t.querySelector("span").textContent = text;
  icons();
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}

function countUp(node, to, digits, ms = 450) {
  if (reduceMotion.matches) { node.textContent = fmt(to, digits); return; }
  const from = parseFloat(node.dataset.v || "0") || 0;
  node.dataset.v = to;
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / ms);
    const e = 1 - Math.pow(1 - k, 3);
    node.textContent = fmt(from + (to - from) * e, digits);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* Segmented control with a sliding pill */
function segmented(root, items, current, onPick) {
  root.innerHTML = `<span class="seg-pill"></span>`;
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = it.label;
    b.dataset.key = it.key;
    b.setAttribute("role", "tab");
    b.addEventListener("click", () => { set(it.key); onPick(it.key); });
    root.appendChild(b);
  }
  const pill = root.querySelector(".seg-pill");
  function set(key) {
    for (const b of $$("button", root)) {
      const on = b.dataset.key === String(key);
      b.classList.toggle("on", on);
      b.setAttribute("aria-selected", on);
      if (on) {
        pill.style.width = `${b.offsetWidth}px`;
        pill.style.transform = `translateX(${b.offsetLeft}px)`;
      }
    }
  }
  requestAnimationFrame(() => set(current));
  return { set };
}

/* ================================================================
   Network lane: the animated topology
   ================================================================ */
const NODES = [
  { id: "snd", x: 0.06, icon: "laptop", label: "Sender" },
  { id: "r1",  x: 0.30, icon: "router", label: "Router" },
  { id: "r2",  x: 0.72, icon: "router", label: "Router" },
  { id: "rcv", x: 0.94, icon: "server", label: "Receiver" },
];

class Lane {
  constructor(parent, { algo, title, queueSize, maxRate, delay, stats = true }) {
    this.algo = algo;
    this.queueSize = queueSize;
    this.maxRate = maxRate;
    this.delay = delay;
    this.state = { capacity: maxRate * 0.7, throughput: maxRate * 0.6, queue: 0, drops: 0, rtt: 0, cwnd: 0 };
    this.shown = { capacity: this.state.capacity, queue: 0 };
    this.packets = [];
    this.debris = [];
    this.emitAcc = 0;
    this.lastDrops = null;

    this.el = document.createElement("div");
    this.el.className = "lane";
    this.el.style.setProperty("--c", `var(--algo-${algo || "rl"})`);
    this.el.innerHTML = `
      <div class="lane-head">
        <div class="lane-name"><span class="dot"></span><span class="lt"></span></div>
        ${stats ? `<div class="lane-stats">
          <span><b data-k="throughput">–</b> Mbps</span>
          <span><b data-k="rtt">–</b> ms</span>
          <span><b data-k="drops">0</b> drops</span>
        </div>` : ""}
      </div>
      <div class="topo">
        <canvas></canvas>
        ${NODES.map((n) => `<div class="node" style="left:${n.x * 100}%"><div class="ico"><i data-lucide="${n.icon}"></i></div><span>${n.label}</span></div>`).join("")}
        <div class="link-label num"></div>
        <div class="queue-label"></div>
      </div>`;
    this.el.querySelector(".lt").textContent = title;
    parent.appendChild(this.el);
    this.canvas = this.el.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.linkLabel = this.el.querySelector(".link-label");
    this.queueLabel = this.el.querySelector(".queue-label");
    this.statEls = Object.fromEntries($$("[data-k]", this.el).map((b) => [b.dataset.k, b]));
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.canvas);
    this.resize();
    this.refreshColors();
    Lane.all.add(this);
    icons();
  }

  destroy() { this.ro.disconnect(); Lane.all.delete(this); this.el.remove(); }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = r.width; this.h = r.height;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  refreshColors() {
    this.colors = {
      algo: algoColor(this.algo || "rl"), pipe: cssVar("--pipe"), edge: cssVar("--pipe-edge"),
      line: cssVar("--line-2"), crit: cssVar("--critical"), panel: cssVar("--panel"),
    };
  }

  set(s) {
    const prevCap = this.state.capacity;
    Object.assign(this.state, s);
    if (this.lastDrops != null && s.drops > this.lastDrops) {
      const n = Math.min(4, s.drops - this.lastDrops);
      for (let i = 0; i < n; i++) this.spawnDrop();
    }
    this.lastDrops = s.drops ?? this.lastDrops;
    const label = `${fmt(this.state.capacity, 1)} Mbps · ${fmt(this.delay, 0)} ms`;
    if (this.linkLabel.textContent !== label) {
      this.linkLabel.textContent = label;
      if (Math.abs(prevCap - this.state.capacity) > 0.4) {
        this.linkLabel.classList.remove("flash");
        void this.linkLabel.offsetWidth;
        this.linkLabel.classList.add("flash");
      }
    }
    if (this.statEls.throughput) {
      this.statEls.throughput.textContent = fmt(s.throughput, 1);
      this.statEls.rtt.textContent = fmt(s.rtt, 0);
      this.statEls.drops.textContent = fmt(s.drops, 0);
    }
  }

  resetDrops() { this.lastDrops = null; this.debris = []; }

  spawnDrop() {
    const x = NODES[1].x * this.w + 30;
    this.debris.push({ x, y: this.h / 2, vx: 20 + Math.random() * 40, vy: -40 - Math.random() * 30, life: 1 });
  }

  tick(dt) {
    const w = this.w, h = this.h, ctx = this.ctx, c = this.colors;
    if (!w) return;
    const cy = h / 2;
    const k = 1 - Math.exp(-dt * 8);  // smoothing toward targets
    this.shown.capacity += (this.state.capacity - this.shown.capacity) * k;
    this.shown.queue += (this.state.queue - this.shown.queue) * k;

    const xs = NODES.map((n) => n.x * w);
    const pad = 24; // half node width
    ctx.clearRect(0, 0, w, h);

    // Access links
    ctx.lineCap = "round";
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xs[0] + pad, cy); ctx.lineTo(xs[1] - pad, cy);
    ctx.moveTo(xs[2] + pad, cy); ctx.lineTo(xs[3] - pad, cy);
    ctx.stroke();

    // Bottleneck pipe: thickness follows capacity
    const frac = Math.max(0.12, Math.min(1, this.shown.capacity / this.maxRate));
    const thick = 6 + 16 * frac;
    const px0 = xs[1] + pad, px1 = xs[2] - pad;
    ctx.fillStyle = c.pipe;
    ctx.strokeStyle = c.edge;
    ctx.lineWidth = 1;
    roundRect(ctx, px0, cy - thick / 2, px1 - px0, thick, thick / 2);
    ctx.fill(); ctx.stroke();

    // Queue buffer at the router's egress
    const cells = Math.min(this.queueSize, 30);
    const per = this.queueSize / cells;
    const qx0 = px0 + 2, qw = Math.min(170, (px1 - px0) * 0.45);
    const qy = cy + 22;
    const cw = qw / cells;
    const filled = this.shown.queue / per;
    for (let i = 0; i < cells; i++) {
      const f = Math.max(0, Math.min(1, filled - i));
      ctx.fillStyle = c.pipe;
      roundRect(ctx, qx0 + i * cw + 1, qy, cw - 2, 10, 2); ctx.fill();
      if (f > 0) {
        ctx.globalAlpha = 0.35 + 0.65 * f;
        ctx.fillStyle = c.algo;
        roundRect(ctx, qx0 + i * cw + 1, qy, cw - 2, 10, 2); ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
    // Beside the buffer when there's room, otherwise underneath it
    const lw = this.queueLabel.offsetWidth || 80;
    const beside = qx0 + qw + 8 + lw < xs[2] - pad - 4;
    this.queueLabel.style.left = `${beside ? qx0 + qw + 8 : qx0}px`;
    this.queueLabel.style.top = `${beside ? qy - 2 : qy + 13}px`;
    this.queueLabel.textContent = `Queue ${Math.round(this.state.queue)} / ${this.queueSize}`;

    // Packets: emitted in proportion to throughput
    if (!reduceMotion.matches) {
      this.emitAcc += dt * Math.min(40, this.state.throughput * 2.2);
      while (this.emitAcc >= 1) { this.emitAcc -= 1; this.packets.push({ x: xs[0] + pad, lane: (Math.random() - 0.5) }); }
      const queueSlow = 1 + Math.min(2.5, this.state.queue / Math.max(10, this.queueSize) * 4);
      for (const p of this.packets) {
        const inPipe = p.x > px0 && p.x < px1;
        const v = inPipe ? 120 + 70 * frac : 300 / (p.x < px0 ? queueSlow : 1);
        p.x += v * dt;
      }
      this.packets = this.packets.filter((p) => p.x < xs[3] - pad);
      ctx.fillStyle = c.algo;
      for (const p of this.packets) {
        const inPipe = p.x > px0 && p.x < px1;
        const y = cy + (inPipe ? p.lane * (thick - 8) * 0.6 : 0);
        ctx.beginPath(); ctx.arc(p.x, y, 3, 0, Math.PI * 2); ctx.fill();
      }
      // Dropped packets fall away from the full queue
      ctx.fillStyle = c.crit;
      for (const d of this.debris) {
        d.vy += 260 * dt; d.x += d.vx * dt; d.y += d.vy * dt; d.life -= dt * 1.2;
        ctx.globalAlpha = Math.max(0, d.life);
        ctx.beginPath(); ctx.arc(d.x, d.y, 3, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      this.debris = this.debris.filter((d) => d.life > 0);
    }
  }
}
Lane.all = new Set();

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}

let lastFrame = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  if (!document.hidden) {
    replay.advance(dt);
    for (const l of Lane.all) if (l.el.offsetParent) l.tick(dt);
  }
  requestAnimationFrame(frame);
}

/* ================================================================
   App state & navigation
   ================================================================ */
const app = { defaults: null, expId: null, summary: null, metric: "throughput", traces: {}, charts: {}, scatter: null };

function showView(name) {
  for (const v of $$(".view")) {
    const on = v.id === `view-${name}`;
    if (on && v.hidden) { v.hidden = false; v.style.animation = "none"; void v.offsetWidth; v.style.animation = ""; }
    if (!on) v.hidden = true;
  }
  const btns = $$(".rail-items .rail-btn");
  const ind = $(".rail-indicator");
  btns.forEach((b, i) => {
    const on = b.dataset.view === name;
    b.classList.toggle("active", on);
    if (on) { ind.style.opacity = 1; ind.style.transform = `translateY(${b.offsetTop}px)`; }
  });
  app.view = name;
  if (name !== "replay") replay.pause();
  if (name === "library") renderLibrary();
  if (name === "replay") replay.ensure();
  for (const l of Lane.all) l.resize();
}

function enableExperimentViews(on) {
  for (const v of ["results", "replay"]) $(`.rail-btn[data-view=${v}]`).disabled = !on;
}

/* ================================================================
   Run view
   ================================================================ */
const form = $("#runForm");
let previewLane = null;
let previewTimer = null;

function buildForm() {
  const d = app.defaults;
  const toggles = $("#algoToggles");
  toggles.innerHTML = "";
  for (const a of ORDER) {
    const lab = document.createElement("label");
    lab.className = "algo-toggle on";
    lab.style.setProperty("--c", `var(--algo-${a})`);
    lab.innerHTML = `<input type="checkbox" name="algo" value="${a}" checked><span class="check"><i data-lucide="check"></i></span><span>${ALGOS[a].name}<small>${ALGOS[a].note}</small></span>`;
    const input = lab.querySelector("input");
    input.addEventListener("change", () => { lab.classList.toggle("on", input.checked); onFormChange(); });
    toggles.appendChild(lab);
  }
  for (const [k, p] of Object.entries(d.params)) {
    const input = form.elements[k];
    if (!input) continue;
    input.value = p.default; input.min = p.min; input.max = p.max;
  }
  const sel = $("#modelSelect");
  sel.innerHTML = d.models.map((m) => `<option value="${m}">${m.replace("checkpoints_ppo_tcp/", "checkpoint: ").replace(".zip", "")}</option>`).join("")
    || `<option value="">No trained model found</option>`;
  for (const b of $$(".stepper button")) {
    b.addEventListener("click", () => {
      const inp = form.elements.runs;
      inp.value = Math.max(+inp.min, Math.min(+inp.max, (+inp.value || 1) + +b.dataset.step));
      onFormChange();
    });
  }
  form.addEventListener("input", onFormChange);
  icons();
  onFormChange();
}

function readForm() {
  const cfg = { algorithms: $$("input[name=algo]:checked").map((i) => i.value), model: $("#modelSelect").value };
  for (const k of Object.keys(app.defaults.params)) cfg[k] = parseFloat(form.elements[k].value);
  return cfg;
}

function onFormChange() {
  const cfg = readForm();
  const p = app.defaults.params;
  for (const [k, lim] of Object.entries(p)) {
    const v = cfg[k];
    form.elements[k]?.classList.toggle("invalid", isNaN(v) || v < lim.min || v > lim.max);
  }
  const tr = app.defaults.training;
  const outside = Object.keys(tr).some((k) => Math.abs(cfg[k] - tr[k]) > 1e-9);
  $("#trainingHint").hidden = !(outside && cfg.algorithms.includes("rl"));
  $("#modelField").hidden = !cfg.algorithms.includes("rl");
  const n = (cfg.runs || 0) * cfg.algorithms.length;
  const per = 0.07 * (cfg.duration || 0) + (cfg.algorithms.includes("rl") ? 3 : 0);
  const est = Math.max(3, Math.round(per * Math.ceil(n / Math.max(1, (navigator.hardwareConcurrency || 4) / 2)) * 1.3));
  $("#runEstimate").textContent = n ? `${n} simulations, about ${est < 60 ? `${est} s` : `${Math.round(est / 60)} min`}.` : "";
  $("#runBtn").disabled = !cfg.algorithms.length;
  updatePreview(cfg);
}

function updatePreview(cfg) {
  const q = Math.max(10, cfg.queueSize || 60), maxR = Math.max(1, cfg.maxRate || 11.5);
  if (!previewLane || previewLane.queueSize !== q || previewLane.maxRate !== maxR || previewLane.delay !== cfg.bottleneckDelay) {
    previewLane?.destroy();
    previewLane = new Lane($("#previewLanes"), { algo: "rl", title: "Preview", queueSize: q, maxRate: maxR, delay: cfg.bottleneckDelay || 20, stats: false });
  }
  previewLane.cfg = cfg;
}

function previewWander() {
  // Gentle stand-in traffic so the diagram is alive before any data exists
  clearTimeout(previewTimer);
  if (previewLane) {
    const c = previewLane.cfg || {};
    const lo = c.minRate || 3.5, hi = c.maxRate || 11.5;
    const cap = lo + Math.random() * (hi - lo);
    previewLane.set({ capacity: cap, throughput: cap * (0.75 + Math.random() * 0.2), queue: Math.random() * (previewLane.queueSize * 0.25), drops: 0, rtt: 0 });
  }
  const holdLo = (previewLane?.cfg?.minHold || 3.5), holdHi = (previewLane?.cfg?.maxHold || 9);
  previewTimer = setTimeout(previewWander, 1000 * Math.min(4, holdLo + Math.random() * (holdHi - holdLo)) / 1.5);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("#formError");
  err.hidden = true;
  try {
    const snap = await api("/api/experiments", { method: "POST", body: readForm() });
    follow(snap);
  } catch (ex) {
    err.textContent = ex.message;
    err.hidden = false;
  }
});

function setRunning(on) {
  $("#progressBlock").hidden = !on;
  for (const el of form.elements) el.disabled = on;
  $("#runBtn").innerHTML = on ? `<i data-lucide="loader-circle" class="spin"></i><span>Running…</span>` : `<i data-lucide="play"></i><span>Run experiment</span>`;
  $("#previewTitle").textContent = on ? "Simulating" : "The test network";
  $("#previewSub").textContent = on
    ? "Each algorithm is being run on the same sequence of capacity changes. You can leave this page; the run continues."
    : "One TCP flow crosses a bottleneck whose speed jumps at random. Every algorithm sees the exact same jumps.";
  icons();
  if (on) $(".spin", $("#runBtn"))?.classList.add("spin");
}

function follow(snap) {
  showView("run");
  setRunning(true);
  const chips = $("#runChips");
  chips.innerHTML = "";
  for (const r of snap.runs) {
    const li = document.createElement("li");
    li.className = "chip";
    li.style.setProperty("--c", `var(--algo-${r.algo})`);
    li.innerHTML = `<span class="dot"></span><span>${ALGOS[r.algo].short} · run ${r.seed - snap.config.seed + 1}</span><span class="st"></span>`;
    chips.appendChild(li);
  }
  render(snap);
  const es = new EventSource(`/api/experiments/${snap.id}/events`);
  $("#cancelBtn").onclick = () => api(`/api/experiments/${snap.id}/cancel`, { method: "POST" }).catch(() => {});
  es.onmessage = (ev) => {
    const s = JSON.parse(ev.data);
    render(s);
    if (s.status !== "running") { es.close(); finish(s); }
  };
  es.onerror = () => { es.close(); setTimeout(() => api("/api/defaults").then((d) => { if (!d.active) finish({ ...snap, status: "unknown" }); }), 800); };

  function render(s) {
    const chipsEls = $$("#runChips .chip");
    let total = 0;
    s.runs.forEach((r, i) => {
      const li = chipsEls[i];
      total += r.progress;
      li.style.setProperty("--p", r.progress);
      li.classList.toggle("done", r.status === "done");
      li.classList.toggle("failed", r.status === "failed");
      const st = r.status === "done" ? "check" : r.status === "failed" ? "x" : r.status === "running" ? "loader-circle" : "";
      const holder = li.querySelector(".st");
      if (holder.dataset.st !== st) {
        holder.dataset.st = st;
        holder.innerHTML = st ? `<i data-lucide="${st}"></i>` : "";
        if (r.status === "failed") li.title = r.error || "Failed";
        icons();
        if (st === "loader-circle") holder.querySelector("svg")?.classList.add("spin");
      }
    });
    const frac = total / s.runs.length;
    $("#progressFill").style.width = `${(frac * 100).toFixed(1)}%`;
    const done = s.runs.filter((r) => r.status === "done").length;
    $("#progressLabel").textContent = `${done} of ${s.runs.length} simulations finished`;
    const eta = frac > 0.04 ? s.elapsed * (1 - frac) / frac : null;
    $("#progressEta").textContent = eta == null ? "" : eta < 1.5 ? "almost done" : `about ${Math.ceil(eta)} s left`;
    // Pulse the preview with real progress
    if (previewLane) previewLane.set({ ...previewLane.state, throughput: Math.max(2, previewLane.state.throughput) });
  }
}

async function finish(s) {
  setRunning(false);
  if (s.status === "done") {
    toast("Experiment finished");
    await openExperiment(s.id, "results");
  } else if (s.status === "cancelled") {
    toast("Experiment cancelled", "circle-slash");
  } else {
    const err = $("#formError");
    err.textContent = s.error || "The experiment stopped unexpectedly. Check the terminal running the dashboard.";
    err.hidden = false;
  }
}

/* ================================================================
   Results view
   ================================================================ */
async function openExperiment(id, view = "results") {
  const summary = await api(`/api/experiments/${id}`);
  app.expId = id;
  app.summary = summary;
  app.traces = {};
  enableExperimentViews(true);
  renderResults();
  replay.reset();
  showView(view);
}

function algosOf(s) { return ORDER.filter((a) => s.algorithms[a]); }

function renderResults() {
  const s = app.summary;
  $("#expName").value = s.name;
  const c = s.config;
  const date = new Date(s.created * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  $("#expMeta").textContent = `${c.runs} run${c.runs > 1 ? "s" : ""} of ${c.duration} s, capacity ${c.minRate}–${c.maxRate} Mbps, ${c.queueSize}-packet queue, ${c.bottleneckDelay} ms link. ${date}.`;
  syncSaveBtn();

  segmented($("#metricSeg"), METRICS.map((m) => ({ key: m.key, label: m.label })), app.metric, (k) => { app.metric = k; renderBars(); });
  renderBars();
  renderScatter();
  renderTable();
}

function bestAlgo(metric) {
  const s = app.summary, m = METRICS.find((x) => x.key === metric);
  const algos = algosOf(s);
  if (algos.length < 2) return null;
  return algos.reduce((best, a) => {
    const v = s.algorithms[a][metric].mean, bv = s.algorithms[best][metric].mean;
    return (m.better === "high" ? v > bv : v < bv) ? a : best;
  });
}

function renderBars() {
  const s = app.summary, m = METRICS.find((x) => x.key === app.metric);
  const algos = algosOf(s);
  $("#metricNote").textContent = `${m.note} ${m.better === "high" ? "Higher" : "Lower"} is better.`;
  const max = Math.max(...algos.map((a) => s.algorithms[a][m.key].mean + s.algorithms[a][m.key].std), 1e-9) * 1.08;
  const best = bestAlgo(m.key);
  const root = $("#metricBars");
  if (root.children.length !== algos.length) {
    root.innerHTML = algos.map((a) => `
      <div class="bar-row" style="--c: var(--algo-${a})" data-a="${a}">
        <span class="name"><span class="dot"></span>${ALGOS[a].short}</span>
        <div class="bar-track"><div class="bar-fill"></div><div class="bar-err"></div></div>
        <span class="val"><span class="v">0</span> <small class="u"></small><span class="b"></span></span>
      </div>`).join("");
  }
  for (const row of $$(".bar-row", root)) {
    const a = row.dataset.a, { mean, std } = s.algorithms[a][m.key];
    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = `${(mean / max) * 100}%`;
      const err = row.querySelector(".bar-err");
      err.style.left = `${(Math.max(0, mean - std) / max) * 100}%`;
      err.style.width = `${(Math.min(2 * std, mean + std) / max) * 100}%`;
      err.style.opacity = std > 0 ? 1 : 0;
    });
    countUp(row.querySelector(".v"), mean, m.digits);
    row.querySelector(".u").textContent = m.unit;
    row.querySelector(".b").innerHTML = a === best ? `<span class="best"><i data-lucide="award"></i>Best</span>` : "";
  }
  icons();
}

function chartTheme() {
  return { ink: cssVar("--ink"), ink2: cssVar("--ink-2"), ink3: cssVar("--ink-3"), grid: cssVar("--line"), panel: cssVar("--panel"), ref: cssVar("--ref") };
}

function baseChartOpts(t) {
  return {
    responsive: true, maintainAspectRatio: false,
    animation: reduceMotion.matches ? false : { duration: 500, easing: "easeOutCubic" },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: t.ink, titleColor: t.panel, bodyColor: t.panel, padding: 10, cornerRadius: 8,
        boxPadding: 4, usePointStyle: true, titleFont: { weight: "600" },
      },
    },
    scales: {
      x: { grid: { color: t.grid, drawTicks: false }, border: { display: false }, ticks: { color: t.ink3, padding: 6 } },
      y: { grid: { color: t.grid, drawTicks: false }, border: { display: false }, ticks: { color: t.ink3, padding: 8 } },
    },
  };
}

// Draws each algorithm's name beside its mean marker (direct labels)
const meanLabels = {
  id: "meanLabels",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    ctx.save();
    ctx.font = "600 12px Geist, sans-serif";
    ctx.fillStyle = cssVar("--ink");
    ctx.textBaseline = "middle";
    // Greedy placement: try right, left, above, below; skip spots that collide
    const placed = [];
    const means = chart.data.datasets.map((ds, i) => ({ ds, i, pt: chart.getDatasetMeta(i).data[0] }))
      .filter(({ ds, i, pt }) => ds.isMean && pt && chart.isDatasetVisible(i));
    const marks = means.map(({ pt }) => ({ x0: pt.x - 10, x1: pt.x + 10, y0: pt.y - 10, y1: pt.y + 10 }));
    const hit = (b) => [...placed, ...marks].some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
    const area = chart.chartArea;
    for (const { ds, pt } of means) {
      const text = ds.label.replace(" mean", "");
      const w = ctx.measureText(text).width, h = 14;
      const spots = [[pt.x + 13, pt.y], [pt.x - 13 - w, pt.y], [pt.x - w / 2, pt.y - 20], [pt.x - w / 2, pt.y + 20]];
      let best = spots[0];
      for (const [x, y] of spots) {
        const b = { x0: x - 2, x1: x + w + 2, y0: y - h / 2, y1: y + h / 2 };
        if (b.x0 >= area.left && b.x1 <= area.right && b.y0 >= area.top && b.y1 <= area.bottom && !hit(b)) { best = [x, y]; break; }
      }
      placed.push({ x0: best[0] - 2, x1: best[0] + w + 2, y0: best[1] - h / 2, y1: best[1] + h / 2 });
      ctx.fillText(text, best[0], best[1]);
    }
    ctx.restore();
  },
};

function renderScatter() {
  const s = app.summary, t = chartTheme();
  const algos = algosOf(s);
  const datasets = [];
  for (const a of algos) {
    const color = algoColor(a);
    datasets.push({
      label: ALGOS[a].short, data: s.runs.filter((r) => r.algo === a).map((r) => ({ x: r.rtt, y: r.throughput, seed: r.seed })),
      pointStyle: ALGOS[a].shape, backgroundColor: color + "88", borderColor: t.panel, borderWidth: 1.5,
      pointRadius: 6, pointHoverRadius: 8,
    });
    datasets.push({
      label: `${ALGOS[a].short} mean`, isMean: true,
      data: [{ x: s.algorithms[a].rtt.mean, y: s.algorithms[a].throughput.mean }],
      pointStyle: ALGOS[a].shape, backgroundColor: color, borderColor: t.panel, borderWidth: 2,
      pointRadius: 9, pointHoverRadius: 10,
    });
  }
  const opts = baseChartOpts(t);
  opts.scales.x.title = { display: true, text: "Average delay (ms)", color: t.ink2 };
  opts.scales.y.title = { display: true, text: "Throughput (Mbps)", color: t.ink2 };
  opts.plugins.tooltip.callbacks = {
    label: (c) => c.dataset.isMean
      ? `${c.dataset.label}: ${fmt(c.parsed.y, 2)} Mbps at ${fmt(c.parsed.x, 1)} ms`
      : `${c.dataset.label}, seed ${c.raw.seed}: ${fmt(c.parsed.y, 2)} Mbps at ${fmt(c.parsed.x, 1)} ms`,
  };
  app.scatter?.destroy();
  app.scatter = new Chart($("#scatter"), { type: "scatter", data: { datasets }, options: opts, plugins: [meanLabels] });

  $("#scatterLegend").innerHTML = algos.map((a) => `<span style="--c: var(--algo-${a})">${shapeIcon(a)}${ALGOS[a].name}</span>`).join("");
}

function shapeIcon(a) {
  const c = `var(--algo-${a})`;
  const shapes = {
    circle: `<circle cx="6" cy="6" r="5" fill="${c}"/>`,
    triangle: `<path d="M6 1 11 11H1Z" fill="${c}"/>`,
    rect: `<rect x="1.5" y="1.5" width="9" height="9" fill="${c}"/>`,
    rectRot: `<path d="M6 0.5 11.5 6 6 11.5 0.5 6Z" fill="${c}"/>`,
  };
  return `<svg viewBox="0 0 12 12" aria-hidden="true">${shapes[ALGOS[a].shape]}</svg>`;
}

function renderTable() {
  const s = app.summary, algos = algosOf(s);
  const head = `<thead><tr><th>Algorithm</th>${METRICS.map((m) => `<th>${m.label} <small>${m.unit}</small></th>`).join("")}</tr></thead>`;
  const best = Object.fromEntries(METRICS.map((m) => [m.key, bestAlgo(m.key)]));
  const rows = algos.map((a) => `<tr><td><span class="name" style="--c: var(--algo-${a})"><span class="dot"></span>${ALGOS[a].name}</span></td>${
    METRICS.map((m) => {
      const { mean, std } = s.algorithms[a][m.key];
      return `<td class="${best[m.key] === a ? "top" : ""}">${fmt(mean, m.digits)}${std > 0 ? ` <small>± ${fmt(std, m.digits)}</small>` : ""}</td>`;
    }).join("")}</tr>`).join("");
  $("#summaryTable").innerHTML = head + `<tbody>${rows}</tbody>`;
}

function syncSaveBtn() {
  const btn = $("#saveBtn");
  const saved = app.summary?.saved && $("#expName").value.trim() === app.summary.name;
  btn.disabled = !!saved;
  btn.innerHTML = saved ? `<i data-lucide="bookmark-check"></i><span>Saved</span>` : `<i data-lucide="bookmark"></i><span>${app.summary?.saved ? "Save name" : "Save experiment"}</span>`;
  icons();
}

$("#expName").addEventListener("input", syncSaveBtn);
$("#expName").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#saveBtn").click(); });
$("#saveBtn").addEventListener("click", async () => {
  try {
    app.summary = await api(`/api/experiments/${app.expId}/save`, { method: "POST", body: { name: $("#expName").value } });
    syncSaveBtn();
    toast("Saved to library", "bookmark-check");
  } catch (ex) { toast(ex.message, "triangle-alert"); }
});
$("#toReplayBtn").addEventListener("click", () => showView("replay"));

/* ================================================================
   Replay view
   ================================================================ */
const SPEEDS = [0.5, 1, 2, 4, 8];
const replay = {
  lanes: [], data: null, idx: 0, playing: false, speed: 2, seed: null, baseline: null,

  reset() { this.data = null; this.seed = null; this.baseline = null; this.pause(); },

  async ensure() {
    const s = app.summary;
    if (!s) return;
    const algos = algosOf(s);
    const seeds = [...new Set(s.runs.map((r) => r.seed))].sort((a, b) => a - b);
    const others = algos.filter((a) => a !== "rl");
    if (this.seed == null) {
      this.seed = seeds[0];
      this.baseline = others[0] || null;
      $("#seedSelect").innerHTML = seeds.map((sd, i) => `<option value="${sd}">Run ${i + 1} (seed ${sd})</option>`).join("");
      const primaryIsRl = algos.includes("rl");
      $("#baselineSelect").innerHTML = (primaryIsRl ? others : algos.slice(1)).map((a) => `<option value="${a}">${ALGOS[a].name}</option>`).join("");
      $("#baselineSelect").closest(".field").hidden = algos.length < 2;
      $("#baselineSelect").previousElementSibling.textContent = primaryIsRl ? "Compare PPO with" : `Compare ${ALGOS[algos[0]].short} with`;
      if (!primaryIsRl) this.baseline = algos[1] || null;
      await this.load();
    }
  },

  primary() { const algos = algosOf(app.summary); return algos.includes("rl") ? "rl" : algos[0]; },

  async load() {
    const key = `${app.expId}:${this.seed}`;
    if (!app.traces[key]) app.traces[key] = await api(`/api/experiments/${app.expId}/traces/${this.seed}`);
    this.data = app.traces[key];
    this.buildLanes();
    this.buildCharts();
    const n = this.data[this.primary()].t.length;
    const sc = $("#scrubber");
    sc.max = n - 1;
    $("#timeEnd").textContent = `${fmt(this.data[this.primary()].t[n - 1], 1)} s`;
    this.idx = Math.min(this.idx, n - 1);
    const c = app.summary.config;
    $("#replayTitle").textContent = app.summary.name;
    $("#replaySub").textContent = `Same bottleneck for both: capacity ${c.minRate}–${c.maxRate} Mbps, ${c.queueSize}-packet queue.`;
    this.render(true);
  },

  shown() { return [this.primary(), this.baseline].filter((a) => a && this.data?.[a]); },

  buildLanes() {
    for (const l of this.lanes) l.destroy();
    const c = app.summary.config;
    this.lanes = this.shown().map((a) => new Lane($("#replayLanes"), {
      algo: a, title: ALGOS[a].name, queueSize: c.queueSize, maxRate: c.maxRate, delay: c.bottleneckDelay,
    }));
  },

  buildCharts() {
    const t = chartTheme(), algos = this.shown(), p = this.data[this.primary()];
    const xy = (arr, ts) => arr.map((v, i) => ({ x: ts[i], y: v }));
    const smooth = (arr, w = 5) => arr.map((_, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - w + 1); j <= i; j++) { s += arr[j]; n++; } return s / n; });
    const line = (a, data) => ({ label: ALGOS[a].short, data, borderColor: algoColor(a), borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.25 });
    const ref = (label, data, dash = [5, 4], stepped = false) => ({ label, data, borderColor: t.ref, borderWidth: 1.5, borderDash: dash, pointRadius: 0, stepped, pointHoverRadius: 0 });
    const baseRtt = 2 * (app.summary.config.bottleneckDelay + 1);
    const specs = {
      cCwnd: [...algos.map((a) => line(a, xy(this.data[a].cwnd, this.data[a].t))), ref("Ideal window", xy(p.bdp, p.t), [5, 4], true)],
      cRtt: [...algos.map((a) => line(a, xy(this.data[a].rtt, this.data[a].t))), ref("Base delay", p.t.map((x) => ({ x, y: baseRtt })), [2, 3])],
      cThr: [...algos.map((a) => line(a, xy(smooth(this.data[a].throughput), this.data[a].t))), ref("Capacity", xy(p.capacity, p.t), [5, 4], true)],
    };
    for (const [id, datasets] of Object.entries(specs)) {
      const opts = baseChartOpts(t);
      opts.interaction = { mode: "index", intersect: false };
      opts.scales.x.type = "linear";
      opts.scales.x.min = 0; opts.scales.x.max = Math.ceil(p.t[p.t.length - 1]);
      opts.scales.x.ticks.callback = (v) => `${v}s`;
      opts.scales.y.beginAtZero = true;
      opts.plugins.tooltip.callbacks = { title: (items) => `${fmt(items[0].parsed.x, 1)} s`, label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 1)}` };
      opts.onClick = (_, __, chart) => {};
      this.charts?.[id]?.destroy();
      this.charts = this.charts || {};
      this.charts[id] = new Chart($(`#${id}`), { type: "line", data: { datasets }, options: opts });
    }
    $("#replayLegend").innerHTML = algos.map((a) => `<span style="--c: var(--algo-${a})"><i></i>${ALGOS[a].name}</span>`).join("")
      + `<span style="--c: var(--ref)"><i class="dash"></i>Reference: ideal window, base delay, capacity</span>`;
    // Click a chart to jump there
    for (const id of Object.keys(specs)) {
      const ch = this.charts[id];
      ch.canvas.onclick = (e) => {
        const r = ch.canvas.getBoundingClientRect();
        const x = ch.scales.x.getValueForPixel(e.clientX - r.left);
        this.seek(Math.round((x - p.t[0]) / 0.1));
      };
    }
  },

  seek(i) {
    const n = this.data?.[this.primary()]?.t.length || 0;
    this.idx = Math.max(0, Math.min(n - 1, i));
    for (const l of this.lanes) l.resetDrops();
    this.render(true);
  },

  render(jump = false) {
    if (!this.data) return;
    const i = Math.floor(this.idx);
    const p = this.data[this.primary()];
    for (const l of this.lanes) {
      const d = this.data[l.algo];
      const j = Math.min(i, d.t.length - 1);
      const from = Math.max(0, j - 4);
      let thr = 0; for (let k = from; k <= j; k++) thr += d.throughput[k]; thr /= (j - from + 1);
      l.set({ capacity: d.capacity[j], throughput: thr, queue: d.queue[j], drops: d.drops[j], rtt: d.rtt[j], cwnd: d.cwnd[j] });
      if (jump) l.lastDrops = d.drops[j];
    }
    const sc = $("#scrubber");
    sc.value = i;
    sc.style.setProperty("--pct", `${(i / Math.max(1, sc.max)) * 100}%`);
    $("#timeNow").textContent = `${fmt(p.t[i], 1)} s`;
    for (const [id, ch] of Object.entries(this.charts || {})) {
      const head = $(`#${id}`).parentElement.querySelector(".playhead");
      const x = ch.scales.x.getPixelForValue(p.t[i]);
      head.style.transform = `translateX(${x}px)`;
      head.style.top = `${ch.chartArea.top}px`;
      head.style.height = `${ch.chartArea.bottom - ch.chartArea.top}px`;
    }
  },

  advance(dt) {
    if (!this.playing || !this.data) return;
    const n = this.data[this.primary()].t.length;
    const prev = Math.floor(this.idx);
    this.idx += dt * 10 * this.speed;
    if (this.idx >= n - 1) { this.idx = n - 1; this.pause(); }
    if (Math.floor(this.idx) !== prev) this.render();
  },

  play() {
    if (!this.data) return;
    const n = this.data[this.primary()].t.length;
    if (this.idx >= n - 1) this.seek(0);
    this.playing = true; this.syncBtn();
  },
  pause() { this.playing = false; this.syncBtn(); },
  toggle() { this.playing ? this.pause() : this.play(); },
  syncBtn() {
    const b = $("#playBtn");
    b.innerHTML = `<i data-lucide="${this.playing ? "pause" : "play"}"></i>`;
    b.setAttribute("aria-label", this.playing ? "Pause" : "Play");
    icons();
  },
};

$("#playBtn").addEventListener("click", () => replay.toggle());
$("#scrubber").addEventListener("input", (e) => { replay.pause(); replay.seek(+e.target.value); });
$("#seedSelect").addEventListener("change", (e) => { replay.seed = +e.target.value; replay.load(); });
$("#baselineSelect").addEventListener("change", (e) => { replay.baseline = e.target.value; replay.load(); });
segmented($("#speedSeg"), SPEEDS.map((s) => ({ key: String(s), label: `${s}×` })), String(replay.speed), (k) => { replay.speed = +k; });
document.addEventListener("keydown", (e) => {
  if (app.view !== "replay" || ["INPUT", "SELECT"].includes(document.activeElement?.tagName) && document.activeElement.type !== "range") return;
  if (e.code === "Space") { e.preventDefault(); replay.toggle(); }
  if (e.key === "ArrowRight") { e.preventDefault(); replay.seek(Math.floor(replay.idx) + 10); }
  if (e.key === "ArrowLeft") { e.preventDefault(); replay.seek(Math.floor(replay.idx) - 10); }
});
window.addEventListener("resize", () => replay.render(true));

/* ================================================================
   Library view
   ================================================================ */
async function renderLibrary() {
  const root = $("#libList");
  let items;
  try { items = await api("/api/experiments"); } catch (ex) { root.innerHTML = `<div class="empty">${ex.message}</div>`; return; }
  if (!items.length) {
    root.innerHTML = `<div class="empty"><i data-lucide="library"></i><p>No saved experiments yet. Run one, then press <b>Save experiment</b> on its results.</p><button class="btn primary" id="libRun"><i data-lucide="flask-conical"></i><span>New experiment</span></button></div>`;
    $("#libRun").onclick = () => showView("run");
    icons();
    return;
  }
  root.innerHTML = "";
  items.forEach((it, n) => {
    const row = document.createElement("div");
    row.className = "lib-row";
    row.style.animationDelay = `${Math.min(n, 8) * 30}ms`;
    const date = new Date(it.created * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
    row.innerHTML = `
      <div>
        <div class="t"></div>
        <div class="meta"><span><i data-lucide="calendar"></i>${date}</span><span><i data-lucide="repeat"></i>${it.runs} run${it.runs > 1 ? "s" : ""}</span><span><i data-lucide="timer"></i>${it.duration} s each</span></div>
      </div>
      <div class="algo-dots">${ORDER.filter((a) => it.algorithms.includes(a)).map((a) => `<span style="--c: var(--algo-${a})"><i></i>${ALGOS[a].short}</span>`).join("")}</div>
      <div class="lib-actions">
        <button class="btn" data-act="open"><i data-lucide="chart-column"></i><span>Results</span></button>
        <button class="btn" data-act="replay"><i data-lucide="circle-play"></i><span>Replay</span></button>
        <button class="btn ghost" data-act="rename" aria-label="Rename"><i data-lucide="pencil"></i></button>
        <button class="btn ghost danger" data-act="delete" aria-label="Delete"><i data-lucide="trash-2"></i></button>
      </div>`;
    row.querySelector(".t").textContent = it.name;
    row.querySelector("[data-act=open]").onclick = () => openExperiment(it.id, "results");
    row.querySelector("[data-act=replay]").onclick = () => openExperiment(it.id, "replay");
    row.querySelector("[data-act=rename]").onclick = () => {
      const t = row.querySelector(".t");
      const input = document.createElement("input");
      input.className = "lib-rename"; input.value = it.name;
      t.replaceChildren(input); input.focus(); input.select();
      const commit = async () => {
        const name = input.value.trim();
        if (name && name !== it.name) {
          await api(`/api/experiments/${it.id}/save`, { method: "POST", body: { name } });
          it.name = name;
          if (app.expId === it.id && app.summary) { app.summary.name = name; $("#expName").value = name; syncSaveBtn(); }
          toast("Renamed");
        }
        t.textContent = it.name;
      };
      input.addEventListener("blur", commit, { once: true });
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") input.blur(); if (e.key === "Escape") { input.value = it.name; input.blur(); } });
    };
    const del = row.querySelector("[data-act=delete]");
    del.onclick = async () => {
      if (!del.classList.contains("armed")) {
        del.classList.add("armed");
        del.innerHTML = `<i data-lucide="trash-2"></i><span>Delete?</span>`;
        icons();
        setTimeout(() => { if (del.isConnected) { del.classList.remove("armed"); del.innerHTML = `<i data-lucide="trash-2"></i>`; icons(); } }, 3000);
        return;
      }
      await api(`/api/experiments/${it.id}`, { method: "DELETE" });
      row.style.transition = "opacity 160ms, transform 160ms";
      row.style.opacity = 0; row.style.transform = "translateX(12px)";
      setTimeout(() => { row.remove(); if (!root.children.length) renderLibrary(); }, 170);
      if (app.expId === it.id) { app.summary.saved = false; enableExperimentViews(false); app.expId = null; }
      toast("Deleted", "trash-2");
    };
    root.appendChild(row);
  });
  icons();
}

/* ================================================================
   Theme
   ================================================================ */
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  try { t ? localStorage.setItem("theme", t) : localStorage.removeItem("theme"); } catch {}
  for (const l of Lane.all) l.refreshColors();
  if (app.summary) {
    renderScatter();
    if (replay.data) { replay.buildCharts(); replay.render(true); }
  }
}
$("#themeToggle").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(dark ? "light" : "dark");
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(document.documentElement.dataset.theme));

/* ================================================================
   Boot
   ================================================================ */
for (const b of $$(".rail-items .rail-btn")) b.addEventListener("click", () => !b.disabled && showView(b.dataset.view));

(async function boot() {
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch {}
  Chart.defaults.font.family = "Geist, ui-sans-serif, system-ui, sans-serif";
  Chart.defaults.font.size = 12;
  icons();
  try {
    app.defaults = await api("/api/defaults");
  } catch (ex) {
    $("#formError").textContent = "Can't reach the dashboard server. Start it with dashboard/run_dashboard.sh.";
    $("#formError").hidden = false;
    return;
  }
  buildForm();
  showView("run");
  previewWander();
  requestAnimationFrame(frame);
  if (!app.defaults.simReady) {
    $("#formError").textContent = "The ns-3 simulator isn't built yet. Run dashboard/build_sim.sh, then reload.";
    $("#formError").hidden = false;
  }
  if (app.defaults.active) {
    const snap = await api(`/api/experiments/${app.defaults.active}/status`).catch(() => null);
    if (snap) follow(snap);
  }
})();
