const DATASETS = {
  seg_023: {
    wav: "example_data/seg_023.wav",
    predCsv: "example_data/seg_023_pred.csv",
  },
  seg_006: {
    wav: "example_data/seg_006.wav",
    predCsv: "example_data/seg_006_pred.csv",
  },
};

const audio = document.getElementById("audio");
const sourceEl = audio.querySelector("source");
const plainTranscript = document.getElementById("plainTranscript");
const prominentTranscript = document.getElementById("prominentTranscript");
const timeline = document.getElementById("timeline");
const viewPlain = document.getElementById("viewPlain");
const viewProm = document.getElementById("viewProm");
const exampleSelect = document.getElementById("exampleSelect");
const chart = document.getElementById("promChart");

let words = [];
let wordEls = [];
let timelineEls = [];
let pointEls = [];

function bucket(prom) {
  if (prom >= 0.7) return "high";
  if (prom >= 0.4) return "mid";
  return "low";
}

function parseCsvLine(line) {
  const out = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  out.push(current);
  return out;
}

function parsePredCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]).map((h) => h.trim().toLowerCase());

  const idxStart = headers.indexOf("start");
  const idxEnd = headers.indexOf("end");
  const idxLabel = headers.indexOf("label");
  const idxPred = headers.indexOf("pred");

  if (idxStart < 0 || idxEnd < 0 || idxLabel < 0 || idxPred < 0) return [];

  return lines.slice(1).map((line) => {
    const cols = parseCsvLine(line);
    return {
      start: Number(cols[idxStart]),
      end: Number(cols[idxEnd]),
      text: String(cols[idxLabel] || "").trim(),
      prom: Number(cols[idxPred]),
    };
  }).filter((w) => Number.isFinite(w.start) && Number.isFinite(w.end) && Number.isFinite(w.prom));
}

function renderTranscripts() {
  wordEls = [];
  timelineEls = [];

  plainTranscript.textContent = words.map((w) => w.text).join(" ");
  prominentTranscript.innerHTML = "";
  timeline.innerHTML = "";

  words.forEach((w, i) => {
    const b = bucket(w.prom);

    const span = document.createElement("span");
    span.className = `word ${b}`;
    span.dataset.i = String(i);
    span.textContent = w.text;
    span.title = `Prominence: ${w.prom.toFixed(2)} | ${w.start.toFixed(2)}-${w.end.toFixed(2)}s`;
    prominentTranscript.appendChild(span);
    wordEls.push(span);

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `timeline-chip ${b}`;
    chip.dataset.i = String(i);
    chip.textContent = `${w.text} (${w.prom.toFixed(2)})`;
    chip.addEventListener("click", () => {
      audio.currentTime = w.start;
      audio.play();
    });
    timeline.appendChild(chip);
    timelineEls.push(chip);
  });
}

function renderChart() {
  pointEls = [];
  chart.innerHTML = "";
  if (!words.length) return;

  const w = 900;
  const h = 260;
  const m = { l: 44, r: 16, t: 12, b: 28 };
  const innerW = w - m.l - m.r;
  const innerH = h - m.t - m.b;

  const tMin = 0;
  const tMax = Math.max(...words.map((x) => x.end));
  const pMax = Math.max(1.2, ...words.map((x) => x.prom));

  const x = (t) => m.l + (Math.max(0, Math.min(tMax, t)) / tMax) * innerW;
  const y = (p) => m.t + (1 - (Math.max(0, Math.min(pMax, p)) / pMax)) * innerH;

  for (let i = 0; i <= 4; i += 1) {
    const gy = m.t + (i / 4) * innerH;
    const gl = document.createElementNS("http://www.w3.org/2000/svg", "line");
    gl.setAttribute("x1", String(m.l));
    gl.setAttribute("x2", String(w - m.r));
    gl.setAttribute("y1", String(gy));
    gl.setAttribute("y2", String(gy));
    gl.setAttribute("class", "grid-line");
    chart.appendChild(gl);
  }

  const xAxis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  xAxis.setAttribute("x1", String(m.l));
  xAxis.setAttribute("x2", String(w - m.r));
  xAxis.setAttribute("y1", String(h - m.b));
  xAxis.setAttribute("y2", String(h - m.b));
  xAxis.setAttribute("class", "axis");
  chart.appendChild(xAxis);

  const yAxis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  yAxis.setAttribute("x1", String(m.l));
  yAxis.setAttribute("x2", String(m.l));
  yAxis.setAttribute("y1", String(m.t));
  yAxis.setAttribute("y2", String(h - m.b));
  yAxis.setAttribute("class", "axis");
  chart.appendChild(yAxis);

  const points = words.map((d) => {
    const cx = x((d.start + d.end) / 2);
    const cy = y(d.prom);
    return `${cx},${cy}`;
  });

  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  poly.setAttribute("points", points.join(" "));
  poly.setAttribute("class", "curve");
  chart.appendChild(poly);

  words.forEach((d, i) => {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", String(x((d.start + d.end) / 2)));
    c.setAttribute("cy", String(y(d.prom)));
    c.setAttribute("r", "3.2");
    c.setAttribute("class", "pt");
    c.setAttribute("data-i", String(i));
    c.appendChild(document.createTitle());
    c.querySelector("title").textContent = `${d.text}: ${d.prom.toFixed(2)}`;
    chart.appendChild(c);
    pointEls.push(c);
  });
}

function updateActiveWord(currentTime) {
  let active = -1;
  for (let i = 0; i < words.length; i += 1) {
    if (currentTime >= words[i].start && currentTime <= words[i].end) {
      active = i;
      break;
    }
  }

  wordEls.forEach((el, i) => el.classList.toggle("active", i === active));
  timelineEls.forEach((el, i) => el.classList.toggle("active", i === active));
  pointEls.forEach((el, i) => el.classList.toggle("active", i === active));
}

function setMode(mode) {
  const showProm = mode === "prom";
  prominentTranscript.style.display = showProm ? "flex" : "none";
  timeline.parentElement.style.display = showProm ? "block" : "none";
  viewPlain.classList.toggle("chip-active", !showProm);
  viewProm.classList.toggle("chip-active", showProm);
}

async function loadExample(key) {
  const cfg = DATASETS[key];
  if (!cfg) return;

  sourceEl.src = cfg.wav;
  audio.load();

  const res = await fetch(cfg.predCsv);
  if (!res.ok) throw new Error(`Failed to load ${cfg.predCsv}`);
  const text = await res.text();
  words = parsePredCsv(text);

  renderTranscripts();
  renderChart();
  updateActiveWord(audio.currentTime);
}

audio.addEventListener("timeupdate", () => updateActiveWord(audio.currentTime));
viewPlain.addEventListener("click", () => setMode("plain"));
viewProm.addEventListener("click", () => setMode("prom"));
exampleSelect.addEventListener("change", async (e) => {
  await loadExample(e.target.value);
});

setMode("prom");
loadExample(exampleSelect.value).catch((err) => {
  plainTranscript.textContent = `Could not load demo data: ${err.message}`;
});
