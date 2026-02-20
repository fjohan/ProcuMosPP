const words = [
  { start: 0.0, end: 0.601, text: "Forskare", prom: 0.86 },
  { start: 0.681, end: 0.801, text: "vid", prom: 0.22 },
  { start: 0.861, end: 1.341, text: "Sveriges", prom: 0.58 },
  { start: 1.441, end: 2.783, text: "lantbruksuniversitet", prom: 0.93 },
  { start: 2.803, end: 2.883, text: "har", prom: 0.28 },
  { start: 2.923, end: 3.123, text: "tagit", prom: 0.44 },
  { start: 3.183, end: 3.363, text: "fram", prom: 0.51 },
  { start: 3.403, end: 3.524, text: "ett", prom: 0.18 },
  { start: 3.604, end: 3.824, text: "nytt", prom: 0.62 },
  { start: 3.844, end: 3.944, text: "och", prom: 0.21 },
  { start: 3.984, end: 4.364, text: "hallbart", prom: 0.69 },
  { start: 4.445, end: 5.165, text: "fiskfoder", prom: 0.83 },
  { start: 5.205, end: 5.666, text: "rapporterar", prom: 0.56 },
  { start: 5.706, end: 6.827, text: "Vetenskapsradion", prom: 0.97 }
];

const audio = document.getElementById("audio");
const plainTranscript = document.getElementById("plainTranscript");
const prominentTranscript = document.getElementById("prominentTranscript");
const timeline = document.getElementById("timeline");
const viewPlain = document.getElementById("viewPlain");
const viewProm = document.getElementById("viewProm");

const wordEls = [];
const timelineEls = [];

function bucket(prom) {
  if (prom >= 0.7) return "high";
  if (prom >= 0.4) return "mid";
  return "low";
}

function render() {
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
}

function setMode(mode) {
  const showProm = mode === "prom";
  prominentTranscript.style.display = showProm ? "flex" : "none";
  timeline.parentElement.style.display = showProm ? "block" : "none";
  viewPlain.classList.toggle("chip-active", !showProm);
  viewProm.classList.toggle("chip-active", showProm);
}

audio.addEventListener("timeupdate", () => updateActiveWord(audio.currentTime));
viewPlain.addEventListener("click", () => setMode("plain"));
viewProm.addEventListener("click", () => setMode("prom"));

render();
setMode("prom");
