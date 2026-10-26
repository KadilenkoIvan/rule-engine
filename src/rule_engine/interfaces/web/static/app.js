// HTML собирает сервер. Здесь только запросы, вставка ответа и переключение кадров прогона.

const $ = (id) => document.getElementById(id);

async function post(url, fields) {
  const response = await fetch(url, { method: "POST", body: new URLSearchParams(fields) });
  return response.text(); // ошибка правил приходит тоже HTML, со статусом 422
}

function debounce(action, delay) {
  let timer;
  return () => {
    clearTimeout(timer);
    timer = setTimeout(action, delay);
  };
}

function showTab(name) {
  for (const tab of document.querySelectorAll("[data-tab]")) {
    const on = tab.dataset.tab === name;
    tab.classList.toggle("on", on);
    $(tab.dataset.tab).hidden = !on;
  }
}

async function loadExample(select, target) {
  if (!select.value) return;
  const response = await fetch(`/example?name=${encodeURIComponent(select.value)}`);
  target.value = await response.text();
}

const refreshGraph = debounce(async () => {
  $("graph").innerHTML = await post("/graph", { rules: $("rules").value });
}, 400);

// кадры прогона уже на странице: ползунок только показывает нужный
function setupPlayer(root) {
  const frames = root.querySelectorAll(".frame");
  const slider = root.querySelector(".player input[type=range]");
  if (!slider) return;
  const position = root.querySelector(".player .position");
  const show = (index) => {
    const current = Math.max(0, Math.min(frames.length - 1, index));
    slider.value = current;
    frames.forEach((frame, i) => { frame.hidden = i !== current; });
    position.textContent = `кадр ${current + 1} из ${frames.length}`;
    for (const item of root.querySelectorAll("[data-frame]")) {
      item.classList.toggle("current", Number(item.dataset.frame) === current);
    }
  };
  slider.addEventListener("input", () => show(Number(slider.value)));
  for (const button of root.querySelectorAll("[data-step]")) {
    button.addEventListener("click", () => show(Number(slider.value) + Number(button.dataset.step)));
  }
  for (const item of root.querySelectorAll("[data-frame]")) {
    item.addEventListener("click", () => show(Number(item.dataset.frame)));
  }
  root.step = (delta) => show(Number(slider.value) + delta);
  show(0);
}

$("rules").addEventListener("input", refreshGraph);
$("rules-file").addEventListener("change", async () => {
  await loadExample($("rules-file"), $("rules"));
  refreshGraph();
});
$("stream-file").addEventListener("change", () => loadExample($("stream-file"), $("stream")));

$("run").addEventListener("click", async () => {
  $("trace").innerHTML = await post("/trace", { rules: $("rules").value, stream: $("stream").value });
  showTab("trace");
  setupPlayer($("trace"));
});

for (const tab of document.querySelectorAll("[data-tab]")) {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
}

// стрелки листают кадры, когда открыт прогон и курсор не в текстовом поле
document.addEventListener("keydown", (event) => {
  const trace = $("trace");
  if (trace.hidden || !trace.step || event.target.closest("textarea, input, select")) return;
  if (event.key === "ArrowLeft") trace.step(-1);
  if (event.key === "ArrowRight") trace.step(1);
});
