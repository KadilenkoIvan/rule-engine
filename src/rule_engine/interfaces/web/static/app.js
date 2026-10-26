// HTML собирает сервер. Здесь только запросы, вставка ответа и переключение кадров прогона.

const $ = (id) => document.getElementById(id);

async function post(url, fields) {
  return fetch(url, { method: "POST", body: new URLSearchParams(fields) });
}

function debounce(action, delay) {
  let timer;
  return () => {
    clearTimeout(timer);
    timer = setTimeout(action, delay);
  };
}

// запросы по одному месту нумеруются: ответ на устаревший запрос отбрасывается, иначе медленный
// старый ответ затёр бы свежий граф
const tickets = {};

async function latest(key, request) {
  const ticket = (tickets[key] = (tickets[key] || 0) + 1);
  const result = await request();
  return ticket === tickets[key] ? result : null;
}

function showTab(name) {
  for (const tab of document.querySelectorAll("[data-tab]")) {
    const on = tab.dataset.tab === name;
    tab.classList.toggle("on", on);
    $(tab.dataset.tab).hidden = !on;
  }
}

const session = document.querySelector(".inputs").dataset.session;
let syncedRules = document.querySelector(".inputs").dataset.rulesValid === "true" ? $("rules").value : null;
let addingScenario = false;
let graphQueue = Promise.resolve();
let graphPending = 0;

function queueGraph(action) {
  graphPending += 1;
  $("graph").classList.add("pending");
  updateScenarioButton();
  const request = graphQueue.then(action).finally(() => {
    graphPending -= 1;
    if (graphPending === 0) $("graph").classList.remove("pending");
    updateScenarioButton();
  });
  graphQueue = request.catch(() => {}); // ошибка запроса не блокирует следующие действия
  return request;
}

function updateScenarioButton() {
  $("add-scenario").disabled = addingScenario || graphPending > 0 || syncedRules !== $("rules").value;
}

function scenarioError(message) {
  $("scenario-error").textContent = message;
  $("scenario-error").hidden = false;
}

async function refreshGraph() {
  try {
    await queueGraph(async () => {
      const source = $("rules").value;
      if (source === syncedRules) return;
      const result = await latest("graph", async () => {
        const response = await post("/graph", { rules: source, session });
        return { ok: response.ok, html: await response.text() };
      });
      if (result === null) return;
      syncedRules = result.ok ? source : null;
      if (source === $("rules").value) $("graph").innerHTML = result.html;
      updateScenarioButton();
    });
  } catch (error) {
    syncedRules = null;
    scenarioError(`Не удалось обновить граф: ${error.message}`);
    updateScenarioButton();
  }
}

updateScenarioButton();

$("scenario-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (addingScenario || graphPending > 0 || syncedRules !== $("rules").value) return;
  $("scenario-error").hidden = true;
  $("scenario-status").textContent = "";
  let scenario;
  try {
    const fields = new FormData(event.currentTarget);
    scenario = { name: fields.get("name"), order: fields.get("order"), patterns: JSON.parse(fields.get("patterns")) };
    if (!Array.isArray(scenario.patterns) || !scenario.patterns.length || scenario.patterns.some((pattern) => !Array.isArray(pattern) || !pattern.length)) {
      throw new Error("Шаблоны должны быть непустым массивом непустых массивов шагов.");
    }
    for (const name of ["max_gap", "min_score", "min_confidence", "cooldown"]) {
      if (fields.get(name) !== "") scenario[name] = Number(fields.get(name));
    }
  } catch (error) {
    scenarioError(`Проверьте шаблоны: ${error.message}`);
    return;
  }
  addingScenario = true;
  $("scenario-fields").disabled = true;
  $("rules").disabled = true;
  $("rules-file").disabled = true;
  $("run").disabled = true;
  $("scenario-status").textContent = "Добавление…";
  updateScenarioButton();
  try {
    await queueGraph(async () => {
      const response = await post("/scenario", { session, scenario: JSON.stringify(scenario) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      $("rules").value = result.rules;
      syncedRules = result.rules;
      $("rules-file").value = "";
      $("graph").innerHTML = result.graph;
      if (traced) runTrace();
      showTab("graph");
      $("scenario-form").reset();
      $("scenario-status").textContent = `Сценарий «${scenario.name}» добавлен.`;
    });
  } catch (error) {
    $("scenario-status").textContent = "";
    scenarioError(error.message);
  } finally {
    addingScenario = false;
    $("scenario-fields").disabled = false;
    $("rules").disabled = false;
    $("rules-file").disabled = false;
    $("run").disabled = false;
    updateScenarioButton();
  }
});

// кадры прогона уже на странице: ползунок только показывает нужный
function setupPlayer(root, start) {
  const frames = root.querySelectorAll(".frame");
  const slider = root.querySelector(".player input[type=range]");
  root.step = null;
  root.current = null;
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
  root.current = () => Number(slider.value);
  show(start);
}

let traced = false; // прогон уже показан: дальше он пересчитывается при каждой правке

async function runTrace() {
  const trace = $("trace");
  trace.classList.add("pending");
  const source = $("rules").value;
  const stream = $("stream").value;
  const html = await latest("trace", async () => {
    const response = await post("/trace", { rules: source, stream });
    return response.text();
  });
  if (html === null || source !== $("rules").value || stream !== $("stream").value) return;
  // после пересчёта остаёмся на том же кадре, если он ещё есть
  const start = trace.current ? trace.current() : 0;
  trace.innerHTML = html;
  trace.classList.remove("pending");
  setupPlayer(trace, start);
  traced = true;
}

const refreshAll = debounce(() => {
  refreshGraph();
  if (traced) runTrace();
}, 400);

const refreshTrace = debounce(() => {
  if (traced) runTrace();
}, 400);

async function loadExample(select, target, refresh) {
  if (!select.value) return;
  const name = select.value;
  const text = await latest(select.id, async () => {
    const response = await fetch(`/example?name=${encodeURIComponent(name)}`);
    return response.text();
  });
  if (text === null) return;
  target.value = text;
  updateScenarioButton();
  refresh();
}

// своя правка — уже не пример из списка
$("rules").addEventListener("input", () => {
  $("rules-file").value = "";
  updateScenarioButton();
  $("scenario-status").textContent = "";
  refreshAll();
});
$("stream").addEventListener("input", () => {
  $("stream-file").value = "";
  refreshTrace();
});
$("rules-file").addEventListener("change", () => {
  syncedRules = null;
  updateScenarioButton();
  loadExample($("rules-file"), $("rules"), refreshAll);
});
$("stream-file").addEventListener("change", () => loadExample($("stream-file"), $("stream"), refreshTrace));

$("run").addEventListener("click", async () => {
  showTab("trace");
  await runTrace();
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
