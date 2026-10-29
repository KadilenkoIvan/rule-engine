// Живой режим: соединение на сессию, ввод токенов, кадры в плеере со слежением, запись в поток.
// HTML кадров собирает сервер; здесь только соединение, хранение кадров и переключение.

import { player } from "./player.js";

const $ = (id) => document.getElementById(id);

// rules() — правила, которые сервер уже проверил, или null; toStream(text) — сессия в «Поток»
export function setupLive({ rules, toStream }) {
  let socket = null; // соединение текущей сессии: сообщения прежних отбрасываются
  let wanted = false; // сессию запустили и не останавливали: смена правил начинает её заново
  let sessionRules = null;
  let note = ""; // что сказать, когда сессия начнётся
  let frames = []; // HTML кадров: в DOM только показанный
  let lines = []; // строки потока для «В поток»
  let control = null;
  let following = true; // показывать каждый новый кадр
  let started = 0;
  let ticker = null;

  const status = (text) => { $("live-status").textContent = text; };
  const error = (text) => {
    $("live-error").textContent = text;
    $("live-error").hidden = !text;
  };

  function setRunning(running) {
    $("live-token").disabled = !running;
    $("live-send").disabled = !running;
    for (const button of $("live-tokens").querySelectorAll("button")) button.disabled = !running;
    $("live-toggle").textContent = wanted ? "Остановить" : "Начать сессию";
    $("live-clock").hidden = !running;
    clearInterval(ticker);
    if (running) {
      // часы примерные: время событий ставит сервер, отсчёт у него начался чуть раньше
      const tick = () => { $("live-clock").textContent = `${((performance.now() - started) / 1000).toFixed(1)} с`; };
      tick();
      ticker = setInterval(tick, 100);
    }
  }

  // кадры и строки потока остаются: их можно листать и записать в поток
  function disconnect() {
    if (socket) socket.close();
    socket = null;
    setRunning(false);
  }

  function connect(text, message) {
    disconnect();
    frames = [];
    lines = [];
    control = null;
    sessionRules = text;
    note = message;
    $("live-view").innerHTML = "";
    $("live-export").disabled = true;
    $("live").step = null;
    error("");
    status("Подключение…");
    const current = (socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/live`));
    current.addEventListener("open", () => current.send(JSON.stringify({ rules: text })));
    current.addEventListener("message", (event) => {
      if (current === socket) receive(JSON.parse(event.data));
    });
    current.addEventListener("close", () => {
      if (current !== socket) return;
      socket = null;
      wanted = false;
      setRunning(false);
      status("Соединение закрыто: сессия остановлена.");
    });
  }

  function show(index) {
    const box = $("live-view").querySelector(".player");
    box.querySelector(".frames").innerHTML = frames[index];
    following = index === frames.length - 1;
    box.querySelector(".follow").hidden = following;
  }

  function receive(message) {
    if (message.type === "start") {
      $("live-tokens").innerHTML = message.tokens;
      $("live-view").innerHTML = message.view;
      const box = $("live-view").querySelector(".player");
      control = player(box, () => frames.length, show);
      box.querySelector(".follow").addEventListener("click", () => control.go(frames.length - 1));
      $("live").step = control.step;
      following = true;
      started = performance.now();
      setRunning(true);
      status(note || "Сессия идёт.");
      $("live-token").focus();
    } else if (message.type === "frame") {
      frames.push(message.html);
      lines.push(message.line);
      $("live-view").querySelector(".log ol").insertAdjacentHTML("beforeend", message.log);
      $("live-export").disabled = false;
      if (following) control.go(frames.length - 1);
      else control.update();
    } else if (message.type === "error") {
      error(message.message);
      if (!control) {
        // правила не приняты: сессии нет
        wanted = false;
        disconnect();
        status("Сессия не началась.");
      }
    }
  }

  function send(token) {
    if (!socket || socket.readyState !== WebSocket.OPEN || !control) return;
    error("");
    socket.send(JSON.stringify({ token, score: Number($("live-score").value) }));
  }

  $("live-toggle").addEventListener("click", () => {
    if (wanted) {
      wanted = false;
      disconnect();
      status("Сессия остановлена. Кадры можно листать и записать в поток.");
      return;
    }
    const text = rules();
    if (text === null) {
      error("Исправьте правила: в них ошибка или они ещё проверяются.");
      return;
    }
    wanted = true;
    connect(text, "");
  });

  $("live-form").addEventListener("submit", (event) => {
    event.preventDefault();
    send($("live-token").value);
    $("live-token").value = "";
    $("live-token").focus();
  });

  $("live-tokens").addEventListener("click", (event) => {
    const button = event.target.closest("[data-token]");
    if (button) send(button.dataset.token);
  });

  $("live-score").addEventListener("input", () => {
    $("live-score-value").textContent = Number($("live-score").value).toFixed(2);
  });

  $("live-export").addEventListener("click", () => {
    toStream(["# живая сессия: время — секунды от её начала, метка — ход таймера", ...lines, ""].join("\n"));
  });

  return {
    // правила прошли проверку сервера (text) или в них ошибка (null)
    rulesChanged(text) {
      if (!wanted) return;
      if (text === null) {
        disconnect();
        status("В правилах ошибка: сессия остановлена и начнётся заново после исправления.");
        return;
      }
      if (text === sessionRules && socket) return;
      connect(text, "Правила изменились: сессия начата заново.");
    },
  };
}
