// Плеер кадров: ◀ ▶, ползунок и клик по срабатыванию в ленте. Как показать кадр, решает `show`:
// в прогоне все кадры уже на странице, в живом режиме они приходят по одному.

export function player(box, count, show) {
  const slider = box.querySelector(".controls input[type=range]");
  const position = box.querySelector(".controls .position");
  let current = 0;
  // счётчик и ползунок без перерисовки кадра: в живом режиме кадров становится больше
  const update = () => {
    const total = count();
    slider.max = Math.max(0, total - 1);
    slider.value = current;
    slider.disabled = total === 0;
    position.textContent = total ? `кадр ${current + 1} из ${total}` : "кадров пока нет";
    for (const item of box.querySelectorAll("[data-frame]")) {
      item.classList.toggle("current", Number(item.dataset.frame) === current);
    }
  };
  const go = (index) => {
    const total = count();
    if (!total) return;
    current = Math.max(0, Math.min(total - 1, index));
    show(current);
    update();
  };
  slider.addEventListener("input", () => go(Number(slider.value)));
  for (const button of box.querySelectorAll("[data-step]")) {
    button.addEventListener("click", () => go(current + Number(button.dataset.step)));
  }
  // лента в живом режиме дописывается, поэтому клик ловит сам плеер
  box.addEventListener("click", (event) => {
    const item = event.target.closest("[data-frame]");
    if (item) go(Number(item.dataset.frame));
  });
  return { go, update, current: () => current, step: (delta) => go(current + delta) };
}

// кадры прогона уже на странице: ползунок только показывает нужный
export function setupPlayer(root, start) {
  const box = root.querySelector(".player");
  root.step = null;
  root.current = null;
  if (!box) return;
  const frames = box.querySelectorAll(".frame");
  const control = player(box, () => frames.length, (index) => {
    frames.forEach((frame, i) => { frame.hidden = i !== index; });
  });
  root.step = control.step;
  root.current = control.current;
  control.go(start);
}
