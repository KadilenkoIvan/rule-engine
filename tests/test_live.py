import threading

import pytest

from rule_engine import LiveEngine


def test_live_timer_callback_continuation_and_close(make_rules):
    rules = make_rules([
        {"name": "end", "patterns": [["a", "b", {"absent": ["x"], "for": 2.0}]]},
        {"name": "more", "patterns": [["a", "b", {"absent": ["x"], "for": 2.0}, "c"]]},
    ])
    now = [0.0]
    updates = []
    timer_arrived = threading.Event()

    def receive(update):
        updates.append(update)
        if update.step is None:
            timer_arrived.set()

    live = LiveEngine(rules, receive, clock=lambda: now[0])
    try:
        first = live.push_now("a", 0.9)
        assert first.step.position == 0
        with live._wake:
            now[0] = 1.0
        live.push_now("b", 0.8)
        # Будим реальный таймер-поток, но время контролируем без sleep и гонки с часами ОС.
        with live._wake:
            now[0] = 4.0
            live._wake.notify()
        assert timer_arrived.wait(3.0), "таймер-поток не отдал обновление"
        [timer_update] = [update for update in updates if update.step is None]
        assert [match.scenario for match in timer_update.matches] == ["end"]
        last = live.push_now("c", 0.7)
        [match] = last.matches
        assert match.scenario == "more"
        assert [step.position for step in match.steps] == [0, 1, 2]
        ordered = sorted(updates, key=lambda update: update.number)
        assert [update.number for update in ordered] == list(range(1, len(ordered) + 1))
    finally:
        live.close()
    assert not live._thread.is_alive()
    with pytest.raises(ValueError, match="закрыт"):
        live.push_now("a", 0.9)
