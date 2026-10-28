import random

import pytest

from rule_engine.domain.timer_wheel import TimerWheel


def small() -> TimerWheel[str]:
    """Маленькое колесо: уровни появляются уже на сроках в несколько секунд."""
    wheel: TimerWheel[str] = TimerWheel(tick=1.0, size=4)
    wheel.advance(0.0)
    return wheel


def payloads(timers) -> list[str]:
    return [timer.payload for timer in timers]


def test_deadline_equal_to_now_does_not_fire() -> None:
    wheel = small()
    wheel.schedule(5.0, "a")
    assert wheel.advance(5.0) == []
    assert payloads(wheel.advance(5.000001)) == ["a"]


def test_exact_boundary_inside_one_cell() -> None:
    wheel: TimerWheel[str] = TimerWheel(tick=0.01, size=64)
    wheel.advance(0.0)
    wheel.schedule(0.105, "a")
    assert wheel.advance(0.103) == []  # та же ячейка [0.10, 0.11), но срок ещё не прошёл
    assert wheel.advance(0.105) == []
    assert wheel.next_deadline() == 0.105
    assert payloads(wheel.advance(0.1051)) == ["a"]


def test_fired_in_deadline_then_schedule_order() -> None:
    wheel = small()
    wheel.schedule(3.0, "c")
    wheel.schedule(2.0, "a")
    wheel.schedule(3.0, "d")
    wheel.schedule(2.5, "b")
    assert payloads(wheel.advance(10.0)) == ["a", "b", "c", "d"]
    assert len(wheel) == 0


def test_cancel() -> None:
    wheel = small()
    keep = wheel.schedule(2.0, "keep")
    drop = wheel.schedule(1.5, "drop")
    assert len(wheel) == 2
    wheel.cancel(drop)
    assert len(wheel) == 1
    assert wheel.next_deadline() == 2.0
    with pytest.raises(ValueError):
        wheel.cancel(drop)
    assert payloads(wheel.advance(3.0)) == ["keep"]
    with pytest.raises(ValueError):
        wheel.cancel(keep)


def test_cascade_through_levels_one_tick_at_a_time() -> None:
    wheel = small()
    # уровень 0 — 4 с, уровень 1 — 16 с, уровень 2 — 64 с
    wheel.schedule(37.5, "far")
    wheel.schedule(9.2, "mid")
    fired = {}
    for second in range(1, 50):
        for timer in wheel.advance(float(second)):
            fired[timer.payload] = second
    assert fired == {"mid": 10, "far": 38}


def test_huge_jump_does_not_walk_ticks() -> None:
    # Unix-время и скачок на 10^12 тиков: перебор тиков здесь бы завис
    wheel: TimerWheel[int] = TimerWheel()
    start = 1.7e9
    wheel.advance(start)
    deadlines = [start + offset for offset in (0.5, 3.0, 4e3, 7e6, 9e9)]
    for number, deadline in enumerate(deadlines):
        wheel.schedule(deadline, number)
    assert payloads(wheel.advance(start + 1e10)) == [0, 1, 2, 3, 4]
    assert wheel.next_deadline() is None


def test_next_deadline() -> None:
    wheel = small()
    assert wheel.next_deadline() is None
    timer = wheel.schedule(2.0, "a")
    wheel.cancel(timer)
    assert wheel.next_deadline() is None
    # ранний таймер ставится на уровень 1, пока время 0; поздний — на уровень 0, когда время 10
    wheel.schedule(12.5, "early")
    wheel.advance(10.0)
    wheel.schedule(13.5, "late")
    assert wheel.next_deadline() == 12.5


def check_against_reference(seed: int, start: float) -> None:
    """Случайные постановки, отмены и сдвиги против простого списка."""
    rng = random.Random(seed)
    wheel: TimerWheel[int] = TimerWheel(tick=0.01, size=8)
    reference: dict[int, float] = {}  # номер → срок ждущего таймера
    timers = {}
    now = start
    wheel.advance(now)
    for number in range(3000):
        action = rng.random()
        if action < 0.5:
            offsets = [rng.uniform(1e-6, 0.05), rng.uniform(0, 5), rng.uniform(0, 500)]
            deadline = now + rng.choice(offsets)
            if deadline <= now:
                continue
            timers[number] = wheel.schedule(deadline, number)
            reference[number] = deadline
        elif action < 0.65 and reference:
            victim = rng.choice(sorted(reference))
            wheel.cancel(timers[victim])
            del reference[victim]
        else:
            now += rng.choice([0.0, rng.uniform(0, 0.02), rng.uniform(0, 3), rng.uniform(0, 300)])
            expected = sorted(
                (deadline, key) for key, deadline in reference.items() if deadline < now
            )
            assert payloads(wheel.advance(now)) == [key for _, key in expected]
            for _, key in expected:
                del reference[key]
        assert len(wheel) == len(reference)
        assert wheel.next_deadline() == min(reference.values(), default=None)


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference_near_zero(seed: int) -> None:
    check_against_reference(seed, 0.0)


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference_near_unix_time(seed: int) -> None:
    check_against_reference(seed, 1.7e9)


def test_errors() -> None:
    with pytest.raises(ValueError):
        TimerWheel(tick=0.0)
    with pytest.raises(ValueError):
        TimerWheel(tick=float("inf"))
    with pytest.raises(ValueError):
        TimerWheel(size=1)
    wheel: TimerWheel[str] = TimerWheel()
    with pytest.raises(ValueError):
        wheel.schedule(1.0, "до первого advance")
    wheel.advance(5.0)
    with pytest.raises(ValueError):
        wheel.advance(4.0)
    with pytest.raises(ValueError):
        wheel.advance(float("nan"))
    with pytest.raises(ValueError):
        wheel.schedule(5.0, "не позже текущего")
    with pytest.raises(ValueError):
        wheel.schedule(float("inf"), "не конечный")
