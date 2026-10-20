"""Проход по рёбрам: задержки, ошибки, сверка состояния, повторы, откаты."""

from __future__ import annotations

from rule_engine.engine import (
    PARTIAL,
    REJECTED,
    ROLLBACK_FAILED,
    ROLLED_BACK,
    SUCCEEDED,
)


def _confirm(engine, text):
    response = engine.handle(text)
    if engine.session.pending is not None:
        response = engine.handle("да")
    return response


def test_transient_errors_are_retried(engine):
    engine.driver.inject("living_tv", "error", 2)
    response = engine.handle("включи телевизор")
    assert response.state == SUCCEEDED
    log = response.execution[0]
    assert log["attempts"] == 3 and log["errors"] == ["DEVICE_ERROR", "DEVICE_ERROR"]
    assert engine.world.state("living_tv") == "on"


def test_timeout_with_late_apply_is_reconciled_not_resent(engine):
    engine.driver.inject("living_tv", "late", 1)
    response = engine.handle("включи телевизор")
    assert response.state == SUCCEEDED
    log = response.execution[0]
    assert log["attempts"] == 1  # чтение показало, что команда применилась — повтора нет
    assert any("применилась" in line for line in log["trace"])
    assert len([s for s in engine.driver.sent if s[1] == "living_tv"]) == 1


def test_fault_rolls_back_previous_steps(engine):
    engine.driver.inject("living_tv", "fault", 1)
    response = engine.handle("включи свет на кухне и включи телевизор")
    assert response.state == ROLLED_BACK
    assert engine.world.state("kitchen_light") == "off"  # первый шаг откатили
    assert engine.driver.physical["kitchen_light"][0] == "off"
    assert [r["status"] for r in response.rollback] == ["compensated"]


def test_retries_exhausted_then_rollback(engine):
    engine.driver.inject("living_tv", "timeout", 10)
    response = engine.handle("включи свет на кухне и включи телевизор")
    assert response.state == ROLLED_BACK
    codes = {e["code"] for e in response.errors}
    assert codes & {"TIMEOUT", "DEADLINE_EXCEEDED"}
    assert engine.world.state("kitchen_light") == "off"


def test_compensation_failure_is_reported(engine):
    engine.driver.inject("living_tv", "fault", 1)
    engine.driver.inject("kitchen_light", "fault", 0)
    # свет включится, телевизор сломается, а откат света тоже сломается
    original_send = engine.driver.send

    def send(step, timeout):
        if step.device == "kitchen_light" and step.action == "turn_off":
            engine.driver.faults.setdefault("kitchen_light", []).append("fault")
        return original_send(step, timeout)

    engine.driver.send = send
    response = engine.handle("включи свет на кухне и включи телевизор")
    assert response.state == ROLLBACK_FAILED
    assert any(e["code"] == "COMPENSATION_FAILED" for e in response.errors)


def test_state_drift_triggers_replan(engine):
    # пылесос кто-то поставил на паузу вручную: движок считает «docked»
    engine.driver.inject("vacuum", "drift:paused")
    response = engine.handle("запусти пылесос")
    assert response.state == SUCCEEDED
    log = response.execution[0]
    assert "STATE_DRIFT" in log["errors"]
    assert engine.world.state("vacuum") == "cleaning"


def test_already_in_target_state_physically(engine):
    engine.driver.inject("living_tv", "drift:on")
    response = engine.handle("включи телевизор")
    assert response.state == SUCCEEDED
    assert response.execution[0]["status"] == "already"
    assert not [s for s in engine.driver.sent if s[1] == "living_tv"]


def test_offline_device_rejected_at_planning(engine):
    engine.world.set_online("living_tv", False)
    response = engine.handle("включи телевизор")
    assert response.state == REJECTED
    assert response.errors[0]["code"] == "DEVICE_OFFLINE"


def test_device_goes_offline_during_execution(engine):
    engine.driver.inject("living_tv", "offline")
    response = engine.handle("включи свет на кухне и включи телевизор")
    assert response.state == ROLLED_BACK
    assert not engine.world.devices["living_tv"].online


def test_noncritical_scenario_step_failure_is_partial(engine):
    engine.driver.inject("vacuum", "fault", 1)
    response = engine.handle("я ухожу")
    assert response.state == PARTIAL
    assert engine.world.state("home_alarm") == "armed"
    assert engine.world.state("vacuum") == "docked"


def test_critical_scenario_step_failure_rolls_back_scenario(engine):
    engine.handle("включи свет на кухне")
    engine.driver.inject("home_alarm", "fault", 1)
    response = engine.handle("я ухожу")
    assert response.state == ROLLED_BACK
    # свет был выключен сценарием и включён обратно при откате
    assert engine.world.state("kitchen_light") == "on"
    assert engine.world.state("home_alarm") == "disarmed"


def test_value_rollback_restores_previous_value(engine):
    engine.handle("включи кондиционер в спальне")
    engine.driver.inject("living_tv", "fault", 1)
    response = engine.handle("поставь температуру 18 в спальне и включи телевизор")
    assert response.state == ROLLED_BACK
    assert engine.world.value("bedroom_ac", "temperature") == 22
    assert engine.driver.physical["bedroom_ac"][1]["temperature"] == 22


def test_undo_last_command(engine):
    engine.handle("включи свет в гостиной")
    assert engine.world.state("living_chandelier") == "on"
    response = engine.handle("отмени")
    assert response.state == ROLLED_BACK
    assert engine.world.state("living_chandelier") == "off"
    assert engine.handle("отмени").state == REJECTED  # отменять больше нечего


def test_virtual_deadline_is_respected(engine):
    engine.executor.config.deadline_ms = 1000
    engine.driver.inject("living_tv", "timeout", 5)
    response = engine.handle("включи телевизор")
    assert response.state == ROLLED_BACK
    assert response.timings_ms["execute_virtual"] <= 1000 + engine.driver.read_latency_ms * 2
