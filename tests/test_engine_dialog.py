"""Диалог: уточнения, подтверждения, синонимы, сцена."""

from __future__ import annotations

from rule_engine.engine import (
    AWAITING_CLARIFICATION,
    AWAITING_CONFIRMATION,
    CANCELLED,
    REJECTED,
    SUCCEEDED,
)
from rule_engine.evaluation import load_dataset, regression_diff


def test_clarify_room_then_answer(engine):
    response = engine.handle("включи свет")
    assert response.state == AWAITING_CLARIFICATION
    assert response.question.startswith("В какой комнате")
    response = engine.handle("на кухне")
    assert response.state == SUCCEEDED
    assert engine.world.state("kitchen_light") == "on"
    assert engine.world.state("hall_light") == "off"


def test_clarify_everywhere_merges_options(engine):
    engine.handle("включи свет")
    response = engine.handle("везде")
    assert response.state == SUCCEEDED
    lights = [d for d in engine.spec.devices.values() if d.type == "light"]
    assert all(engine.world.state(d.id) == "on" for d in lights)


def test_clarify_by_ordinal(engine):
    engine.handle("возьми чашку")
    response = engine.handle("второй")
    assert response.state == SUCCEEDED
    assert engine.world.devices["lab_arm"].holding == "cup_blue"


def test_clarify_by_attribute(engine):
    engine.handle("возьми чашку")
    engine.handle("красную")
    assert engine.world.devices["lab_arm"].holding == "cup_red"


def test_missing_value_answer(engine):
    response = engine.handle("яркость торшера")
    assert response.state == AWAITING_CLARIFICATION and not response.options
    response = engine.handle("тридцать")
    assert response.state == SUCCEEDED
    assert engine.world.value("living_floor_lamp", "brightness") == 30


def test_antonym_conflict_is_clarified(engine):
    response = engine.handle("ыключи свет на кухне")
    assert response.state == AWAITING_CLARIFICATION
    assert response.question == "Включить или выключить?"
    assert engine.handle("выключи").state == SUCCEEDED


def test_dangerous_requires_confirmation(engine):
    response = engine.handle("открой дверь")
    assert response.state == AWAITING_CONFIRMATION
    assert engine.world.state("front_door") == "locked"
    assert engine.handle("нет").state == CANCELLED
    assert engine.world.state("front_door") == "locked"
    engine.handle("открой дверь")
    assert engine.handle("да").state == SUCCEEDED
    assert engine.world.state("front_door") == "unlocked"


def test_plan_level_risk_gate(engine):
    # «температура духовки 200» — сама установка температуры не опасна, но духовка
    # выключена, и план автоматически вставляет опасное ребро «включить духовку»
    response = engine.handle("нагрей духовку до 200")
    assert response.state == AWAITING_CONFIRMATION
    assert "kitchen_oven: off --turn_on--> on" in response.question


def test_new_command_while_pending_cancels_pending(engine):
    engine.handle("включи свет")
    response = engine.handle("выключи телевизор")
    assert response.state == SUCCEEDED
    assert engine.session.pending is None
    assert engine.session.history[0].state == CANCELLED


def test_negation_rejected(engine):
    assert engine.handle("не включай свет на кухне").state == REJECTED
    assert engine.world.state("kitchen_light") == "off"


def test_added_synonym_changes_only_target_queries(engine):
    examples = load_dataset()
    assert engine.handle("потуши свет в ванной").state == AWAITING_CLARIFICATION
    engine.session.pending = None
    report, diff = regression_diff(engine, examples, lambda e: e.add_alias("action:turn_off", "потуши"))
    assert report.entry is not None
    assert [q for q, _, _ in diff] == ["потуши свет в ванной"]
    response = engine.handle("потуши свет в ванной")
    assert response.state == SUCCEEDED
    # откат к исходной версии возвращает прежнее поведение
    engine.rollback_aliases(0)
    assert engine.understand("потуши свет в ванной").decision.status.value == "CLARIFY"


def test_alias_conflict_leads_to_clarification(engine):
    report = engine.add_alias("device:bedside_lamp", "торшер")
    assert [c.ref for c in report.conflicts] == ["device:living_floor_lamp"]
    assert engine.understand("включи торшер").decision.status.value == "CLARIFY"


def test_alias_warning_for_antonym_proximity(engine):
    # оператор по ошибке добавляет к «включить» синоним, близкий к «выключить»
    assert "выруби" in " ".join(engine.alias_warnings("action:turn_on", "вырубай"))
    assert "выключи" in " ".join(engine.alias_warnings("action:turn_on", "выключ"))
    assert engine.alias_warnings("action:turn_on", "зажигай") == []


def test_overlay_persists_aliases(tmp_path):
    from rule_engine.engine import Engine

    overlay = tmp_path / "overlay.json"
    first = Engine.from_files(overlay_path=overlay)
    first.add_alias("device:living_tv", "зомбоящик")
    second = Engine.from_files(overlay_path=overlay)
    decision = second.understand("включи зомбоящик").decision
    assert decision.status.value == "OK"
    assert decision.chosen.commands[0].targets == ("living_tv",)


def test_scene_update(engine):
    engine.update_scene([{"id": "bottle_1", "type": "bottle", "attrs": {"color": "green"}, "confidence": 0.9}])
    response = engine.handle("возьми бутылку")
    assert response.state == SUCCEEDED
    # объект в захвате не исчезает, даже если детектор его больше не видит
    engine.update_scene([])
    assert "bottle_1" in engine.world.scene
    assert engine.understand("возьми красную чашку").decision.status.value == "NOT_FOUND"
