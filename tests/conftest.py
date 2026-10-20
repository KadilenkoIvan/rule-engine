from __future__ import annotations

import pytest

from rule_engine.engine import DEFAULT_SPEC, Engine
from rule_engine.rules import RuleCompiler, RuleSet
from rule_engine.spec import Spec


@pytest.fixture(scope="session")
def spec() -> Spec:
    return Spec.load(DEFAULT_SPEC)


@pytest.fixture(scope="session")
def ruleset(spec: Spec) -> RuleSet:
    """Общий скомпилированный набор правил — только для тестов, которые его не меняют."""
    return RuleCompiler().compile(spec)


@pytest.fixture
def engine() -> Engine:
    """Свежий движок: у него своё состояние мира, словарь и симулятор."""
    return Engine.from_files()
