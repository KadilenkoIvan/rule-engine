"""Что считать ошибкой и как её обрабатывать.

Каждая ошибка имеет код, этап, на котором возникает, признак «временная»
(имеет ли смысл повтор) и стратегию обработки. Таблица ``POLICIES`` — единый
источник правды: и движок, и отчёты, и документация берут стратегию отсюда.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Stage(StrEnum):
    INPUT = "input"  # разбор текста
    RESOLUTION = "resolution"  # выбор команды и цели
    PLANNING = "planning"  # построение пути по автомату
    EXECUTION = "execution"  # проход по ребру на устройстве
    ROLLBACK = "rollback"  # компенсация


class Strategy(StrEnum):
    CLARIFY = "clarify"  # переспросить с вариантами
    CONFIRM = "confirm"  # спросить «да/нет»
    REJECT = "reject"  # безопасно отказать, ничего не делать
    RETRY = "retry"  # повторить с backoff, если позволяет дедлайн
    REPLAN = "replan"  # перечитать состояние и перестроить путь
    ROLLBACK = "rollback"  # откатить уже выполненные шаги (сага)
    SKIP = "skip"  # пропустить некритичный шаг
    ALERT = "alert"  # требуется вмешательство человека


class ErrorCode(StrEnum):
    # ввод
    EMPTY_INPUT = "EMPTY_INPUT"
    NEGATED = "NEGATED"
    NOT_FOUND = "NOT_FOUND"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    # выбор команды
    AMBIGUOUS = "AMBIGUOUS"
    ANTONYM_CONFLICT = "ANTONYM_CONFLICT"
    UNSAFE_FUZZY = "UNSAFE_FUZZY"
    HIGH_RISK = "HIGH_RISK"
    MISSING_ACTION = "MISSING_ACTION"
    MISSING_TARGET = "MISSING_TARGET"
    MISSING_VALUE = "MISSING_VALUE"
    VALUE_OUT_OF_RANGE = "VALUE_OUT_OF_RANGE"
    INCOMPATIBLE = "INCOMPATIBLE"
    # планирование
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    NO_PATH = "NO_PATH"
    DEVICE_OFFLINE = "DEVICE_OFFLINE"
    # исполнение
    TIMEOUT = "TIMEOUT"
    DEVICE_ERROR = "DEVICE_ERROR"
    DEVICE_FAULT = "DEVICE_FAULT"
    STATE_DRIFT = "STATE_DRIFT"
    VERIFY_FAILED = "VERIFY_FAILED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    # откат
    COMPENSATION_FAILED = "COMPENSATION_FAILED"
    COMPENSATION_UNAVAILABLE = "COMPENSATION_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ErrorPolicy:
    stage: Stage
    transient: bool
    strategy: Strategy
    message: str


POLICIES: dict[ErrorCode, ErrorPolicy] = {
    ErrorCode.EMPTY_INPUT: ErrorPolicy(Stage.INPUT, False, Strategy.REJECT, "Пустая команда."),
    ErrorCode.NEGATED: ErrorPolicy(
        Stage.INPUT, False, Strategy.REJECT, "В команде есть отрицание — ничего не делаю."
    ),
    ErrorCode.NOT_FOUND: ErrorPolicy(Stage.INPUT, False, Strategy.REJECT, "Не понял команду."),
    ErrorCode.LOW_CONFIDENCE: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.CLARIFY, "Не уверен, что понял правильно."
    ),
    ErrorCode.AMBIGUOUS: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.CLARIFY, "Команда неоднозначна — уточните."
    ),
    ErrorCode.ANTONYM_CONFLICT: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.CLARIFY, "Похоже на противоположные команды — уточните."
    ),
    ErrorCode.UNSAFE_FUZZY: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.CONFIRM,
        "Опасная команда распознана неточно — нужно подтверждение.",
    ),
    ErrorCode.HIGH_RISK: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.CONFIRM, "Опасная команда — нужно подтверждение."
    ),
    ErrorCode.MISSING_ACTION: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.CLARIFY, "Что сделать с устройством?"
    ),
    ErrorCode.MISSING_TARGET: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.CLARIFY, "С каким устройством?"
    ),
    ErrorCode.MISSING_VALUE: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.CLARIFY, "Какое значение установить?"
    ),
    ErrorCode.VALUE_OUT_OF_RANGE: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.REJECT, "Значение вне допустимого диапазона."
    ),
    ErrorCode.INCOMPATIBLE: ErrorPolicy(
        Stage.RESOLUTION, False, Strategy.REJECT, "Это устройство так не умеет."
    ),
    ErrorCode.PRECONDITION_FAILED: ErrorPolicy(
        Stage.PLANNING, False, Strategy.REJECT, "Не выполнено условие для команды."
    ),
    ErrorCode.NO_PATH: ErrorPolicy(
        Stage.PLANNING, False, Strategy.REJECT, "Из текущего состояния так сделать нельзя."
    ),
    ErrorCode.DEVICE_OFFLINE: ErrorPolicy(
        Stage.PLANNING, False, Strategy.REJECT, "Устройство недоступно."
    ),
    ErrorCode.TIMEOUT: ErrorPolicy(
        Stage.EXECUTION, True, Strategy.RETRY, "Устройство не ответило вовремя."
    ),
    ErrorCode.DEVICE_ERROR: ErrorPolicy(
        Stage.EXECUTION, True, Strategy.RETRY, "Временная ошибка устройства."
    ),
    ErrorCode.DEVICE_FAULT: ErrorPolicy(
        Stage.EXECUTION, False, Strategy.ROLLBACK, "Неисправность устройства."
    ),
    ErrorCode.STATE_DRIFT: ErrorPolicy(
        Stage.EXECUTION, True, Strategy.REPLAN, "Состояние устройства изменилось извне."
    ),
    ErrorCode.VERIFY_FAILED: ErrorPolicy(
        Stage.EXECUTION, True, Strategy.RETRY, "Устройство не подтвердило переход."
    ),
    ErrorCode.DEADLINE_EXCEEDED: ErrorPolicy(
        Stage.EXECUTION, False, Strategy.ROLLBACK, "Не уложились во время."
    ),
    ErrorCode.COMPENSATION_FAILED: ErrorPolicy(
        Stage.ROLLBACK, False, Strategy.ALERT, "Не удалось откатить изменения."
    ),
    ErrorCode.COMPENSATION_UNAVAILABLE: ErrorPolicy(
        Stage.ROLLBACK, False, Strategy.ALERT, "Шаг необратим — откат невозможен."
    ),
}


@dataclass(frozen=True, slots=True)
class EngineError:
    code: ErrorCode
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def policy(self) -> ErrorPolicy:
        return POLICIES[self.code]

    @property
    def strategy(self) -> Strategy:
        return self.policy.strategy

    def message(self) -> str:
        return f"{self.policy.message} {self.detail}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "stage": self.policy.stage.value,
            "strategy": self.strategy.value,
            "transient": self.policy.transient,
            "message": self.message(),
            **({"data": self.data} if self.data else {}),
        }
