"""Порты: что сценарии использования ждут от интерфейса."""

from typing import Protocol

from rule_engine.application.views import Frame


class LiveListener(Protocol):
    def on_frame(self, frame: Frame) -> None:
        """Кадр живой сессии: после события или хода таймера.

        Зовётся из потока, отправившего токен, или из таймер-потока движка, но всегда по одному
        и строго по порядку кадров. Пока он работает, следующий кадр ждёт.
        """
        ...
