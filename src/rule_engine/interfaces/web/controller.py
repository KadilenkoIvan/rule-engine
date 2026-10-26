"""Контроллер веба: строки на входе, `Reply` на выходе. О фреймворке не знает.

Ошибку правил или потока показывает пользователю: это ответ на неверный вход, а не обход.
"""

import json
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import NamedTuple
from uuid import uuid4

from rule_engine.application.service import RuleEditor, RuleEngineService
from rule_engine.application.views import Frame
from rule_engine.interfaces.web.presenter import Presenter

HTML = "text/html"


class Reply(NamedTuple):
    status: int
    body: str
    media_type: str


def describe(error: ValueError | KeyError | TypeError) -> str:
    if isinstance(error, KeyError):
        return f"в JSON нет ключа {error}"
    if isinstance(error, TypeError):
        return "Неверная структура сценария. Проверьте шаблоны и настройки."
    return str(error)


class WebController:
    def __init__(self, service: RuleEngineService, presenter: Presenter, examples: Path) -> None:
        self._service = service
        self._presenter = presenter
        self._examples = examples  # папка с правилами *.json и потоками *.txt
        self._editors: dict[str, tuple[float, RuleEditor | None]] = {}
        self._editor_lock = Lock()

    def _expire_editors(self, now: float) -> None:
        for session, (touched, _) in list(self._editors.items()):
            if now - touched >= 3600:
                del self._editors[session]

    def _editor(self, session: str) -> RuleEditor | None:
        """Вызывается под блокировкой; вкладки имеют независимые сессии редактирования."""
        now = monotonic()
        self._expire_editors(now)
        if session not in self._editors:
            raise ValueError("Сессия редактирования завершена. Обновите страницу.")
        _, editor = self._editors[session]
        self._editors[session] = (now, editor)
        return editor

    def _files(self, pattern: str) -> dict[str, Path]:
        return {path.name: path for path in sorted(self._examples.glob(pattern))}

    def index(self) -> Reply:
        rules_files, stream_files = self._files("*.json"), self._files("*.txt")
        # на странице сразу открыты первые примеры; если их нет, поля пустые
        rules = next(iter(rules_files.values()), None)
        stream = next(iter(stream_files.values()), None)
        rules_text = "" if rules is None else rules.read_text(encoding="utf-8")
        stream_text = "" if stream is None else stream.read_text(encoding="utf-8")
        graph, error, editor = None, None, None
        try:
            editor = self._service.editor(rules_text)
            graph = editor.graph()
        except (ValueError, KeyError) as caught:
            error = describe(caught)
        session = uuid4().hex
        with self._editor_lock:
            now = monotonic()
            self._expire_editors(now)
            self._editors[session] = (now, editor)
        body = self._presenter.page(
            list(rules_files), list(stream_files), rules_text, stream_text, graph, error, session
        )
        return Reply(200, body, HTML)

    def example(self, name: str) -> Reply:
        """Текст примера по имени файла. Имя ищется среди файлов папки, путь из запроса не строится."""
        path = {**self._files("*.json"), **self._files("*.txt")}.get(name)
        if path is None:
            return Reply(404, f"примера {name!r} нет", "text/plain")
        return Reply(200, path.read_text(encoding="utf-8"), "text/plain")

    def graph(self, rules: str, session: str) -> Reply:
        try:
            with self._editor_lock:
                self._editor(session)
                editor = self._service.editor(rules)
                graph = editor.graph()
                self._editors[session] = (monotonic(), editor)
        except (ValueError, KeyError) as error:
            return Reply(422, self._presenter.error(describe(error)), HTML)
        return Reply(200, self._presenter.graph(graph), HTML)

    def add_scenario(self, session: str, scenario: str) -> Reply:
        try:
            with self._editor_lock:
                editor = self._editor(session)
                if editor is None:
                    raise ValueError("Исправьте правила перед добавлением сценария.")
                editor.add_scenario(scenario)
                payload = {
                    "rules": editor.rules_text,
                    "graph": self._presenter.graph(editor.graph()),
                }
        except (ValueError, KeyError, TypeError) as error:
            body = json.dumps({"error": describe(error)}, ensure_ascii=False)
            return Reply(422, body, "application/json")
        return Reply(200, json.dumps(payload, ensure_ascii=False), "application/json")

    def trace(self, rules: str, stream: str) -> Reply:
        try:
            trace = self._service.trace(rules, stream.splitlines())
        except (ValueError, KeyError) as error:
            return Reply(422, self._presenter.error(describe(error)), HTML)
        frames: list[Frame] = []
        try:
            for frame in trace.frames:
                frames.append(frame)
        except (ValueError, KeyError) as error:
            # кадры до ошибки показываются: по ним видно, где поток сломался
            return Reply(422, self._presenter.trace(trace.graph, frames, describe(error)), HTML)
        return Reply(200, self._presenter.trace(trace.graph, frames, None), HTML)
