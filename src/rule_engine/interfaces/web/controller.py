"""Контроллер веба: строки на входе, `Reply` на выходе. О фреймворке не знает.

Ошибку правил или потока показывает пользователю: это ответ на неверный вход, а не обход.
"""

from pathlib import Path
from typing import NamedTuple

from rule_engine.application.service import RuleEngineService
from rule_engine.application.views import Frame
from rule_engine.interfaces.web.presenter import Presenter

HTML = "text/html"


class Reply(NamedTuple):
    status: int
    body: str
    media_type: str


def describe(error: ValueError | KeyError) -> str:
    if isinstance(error, KeyError):
        return f"в JSON нет ключа {error}"
    return str(error)


class WebController:
    def __init__(self, service: RuleEngineService, presenter: Presenter, examples: Path) -> None:
        self._service = service
        self._presenter = presenter
        self._examples = examples  # папка с правилами *.json и потоками *.txt

    def _files(self, pattern: str) -> dict[str, Path]:
        return {path.name: path for path in sorted(self._examples.glob(pattern))}

    def index(self) -> Reply:
        rules_files, stream_files = self._files("*.json"), self._files("*.txt")
        # на странице сразу открыты первые примеры; если их нет, поля пустые
        rules = next(iter(rules_files.values()), None)
        stream = next(iter(stream_files.values()), None)
        rules_text = "" if rules is None else rules.read_text(encoding="utf-8")
        stream_text = "" if stream is None else stream.read_text(encoding="utf-8")
        graph, error = None, None
        try:
            graph = self._service.graph(rules_text)
        except (ValueError, KeyError) as caught:
            error = describe(caught)
        body = self._presenter.page(
            list(rules_files), list(stream_files), rules_text, stream_text, graph, error
        )
        return Reply(200, body, HTML)

    def example(self, name: str) -> Reply:
        """Текст примера по имени файла. Имя ищется среди файлов папки, путь из запроса не строится."""
        path = {**self._files("*.json"), **self._files("*.txt")}.get(name)
        if path is None:
            return Reply(404, f"примера {name!r} нет", "text/plain")
        return Reply(200, path.read_text(encoding="utf-8"), "text/plain")

    def graph(self, rules: str) -> Reply:
        try:
            graph = self._service.graph(rules)
        except (ValueError, KeyError) as error:
            return Reply(422, self._presenter.error(describe(error)), HTML)
        return Reply(200, self._presenter.graph(graph), HTML)

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
