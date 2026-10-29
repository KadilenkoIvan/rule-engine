"""Web controller: strings in, `Reply` out; it knows nothing of the framework.

Errors in rules or streams are shown to the user: they answer invalid input.
"""

import json
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import NamedTuple
from uuid import uuid4

from rule_engine.application.service import LiveSession, RuleEditor, RuleEngineService
from rule_engine.application.stream import stream_line
from rule_engine.application.views import Frame
from rule_engine.interfaces.web.presenter import Board, Presenter

HTML = "text/html"


class Reply(NamedTuple):
    """Controller answer: HTTP status, body and media type."""

    status: int
    body: str
    media_type: str


def describe(error: ValueError | KeyError | TypeError) -> str:
    """Error message for the user."""
    if isinstance(error, KeyError):
        return f"в JSON нет ключа {error}"
    if isinstance(error, TypeError):
        return "Неверная структура сценария. Проверьте шаблоны и настройки."
    return str(error)


def fields(message: object, *keys: str) -> list:
    """Values of the keys of a live mode message.

    Raises:
        ValueError: the message is not an object with exactly these keys.
    """
    if not isinstance(message, dict) or message.keys() != set(keys):
        # у сообщения с правилами значение — весь JSON правил: в ошибке хватит ключей
        got = f"ключи {sorted(message)}" if isinstance(message, dict) else repr(message)
        raise ValueError(f"ожидается объект JSON с ключами {list(keys)}, получено: {got}")
    return [message[key] for key in keys]


def _first_text(files: dict[str, Path]) -> str:
    """Text of the first file; empty if there are none."""
    path = next(iter(files.values()), None)
    return "" if path is None else path.read_text(encoding="utf-8")


class LiveChannel:
    """Live mode connection: the first message is the rules, then tokens.

    Replies and frames go out through `send` as JSON strings with HTML inside. `send` is called
    from the receiving thread and from the engine's timer thread; `LiveSession` keeps the order.
    """

    def __init__(
        self, service: RuleEngineService, presenter: Presenter, send: Callable[[str], None]
    ) -> None:
        self._service = service
        self._presenter = presenter
        self._send = send
        self._session: LiveSession | None = None
        self._board: Board | None = None
        self._frames = 0  # номер следующего кадра: по нему строка ленты открывает свой кадр

    def receive(self, text: str) -> None:
        """Client message; an input error goes back to the client, the session stays."""
        try:
            message = json.loads(text)
            if self._session is None:
                self._start(*fields(message, "rules"))
            else:
                self._push(*fields(message, "token", "score"))
        except (ValueError, KeyError) as error:
            self._reply({"type": "error", "message": describe(error)})

    def on_frame(self, frame: Frame) -> None:
        """Sends a frame of the live session."""
        html, log = self._presenter.live_frame(self._board, frame, self._frames)
        self._frames += 1
        self._reply({"type": "frame", "html": html, "log": log, "line": stream_line(frame)})

    def close(self) -> None:
        """The connection closed: stops the session's timer thread."""
        if self._session is not None:
            self._session.close()

    def _start(self, rules: object) -> None:
        """Starts the session with the rules and sends the token buttons and the empty graph."""
        if not isinstance(rules, str):
            raise ValueError("правила должны быть строкой с JSON")
        session = self._service.live(rules, self)
        # кадров до первого токена нет, поэтому граф готов раньше, чем он понадобится `on_frame`
        self._board = self._presenter.board(session.graph)
        self._session = session
        tokens, view = self._presenter.live_start(self._board)
        self._reply({"type": "start", "tokens": tokens, "view": view})

    def _push(self, token: object, score: object) -> None:
        """Pushes a token of the client; its frame comes through `on_frame`."""
        if not isinstance(token, str):
            raise ValueError(f"токен должен быть строкой, получено {token!r}")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError(f"уверенность должна быть числом, получено {score!r}")
        self._session.push(token, float(score))

    def _reply(self, message: dict) -> None:
        """Sends a message as JSON."""
        self._send(json.dumps(message, ensure_ascii=False))


class WebController:
    """Pages, examples, graphs, stream runs and live sessions of the web interface."""

    def __init__(self, service: RuleEngineService, presenter: Presenter, examples: Path) -> None:
        self._service = service
        self._presenter = presenter
        self._examples = examples  # папка с правилами *.json и потоками *.txt
        self._editors: dict[str, tuple[float, RuleEditor | None]] = {}
        self._editor_lock = Lock()

    def _expire_editors(self, now: float) -> None:
        """Drops editing sessions idle for an hour."""
        for session, (touched, _) in list(self._editors.items()):
            if now - touched >= 3600:
                del self._editors[session]

    def _editor(self, session: str) -> RuleEditor | None:
        """Editor of the session; call under the lock. Each tab has its own session.

        Raises:
            ValueError: the session expired.
        """
        now = monotonic()
        self._expire_editors(now)
        if session not in self._editors:
            raise ValueError("Сессия редактирования завершена. Обновите страницу.")
        _, editor = self._editors[session]
        self._editors[session] = (now, editor)
        return editor

    def _files(self, pattern: str) -> dict[str, Path]:
        """Example files matching `pattern`, by name in alphabetical order."""
        return {path.name: path for path in sorted(self._examples.glob(pattern))}

    def index(self) -> Reply:
        """Main page with the first examples open and a new editing session."""
        rules_files, stream_files = self._files("*.json"), self._files("*.txt")
        # на странице сразу открыты первые примеры; если их нет, поля пустые
        rules_text = _first_text(rules_files)
        stream_text = _first_text(stream_files)
        graph, error, editor = None, None, None
        try:
            editor = self._service.editor(rules_text)
            graph = editor.graph()
        except (ValueError, KeyError) as caught:
            error = describe(caught)
        session = self._open_session(editor)
        body = self._presenter.page(
            list(rules_files), list(stream_files), rules_text, stream_text, graph, error, session
        )
        return Reply(200, body, HTML)

    def example(self, name: str) -> Reply:
        """Text of an example by file name; the name is looked up, no path is built from it."""
        path = {**self._files("*.json"), **self._files("*.txt")}.get(name)
        if path is None:
            return Reply(404, f"примера {name!r} нет", "text/plain")
        return Reply(200, path.read_text(encoding="utf-8"), "text/plain")

    def _open_session(self, editor: RuleEditor | None) -> str:
        """New editing session with `editor`; returns its id."""
        session = uuid4().hex
        with self._editor_lock:
            now = monotonic()
            self._expire_editors(now)
            self._editors[session] = (now, editor)
        return session

    def graph(self, rules: str, session: str) -> Reply:
        """Graph of the rules; they become the session's rules."""
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
        """Adds a scenario to the session's rules; replies with the rules and the graph."""
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

    def live(self, send: Callable[[str], None]) -> LiveChannel:
        """Live session of one connection: its own engine, clock and timer."""
        return LiveChannel(self._service, self._presenter, send)

    def trace(self, rules: str, stream: str) -> Reply:
        """Stream run; frames before an error in the stream are shown too."""
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
