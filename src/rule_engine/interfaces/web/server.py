"""Тонкий слой фреймворка: единственный модуль, который знает о FastAPI и uvicorn.

Маршруты разбирают запрос, зовут контроллер и оборачивают его `Reply`. Логики и HTML здесь нет:
замена фреймворка — правка этого файла.
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from rule_engine.interfaces.web.controller import Reply, WebController


def respond(reply: Reply) -> Response:
    return Response(reply.body, status_code=reply.status, media_type=reply.media_type)


def create_app(controller: WebController, static: Path) -> FastAPI:
    app = FastAPI(title="rule-engine", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index() -> Response:
        return respond(controller.index())

    @app.get("/example")
    def example(name: str) -> Response:
        return respond(controller.example(name))

    @app.post("/graph")
    def graph(rules: str = Form(""), session: str = Form("")) -> Response:
        return respond(controller.graph(rules, session))

    @app.post("/scenario")
    def add_scenario(session: str = Form(""), scenario: str = Form("")) -> Response:
        return respond(controller.add_scenario(session, scenario))

    @app.post("/trace")
    def trace(rules: str = Form(""), stream: str = Form("")) -> Response:
        return respond(controller.trace(rules, stream))

    return app


def serve(controller: WebController, static: Path, host: str, port: int) -> None:
    uvicorn.run(create_app(controller, static), host=host, port=port)
