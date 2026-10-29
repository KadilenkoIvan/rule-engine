"""Тонкий слой фреймворка: единственный модуль, который знает о FastAPI и uvicorn.

Маршруты разбирают запрос, зовут контроллер и оборачивают его `Reply`. Логики и HTML здесь нет:
замена фреймворка — правка этого файла.
"""

import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
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

    @app.websocket("/live")
    async def live(socket: WebSocket) -> None:
        await socket.accept()
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue[str] = asyncio.Queue()
        # кадры приходят и из таймер-потока движка: в цикл событий — только так
        channel = controller.live(lambda text: loop.call_soon_threadsafe(outbox.put_nowait, text))

        async def deliver() -> None:
            while True:
                await socket.send_text(await outbox.get())

        sender = asyncio.create_task(deliver())
        try:
            while True:
                text = await socket.receive_text()
                # движок ждёт свою блокировку и рисует кадр: цикл событий в это время свободен
                await asyncio.to_thread(channel.receive, text)
        except WebSocketDisconnect:
            pass
        finally:
            await asyncio.to_thread(channel.close)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    return app


def serve(controller: WebController, static: Path, host: str, port: int) -> None:
    uvicorn.run(create_app(controller, static), host=host, port=port)
