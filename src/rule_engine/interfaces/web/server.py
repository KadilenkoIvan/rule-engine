"""Thin framework layer: the only module that knows FastAPI and uvicorn.

Routes parse the request, call the controller and wrap its `Reply`. No logic and no HTML here:
replacing the framework means editing this file.
"""

import asyncio
from functools import partial
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from rule_engine.interfaces.web.controller import Reply, WebController


def respond(reply: Reply) -> Response:
    """HTTP response of a controller reply."""
    return Response(reply.body, status_code=reply.status, media_type=reply.media_type)


def create_app(controller: WebController, static: Path) -> FastAPI:
    """FastAPI app with the routes of `controller` and the static files of `static`."""
    app = FastAPI(title="rule-engine", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=static), name="static")
    routes = _Routes(controller)
    app.get("/")(routes.index)
    app.get("/example")(routes.example)
    app.post("/graph")(routes.graph)
    app.post("/scenario")(routes.add_scenario)
    app.post("/trace")(routes.trace)
    app.websocket("/live")(routes.live)
    return app


class _Routes:
    """Route handlers over the controller."""

    def __init__(self, controller: WebController) -> None:
        self._controller = controller

    def index(self) -> Response:
        return respond(self._controller.index())

    def example(self, name: str) -> Response:
        return respond(self._controller.example(name))

    def graph(self, rules: str = Form(""), session: str = Form("")) -> Response:
        return respond(self._controller.graph(rules, session))

    def add_scenario(self, session: str = Form(""), scenario: str = Form("")) -> Response:
        return respond(self._controller.add_scenario(session, scenario))

    def trace(self, rules: str = Form(""), stream: str = Form("")) -> Response:
        return respond(self._controller.trace(rules, stream))

    async def live(self, socket: WebSocket) -> None:
        """Live session over a websocket: messages go to the channel, frames come back."""
        await socket.accept()
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue[str] = asyncio.Queue()
        # кадры приходят и из таймер-потока движка: в цикл событий — только так
        channel = self._controller.live(partial(loop.call_soon_threadsafe, outbox.put_nowait))
        sender = asyncio.create_task(_deliver(socket, outbox))
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


async def _deliver(socket: WebSocket, outbox: "asyncio.Queue[str]") -> None:
    """Sends the outgoing messages of a live session in order."""
    while True:
        await socket.send_text(await outbox.get())


def serve(controller: WebController, static: Path, host: str, port: int) -> None:
    """Runs the web server."""
    uvicorn.run(create_app(controller, static), host=host, port=port)
