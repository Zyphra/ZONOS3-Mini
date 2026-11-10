"""The FastAPI app of the MLX server: the v2 API of servers/api/openapi.yaml, the access-token gate, CORS, the keepalive
wrapper, studio projects and the studio's production build at / (the sgl api_server app, route for route)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from zonos3.server import plugins
from zonos3.server.edit import make_router as edit_router
from zonos3.server.keepalive import KEEPALIVE_PATHS, KeepaliveMiddleware
from zonos3.server.middleware import install_cors, install_private_access
from zonos3.server.projects import make_router as projects_router
from zonos3.server.studio import install_studio

from zonos3_mlx.server import capabilities, edit, generate, host, speakers
from zonos3_mlx.server.state import ServerState


def create_app(state: ServerState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # with a steering bundle, every default voice's embedding up front (the studio's Who map), as the sgl server does
        if state.engine.steer_bundle is not None:
            asyncio.get_running_loop().create_task(speakers.warm_default_voices(state))
        try:
            yield
        finally:
            state.close()

    # The spec of this API is servers/api/openapi.yaml; FastAPI's generated docs would describe a second one.
    app = FastAPI(
        title="Zonos3 API Server (MLX)",
        version="2.0.0",
        lifespan=lifespan,
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    app.state.zonos3 = state

    # Innermost, so CORS and the access check wrap its early status line; plugins.mount extends the list with the
    # plugins' paths.
    keepalive = list(KEEPALIVE_PATHS)
    app.add_middleware(KeepaliveMiddleware, paths=keepalive)
    install_cors(app)
    install_private_access(app)

    app.include_router(projects_router(lambda: state.projects), prefix="/v2")
    for module in (capabilities, speakers, generate):
        app.include_router(module.router(state))
    app.include_router(edit_router(lambda: edit.runtime(state), lambda: list(state.engine.decoders)))
    plugins.mount(app, state.plugins, host.MlxHost(state), keepalive)
    # Registered last, so every API route matches first (zonos3.server.studio).
    install_studio(app, lambda: state.options.studio_dir)
    return app
