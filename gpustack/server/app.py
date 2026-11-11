from fastapi import FastAPI
from gpustack import __version__
from gpustack.config.config import Config
from gpustack.routes import probes
from contextlib import asynccontextmanager
import aiohttp
from gpustack.api import exceptions
from gpustack import envs
from gpustack.routes.routes import api_router


def create_app(cfg: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.server_config = cfg
        connector = aiohttp.TCPConnector(
            limit=envs.TCP_CONNECTOR_LIMIT,
            force_close=True,
        )
        app.state.http_client = aiohttp.ClientSession(
            connector=connector, trust_env=True
        )
        app.state.http_client_no_proxy = aiohttp.ClientSession(connector=connector)
        yield
        await app.state.http_client.close()
        await app.state.http_client_no_proxy.close()

    app = FastAPI(
        title="GPUStack",
        lifespan=lifespan,
        response_model_exclude_unset=True,
        version=__version__,
        docs_url=None if (cfg and cfg.disable_openapi_docs) else "/docs",
        redoc_url=None if (cfg and cfg.disable_openapi_docs) else "/redoc",
        openapi_url=None if (cfg and cfg.disable_openapi_docs) else "/openapi.json",
    )
    app.include_router(api_router)
    exceptions.register_handlers(app)

    return app
