from contextlib import asynccontextmanager
import aiohttp
from fastapi import FastAPI

from gpustack import __version__
from gpustack.api import exceptions
from gpustack.config.config import Config
from gpustack import envs
from gpustack.routes.routes import api_router


def create_app(cfg: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        FastAPI 的生命周期钩子。`yield` 之前的代码在服务启动时执行一次，`yield` 之后的在关闭时执行
        """
        app.state.server_config = cfg
        connector = aiohttp.TCPConnector(
            limit=envs.TCP_CONNECTOR_LIMIT,
            force_close=True,
        )
        """
        aiohttp.ClientSession 内部有连接池和 DNS 缓存，每个请求新建一个，等于每次都重新握手，还会泄漏文件描述符
        """
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