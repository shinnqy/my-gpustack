from typing import Optional
import aiohttp
from contextlib import asynccontextmanager
from typing import AsyncIterator
from aiohttp.connector import UnixClientConnectorError
from aiohttp.helpers import ceil_timeout
import asyncio
import ssl

"""
本文件是发送请求时使用的通信工具
(用来把'哪些集群需要订阅，何时订阅、更新或取消'的决定转换成 HTTP 请求，发送给 Operator 网关)
"""

_BASE_URL = "https://gpustack-operator/apis"


_CONTROL_TIMEOUT = aiohttp.ClientTimeout(total=30, sock_connect=10)


_unix_path: Optional[str] = None


def set_gateway_unix_path(path: Optional[str]) -> None:
    """
    调用 wg 的客户端
    """
    global _unix_path
    _unix_path = path


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class _UnixTLSConnector(aiohttp.UnixConnector):
    """
    负责在 Unix socket 上建立 TLS 连接
    """

    def __init__(self, path: str, ssl_context: ssl.SSLContext) -> None:
        super().__init__(path=path)
        self._ssl_context = ssl_context

    async def _create_connection(self, req, traces, timeout):  # type: ignore[override]
        try:
            async with ceil_timeout(
                timeout.sock_connect, ceil_threshold=timeout.ceil_threshold
            ):
                _, proto = await self._loop.create_unix_connection(
                    self._factory,
                    self._path,
                    ssl=self._ssl_context,
                    server_hostname="localhost",
                )
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise UnixClientConnectorError(self.path, req.connection_key, exc) from exc
        return proto


@asynccontextmanager
async def _session() -> AsyncIterator[aiohttp.ClientSession]:
    if not _unix_path:
        raise RuntimeError(
            "operator worker gateway unix path is not configured; "
            "is gpustack-operator running?"
        )
    connector = _UnixTLSConnector(path=_unix_path, ssl_context=_ssl_context())
    async with aiohttp.ClientSession(connector=connector) as session:
        yield session


def _query(
    clusters: Optional[list[str]] = None,
    token: Optional[str] = None,
    namespace: Optional[str] = None,
    watch: bool = False,
    aggregated: bool = False,
    force: bool = False,
) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    for c in clusters or []:
        params.append(("cluster", c))
    if token:
        params.append(("token", token))
    if namespace:
        params.append(("namespace", namespace))
    if watch:
        params.append(("watch", "true"))
    if aggregated:
        params.append(("aggregated", "true"))
    if force:
        params.append(("force", "true"))
    return params


async def subscribe_worker(
    cluster: str,
    token: str,
    force: bool = False,
    gvk: Optional[list[tuple[str, str, str]]] = None,
) -> None:
    """POST /apis/workers — subscribe a worker cluster to the gateway."""
    async with _session() as session:
        params = _query(clusters=[cluster], token=token, force=force)
        body = {"gvks": [f"{g}/{v}/{k}" if g else f"{v}/{k}" for g, v, k in gvk or []]}
        async with session.post(
            f"{_BASE_URL}/workers", params=params, json=body, timeout=_CONTROL_TIMEOUT
        ) as resp:
            resp.raise_for_status()


async def unsubscribe_worker(
    cluster: str,
) -> None:
    """DELETE /apis/workers — unsubscribe a worker cluster from the gateway."""
    async with _session() as session:
        params = _query(clusters=[cluster])
        async with session.delete(
            f"{_BASE_URL}/workers", params=params, timeout=_CONTROL_TIMEOUT
        ) as resp:
            resp.raise_for_status()
