import asyncio
import functools
import logging
import os
import ssl
from pathlib import Path
from typing import Optional, Tuple

import aiohttp
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import (
    InternalServerErrorException,
    GatewayTimeoutException,
    ServiceUnavailableException,
)
from gpustack.http_proxy.headers import filter_response_headers

"""
本文件的作用：让集群里的 Worker 帮 Server 访问 Kubernetes

假设 GPUStack Server 部署在机器 A，而 Kubernetes 集群在另一套内网里。Server 想查看 Pod、创建资源，就可能遇到两个问题：
- 访问不到：Kubernetes API 的内部地址，Server 未必能连接。
- 没有凭据：访问 Kubernetes 需要这个集群认可的身份。

可以把它理解成：Server 是总部，Worker 是驻场人员，这个文件就是驻场人员办理集群内事务的窗口。
"""

router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)


SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
TOKEN_PATH = SERVICE_ACCOUNT_DIR / "token"
CA_PATH = SERVICE_ACCOUNT_DIR / "ca.crt"

# Hop-by-hop and connection-control headers we never forward in either direction.
_REQUEST_HEADER_SKIP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
    # Drop the inbound auth — the API server is reached with the SA token below.
    "authorization",
    "x-api-key",
    "x-gpustack-model-instance",
    "cookie",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
}



_session_lock = asyncio.Lock()


def _read_token() -> str:
    """
    取出当前有效的凭证
    (Kubernetes 会定期更换 token。每次请求都重新读文件，才能使用最新凭据；如果一直使用启动时记住的旧 token，过期后就会访问失败)
    """
    return TOKEN_PATH.read_text().strip()


@functools.lru_cache(maxsize=1)
def _resolve_kube_target() -> Tuple[str, ssl.SSLContext]:
    """
    它准备两样东西：
    - Kubernetes API Server 的地址：从环境变量读取 IP、端口。
    - HTTPS 信任配置：读取集群的 CA 证书，用来验证连接的服务器是否可信。
    """

    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS") or os.environ.get(
        "KUBERNETES_SERVICE_PORT"
    )
    if not host or not port:
        raise InternalServerErrorException(
            message=(
                "Worker is not running inside a Kubernetes pod: "
                "KUBERNETES_SERVICE_HOST/PORT environment variables are not set."
            )
        )
    if not TOKEN_PATH.is_file() or not CA_PATH.is_file():
        raise InternalServerErrorException(
            message=(
                "Worker pod ServiceAccount credentials not found at "
                f"{SERVICE_ACCOUNT_DIR}; ensure automountServiceAccountToken is enabled."
            )
        )
    # IPv6 host literal needs brackets in URL.
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    base_url = f"https://{host}:{port}"
    ssl_ctx = ssl.create_default_context(cafile=str(CA_PATH))
    return base_url, ssl_ctx


async def _get_kube_session(request: Request) -> aiohttp.ClientSession:
    session: Optional[aiohttp.ClientSession] = getattr(
        request.app.state, "kube_api_session", None
    )
    if session is not None and not session.closed:
        return session

    async with _session_lock:
        session = getattr(request.app.state, "kube_api_session", None)
        if session is not None and not session.closed:
            return session

        _, ssl_ctx = _resolve_kube_target()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=64)
        # No explicit total timeout — keep watch / log streams open until the
        # client disconnects. Connect timeout still applies.
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
        )
        request.app.state.kube_api_session = session
        return session


@router.api_route(
    "/cluster-proxy/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def cluster_proxy(path: str, request: Request):
    """
    接受 Server 的命令，去集群里执行
    (这里负责理解并转交具体任务，例如查 Pod、创建资源、删除资源)
    """
    base_url, _ = _resolve_kube_target()
    target_url = f"{base_url}/{path}"

    connection_headers = {part.strip().lower() for part in request.headers.get("connection", "").split(",")}
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _REQUEST_HEADER_SKIP | connection_headers
    }
    headers["Authorization"] = f"Bearer {_read_token()}"

    # Stream the request body through to avoid buffering large payloads
    # (e.g. apply of big manifests) in worker memory.
    body = (
        request.stream() if request.method not in ("GET", "HEAD", "OPTIONS") else None
    )

    params = list(request.query_params.multi_items()) or None

    session = await _get_kube_session(request)
    try:
        resp = await session.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=body,
            params=params,
            allow_redirects=False,
        )
    except asyncio.TimeoutError as error:
        raise GatewayTimeoutException(message="Kubernetes API connection timed out") from error
    except aiohttp.ClientError as error:
        raise ServiceUnavailableException(message="Kubernetes API is unavailable") from error

    async def streamer():
        """结果到一点，就汇报一点"""

        try:
            async for chunk in resp.content.iter_any():
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception:
            # The status code and headers were already committed to the
            # caller from the upstream's initial response, so a mid-body
            # failure can no longer be turned into a non-200. Re-raise
            # instead of returning normally: letting the generator complete
            # cleanly here made StreamingResponse close out the body as if
            # it had ended successfully, so the caller received a truncated
            # payload under a 200 with no indication anything went wrong.
            raise
        finally:
            resp.close()

    response = StreamingResponse(
        streamer(),
        status_code=resp.status,
    )
    for key, value in filter_response_headers(resp.headers):
        response.headers.append(key, value)
    return response
