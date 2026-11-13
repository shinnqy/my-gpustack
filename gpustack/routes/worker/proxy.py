from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from gpustack.api.exceptions import ErrorResponse
from gpustack.api.exceptions import NotFoundException
from gpustack.api.exceptions import HTTPException as APIException
from gpustack.gateway.utils import router_header_key
from gpustack.gateway.utils import get_instance_id_from_header
from typing import Optional
from typing import Tuple
from fastapi.responses import StreamingResponse
from gpustack import envs
from gpustack.api.exceptions import GatewayTimeoutException
from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.utils.network import use_proxy_env_for_url
from starlette.background import BackgroundTask
from typing import Callable
import aiohttp
import asyncio
from typing import List
import logging
from fastapi import APIRouter
from fastapi import Depends
from gpustack.api.auth import bearer_auth, worker_auth
from gpustack.http_proxy.headers import filter_response_headers

"""
完成 Worker → 推理引擎这一跳
    - 承担实际转发工作的主体
"""

router = APIRouter(dependencies=[Depends(worker_auth)])


logger = logging.getLogger(__name__)


def _filter_response_headers(resp_headers) -> List[Tuple[str, str]]:
    # Return a list of tuples (not a dict) so multi-value headers such as
    # Set-Cookie are preserved; aiohttp's CIMultiDictProxy emits one item per
    # occurrence.
    return filter_response_headers(resp_headers)


@router.api_route(
    "/proxy/{path:path}",
    methods=["GET", "POST", "OPTIONS", "HEAD"],
)
async def proxy(path: str, request: Request):  # noqa: C901
    worker_ip_getter: Callable[[], str] = request.app.state.worker_ip_getter
    if worker_ip_getter is None:
        worker_ip_getter = localhost_fallback
    target_service_port = getattr(request.state, "x_target_port", None)
    if not target_service_port:
        raise HTTPException(
            status_code=400,
            detail="Missing target port; ensure the request includes the routing header",
        )

    resp = None
    try:
        logger.debug(
            f"Proxying request to worker at port {target_service_port} for path: {path}"
        )
        url = f"http://{worker_ip_getter()}:{target_service_port}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = dict(request.headers)
        chunked = "chunked" in headers.get("transfer-encoding", "").lower()
        for key in ("host", "transfer-encoding", "authorization", router_header_key.lower()):
            headers.pop(key, None)
        if chunked:

            async def body_generator():
                async for chunk in request.stream():
                    yield chunk

            content = body_generator()
        else:
            content = await request.body()

        async def stream_response(resp):
            try:
                async for chunk in resp.content.iter_chunked(1024):
                    yield chunk
            finally:
                resp.close()

        use_proxy_env = use_proxy_env_for_url(url)
        http_client: aiohttp.ClientSession = (
            request.app.state.http_client
            if use_proxy_env
            else request.app.state.http_client_no_proxy
        )
        timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT)
        resp = await http_client.request(
            method=request.method,
            url=url,
            headers=headers,
            data=content,
            timeout=timeout,
        )

        # Heuristic: treat a non-error HTTP status as a successful inference
        # signal so the active health-check loop can skip this instance.
        # For streaming responses the status is available before body
        # transfer, so a mid-stream failure will still be counted — this is
        # acceptable as a best-effort optimisation.
        target_instance_id = getattr(request.state, "x_target_instance_id", None)
        if resp.status < 400 and target_instance_id:
            record_fn = getattr(request.app.state, "record_successful_inference", None)
            if record_fn:
                record_fn(int(target_instance_id))

        response = StreamingResponse(
            stream_response(resp),
            status_code=resp.status,
            background=BackgroundTask(resp.close),
        )
        # Use append (not the headers= constructor kwarg) so duplicate header
        # names like Set-Cookie survive instead of being overwritten by
        # Starlette's MutableHeaders.update.
        for k, v in _filter_response_headers(resp.headers):
            response.headers.append(k, v)
        return response

    except asyncio.TimeoutError as e:
        if resp is not None:
            resp.close()
        error_message = f"Request to {url} timed out"
        if str(e):
            error_message += f": {e}"
        raise GatewayTimeoutException(
            message=error_message,
            is_openai_exception=True,
        )
    except Exception as e:
        if resp is not None:
            resp.close()
        error_message = "An unexpected error occurred"
        if str(e):
            error_message += f": {e}"
        raise ServiceUnavailableException(
            message=error_message,
            is_openai_exception=True,
        )


def localhost_fallback() -> str:
    return "127.0.0.1"


def get_model_instance_info_from_model_name(
    request: Request,
) -> Tuple[int, int]:
    """
    Get model instance port and instance id from model name in header
    "x-gpustack-model-instance".

    Return (port, model_instance_id).
    """
    model_instance_id = get_instance_id_from_header(request.headers)
    port: Optional[int] = request.app.state.get_instance_port_by_model_instance_id(
        model_instance_id
    )
    if not port:
        raise NotFoundException(
            message=f"No running model instance found for model name: {model_instance_id}",
        )
    logger.debug(f"Found port {port} from model instance id {model_instance_id}")
    return port, model_instance_id


async def set_port_from_model_name(request: Request, call_next):
    """
    中间件：先验证 Worker Token，再解析 X-GPUStack-Model-Instance 路由头
    """

    model_name = request.headers.get(router_header_key, None)
    inference_path = request.url.path.startswith(("/v1/", "/v1-openai/", "/proxy/"))
    if model_name is None and not inference_path:
        return await call_next(request)
    try:
        # 先验证令牌，再解析路由，避免未认证请求读取实例信息。
        await worker_auth(request, await bearer_auth(request))
        port, model_instance_id = get_model_instance_info_from_model_name(request)
        if not request.url.path.startswith("/proxy/"):
            request.scope["path"] = f"/proxy{request.url.path}"
        request.state.x_target_port = str(port)
        request.state.x_target_instance_id = model_instance_id
        return await call_next(request)
    except APIException as e:
        logger.debug("failed to find model instance for proxying: %s", e.message)
        return JSONResponse(
            status_code=e.status_code,
            content=ErrorResponse(
                code=e.status_code,
                reason=e.reason,
                message=e.message,
            ).model_dump(),
        )
    except HTTPException as e:
        logger.debug("failed to find model instance for proxying: %s", e.detail)
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
