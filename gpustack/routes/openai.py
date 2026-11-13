from fastapi import APIRouter
from gpustack.gateway.utils import openai_model_prefixes
from fastapi import Request
from gpustack.api.exceptions import BadRequestException
from starlette.datastructures import UploadFile
from typing import Tuple
import aiohttp
from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.server.services import ModelInstanceService
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Optional
from fastapi import Response
from gpustack import envs
from gpustack.api.exceptions import GatewayTimeoutException
from gpustack.api.exceptions import InternalServerErrorException
from gpustack.api.exceptions import NotFoundException
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.gateway.utils import model_instance_prefix
from gpustack.gateway.utils import router_header_key
from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session
from gpustack.server.services import ModelService
from gpustack.server.services import WorkerService
from gpustack.server.worker_request import request_to_worker
import asyncio
import re
from typing import Dict
from typing import Union
import json
from fastapi import status
from gpustack.api.exceptions import OpenAIAPIError
from gpustack.api.exceptions import OpenAIAPIErrorResponse
from gpustack.server.worker_request import stream_to_worker
from typing import AsyncGenerator
from gpustack.schemas.models import Model
from openai.pagination import SyncPage
from openai.types import Model as OAIModel
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.http_proxy.headers import filter_response_headers
import logging


logger = logging.getLogger(__name__)


load_balancer = LoadBalancer()


_server_managed_elsewhere = {"/rerank"}


def get_api_router() -> APIRouter:
    """Full OpenAI-compatible endpoint set, mounted at /v1."""
    router = APIRouter()
    router.add_api_route("/models", list_models, methods=["GET"])
    for rp in openai_model_prefixes:
        for endpoint in rp.prefixes:
            if endpoint in _server_managed_elsewhere:
                continue
            router.add_api_route(endpoint, proxy_request_by_model, methods=["POST"])
    return router


def get_legacy_api_router() -> APIRouter:
    """Legacy subset, mounted at /v1-openai. Frozen — new endpoints go to /v1 only."""
    router = APIRouter()
    router.add_api_route("/models", list_models, methods=["GET"])
    for rp in openai_model_prefixes:
        if not rp.support_legacy:
            continue
        for endpoint in rp.prefixes:
            if endpoint in _server_managed_elsewhere:
                continue
            router.add_api_route(endpoint, proxy_request_by_model, methods=["POST"])
    return router


async def list_models(request: Request):
    async with async_session() as session:
        models = await Model.all_by_fields(session, {"deleted_at": None})
    data = [
        OAIModel(id=m.name, object="model", created=int(m.created_at.timestamp()), owned_by="gpustack")
        for m in models
        if m.ready_replicas > 0
    ]
    return SyncPage[OAIModel](data=data, object="list")


async def proxy_request_by_model(
    request: Request,
):
    """
    Proxy the request to the model instance that is running the model specified in the
    request body.

    Uses an inline session instead of SessionDep so the session is released
    after the initial lookups, preventing long-lived streaming inference
    responses from holding a database connection.
    """
    endpoint = re.sub(r"^/(v1|v1-openai)/", "", request.url.path)
    model_name, stream, body_json, form_data = await parse_request_body(request)

    async with async_session() as session:
        # 真实版本：访问控制 + 模型路由解析（ModelRouteService.resolve_route_targets），
        # 按权重选出一个 target，再取 target.model_id 对应的模型
        model = await ModelService(session).get_by_name(model_name)
        if not model:
            raise NotFoundException(
                message="Model not found",
                is_openai_exception=True,
            )
        request.state.stream = stream
        request.state.model = model

        # 找实例
        instance = await get_running_instance(session, model.id)
        worker: Worker = await WorkerService(session).get_by_id(instance.worker_id)
        if not worker:
            raise InternalServerErrorException(
                message=f"Worker with ID {instance.worker_id} not found",
                is_openai_exception=True,
            )
    extra_headers = {
        router_header_key: f"{model_instance_prefix(instance)}.static",
    }
    path = f"v1/{endpoint}"
    logger.debug(
        f"proxying to {instance.worker_ip}:{instance.port}, instance port: {instance.port}"
    )

    try:
        headers, data = _prepare_proxy_request(
            request,
            body_json,
            form_data,
            extra_headers,
            add_stream_options=stream,
        )
        if stream:
            return StreamingResponseWithStatusCode(
                _stream_response(
                    worker,
                    request.method,
                    path,
                    headers,
                    data,
                    request.app.state.http_client,
                    request.app.state.http_client_no_proxy,
                    params=request.query_params.multi_items(),
                ),
                media_type="text/event-stream",
            )
        else:
            resp, body = await request_to_worker(
                worker=worker,
                method=request.method,
                path=path,
                proxy_client=request.app.state.http_client,
                no_proxy_client=request.app.state.http_client_no_proxy,
                data=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT),
                params=request.query_params.multi_items(),
                raise_on_error=False,
            )
            response = Response(
                status_code=resp.status,
                content=body,
            )
            for key, value in filter_response_headers(resp.headers):
                response.headers.append(key, value)
            return response
    except asyncio.TimeoutError as e:
        error_message = f"Request to worker {worker.id} timed out"
        if str(e):
            error_message += f": {e}"
        raise GatewayTimeoutException(
            message=error_message,
            is_openai_exception=True,
        )
    except Exception as e:
        error_message = "An unexpected error occurred"
        if str(e):
            error_message += f": {e}"
        raise ServiceUnavailableException(
            message=error_message,
            is_openai_exception=True,
        )


async def parse_request_body(request: Request):
    model_name = None
    stream = False
    body_json = None
    form_data = None
    content_type = request.headers.get("content-type", "application/json").lower()

    if request.method == "GET":
        model_name = request.query_params.get("model")
    elif content_type.startswith("multipart/form-data"):
        form_data, model_name, stream = await parse_form_data(request)
    else:
        body_json, model_name, stream = await parse_json_body(request)

    if not model_name:
        raise BadRequestException(
            message="Missing 'model' field",
            is_openai_exception=True,
        )
    if not isinstance(model_name, str):
        raise BadRequestException(
            message="The 'model' field must be a non-empty string",
            is_openai_exception=True,
        )

    return model_name, stream, body_json, form_data


async def parse_form_data(request: Request) -> Tuple[aiohttp.FormData, str, bool]:
    form = None
    try:
        form = await request.form()
        model_name = form.get("model")
        stream_value = form.get("stream", "false")
        if not isinstance(stream_value, str) or stream_value.lower() not in {"true", "false", "1", "0"}:
            raise ValueError("The 'stream' field must be a boolean")
        stream = stream_value.lower() in {"true", "1"}

        form_data = aiohttp.FormData()
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                form_data.add_field(
                    key,
                    await value.read(),
                    filename=value.filename,
                    content_type=value.content_type,
                )
            else:
                form_data.add_field(key, value)

        return form_data, model_name, stream
    except Exception as e:
        raise BadRequestException(
            message=f"We could not parse the form body of your request: {e}",
            is_openai_exception=True,
        )
    finally:
        if form is not None:
            await form.close()


async def parse_json_body(request: Request):
    try:
        body_json = await request.json()
        model_name = body_json.get("model")
        stream = body_json.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("The 'stream' field must be a boolean")
        return body_json, model_name, stream
    except Exception as e:
        raise BadRequestException(
            message=f"We could not parse the JSON body of your request: {e}",
            is_openai_exception=True,
        )


def _prepare_proxy_request(
    request: Request,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
    extra_headers: Optional[dict] = None,
    add_stream_options: bool = False,
) -> Tuple[Dict[str, str], Optional[Union[bytes, aiohttp.FormData]]]:
    """
    Prepare headers and body for proxy requests.

    Returns (headers, data) tuple.
    """
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    if add_stream_options and body_json and "stream_options" not in body_json:
        # Defaults to include usage.
        # TODO Record usage without client awareness.
        body_json["stream_options"] = {"include_usage": True}

    # Convert body to data
    data = (
        form_data
        if form_data
        else (json.dumps(body_json).encode() if body_json else None)
    )

    # When using data=bytes (instead of json=), aiohttp doesn't set Content-Type
    if body_json and not form_data:
        headers["Content-Type"] = "application/json"

    return headers, data


async def _stream_response(
    worker: Worker,
    method: str,
    path: str,
    headers: Dict[str, str],
    data: Optional[Union[bytes, aiohttp.FormData]],
    proxy_client: aiohttp.ClientSession,
    no_proxy_client: aiohttp.ClientSession,
    params=None,
) -> AsyncGenerator[Tuple[Union[bytes, str], Dict[str, str], int], None]:
    """
    Stream response from worker. Yields (chunk, headers, status) tuples.
    """
    yielded_any = False
    upstream = stream_to_worker(
        worker=worker,
        method=method,
        path=path,
        proxy_client=proxy_client,
        no_proxy_client=no_proxy_client,
        data=data,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT),
        params=params,
        raw=True,
    )
    try:
        async for chunk, resp_headers, resp_status in upstream:
            yielded_any = True
            yield chunk, resp_headers, resp_status
    except asyncio.TimeoutError:
        error_response = OpenAIAPIErrorResponse(
            error=OpenAIAPIError(
                message="Request to worker timed out",
                code=status.HTTP_504_GATEWAY_TIMEOUT,
                type="GatewayTimeout",
            ),
        )
        yield _error_chunk(error_response, yielded_any), {}, status.HTTP_504_GATEWAY_TIMEOUT
    except aiohttp.ClientError as e:
        logger.error(f"Error streaming from worker {worker.id}: {e}", exc_info=True)
        error_response = OpenAIAPIErrorResponse(
            error=OpenAIAPIError(
                message="Service unavailable. Please retry your requests after a brief wait.",
                code=status.HTTP_503_SERVICE_UNAVAILABLE,
                type="ServiceUnavailable",
            ),
        )
        yield _error_chunk(
            error_response, yielded_any
        ), {}, status.HTTP_503_SERVICE_UNAVAILABLE
    except Exception as e:
        logger.error(f"Error streaming from worker {worker.id}: {e}", exc_info=True)
        error_response = OpenAIAPIErrorResponse(
            error=OpenAIAPIError(
                message="Internal server error.",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                type="InternalServerError",
            ),
        )
        yield _error_chunk(
            error_response, yielded_any
        ), {}, status.HTTP_500_INTERNAL_SERVER_ERROR
    finally:
        await upstream.aclose()


def _error_chunk(error_response: OpenAIAPIErrorResponse, mid_stream: bool) -> str:
    """
    Render an error body for the streaming proxy. Once the SSE stream has
    started (mid_stream), the error must be sent as an SSE data event so the
    client keeps parsing; before any chunk is sent it is returned as raw JSON
    with an error status code.
    """
    error_json = error_response.model_dump_json()
    return f"data: {error_json}\n\n" if mid_stream else error_json


def filter_headers(headers):
    return {
        key: value
        for key, value in headers.items()
        if key.lower() != "content-length"
        and key.lower() != "host"
        and key.lower() != "content-type"
        and key.lower() != "transfer-encoding"
        and key.lower() != "authorization"
        and key.lower() != router_header_key.lower()
    }


async def get_running_instance(
    session: AsyncSession,
    model_id: int,
    overridden_model_name: Optional[str] = None,
):
    """
    找实例

    Pick a RUNNING instance, narrowing by ``mounted_loras`` when a
    LoRA ``overridden_model_name`` is given. The filter is needed
    because ``mounted_loras`` is a one-shot snapshot taken at STARTING
    and never hot-reloaded
    """
    running_instances = await ModelInstanceService(session).get_running_instances(
        model_id
    )
    if not running_instances:
        raise ServiceUnavailableException(
            message="No running instances available",
            is_openai_exception=True,
        )
    if overridden_model_name:
        running_instances = [
            inst
            for inst in running_instances
            if inst.mounted_loras
            and any(m.lora_name == overridden_model_name for m in inst.mounted_loras)
        ]
        if not running_instances:
            raise ServiceUnavailableException(
                message=(
                    f"No running instances with LoRA '{overridden_model_name}' mounted. "
                    f"Restart instances after updating lora_list to apply changes."
                ),
                is_openai_exception=True,
            )
    return await load_balancer.get_instance(running_instances)
