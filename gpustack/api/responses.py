import logging
import asyncio
import aiohttp

from fastapi import status
from fastapi.responses import StreamingResponse
from starlette.types import Send

from gpustack.api.exceptions import OpenAIAPIError, OpenAIAPIErrorResponse
from gpustack.http_proxy.headers import filter_response_headers


logger = logging.getLogger(__name__)


class StreamingResponseWithStatusCode(StreamingResponse):
    """
    等上游第一块数据到达后，才确定响应头和状态码。

    生成器产出 (content, headers, status_code)，HTTP 响应只启动一次。
    """

    async def stream_response(self, send: Send) -> None:
        started = False
        try:
            try:
                content, headers, self.status_code = await anext(self.body_iterator)
            except StopAsyncIteration:
                self.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                content = self._error_body(self.status_code, "ServiceUnavailable")
                headers = {}

            response_headers = filter_response_headers(headers)
            if not any(key.lower() == "content-type" for key, _ in response_headers):
                content_type = (
                    "application/json" if self.status_code >= 400
                    else self.media_type or "application/octet-stream"
                )
                response_headers.append(("content-type", content_type))
            await send({
                "type": "http.response.start",
                "status": self.status_code,
                "headers": [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in response_headers],
            })
            started = True
            await send({
                "type": "http.response.body",
                "body": self._encode(content),
                "more_body": True,
            })
            async for content, _, _ in self.body_iterator:
                await send({
                    "type": "http.response.body",
                    "body": self._encode(content),
                    "more_body": True,
                })
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except Exception as error:
            # 发送端断开由 Starlette 处理；上游连接/超时错误仍需返回代理状态码。
            if isinstance(error, OSError) and not isinstance(error, (asyncio.TimeoutError, aiohttp.ClientError)):
                raise
            logger.exception("Error while streaming response")
            # Kubernetes watch/protobuf 使用原始字节流；中途失败应断开，不能插入 SSE 数据。
            if started and self.media_type != "text/event-stream":
                raise
            code, error_type = status.HTTP_500_INTERNAL_SERVER_ERROR, "InternalServerError"
            if isinstance(error, asyncio.TimeoutError):
                code, error_type = status.HTTP_504_GATEWAY_TIMEOUT, "GatewayTimeout"
            elif isinstance(error, aiohttp.ClientError):
                code, error_type = status.HTTP_503_SERVICE_UNAVAILABLE, "ServiceUnavailable"
            body = self._error_body(code, error_type)
            if not started:
                self.status_code = code
                await send({
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": [(b"content-type", b"application/json")],
                })
            else:
                body = b"data: " + body + b"\n\n"
            await send({"type": "http.response.body", "body": body, "more_body": False})
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close:
                await close()

    def _encode(self, content):
        return content if isinstance(content, bytes) else content.encode(self.charset)

    @staticmethod
    def _error_body(code, error_type):
        return OpenAIAPIErrorResponse(error=OpenAIAPIError(
            message={503: "Service unavailable. Please retry your requests after a brief wait.",
                     504: "Gateway request timed out."}.get(code, "Internal server error."),
            code=code,
            type=error_type,
        )).model_dump_json().encode()
