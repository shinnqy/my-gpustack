import logging

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
        except OSError:
            # 客户端断开时交给 Starlette 处理，同时在 finally 关闭上游。
            raise
        except Exception:
            logger.exception("Error while streaming response")
            body = self._error_body(status.HTTP_500_INTERNAL_SERVER_ERROR, "InternalServerError")
            if not started:
                self.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
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
            message="Service unavailable. Please retry your requests after a brief wait."
            if code == 503 else "Internal server error.",
            code=code,
            type=error_type,
        )).model_dump_json().encode()
