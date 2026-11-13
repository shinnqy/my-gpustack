from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
from pydantic import BaseModel
import httpx


logger = logging.getLogger(__name__)


HTTP_422_UNPROCESSABLE = (
    getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None)
    or status.HTTP_422_UNPROCESSABLE_ENTITY
)


class HTTPException(Exception):
    def __init__(
        self,
        status_code: int,
        reason: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        log_level: int = logging.ERROR,
    ):
        self.status_code = status_code
        self.reason = reason
        self.message = message
        self.details = details
        self.log_level = log_level


class OpenAIAPIException(HTTPException):
    pass


def http_exception_factory(
    status_code: int,
    reason: str,
    default_message: str,
):
    class_name = reason + "Exception"

    def init(
        self,
        message=default_message,
        is_openai_exception=False,
        details=None,
        *,
        log_level=logging.ERROR,
    ):
        self.is_openai_exception = is_openai_exception
        HTTPException.__init__(
            self,
            status_code, reason, message, details, log_level=log_level
        )

    return type(
        class_name,
        (HTTPException,),
        {"__init__": init},
    )


AlreadyExistsException = http_exception_factory(
    status.HTTP_409_CONFLICT, "AlreadyExists", "Already exists"
)


ConflictException = http_exception_factory(
    status.HTTP_409_CONFLICT, "Conflict", "Conflict with existing resource"
)


NotFoundException = http_exception_factory(
    status.HTTP_404_NOT_FOUND, "NotFound", "Not found"
)


UnauthorizedException = http_exception_factory(
    status.HTTP_401_UNAUTHORIZED, "Unauthorized", "Unauthorized"
)


ForbiddenException = http_exception_factory(
    status.HTTP_403_FORBIDDEN, "Forbidden", "Forbidden"
)


InvalidException = http_exception_factory(
    HTTP_422_UNPROCESSABLE, "Invalid", "Invalid input"
)


BadRequestException = http_exception_factory(
    status.HTTP_400_BAD_REQUEST, "BadRequest", "Bad request"
)


InternalServerErrorException = http_exception_factory(
    status.HTTP_500_INTERNAL_SERVER_ERROR,
    "InternalServerError",
    "Internal server error",
)


ServiceUnavailableException = http_exception_factory(
    status.HTTP_503_SERVICE_UNAVAILABLE, "ServiceUnavailable", "Service unavailable"
)


GatewayTimeoutException = http_exception_factory(
    status.HTTP_504_GATEWAY_TIMEOUT, "GatewayTimeout", "Gateway timeout"
)


async def async_raise_if_response_error(response: httpx.Response):  # noqa: C901
    if response.status_code < status.HTTP_400_BAD_REQUEST:
        return
    try:
        await response.aread()
    except httpx.ReadError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            reason="Unknown",
            message=str(e),
        )
    raise_errors(response)


def raise_if_response_error(response: httpx.Response):  # noqa: C901
    if response.status_code < status.HTTP_400_BAD_REQUEST:
        return
    raise_errors(response)


def raise_errors(response: httpx.Response):
    try:
        response_json = response.json()
        # Compatible with OpenAI API error format
        if "error" in response_json and isinstance(response_json["error"], dict):
            response_json = response_json["error"]
            if "type" in response_json and isinstance(response_json["type"], str):
                response_json["reason"] = response_json["type"]
        error = ErrorResponse.model_validate(response_json)
    except Exception:
        raise HTTPException(response.status_code, "Unknown", response.text)

    if response.status_code == status.HTTP_404_NOT_FOUND:
        raise NotFoundException(error.message)

    if (
        response.status_code == status.HTTP_409_CONFLICT
        and error.reason == "AlreadyExists"
    ):
        raise AlreadyExistsException(error.message)

    if response.status_code == status.HTTP_401_UNAUTHORIZED:
        raise UnauthorizedException(error.message)

    if response.status_code == status.HTTP_403_FORBIDDEN:
        raise ForbiddenException(error.message)

    if response.status_code == HTTP_422_UNPROCESSABLE:
        raise InvalidException(error.message)

    if response.status_code == status.HTTP_400_BAD_REQUEST:
        raise BadRequestException(error.message)

    if response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        raise InternalServerErrorException(error.message)

    if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        raise ServiceUnavailableException(error.message)

    if response.status_code == status.HTTP_504_GATEWAY_TIMEOUT:
        raise GatewayTimeoutException(error.message)

    raise HTTPException(error.code, error.reason, error.message)


class ErrorResponse(BaseModel):
    code: int
    reason: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


error_responses = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def register_handlers(app: FastAPI):
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if exc.status_code >= 500:
            logger.log(
                exc.log_level,
                "HTTP server error occurred: %s %s - %s (path=%s, method=%s)",
                exc.status_code,
                exc.reason,
                exc.message,
                request.url.path,
                request.method,
            )
        if getattr(exc, "is_openai_exception", False):
            return await openai_api_exception_handler(request, exc)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                code=exc.status_code,
                reason=exc.reason,
                message=exc.message,
                details=getattr(exc, "details", None),
            ).model_dump(),
        )

    @app.exception_handler(OpenAIAPIException)
    async def openai_api_exception_handler(request: Request, exc: OpenAIAPIException):
        """
        This handler is used to return error response in OpenAI API format.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "code": exc.status_code,
                    "type": exc.reason,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        message = f"{len(exc.errors())} validation errors:\n"
        for err in exc.errors():
            message += f"  {err}\n"
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE,
            content=ErrorResponse(
                code=HTTP_422_UNPROCESSABLE,
                reason="Invalid",
                message=message,
            ).model_dump(),
        )


class OpenAIAPIError(BaseModel):
    message: str
    type: Optional[str] = None
    code: Optional[int] = None
    param: Optional[str] = None


class OpenAIAPIErrorResponse(BaseModel):
    error: OpenAIAPIError


openai_api_error_responses = {
    404: {"model": OpenAIAPIErrorResponse},
    409: {"model": OpenAIAPIErrorResponse},
    401: {"model": OpenAIAPIErrorResponse},
    403: {"model": OpenAIAPIErrorResponse},
    422: {"model": OpenAIAPIErrorResponse},
    400: {"model": OpenAIAPIErrorResponse},
    500: {"model": OpenAIAPIErrorResponse},
    503: {"model": OpenAIAPIErrorResponse},
}
