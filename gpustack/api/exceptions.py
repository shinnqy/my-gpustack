from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
from pydantic import BaseModel


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
        if is_openai_exception:
            self.__class__.__bases__ = (OpenAIAPIException,)
        super(self.__class__, self).__init__(
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
NotFoundException = http_exception_factory(
    status.HTTP_404_NOT_FOUND, "NotFound", "Not found"
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
ConflictException = http_exception_factory(
    status.HTTP_409_CONFLICT, "Conflict", "Conflict with existing resource"
)


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