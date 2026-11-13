from fastapi import Depends
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
from gpustack.api.exceptions import UnauthorizedException
from gpustack.config.config import Config
from typing import Annotated
from typing import Optional
from fastapi.security import HTTPBearer

"""使用 auth 保护 Worker 代理入口"""

bearer_auth = HTTPBearer(auto_error=False)


async def worker_auth(
    request: Request,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
):
    token_value = bearer_token.credentials if bearer_token else None
    if not token_value:
        raise UnauthorizedException(message="Invalid authentication credentials")
    token = request.app.state.token
    config: Config = request.app.state.config
    registration_token = config.token
    if token_value in [token, registration_token]:
        return
    raise UnauthorizedException(message="Invalid authentication credentials")
