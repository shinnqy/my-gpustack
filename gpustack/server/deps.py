from typing import Annotated
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.server.db import get_session
from gpustack.schemas.common import ListParams

"""这里的意思是：类型是 AsyncSession，值由 `get_session` 提供"""
SessionDep = Annotated[AsyncSession, Depends(get_session)]
ListParamsDep = Annotated[ListParams, Depends(ListParams)]