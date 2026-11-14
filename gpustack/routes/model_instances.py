from typing import Optional
from typing import Any, Dict
import asyncio
from weakref import WeakValueDictionary
from pydantic import ValidationError
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from gpustack.api.exceptions import (
    InternalServerErrorException,
    NotFoundException,
    BadRequestException,
    ConflictException,
)
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceCreate,
    ModelInstancePublic,
    ModelInstanceUpdate,
    ModelInstancesPublic,
    ModelInstanceSubordinateWorker,
)
from gpustack.utils.attrs import set_attr
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.server.services import ModelInstanceService


router = APIRouter()
_patch_locks = WeakValueDictionary()


@router.get("", response_model=ModelInstancesPublic)
async def get_model_instances(
    params: ListParamsDep,
    id: Optional[int] = None,
    model_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    state: Optional[str] = None,
    search: Optional[str] = None,
):
    fields = {}
    search = search.strip() if search else None
    fuzzy_fields = {"name": search} if search else {}
    if id:
        fields["id"] = id

    if model_id:
        fields["model_id"] = model_id

    if worker_id:
        fields["worker_id"] = worker_id

    if state:
        fields["state"] = state

    if params.watch:
        return StreamingResponse(
            ModelInstance.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await ModelInstance.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=ModelInstancePublic)
async def get_model_instance(session: SessionDep, id: int):
    model_instance = await ModelInstance.one_by_id(session, id)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")
    return model_instance


@router.post("", response_model=ModelInstancePublic)
async def create_model_instance(
    session: SessionDep, model_instance_in: ModelInstanceCreate
):
    try:
        model_instance = await ModelInstanceService(session).create(model_instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create model instance: {e}"
        )
    return model_instance


@router.put("/{id}", response_model=ModelInstancePublic)
async def update_model_instance(
    session: SessionDep, id: int, model_instance_in: ModelInstanceUpdate
):
    model_instance = await ModelInstance.one_by_id(session, id, for_update=True)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")

    try:
        await ModelInstanceService(session).update(model_instance, model_instance_in)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update model instance: {e}"
        )
    return model_instance


@router.patch("/{id}", response_model=ModelInstancePublic)
async def patch_model_instance(
    session: SessionDep, id: int, updates: Dict[str, Any],
    expected_restart_count: Optional[int] = None,
):
    """
    IMPORTANT:

    状态更新：新增局部 PATCH，避免多台 Worker 互相覆盖。

        - 多台 Worker 同时修改同一条实例记录后，原来的“读取整个对象、修改、整条 PUT 回去”容易产生覆盖:
            A 读取旧记录
            B 更新自己的状态
            A 把旧记录整条写回 → B 的新状态被覆盖
        - 解决方法：
            主 Worker 更新：
            state、port、pid 等主实例字段

            从 Worker 更新：
            distributed_servers.subordinate_workers.0.state
            distributed_servers.subordinate_workers.0.pid

    这部分解决的是多机状态一致性，Ray 本身不会替 GPUStack 维护这些数据库记录
    """
    # 主从 Worker 只回写自己的字段；锁内重新读取，避免整条记录互相覆盖。
    lock = _patch_locks.setdefault(id, asyncio.Lock())
    async with lock:
        instance = await ModelInstance.one_by_id(session, id, for_update=True)
        if instance is None:
            raise NotFoundException(message="Model instance not found")
        if expected_restart_count is not None and (instance.restart_count or 0) != expected_restart_count:
            raise ConflictException(message="Model instance has entered a newer restart generation")
        data = instance.model_dump(round_trip=True)
        for key, value in updates.items():
            parts = key.split(".")
            if len(parts) == 1:
                valid = key in ModelInstanceUpdate.model_fields
            else:
                valid = (
                    len(parts) in {3, 4}
                    and parts[:2] == ["distributed_servers", "subordinate_workers"]
                    and parts[2].isdigit()
                    and instance.distributed_servers is not None
                    and int(parts[2]) < len(instance.distributed_servers.subordinate_workers or [])
                    and (len(parts) == 3 or parts[3] in ModelInstanceSubordinateWorker.model_fields)
                )
            if not valid:
                raise BadRequestException(message=f"Invalid model instance update field: {key}")
            set_attr(data, parts, value)
        try:
            update = ModelInstanceUpdate.model_validate(data)
        except ValidationError as error:
            raise BadRequestException(message=f"Invalid model instance update: {error}") from error
        await ModelInstanceService(session).update(instance, update)
        return instance


@router.delete("/{id}")
async def delete_model_instance(session: SessionDep, id: int):
    model_instance = await ModelInstance.one_by_id(session, id, for_update=True)
    if not model_instance:
        raise NotFoundException(message="Model instance not found")

    try:
        await ModelInstanceService(session).delete(model_instance)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete model instance: {e}"
        )
