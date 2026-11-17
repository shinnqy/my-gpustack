from fastapi import APIRouter, Depends
from sqlalchemy.orm import selectinload
from gpustack.api.exceptions import (
    ForbiddenException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.clusters import (
    WorkerPool,
    WorkerPoolPublic,
    WorkerPoolsPublic,
    WorkerPoolUpdate,
)
from gpustack.schemas.common import ListParams
from gpustack.server.deps import SessionDep


WORKER_POOL_LOAD_OPTIONS = [selectinload(WorkerPool.pool_workers)]


router = APIRouter()


@router.get("", response_model=WorkerPoolsPublic)
async def list(session: SessionDep, params: ListParams = Depends(), cluster_id: int = None):
    fields = {"deleted_at": None}
    if cluster_id:
        fields["cluster_id"] = cluster_id
    return await WorkerPool.paginated_by_query(
        session=session, fields=fields, page=params.page, per_page=params.perPage,
        options=WORKER_POOL_LOAD_OPTIONS,
    )


@router.get("/{id}", response_model=WorkerPoolPublic)
async def get(session: SessionDep, id: int):
    existing = await WorkerPool.one_by_id(session, id, options=WORKER_POOL_LOAD_OPTIONS)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")
    return existing


@router.put("/{id}", response_model=WorkerPoolPublic)
async def update(session: SessionDep, id: int, input: WorkerPoolUpdate):
    existing = await WorkerPool.one_by_id(session, id, for_update=True)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")

    try:
        await WorkerPool.update(existing, session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update worker pool {id}: {e}"
        )

    return await WorkerPool.one_by_id(session, id, options=WORKER_POOL_LOAD_OPTIONS)


@router.delete("/{id}")
async def delete(session: SessionDep, id: int):
    existing = await WorkerPool.one_by_id(session, id, options=WORKER_POOL_LOAD_OPTIONS)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"worker pool {id} not found")
    if len(existing.pool_workers) > 0:
        raise ForbiddenException(
            message=f"worker pool {id} has workers, cannot be deleted"
        )
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete worker pool: {e}")
