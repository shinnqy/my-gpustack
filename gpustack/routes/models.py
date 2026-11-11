from fastapi import APIRouter, Depends
from sqlalchemy.orm import selectinload
from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.common import ListParams
from gpustack.schemas.models import (
    Model,
    ModelCreate,
    ModelPublic,
    ModelUpdate,
    ModelsPublic,
)
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep
from gpustack.schemas.clusters import Cluster


router = APIRouter()


@router.get("", response_model=ModelsPublic)
async def get_models(params: ListParams = Depends()):
    async with async_session() as session:
        return await Model.paginated_by_query(
            session=session,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=ModelPublic)
async def get_model(session: SessionDep, id: int):
    model = await Model.one_by_id(session, id, options=[selectinload(Model.instances)])
    if not model:
        raise NotFoundException(message="Model not found")
    return model


@router.post("", response_model=ModelPublic)
async def create_model(session: SessionDep, model_in: ModelCreate):
    existing = await Model.one_by_field(session, "name", model_in.name)
    if existing:
        raise AlreadyExistsException(
            message=f"Model with name '{model_in.name}' already exists."
        )

    if model_in.cluster_id is None:
        default_cluster = await Cluster.one_by_fields(
            session, {"is_default": True, "deleted_at": None}
        )
        if default_cluster is None:
            raise BadRequestException(message="cluster_id is required")
        model_in.cluster_id = default_cluster.id
    elif await Cluster.one_by_id(session, model_in.cluster_id) is None:
        raise NotFoundException(message=f"Cluster {model_in.cluster_id} not found")

    try:
        model = await Model.create(session, source=model_in)
    except BadRequestException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to create model: {e}")

    return model


@router.put("/{id}", response_model=ModelPublic)
async def update_model(session: SessionDep, id: int, model_in: ModelUpdate):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    try:
        await model.update(session, model_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model: {e}")

    return model


@router.delete("/{id}")
async def delete_model(session: SessionDep, id: int):
    model = await Model.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    try:
        await model.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model: {e}")
