from typing import Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import selectinload
from gpustack.api.exceptions import (
    ConflictException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.model_files import (
    ModelFile,
    ModelFilePublic,
    ModelFileUpdate,
    ModelFilesPublic,
)
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep


router = APIRouter()


@router.get("", response_model=ModelFilesPublic)
async def get_model_files(params: ListParamsDep, worker_id: int = None):
    fields = {"worker_id": worker_id} if worker_id else {}

    if params.watch:
        return StreamingResponse(
            ModelFile.streaming(fields=fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await ModelFile.paginated_by_query(
            session=session,
            fields=fields,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=ModelFilePublic)
async def get_model_file(session: SessionDep, id: int):
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")
    return model_file


@router.put("/{id}", response_model=ModelFilePublic)
async def update_model_file(session: SessionDep, id: int, model_file_in: ModelFileUpdate):
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    try:
        await model_file.update(session, model_file_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model file: {e}")

    return model_file


@router.delete("/{id}")
async def delete_model_file(session: SessionDep, id: int, cleanup: Optional[bool] = None):
    model_file = await ModelFile.one_by_id(
        session, id, options=[
            selectinload(ModelFile.instances), selectinload(ModelFile.draft_instances),
        ]
    )
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    if model_file.instances or model_file.draft_instances:
        model_instance_names = ", ".join(
            [model_instance.name for model_instance in model_file.instances + model_file.draft_instances]
        )
        raise ConflictException(
            message=f"Cannot delete the model file. It's being used by model instances: {model_instance_names}.",
        )

    try:
        if cleanup is not None and model_file.cleanup_on_delete != cleanup:
            model_file.cleanup_on_delete = cleanup
            await model_file.update(session)

        await model_file.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model file: {e}")
