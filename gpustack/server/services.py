import logging
from typing import List, Optional, Union
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.api.exceptions import InternalServerErrorException
from gpustack.schemas.models import (
    Model,
    ModelInstance,
)
from gpustack.schemas.workers import Worker


logger = logging.getLogger(__name__)


class WorkerService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def update(
        self, worker: Worker, source: Union[dict, SQLModel, None] = None, **kwargs
    ):
        result = await worker.update(self.session, source, **kwargs)
        return result

    async def batch_update(
        self,
        workers: List[Worker],
        source: Union[dict, SQLModel, None] = None,
        **kwargs,
    ) -> int:
        if not workers:
            return 0

        updated = await Worker.batch_update(self.session, workers)

        return updated


class ModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def update(
        self,
        model: Model,
        source: Union[dict, SQLModel, None] = None,
        *,
        auto_commit: bool = True,
    ):
        result = await model.update(self.session, source, auto_commit=auto_commit)
        return result

    async def delete(self, model: Model):
        result = await model.delete(self.session)
        return result


class ModelInstanceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, model_instance):
        result = await ModelInstance.create(self.session, model_instance)
        return result

    async def update(
        self, model_instance: ModelInstance, source: Union[dict, SQLModel, None] = None
    ):
        result = await model_instance.update(self.session, source)
        return result

    async def delete(self, model_instance: ModelInstance):
        result = await model_instance.delete(self.session)
        return result

    async def batch_delete(self, model_instances: List[ModelInstance]):
        if not model_instances:
            return []

        names = [mi.name for mi in model_instances]
        try:
            for m in model_instances:
                await m.delete(self.session, auto_commit=False)
            await self.session.commit()

            return names
        except Exception as e:
            await self.session.rollback()
            raise InternalServerErrorException(
                message=f"Failed to delete model instances {names}: {e}"
            )
