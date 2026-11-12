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
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.cache import delete_cache_by_key
from gpustack.server.cache import locked_cached
from gpustack.schemas.model_files import ModelFile


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

    @locked_cached()
    async def get_by_id(self, model_id: int) -> Optional[Model]:
        result = await Model.one_by_id(self.session, model_id)
        if result is not None:
            self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_name(self, name: str) -> Optional[Model]:
        result = await Model.one_by_field(self.session, "name", name)
        if result is not None:
            self.session.expunge(result)
        return result

    async def update(
        self,
        model: Model,
        source: Union[dict, SQLModel, None] = None,
        *,
        auto_commit: bool = True,
    ):
        old_name = model.name
        result = await model.update(self.session, source, auto_commit=auto_commit)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, old_name)
        if model.name != old_name:
            await delete_cache_by_key(self.get_by_name, model.name)
        return result

    async def delete(self, model: Model):
        instances = await ModelInstance.all_by_field(self.session, "model_id", model.id)
        for instance in instances:
            await instance.delete(self.session, auto_commit=False)
        result = await model.delete(self.session)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)
        instance_service = ModelInstanceService(self.session)
        await delete_cache_by_key(instance_service.get_running_instances, model.id)
        for instance in instances:
            await delete_cache_by_key(instance_service.get_by_id, instance.id)
        await ModelInstance._invalidate_cached_all()
        return result


class ModelInstanceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, id: int) -> Optional[ModelInstance]:
        result = await ModelInstance.one_by_id(self.session, id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_running_instances(self, model_id: int) -> List[ModelInstance]:
        results = await ModelInstance.all_by_fields(
            self.session,
            fields={"model_id": model_id, "state": ModelInstanceStateEnum.RUNNING},
        )
        if results is None:
            return []

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_instance):
        result = await ModelInstance.create(self.session, model_instance)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        return result

    async def update(
        self, model_instance: ModelInstance, source: Union[dict, SQLModel, None] = None
    ):
        result = await model_instance.update(self.session, source)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        await delete_cache_by_key(self.get_by_id, model_instance.id)
        return result

    async def delete(self, model_instance: ModelInstance):
        result = await model_instance.delete(self.session)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        await delete_cache_by_key(self.get_by_id, model_instance.id)
        return result

    async def batch_delete(self, model_instances: List[ModelInstance]):
        if not model_instances:
            return []

        names = [mi.name for mi in model_instances]
        try:
            for m in model_instances:
                await m.delete(self.session, auto_commit=False)
            await self.session.commit()

            # delete(auto_commit=False) returns before invalidating cached_all,
            # so the batch commit must do it. Otherwise subscribe()'s replay
            # snapshot keeps serving the deleted rows and watch reconnects
            # resurrect them as ghost CREATED events.
            await ModelInstance._invalidate_cached_all()
            for model_id in {m.model_id for m in model_instances}:
                await delete_cache_by_key(self.get_running_instances, model_id)
            for instance in model_instances:
                await delete_cache_by_key(self.get_by_id, instance.id)

            return names
        except Exception as e:
            await self.session.rollback()
            raise InternalServerErrorException(
                message=f"Failed to delete model instances {names}: {e}"
            )


class ModelFileService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_resolved_path(self, path: str) -> List[ModelFile]:
        """查找实际路径列表中包含该路径的文件记录，供本地路径复用使用"""

        results = await ModelFile.all_by_fields(
            self.session,
        )
        filtered_results = []
        for result in results:
            self.session.expunge(result)
            if path in result.resolved_paths:
                filtered_results.append(result)

        return filtered_results

    async def get_by_source_index(self, source_index: str) -> List[ModelFile]:
        """查找同一模型来源的文件记录，供文件复用和大小查询使用"""

        results = await ModelFile.all_by_field(
            self.session, "source_index", source_index
        )
        if results is None:
            return None

        for result in results:
            self.session.expunge(result)
        return results
