import logging
import random
import string
import asyncio
from typing import List
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.config.config import (
    Config,
)
from gpustack.policies.base import ModelInstanceScore
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceCreate,
    ModelInstanceStateEnum,
    get_backend,
)
from gpustack.server.bus import (
    Event,
    EventType,
    event_bus,
    event_field,
    resolve_event_id,
)
from gpustack.server.db import async_session
from gpustack.server.services import (
    ModelInstanceService,
    ModelService,
)


logger = logging.getLogger(__name__)


class ModelController:
    def __init__(self, cfg: Config):
        self._config = cfg

        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in Model.subscribe(source="model_controller"):
            if event.type == EventType.HEARTBEAT:
                continue

            await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """
        model: Model = event.data
        if not isinstance(model, Model):
            logger.warning(
                f"Model {resolve_event_id(event)} {event.type} not reconciled: "
                f"the event carries only an id and the row is gone"
            )
            return
        try:
            async with async_session() as session:
                await sync_replicas(session, model)
        except Exception as e:
            logger.error(f"Failed to reconcile model {model.name}: {e}")


class ModelInstanceController:
    def __init__(self, cfg: Config):
        self._config = cfg

        pass

    async def start(self):
        """
        Start the controller.
        """

        async for event in ModelInstance.subscribe(source="model_instance_controller"):
            if event.type == EventType.HEARTBEAT:
                continue

            await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        Reconcile the model.
        """

        model_instance: ModelInstance = event.data
        instance_id = resolve_event_id(event)
        model_id = event_field(model_instance, "model_id")
        try:
            async with async_session() as session:
                if model_id is None:
                    return
                model = await Model.one_by_id(session, model_id)
                if not model:
                    return
                model_deleting = model.deleted_at is not None

                if event.type == EventType.DELETED:
                    # trigger model replica sync, but only if model is not deleted
                    if not model_deleting:
                        copied_model = Model.model_validate(model.model_dump())
                        asyncio.create_task(
                            event_bus.publish(
                                Model.__name__.lower(),
                                Event(type=EventType.UPDATED, data=copied_model),
                            )
                        )

                if model_deleting:
                    return

                await model.refresh(session)
                replicas_updated = await sync_ready_replicas(session, model)
        except Exception as e:
            logger.error(
                "Failed to reconcile model instance "
                f"{event_field(model_instance, 'name', instance_id)}: {e}"
            )


async def sync_replicas(session: AsyncSession, model: Model):
    """
    Synchronize the replicas.
    """

    # Re-fetch model from database to ensure we have latest state
    # (event data may be from a different session or stale)
    fresh_model = await Model.one_by_id(session, model.id)
    if not fresh_model or fresh_model.deleted_at is not None:
        return
    model = fresh_model

    instances = await ModelInstance.all_by_field(session, "model_id", model.id)
    if len(instances) < model.replicas:
        for _ in range(model.replicas - len(instances)):
            name_prefix = ''.join(
                random.choices(string.ascii_lowercase + string.digits, k=5)
            )
            instance = ModelInstanceCreate(
                name=f"{model.name}-{name_prefix}",
                model_id=model.id,
                model_name=model.name,
                source=model.source,
                huggingface_repo_id=model.huggingface_repo_id,
                huggingface_filename=model.huggingface_filename,
                model_scope_model_id=model.model_scope_model_id,
                model_scope_file_path=model.model_scope_file_path,
                local_path=model.local_path,
                state=ModelInstanceStateEnum.PENDING,
                cluster_id=model.cluster_id,
                backend=get_backend(model),
                backend_version=model.backend_version,
            )

            await ModelInstanceService(session).create(instance)
            logger.debug(f"Created model instance for model {model.name}")

    elif len(instances) > model.replicas:
        # Get instances for update lock, to avoid race condition with scheduler
        instances = await ModelInstance.all_by_field(
            session, "model_id", model.id, for_update=True
        )
        candidates = await find_scale_down_candidates(instances, model)

        scale_down_count = len(candidates) - model.replicas
        if scale_down_count > 0:
            scale_down_instances = []
            for candidate in candidates[:scale_down_count]:
                scale_down_instances.append(candidate.model_instance)

            scale_down_instance_names = await ModelInstanceService(
                session
            ).batch_delete(scale_down_instances)
            if scale_down_instance_names:
                logger.debug(f"Deleted model instances: {scale_down_instance_names}")


async def find_scale_down_candidates(
    instances: List[ModelInstance],
    model: Model,
) -> List[ModelInstanceScore]:
    # 阶段版：running 的得 1 分，其余得 0 分；分低的、id 大的（新建的）排前面
    scored = [
        ModelInstanceScore(
            model_instance=instance,
            score=1 if instance.state == ModelInstanceStateEnum.RUNNING else 0,
        )
        for instance in instances
    ]
    return sorted(scored, key=lambda x: (x.score, -x.model_instance.id))


async def sync_ready_replicas(session: AsyncSession, model: Model) -> bool:
    """
    Synchronize the ready replicas.

    Returns True if the model row was updated (and the session was committed).
    """

    if model.deleted_at is not None:
        return False

    instances = await ModelInstance.all_by_field(session, "model_id", model.id)

    ready_replicas: int = 0
    for _, instance in enumerate(instances):
        if instance.state == ModelInstanceStateEnum.RUNNING:
            ready_replicas += 1

    if model.ready_replicas != ready_replicas:
        model.ready_replicas = ready_replicas
        await ModelService(session).update(model)
        return True
    return False
