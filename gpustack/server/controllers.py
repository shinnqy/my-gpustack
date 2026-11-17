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
from gpustack.server.cache import delete_cache_by_key
from gpustack.server.services import (
    ModelInstanceService,
    ModelService,
)
from gpustack import envs
from gpustack.policies.scorers.offload_layer_scorer import OffloadLayerScorer
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.policies.scorers.placement_scorer import ScaleTypeEnum
from gpustack.policies.scorers.score_chain import ModelInstanceScoreChain
from gpustack.policies.scorers.status_scorer import StatusScorer
from typing import Optional
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.model_files import ModelFileStateEnum
from sqlalchemy.orm import selectinload
from gpustack.schemas.models import ModelSource
from gpustack.schemas.models import SourceEnum
from gpustack.server.services import ModelFileService
from gpustack.schemas.links import ModelInstanceModelFileLink
from sqlmodel import select
from gpustack.schemas.links import ModelInstanceDraftModelFileLink
from sqlalchemy.orm.attributes import flag_modified
from gpustack.schemas.models import ModelInstanceSubordinateWorker
from gpustack.schemas.workers import Worker
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.utils.model_instance_workers import get_model_instance_worker_match
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.clusters import ClusterStateEnum
from gpustack.schemas.config import GatewayModeEnum
from gpustack.schemas.clusters import WorkerPool
from gpustack.schemas.workers import WorkerStatus
from gpustack.cloud_providers.abstract import CloudInstance
from gpustack.cloud_providers.abstract import InstanceProvisioningFailed
from gpustack.cloud_providers.abstract import InstanceState
from gpustack.cloud_providers.abstract import ProviderClientBase
from gpustack.cloud_providers.common import construct_cloud_instance
from gpustack.cloud_providers.common import generate_ssh_key_pair
from gpustack.cloud_providers.common import get_client_from_provider
from gpustack.config.config import get_cluster_image_name
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.schemas.clusters import CloudCredential
from gpustack.schemas.clusters import Credential
from gpustack.schemas.clusters import CredentialType
from gpustack.schemas.clusters import SSHKeyOptions
from gpustack.schemas.config import SensitivePredefinedConfig
from gpustack.server.services import WorkerService
from typing import Tuple
from typing import Any
from typing import Dict


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
                if event.type == EventType.DELETED:
                    service = ModelInstanceService(session)
                    if instance_id is not None:
                        await delete_cache_by_key(service.get_by_id, instance_id)
                    if model_id is not None:
                        await delete_cache_by_key(service.get_running_instances, model_id)
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
                elif model_instance.state == ModelInstanceStateEnum.INITIALIZING:
                    """这时候 Worker 真正接手了，才值得开始下载"""
                    await ensure_instance_model_file(session, model_instance)
                    return

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


async def _link_instance_primary_model_files(
    session: AsyncSession, instance_id: int, files: List[ModelFile]
):
    """
    补上实例与主模型文件之间缺失的关联记录，交给调用者统一提交
    """
    for f in files:
        if f.id is None:
            continue
        stmt = select(ModelInstanceModelFileLink).where(
            ModelInstanceModelFileLink.model_instance_id == instance_id,
            ModelInstanceModelFileLink.model_file_id == f.id,
        )
        if (await session.exec(stmt)).first() is None:
            session.add(
                ModelInstanceModelFileLink(
                    model_instance_id=instance_id, model_file_id=f.id
                )
            )


async def _link_instance_draft_model_files(
    session: AsyncSession, instance_id: int, files: List[ModelFile]
):
    """
    补上实例与草稿模型文件之间缺失的关联记录
    """
    for f in files:
        if f.id is None:
            continue
        stmt = select(ModelInstanceDraftModelFileLink).where(
            ModelInstanceDraftModelFileLink.model_instance_id == instance_id,
            ModelInstanceDraftModelFileLink.model_file_id == f.id,
        )
        if (await session.exec(stmt)).first() is None:
            session.add(
                ModelInstanceDraftModelFileLink(
                    model_instance_id=instance_id, model_file_id=f.id
                )
            )


def _is_primary_instance_model_file(
    file: ModelFile, instance: ModelInstance, is_draft_model: bool
) -> bool:
    """排除草稿和 LoRA 文件，判断文件是否属于主模型。"""
    if is_draft_model:
        return False
    if file.is_lora:
        return False
    return True


async def ensure_instance_model_file(session: AsyncSession, instance: ModelInstance):
    """
    整个准备模型文件过程的入口
        - 为每台从属机器也建一条 ModelFile，各自下载
    """
    if instance.worker_id is None:
        # Not scheduled yet
        return

    instance = await ModelInstance.one_by_id(
        session,
        instance.id,
        options=[
            selectinload(ModelInstance.model_files),
            selectinload(ModelInstance.draft_model_files),
        ],
    )
    if not instance:
        return

    retry_model_files = []
    # 为每个需要下载的 Worker 查找文件记录，缺少时创建 DOWNLOADING 记录。
    model_files = await get_or_create_model_files_for_instance(session, instance)
    draft_model_files = []
    if instance.draft_model_source:
        # 为每个需要下载的 Worker 查找文件记录，缺少时创建 DOWNLOADING 记录。
        draft_model_files = await get_or_create_model_files_for_instance(
            session, instance, is_draft_model=True
        )
    lora_model_files: List[ModelFile] = []

    for model_file in model_files + draft_model_files + lora_model_files:
        if model_file.state == ModelFileStateEnum.ERROR:
            # Retry the download
            retry_model_files.append(model_file.readable_source)

            model_file.state = ModelFileStateEnum.DOWNLOADING
            model_file.download_progress = 0
            model_file.state_message = ""
            await model_file.update(session, auto_commit=False)

    # 补上实例与主模型文件之间缺失的关联记录，交给调用者统一提交
    await _link_instance_primary_model_files(
        session, instance.id, model_files + lora_model_files
    )
    # 补上实例与草稿模型文件之间缺失的关联记录
    await _link_instance_draft_model_files(session, instance.id, draft_model_files)
    # Commit file creation, retry resets, and instance <--> file links in one
    # transaction.  ModelFile events are published by the after_commit hook,
    # so ModelFileController._reconcile will only see these files after the
    # links already exist — preventing a race where reconcile queries
    # file.instances before the links are committed.
    await session.commit()

    await ModelFile._invalidate_cached_all()

    if retry_model_files:
        logger.info(
            f"Retrying download for model files {retry_model_files} for model instance {instance.name}"
        )

    instance = await ModelInstance.one_by_id(session, instance.id)
    await sync_instance_files_state(
        session,
        instance,
        model_files + draft_model_files + lora_model_files,
    )


async def get_or_create_model_files_for_instance(
    session: AsyncSession, instance: ModelInstance, is_draft_model: bool = False
) -> List[ModelFile]:
    """
    为每个需要下载的 Worker 查找文件记录，缺少时创建 DOWNLOADING 记录。
    """

    model_files = await get_model_files_for_instance(session, instance, is_draft_model)
    worker_ids = _get_worker_ids_for_file_download(instance)

    # Return early if all model files are already created for the workers
    if len(model_files) == len(worker_ids):
        return model_files

    # Get the worker IDs that are missing model files.
    missing_worker_ids = set(worker_ids) - {
        model_file.worker_id for model_file in model_files
    }
    if not missing_worker_ids:
        return model_files

    model_source = instance.draft_model_source if is_draft_model else instance
    # Create model files for the missing worker IDs.
    for worker_id in missing_worker_ids:
        model_file = ModelFile(
            source=model_source.source,
            huggingface_repo_id=model_source.huggingface_repo_id,
            huggingface_filename=model_source.huggingface_filename,
            model_scope_model_id=model_source.model_scope_model_id,
            model_scope_file_path=model_source.model_scope_file_path,
            local_path=model_source.local_path,
            state=ModelFileStateEnum.DOWNLOADING,
            worker_id=worker_id,
            source_index=model_source.model_source_index,
        )
        await ModelFile.create(session, model_file, auto_commit=False)
        logger.info(
            f"Created model file for model instance {instance.name} and worker {worker_id}"
        )

    # After creating the model files, fetch them again to return the complete list.
    return await get_model_files_for_instance(session, instance, is_draft_model)


async def get_model_files_for_instance(
    session: AsyncSession, instance: ModelInstance, is_draft_model: bool = False
) -> List[ModelFile]:
    """
    按来源标识查文件，再筛选实例所使用的 Worker
    """
    worker_ids = _get_worker_ids_for_file_download(instance)

    model_source: ModelSource = instance
    if is_draft_model:
        model_source = instance.draft_model_source

    model_files = await ModelFileService(session).get_by_source_index(
        model_source.model_source_index
    )
    model_files = [
        model_file for model_file in model_files if model_file.worker_id in worker_ids
    ]

    if model_source.source == SourceEnum.LOCAL_PATH and model_source.local_path:
        # 查找实际路径列表中包含该路径的文件记录，供本地路径复用使用
        local_path_model_files = await ModelFileService(session).get_by_resolved_path(
            model_source.local_path
        )
        local_path_model_files = [
            model_file
            for model_file in local_path_model_files
            if model_file.worker_id in worker_ids
        ]
        existing_worker_ids = {mf.worker_id for mf in model_files}
        additional_files = [
            model_file
            for model_file in local_path_model_files
            if model_file.worker_id not in existing_worker_ids
        ]
        model_files.extend(additional_files)

    return model_files


async def find_scale_down_candidates(
    instances: List[ModelInstance],
    model: Model,
    *,
    status_max_score: Optional[float] = None,
    offload_max_score: Optional[float] = None,
    placement_max_score: Optional[float] = None,
    total_max_score: Optional[float] = None,
) -> List[ModelInstanceScore]:
    """缩容时，三个打分器，分数越低越先删"""
    try:
        if status_max_score is None:
            status_max_score = envs.SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE
        if offload_max_score is None:
            offload_max_score = envs.SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE
        if placement_max_score is None:
            placement_max_score = envs.SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE

        # 按权重归一化后相加
        chain = ModelInstanceScoreChain(
            scorers=[
                StatusScorer(model, max_score=status_max_score),
                OffloadLayerScorer(model, max_score=offload_max_score),
                PlacementScorer(
                    model,
                    instances,
                    scale_type=ScaleTypeEnum.SCALE_DOWN,
                    max_score=placement_max_score,
                ),
            ],
            # 把总分缩放到指定范围
            total_max_score=total_max_score,
        )
        final_candidates = await chain.score(instances)
        final_candidates = sorted(
            final_candidates, key=lambda x: x.score, reverse=False
        )
        return final_candidates
    except Exception as e:
        state_message = (
            f"Failed to find scale down candidates for model {model.name}: {e}"
        )
        logger.error(state_message)
        return []


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


class WorkerController:
    def __init__(self, cfg: Config):
        self._config = cfg
        self._provisioning = WorkerProvisioningController(cfg)

    async def start(self):
        """
        Start the controller.
        """

        async for event in Worker.subscribe(source="worker_controller"):
            if event.type == EventType.HEARTBEAT:
                continue
            if not isinstance(event.data, Worker):
                continue
            try:
                await self._reconcile(event)
                await self._provisioning._reconcile(event)
                await self._notify_relatives(event)
            except Exception as e:
                logger.error(f"Failed to reconcile worker: {e}")

    async def _reconcile(self, event: Event):
        """
        Delete instances base on the worker state and event type.
        """
        if event.type not in (EventType.UPDATED, EventType.DELETED):
            return
        worker: Worker = event.data
        if not worker:
            return

        if worker.state.is_provisioning and worker.state != WorkerStateEnum.DELETING:
            # Skip reconciliation for provisioning and deleting workers.
            # There is a dedicated controller to handle provisioning.
            return

        if event.type == EventType.UPDATED:
            changed_fields = event.changed_fields
            if not changed_fields or "state" not in changed_fields:
                # No state change
                return

        async with async_session() as session:
            all_instances = await ModelInstance.all_by_field(
                session, "cluster_id", worker.cluster_id
            )
            if not all_instances:
                return
            matched_instances = []
            for instance in all_instances:
                match = get_model_instance_worker_match(
                    instance,
                    worker_name=worker.name,
                    worker_id=worker.id,
                )
                if match.matched:
                    matched_instances.append((instance, match))
            if not matched_instances:
                return

            if event.type == EventType.DELETED:
                instance_names = await ModelInstanceService(session).batch_delete(
                    [instance for instance, _ in matched_instances]
                )
                if instance_names:
                    logger.info(
                        f"Delete instance {', '.join(instance_names)} "
                        f"since worker {worker.name} is deleted"
                    )
                return

            if (
                worker.unreachable
                or worker.state == WorkerStateEnum.UNREACHABLE
                or worker.state == WorkerStateEnum.NOT_READY
            ):
                await self.update_impacted_instance_states_to_unreachable(
                    session,
                    matched_instances,
                    worker.name,
                )
                return

    async def update_impacted_instance_states_to_unreachable(
        self,
        session,
        matched_instances,
        worker_name,
    ):
        instance_names = set()
        subordinate_worker_names = set()
        for instance, match in matched_instances:
            patch = {}
            distributed_servers_changed = False
            if (
                match.is_main_worker
                and instance.state == ModelInstanceStateEnum.RUNNING
            ):
                patch["state"] = ModelInstanceStateEnum.UNREACHABLE
                patch["state_message"] = "Worker is unreachable from the server"
                instance_names.add(instance.name)

            for index in match.subordinate_worker_indexes:
                subordinate_worker = instance.distributed_servers.subordinate_workers[
                    index
                ]
                if subordinate_worker.state == ModelInstanceStateEnum.UNREACHABLE:
                    continue
                subordinate_worker.state = ModelInstanceStateEnum.UNREACHABLE
                subordinate_worker.state_message = (
                    "Worker is unreachable from the server"
                )
                subordinate_worker_names.add(
                    f"{instance.name}:{subordinate_worker.worker_name}"
                )
                distributed_servers_changed = True

            if distributed_servers_changed:
                patch["distributed_servers"] = instance.distributed_servers
                flag_modified(instance, "distributed_servers")

            if patch:
                await ModelInstanceService(session).update(instance, patch)
        if instance_names:
            logger.info(
                f"Marked instance {', '.join(instance_names)} unreachable "
                f"since worker {worker_name} is unreachable from the server"
            )
        if subordinate_worker_names:
            logger.info(
                f"Marked subordinate workers {', '.join(subordinate_worker_names)} unreachable "
                f"since worker {worker_name} is unreachable from the server"
            )

    async def _notify_relatives(self, event: Event):
        if event.type not in (EventType.UPDATED, EventType.DELETED):
            return
        worker: Worker = event.data
        changed_fields = event.changed_fields
        if not worker or (not changed_fields and event.type != EventType.DELETED):
            return
        state_changed: Optional[Tuple[Any, Any]] = (changed_fields or {}).get(
            "state", None
        )
        proxy_mode_changed: Optional[Tuple[Any, Any]] = (changed_fields or {}).get(
            "proxy_mode", None
        )
        should_notify_parents = (
            state_changed is not None
            or proxy_mode_changed is not None
            or event.type == EventType.DELETED
        )
        proxy_address_changed: Optional[Tuple[Any, Any]] = (changed_fields or {}).get(
            "proxy_address", None
        )
        should_notify_children = (
            proxy_address_changed is not None or proxy_mode_changed is not None
        )

        if not should_notify_parents and not should_notify_children:
            return
        async with async_session() as session:
            if should_notify_parents and worker.worker_pool_id is not None:
                worker_pool = await WorkerPool.one_by_id(
                    session,
                    worker.worker_pool_id,
                    options=[selectinload(WorkerPool.pool_workers)],
                )
                if worker_pool is not None:
                    copied_pool = WorkerPool(**worker_pool.model_dump())
                    await event_bus.publish(
                        copied_pool.__class__.__name__.lower(),
                        Event(
                            type=EventType.UPDATED,
                            data=copied_pool,
                        ),
                    )
            if should_notify_parents and worker.cluster_id is not None:
                cluster = await Cluster.one_by_id(
                    session,
                    worker.cluster_id,
                    options=[
                        selectinload(Cluster.cluster_workers),
                        selectinload(Cluster.cluster_models),
                    ],
                )
                if cluster is not None:
                    copied_cluster = Cluster(**cluster.model_dump())
                    await event_bus.publish(
                        copied_cluster.__class__.__name__.lower(),
                        Event(
                            type=EventType.UPDATED,
                            data=copied_cluster,
                        ),
                    )

            if should_notify_children:
                instances = await ModelInstance.all_by_fields(
                    session,
                    fields={"worker_id": worker.id},
                    options=[selectinload(ModelInstance.model)],
                )
                notified_model = set()
                for instance in instances:
                    if instance.model_id in notified_model:
                        continue
                    notified_model.add(instance.model_id)
                    copied_model = Model(**instance.model.model_dump())
                    await event_bus.publish(
                        copied_model.__class__.__name__.lower(),
                        Event(
                            type=EventType.UPDATED,
                            data=copied_model,
                        ),
                    )


class ModelFileController:
    """
    把文件变化传播给实例
    """

    async def start(self):
        """
        Start the controller.
        """

        async for event in ModelFile.subscribe(source="model_file_controller"):
            if event.type == EventType.CREATED or event.type == EventType.UPDATED:
                await self._reconcile(event)

    async def _reconcile(self, event: Event):
        """
        查询文件最新状态及引用它的实例，将文件变化同步给这些实例。
        """

        file: ModelFile = event.data
        try:
            async with async_session() as session:
                file = await ModelFile.one_by_id(
                    session,
                    file.id,
                    options=[
                        selectinload(ModelFile.instances),
                        selectinload(ModelFile.draft_instances),
                    ],
                )

            if not file:
                # In case the file is deleted
                return

            for instance in file.instances + file.draft_instances:
                async with async_session() as session:
                    await sync_instance_files_state(session, instance, [file])
        except Exception as e:
            logger.error(f"Failed to reconcile model file {file.id}: {e}")


async def sync_instance_files_state(
    session: AsyncSession, instance: ModelInstance, files: List[ModelFile]
):
    """状态同步分发入口：区分主 Worker、从属 Worker，以及主模型和草稿文件，调用对应处理函数。"""

    for file in files:
        if file.worker_id == instance.worker_id:
            is_draft_model = _is_draft_model_file(file, instance)
            if is_draft_model:
                await sync_main_worker_model_file_state(
                    session, file, instance, is_draft_model=True
                )
            else:
                await sync_main_worker_model_file_state(session, file, instance)
        else:
            await sync_distributed_model_file_state(session, file, instance)


def _is_draft_model_file(file: ModelFile, instance: ModelInstance) -> bool:
    """
    根据来源标识或本地实际路径，判断文件是否属于实例的草稿模型。
    """
    if not instance.draft_model_source:
        return False

    if file.model_source_index == instance.draft_model_source.model_source_index:
        return True

    # The model uses a local path as its draft source, but the model file may come from a remote source.
    # Match by resolved path.
    if (
        instance.draft_model_source.source == SourceEnum.LOCAL_PATH
        and file.resolved_paths
        and file.resolved_paths[0] == instance.draft_model_source.local_path
    ):
        return True

    return False


def _aggregate_instance_download_progress(
    instance: ModelInstance,
    current_file: ModelFile,
    override_progress: Optional[float] = None,
    override_state: Optional[ModelFileStateEnum] = None,
) -> Optional[float]:
    """
    计算主 Worker 尚未 READY 的关联文件的平均进度，全部就绪时返回 100
    """
    # Main worker only; subordinate progress is tracked per-worker elsewhere.
    files = [f for f in instance.model_files or [] if f.worker_id == instance.worker_id]
    if not files:
        return None
    active_values: List[float] = []
    for f in files:
        if current_file.id is not None and f.id == current_file.id:
            p = (
                override_progress
                if override_progress is not None
                else current_file.download_progress
            )
            s = override_state if override_state is not None else current_file.state
        else:
            p = f.download_progress
            s = f.state
        if s == ModelFileStateEnum.READY:
            continue
        active_values.append(float(p) if p is not None else 0.0)
    if not active_values:
        return 100.0
    return sum(active_values) / len(active_values)


def _refresh_instance_download_progress(
    instance: ModelInstance, file: ModelFile, *, file_ready: bool = False
) -> bool:
    """
    将汇总进度写入 instance.download_progress，返回是否发生变化
    """
    if instance.download_progress == 100:
        return False
    if file_ready:
        aggregate = _aggregate_instance_download_progress(
            instance,
            file,
            override_progress=100.0,
            override_state=ModelFileStateEnum.READY,
        )
    else:
        aggregate = _aggregate_instance_download_progress(instance, file)
    if aggregate is not None and aggregate != instance.download_progress:
        instance.download_progress = aggregate
        return True
    return False


def _first_resolved_path(
    files: Optional[List[ModelFile]], *, exclude_lora: bool = False
) -> Optional[str]:
    """
    从文件列表中取第一个实际路径，可选择跳过 LoRA 文件
    """
    for file in files or []:
        if exclude_lora and file.is_lora:
            continue
        if file.resolved_paths:
            return file.resolved_paths[0]
    return None


async def _promote_to_starting_if_complete(
    session: AsyncSession, instance: ModelInstance
) -> bool:
    """
    启动前检查入口：全部文件就绪后，补齐主模型和草稿模型路径，将实例设为 STARTING
    """
    loaded = await ModelInstance.one_by_id_with_model_files(session, instance.id)
    if not _download_completed(loaded):
        return False
    # Promotion is the single choke point into STARTING, so backfill the paths
    # here: the subordinate path never sets them and concurrent events may carry
    # a None snapshot, which crashes the worker on Path(None).
    if not instance.resolved_path:
        instance.resolved_path = _first_resolved_path(
            [file for file in loaded.model_files if file.worker_id == instance.worker_id], exclude_lora=True
        )
    if instance.draft_model_source and not instance.draft_model_resolved_path:
        instance.draft_model_resolved_path = _first_resolved_path(
            [file for file in loaded.draft_model_files if file.worker_id == instance.worker_id]
        )
    instance.state = ModelInstanceStateEnum.STARTING
    return True


def _sync_main_worker_downloading(
    instance: ModelInstance, file: ModelFile, is_draft_model: bool
) -> bool:
    """
    处理主 Worker 文件正在下载的情况，更新实例状态及主模型或草稿模型进度。
    """
    # First file to start: flip to DOWNLOADING and seed the bar. Draft seeds 0
    # (tracked separately); primary/LoRA seeds the active-file aggregate.
    if instance.state == ModelInstanceStateEnum.INITIALIZING:
        instance.state = ModelInstanceStateEnum.DOWNLOADING
        instance.state_message = ""
        if is_draft_model:
            instance.download_progress = 0
        else:
            aggregate = _aggregate_instance_download_progress(instance, file)
            instance.download_progress = aggregate if aggregate is not None else 0
        return True

    if instance.state != ModelInstanceStateEnum.DOWNLOADING:
        return False

    if is_draft_model:
        if (
            file.download_progress != instance.draft_model_download_progress
            and instance.draft_model_download_progress != 100
        ):
            instance.draft_model_download_progress = file.download_progress
            return True
        return False

    # Primary/LoRA file: feed the aggregate bar.
    return _refresh_instance_download_progress(instance, file)


async def _sync_main_worker_ready(
    session: AsyncSession,
    instance: ModelInstance,
    file: ModelFile,
    is_draft_model: bool,
) -> bool:
    """
    处理主 Worker 文件就绪，更新路径和进度，并检查整个实例是否已准备完成。
    """
    need_update = False

    if is_draft_model:
        if (
            instance.draft_model_download_progress != 100
            or not instance.draft_model_resolved_path
        ):
            instance.draft_model_download_progress = 100
            if file.resolved_paths:
                instance.draft_model_resolved_path = file.resolved_paths[0]
            need_update = True
    else:
        # Only the primary file owns resolved_path; LoRA files use mounted_loras.
        if (
            _is_primary_instance_model_file(file, instance, is_draft_model)
            and not instance.resolved_path
        ):
            if file.resolved_paths:
                instance.resolved_path = file.resolved_paths[0]
            need_update = True
        if _refresh_instance_download_progress(instance, file, file_ready=True):
            need_update = True

    if await _promote_to_starting_if_complete(session, instance):
        need_update = True
    elif instance.state == ModelInstanceStateEnum.INITIALIZING:
        # Some but not all files done.
        instance.state = ModelInstanceStateEnum.DOWNLOADING
        instance.state_message = ""
        need_update = True

    return need_update


async def sync_main_worker_model_file_state(
    session: AsyncSession,
    file: ModelFile,
    instance: ModelInstance,
    is_draft_model: bool = False,
):
    """
    重新加载实例，根据主 Worker 文件的 DOWNLOADING / READY / ERROR 状态更新并保存实例。
    """

    # Re-load (with model_files) to avoid identity-map conflicts with a detached
    # instance from the caller, and to let progress aggregation read sibling rows.
    instance = await ModelInstance.one_by_id_with_model_files(session, instance.id)
    if not instance or instance.state == ModelInstanceStateEnum.ERROR:
        return

    logger.trace(
        f"Syncing model file {file.id} with model instance {instance.id}, file state: {file.state}, "
        f"progress: {file.download_progress}, message: {file.state_message}, instance state: {instance.state}"
    )

    need_update = False
    if file.state == ModelFileStateEnum.DOWNLOADING:
        need_update = _sync_main_worker_downloading(instance, file, is_draft_model)
    elif file.state == ModelFileStateEnum.READY and instance.state in (
        ModelInstanceStateEnum.DOWNLOADING,
        ModelInstanceStateEnum.INITIALIZING,
    ):
        need_update = await _sync_main_worker_ready(
            session, instance, file, is_draft_model
        )
    elif file.state == ModelFileStateEnum.ERROR:
        instance.state = ModelInstanceStateEnum.ERROR
        instance.state_message = file.state_message
        need_update = True

    if need_update:
        await ModelInstanceService(session).update(instance)


async def _sync_subordinate_worker(
    session: AsyncSession,
    instance: ModelInstance,
    subordinate: ModelInstanceSubordinateWorker,
    file: ModelFile,
) -> bool:
    """
    [$14 多机多卡] 更新某个从属 Worker 的进度；文件就绪时检查整体完成情况，失败时将实例设为 ERROR。
    """
    if file.state == ModelFileStateEnum.DOWNLOADING:
        if file.download_progress == subordinate.download_progress:
            return False
        subordinate.download_progress = file.download_progress
        return True

    if file.state == ModelFileStateEnum.READY:
        # progress may already be 100 from the final DOWNLOADING report, so a
        # READY event must still re-check completion — gating on the progress
        # value would skip the STARTING transition and stick at 100%.
        need_update = subordinate.download_progress != 100
        worker_id = subordinate.worker_id
        if instance.state in (
            ModelInstanceStateEnum.DOWNLOADING,
            ModelInstanceStateEnum.INITIALIZING,
        ):
            if await _promote_to_starting_if_complete(session, instance):
                need_update = True
        # 完成检查会重新加载 JSON 列，之后再定位并更新本从属节点，避免进度被刷新覆盖。
        subordinate = next(sw for sw in instance.distributed_servers.subordinate_workers if sw.worker_id == worker_id)
        subordinate.download_progress = 100
        return need_update

    if file.state == ModelFileStateEnum.ERROR:
        instance.state = ModelInstanceStateEnum.ERROR
        instance.state_message = file.state_message
        return True

    return False


async def sync_distributed_model_file_state(
    session: AsyncSession, file: ModelFile, instance: ModelInstance
):
    """
    [$14 多机多卡] 找到文件对应的从属 Worker，执行同步，再标记 distributed_servers JSON 字段发生变化并保存。
    """

    # Re-load to avoid identity-map conflicts with a detached caller instance.
    instance = await ModelInstance.one_by_id(session, instance.id)
    if not instance or instance.state == ModelInstanceStateEnum.ERROR:
        return

    if (
        not instance.distributed_servers
        or not instance.distributed_servers.download_model_files
    ):
        return

    subordinate = next(
        (
            item
            for item in instance.distributed_servers.subordinate_workers or []
            if item.worker_id == file.worker_id
        ),
        None,
    )
    if subordinate is None:
        return

    logger.trace(
        f"Syncing distributed model file {file.id} with model instance {instance.name}, file state: {file.state}, "
        f"progress: {file.download_progress}, message: {file.state_message}, instance state: {instance.state}"
    )

    if await _sync_subordinate_worker(session, instance, subordinate, file):
        flag_modified(instance, "distributed_servers")
        await ModelInstanceService(session).update(instance)


def _download_completed(instance: Optional[ModelInstance]) -> bool:
    """
    检查所有关联文件，包括从属 Worker 的文件；配置了草稿模型时，还要检查草稿文件
    """
    if instance is None:
        return False

    if not instance.model_files and not instance.draft_model_source:
        return False

    for model_file in instance.model_files or []:
        if model_file.state != ModelFileStateEnum.READY:
            return False

    if instance.draft_model_source:
        draft_files = instance.draft_model_files or []
        if not draft_files:
            return False
        for draft_file in draft_files:
            if draft_file.state != ModelFileStateEnum.READY:
                return False

    # Subordinate files are in instance.model_files (checked above) — the single
    # source of truth. The old subordinate_workers progress check raced with the
    # distributed sync path and could stick at DOWNLOADING after all hit 100%.
    return True


def _get_worker_ids_for_file_download(
    instance: ModelInstance,
) -> List[str]:
    """
    返回需要准备文件的 Worker ID：主 Worker，加上启用文件下载的从属 Worker。
    """

    worker_ids = [instance.worker_id] if instance.worker_id else []

    if (
        instance.distributed_servers
        and instance.distributed_servers.download_model_files
    ):
        worker_ids += [
            item.worker_id
            for item in instance.distributed_servers.subordinate_workers or []
            if item.worker_id
        ]

    return worker_ids


async def new_workers_from_pool(
    session: AsyncSession, pool: WorkerPool
) -> List[Worker]:
    fields = {"deleted_at": None, "worker_pool_id": pool.id}
    current_workers = await Worker.all_by_fields(session, fields=fields)
    current_workers = [
        worker
        for worker in current_workers
        if worker.state not in [WorkerStateEnum.DELETING]
    ]
    # if has enough workers, no need to create more
    if len(current_workers) >= pool.replicas:
        return []
    delta = pool.replicas - len(current_workers)
    provisioning_workers = [
        worker
        for worker in current_workers
        if worker.state in [WorkerStateEnum.PENDING, WorkerStateEnum.PROVISIONING]
    ]
    if pool.batch_size is not None and pool.batch_size <= len(provisioning_workers):
        return []
    if pool.batch_size is not None:
        delta = min(delta, pool.batch_size - len(provisioning_workers))
    new_workers = []
    for _ in range(delta):
        new_worker = Worker(
            hostname="",
            ip="",
            ifname="",
            port=0,
            worker_uuid="",
            cluster=pool.cluster,
            worker_pool=pool,
            provider=pool.cluster.provider,
            name=f"pool-{pool.id}-"
            + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)),
            labels={
                **(pool.labels or {}),
                "provider": pool.cluster.provider.value,
                "instance_type": pool.instance_type or "unknown",
            },
            state=WorkerStateEnum.PENDING,
            status=WorkerStatus.get_default_status(),
        )
        new_workers.append(new_worker)
    return new_workers


class WorkerPoolController:
    """Worker pool controller creates new workers based on the worker pool configuration."""

    async def start(self):
        async for event in WorkerPool.subscribe(source="worker_pool_controller"):
            if event.type == EventType.HEARTBEAT:
                async with async_session() as session:
                    pools = await WorkerPool.all_by_fields(session, {"deleted_at": None})
                for pool in pools:
                    try:
                        await self._reconcile(Event(type=EventType.UPDATED, data={"id": pool.id}))
                    except Exception:
                        logger.exception("Failed to reconcile worker pool %s on heartbeat", pool.id)
                continue
            try:
                await self._reconcile(event)
            except Exception:
                logger.exception("Failed to reconcile worker pool")

    async def _reconcile(self, event: Event):
        """
        Reconcile the worker pool state with the current event.
        """
        # Only the id is needed -- the pool is re-read below either way -- so
        # an id-only payload (see Event) is as good as a hydrated one here.
        # On a DELETE the row is gone and one_by_id returns None.
        pool_event_id = resolve_event_id(event)
        if pool_event_id is None:
            # No row to reconcile. Return rather than passing None to
            # one_by_id, which warns ("fully NULL primary key identity cannot
            # load any object") and may raise in a future SQLAlchemy.
            return
        logger.info(f"Reconcile worker pool {pool_event_id} with event {event.type}")
        async with async_session() as session:
            pool = await WorkerPool.one_by_id(
                session, pool_event_id, for_update=True, options=[selectinload(WorkerPool.cluster)]
            )
            if pool is None or pool.deleted_at is not None:
                return
            # mark the data to avoid read after commit
            cluster_name = pool.cluster.name
            cluster = pool.cluster
            pool_id = pool.id
            current = await Worker.all_by_fields(session, {"worker_pool_id": pool.id, "deleted_at": None})
            current = [worker for worker in current if worker.state != WorkerStateEnum.DELETING]
            if len(current) > pool.replicas:
                instances = await ModelInstance.all_by_field(session, "cluster_id", pool.cluster_id)
                def deletion_priority(worker):
                    occupied = any(get_model_instance_worker_match(instance, worker_id=worker.id).matched for instance in instances)
                    creating = worker.state in {WorkerStateEnum.PENDING, WorkerStateEnum.ERROR}
                    return occupied, not creating, worker.id
                for worker in sorted(current, key=deletion_priority)[:len(current)-pool.replicas]:
                    await WorkerService(session).request_delete(worker, auto_commit=False)
                await session.commit()
                return
            workers = await new_workers_from_pool(session, pool)
            if len(workers) == 0:
                return
            ids = []
            for worker in workers:
                created_worker: Worker = await Worker.create(
                    session=session, source=worker, auto_commit=False
                )
                ids.append(created_worker.id)
            if cluster.state == ClusterStateEnum.PENDING:
                cluster.state = ClusterStateEnum.PROVISIONING
                cluster.state_message = None
                await cluster.update(session=session, auto_commit=False)
            await session.commit()
            logger.info(
                f"Created {len(ids)} new workers {ids} for cluster {cluster_name} worker pool {pool_id}"
            )


def _record_ssh_endpoint(
    provider_config: Dict[str, Any], instance: Optional[CloudInstance]
) -> bool:
    """Note where SSH answers, when the provider doesn't serve it on the
    instance's own address. Returns whether anything was written.

    Only providers that publish instances behind a mapped address report one;
    for the rest ``advertise_address:22`` is the answer and nothing is stored.
    Already-recorded endpoints are left alone so a later poll that happens to
    come back without the mapping can't erase it.
    """
    if instance is None or not instance.ssh_endpoint:
        return False
    if "ssh_endpoint" in provider_config:
        return False
    host, port = instance.ssh_endpoint
    ssh_endpoint: Dict[str, Any] = {"host": host, "port": port}
    # Omit an unknown user rather than storing "": the provider only fills it
    # in once the instance is up, and a consumer shouldn't have to tell an
    # empty string apart from an absent key.
    if instance.ssh_user:
        ssh_endpoint["user"] = instance.ssh_user
    provider_config["ssh_endpoint"] = ssh_endpoint
    return True


class WorkerProvisioningController:
    def __init__(self, cfg: Config):
        self._cfg = cfg

    @classmethod
    async def _create_ssh_key(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
    ) -> int:
        """
        Generate a new ssh key pair,
        And Create ssh_key in cloud provider.
        Create SSHKey record without commit and returns it.
        """
        logger.info(f"Creating ssh key for worker {worker.name}")
        private_key, public_key = generate_ssh_key_pair()
        ssh_key = Credential(
            credential_type=CredentialType.SSH,
            public_key=public_key,
            encoded_private_key=private_key,
            ssh_key_options=SSHKeyOptions(
                algorithm="ED25519",
                length=0,
            ),
        )
        ssh_key_id = await client.create_ssh_key(worker.name, public_key)
        # None means the provider registered nothing (no SSH key API); leave
        # external_id NULL rather than storing the string "None", so teardown
        # knows there is no upstream key to delete.
        ssh_key.external_id = str(ssh_key_id) if ssh_key_id is not None else None
        ssh_key_rtn = await Credential.create(session, ssh_key, auto_commit=False)
        return ssh_key_rtn.id

    @classmethod
    async def _create_instances(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
        cfg: Config,
    ) -> str:
        secret_fields = set(SensitivePredefinedConfig.model_fields.keys())
        secret_configs = (
            worker.cluster.worker_config.model_dump(include=secret_fields)
            if worker.cluster.worker_config
            else {}
        )
        ssh_key = await Credential.one_by_id(session, worker.ssh_key_id)
        if ssh_key is None:
            raise ValueError(f"SSH key {worker.ssh_key_id} not found")
        image_config = cfg.model_copy(update={"system_default_container_registry":
            worker.cluster.system_default_container_registry or cfg.system_default_container_registry})
        image_name = apply_registry_override_to_image(image_config, worker.worker_pool.image_name, None)
        user_data = await client.construct_user_data(
            server_url=worker.cluster.server_url or cfg.server_external_url,
            token=worker.cluster.registration_token,
            image_name=image_name,
            os_image=worker.worker_pool.os_image,
            secret_configs=secret_configs,
            worker_name=worker.name,
            # Providers with no SSH key API embed this in the cloud-config;
            # the ones that have one attach it by id instead and ignore it.
            ssh_public_key=ssh_key.public_key,
        )
        to_create = construct_cloud_instance(worker, ssh_key, user_data.format())
        logger.info(f"Creating cloud instance for worker {worker.name}")
        logger.debug("Cloud instance type=%s region=%s", to_create.type, to_create.region)
        return await client.create_instance(to_create)

    @classmethod
    async def _provisioning_started(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
        instance: CloudInstance,
    ) -> bool:
        changed = True
        # Copy rather than alias: provider_config is a plain JSON column, so
        # SQLAlchemy only notices a write when the attribute is set to a
        # different object. Mutating the loaded dict in place and assigning it
        # back would leave the change unsaved.
        provider_config = dict(worker.provider_config or {})
        volumes = list(
            (getattr(worker.worker_pool.cloud_options, "volumes", None) or [])
        )
        volume_ids = provider_config.get("volume_ids", [])
        # Backfill a mapping that wasn't published yet the first time round.
        # wait_for_public_ip returns as soon as an address appears, which for a
        # provider that maps SSH elsewhere can be before the mapping exists —
        # and the branch below only runs while advertise_address is still
        # empty, so without this the hint would stay missing for good.
        backfilled = _record_ssh_endpoint(provider_config, instance)
        if backfilled:
            worker.provider_config = provider_config
        if worker.advertise_address is None or worker.advertise_address == "":
            try:
                instance = await client.wait_for_public_ip(worker.external_id)
                worker.advertise_address = (
                    instance.ip_address if instance.ip_address else ""
                )
                worker.state_message = "Waiting for volumes to attach"
                # Last in the branch on purpose: this is a display aid, and
                # failing to read it must not hold back the state machine.
                if _record_ssh_endpoint(provider_config, instance):
                    worker.provider_config = provider_config
            except InstanceProvisioningFailed:
                # Terminal on the provider side; see _provisioning_before_started.
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to wait for instance {worker.external_id} to get public ip: {e}"
                )
        elif len(volumes) != len(volume_ids) and len(volumes) > 0:
            volume_ids = await client.create_volumes_and_attach(
                worker.id, worker.external_id, worker.cluster.region, *volumes
            )
            provider_config["volume_ids"] = volume_ids
            worker.provider_config = provider_config
        elif (
            len(volumes) == len(volume_ids)
            and worker.state in {WorkerStateEnum.PROVISIONING, WorkerStateEnum.READY}
        ):
            # Membership, not hasattr: provider_config is a dict, so hasattr
            # was always False and this overwrote the ids of volumes that had
            # just been created -- losing the only record of them, so teardown
            # could never find the volumes to delete.
            if "volume_ids" not in provider_config:
                provider_config["volume_ids"] = []
            worker.provider_config = provider_config
            if worker.state != WorkerStateEnum.READY:
                worker.state = WorkerStateEnum.READY if provider_config.get("registered") else WorkerStateEnum.INITIALIZING
            provider_config["provisioning_complete"] = True
            worker.provider_config = provider_config
            if worker.cluster.state not in {ClusterStateEnum.PROVISIONED, ClusterStateEnum.READY}:
                worker.cluster.state = ClusterStateEnum.PROVISIONED
                await worker.cluster.update(session=session, auto_commit=False)
            if worker.state != WorkerStateEnum.READY:
                worker.state_message = "Initializing: installing required drivers and software. The worker will start automatically after setup."
            else:
                worker.state_message = None
        else:
            changed = backfilled
        return changed

    @classmethod
    async def _provisioning_before_started(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
        cfg: Config,
    ) -> Tuple[Optional[CloudInstance], bool]:
        """
        return started and changed
        """
        instance = None
        changed = False
        if worker.external_id is not None:
            instance = await client.get_instance(worker.external_id)
            # TODO should handle instance not exist problem
            if instance is None:
                raise InstanceProvisioningFailed(f"Cloud instance {worker.external_id} no longer exists")
            if instance.status == InstanceState.RUNNING:
                return instance, changed
        changed = True
        if worker.state == WorkerStateEnum.PENDING:
            worker.state = WorkerStateEnum.PROVISIONING
            worker.state_message = "Creating SSH key"
        elif worker.ssh_key_id is None:
            worker.ssh_key_id = await cls._create_ssh_key(session, client, worker)
            worker.state_message = "Creating cloud instance"
        elif worker.external_id is None:
            worker.external_id = await cls._create_instances(
                session, client, worker, cfg
            )
            worker.state_message = "Waiting for cloud instance started"
        elif worker.external_id is not None:
            try:
                # depress the timeout exception
                instance = await client.wait_for_started(worker.external_id)
                worker.state_message = "Waiting for instance's public ip"
            except InstanceProvisioningFailed:
                # The provider gave up on the instance, so polling it again on
                # the next reconcile would spin forever. Let it through to the
                # caller, which parks the worker in ERROR.
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to wait for instance {worker.external_id} to start: {e}"
                )
        return instance, changed

    @classmethod
    async def _provisioning_instance(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
        cfg: Config,
    ):
        # provider_config = worker.provider_config or {}
        # Phase I is to ensure instance running.
        instance, changed = await cls._provisioning_before_started(
            session, client, worker, cfg
        )
        if (
            not changed
            and instance is not None
            and instance.status == InstanceState.RUNNING
        ):
            # Phase II is to wait for instance infomation and attach volume.
            changed = await cls._provisioning_started(session, client, worker, instance)
        if changed:
            await WorkerService(session).update(
                worker=worker, source=None, auto_commit=False
            )

    @classmethod
    async def _deleting_instance(
        cls,
        session: AsyncSession,
        client: ProviderClientBase,
        worker: Worker,
    ):
        ssh_key = await Credential.one_by_id(session, worker.ssh_key_id)
        if worker.external_id is not None:
            await client.delete_instance(worker.external_id)
        if ssh_key and ssh_key.external_id:
            await client.delete_ssh_key(ssh_key.external_id)
        # if using soft delete here, skip deletion and remove external_id
        if ssh_key:
            await ssh_key.delete(session, auto_commit=False)
        if worker.deleted_at is not None:
            await WorkerService(session).delete(worker, auto_commit=False)

    async def check_server_external_url(self, cluster_server_url: Optional[str] = None):
        server_url = cluster_server_url or self._cfg.server_external_url
        if server_url is None or server_url == "":
            raise ValueError(
                "Cluster's server_url is not configured, Please edit cluster first."
            )
        import aiohttp
        from yarl import URL

        healthz_url = str(URL(server_url) / "healthz")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(healthz_url, timeout=10) as resp:
                    if resp.status != 200:
                        raise ValueError(
                            f"External server healthz url {healthz_url} is not reachable, status code: {resp.status}"
                        )
        except Exception as e:
            raise ValueError(
                f"Failed to check external server healthz url {healthz_url}: {e}"
            )

    async def _reconcile(self, event: Event):
        """重新读取持久化进度，每个事件推进一项云资源操作。"""
        worker_id = resolve_event_id(event)
        if worker_id is None:
            return
        async with async_session() as session:
            worker = await Worker.one_by_id(session, worker_id, for_update=True, options=[
                selectinload(Worker.cluster), selectinload(Worker.worker_pool),
            ])
            if worker is None or worker.worker_pool_id is None:
                return
            finishing = (worker.state == WorkerStateEnum.READY and worker.external_id is not None
                         and not (worker.provider_config or {}).get("provisioning_complete"))
            if worker.state not in {WorkerStateEnum.PENDING, WorkerStateEnum.PROVISIONING, WorkerStateEnum.DELETING} and not finishing:
                return
            client = None
            try:
                if worker.cluster is None or worker.worker_pool is None or worker.cluster_id != worker.worker_pool.cluster_id:
                    raise ValueError("Managed worker has no matching cloud cluster and worker pool")
                credential = await CloudCredential.one_by_id(session, worker.cluster.credential_id)
                if credential is None or credential.deleted_at is not None:
                    raise ValueError("Cloud credential not found")
                client = get_client_from_provider(worker.cluster.provider, credential)
                if worker.state == WorkerStateEnum.PENDING:
                    await self.check_server_external_url(worker.cluster.server_url)
                if worker.state in {WorkerStateEnum.PENDING, WorkerStateEnum.PROVISIONING}:
                    await self._provisioning_instance(session, client, worker, self._cfg)
                elif finishing:
                    instance = await client.get_instance(worker.external_id)
                    if instance is not None and instance.status == InstanceState.RUNNING:
                        if await self._provisioning_started(session, client, worker, instance):
                            await WorkerService(session).update(worker, auto_commit=False)
                elif worker.state == WorkerStateEnum.DELETING:
                    await self._deleting_instance(session, client, worker)
                await session.commit()
            except Exception as error:
                message = f"Failed to provision or delete worker {worker.name}: {error}"
                logger.exception(message)
                await session.rollback()
                await session.refresh(worker)
                worker.state = WorkerStateEnum.ERROR
                worker.state_message = message
                await WorkerService(session).update(worker)
            finally:
                if client is not None:
                    await client.aclose()


class ClusterController:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._disable_gateway = cfg.gateway_mode == GatewayModeEnum.disabled
        pass

    async def start(self):
        """
        Start the controller.
        """
        async for event in Cluster.subscribe(source="cluster_controller"):
            if event.type == EventType.HEARTBEAT:
                continue
            try:
                await self._reconcile(event)
            except Exception as e:
                logger.error(f"Failed to reconcile cluster: {e}")

    async def _reconcile(self, event: Event):
        """
        Reconcile the cluster state.
        """
        await self._sync_cluster_state(event)
        if self._disable_gateway:
            return

    async def _sync_cluster_state(self, event: Event):
        if event.type == EventType.DELETED:
            return
        cluster: Cluster = event.data
        if not cluster:
            return
        async with async_session() as session:
            cluster: Cluster = await Cluster.one_by_id(
                session, cluster.id, options=[selectinload(Cluster.cluster_workers)]
            )
            if not cluster or cluster.provider in [
                ClusterProvider.Kubernetes,
                ClusterProvider.Docker,
            ]:
                return
            if cluster.workers == 0:
                target = ClusterStateEnum.PENDING
                message = "No workers have been provisioned for this cluster yet."
            elif cluster.ready_workers > 0:
                target, message = ClusterStateEnum.READY, None
            elif any(worker.external_id for worker in cluster.cluster_workers):
                target, message = ClusterStateEnum.PROVISIONED, None
            else:
                target, message = ClusterStateEnum.PROVISIONING, None
            if cluster.state != target or cluster.state_message != message:
                cluster.state, cluster.state_message = target, message
                await cluster.update(session=session, auto_commit=True)
