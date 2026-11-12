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
            loaded.model_files, exclude_lora=True
        )
    if instance.draft_model_source and not instance.draft_model_resolved_path:
        instance.draft_model_resolved_path = _first_resolved_path(
            loaded.draft_model_files
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
    更新某个从属 Worker 的进度；文件就绪时检查整体完成情况，失败时将实例设为 ERROR。
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
        subordinate.download_progress = 100
        if instance.state in (
            ModelInstanceStateEnum.DOWNLOADING,
            ModelInstanceStateEnum.INITIALIZING,
        ):
            if await _promote_to_starting_if_complete(session, instance):
                need_update = True
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
    找到文件对应的从属 Worker，执行同步，再标记 distributed_servers JSON 字段发生变化并保存。
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
