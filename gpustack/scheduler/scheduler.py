from gpustack.config.config import Config
from gpustack.scheduler.queue import AsyncUniqueQueue
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import timezone
from gpustack.schemas.models import ModelInstance
from gpustack.server.bus import EventType
import asyncio
from gpustack.server.db import async_session
from typing import Optional
from datetime import datetime
from datetime import timedelta
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.schemas.models import Model
from gpustack.server.services import ModelInstanceService
import queue
from gpustack.schemas.workers import Worker
from sqlalchemy.orm import selectinload
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.base import WorkerFilterChain
from gpustack.policies.utils import ListMessageBuilder
from gpustack.policies.worker_filters.cluster_filter import ClusterFilter
from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.policies.worker_filters.status_filter import StatusFilter
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import List
from typing import Tuple
from gpustack.schemas.models import ComputedResourceClaim
import logging
from gpustack.policies.candidate_selectors import VLLMResourceFitSelector
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.policies.scorers.score_chain import CandidateScoreChain
from gpustack.scheduler.calculator import get_pretrained_config_with_workers
from gpustack.scheduler.meta_registry import get_model_meta
from gpustack.scheduler.model_registry import detect_model_type
from gpustack.schemas.models import BackendEnum
from gpustack.schemas.models import CategoryEnum
from gpustack.server.services import ModelService
from gpustack.schemas.models import get_backend
from gpustack.utils.command import find_parameter
import json
from gpustack.policies.utils import should_skip_gpu_count_check
from gpustack.utils.math import largest_power_of_2_leq
from sqlalchemy.orm.attributes import flag_modified
from gpustack.schemas.models import DistributedServerCoordinateModeEnum
from gpustack.schemas.models import DistributedServers


logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, cfg: Config, check_interval: int = 180):
        self._id = "model-instance-scheduler"
        self._config = cfg
        self._check_interval = check_interval
        self._queue = AsyncUniqueQueue()

    async def start(self):
        consumer = None
        scheduler = None
        subscription = None
        try:
            """先把唯一的队列消费者跑起来"""
            consumer = asyncio.create_task(self._schedule_cycle())

            trigger = IntervalTrigger(
                seconds=self._check_interval, timezone=timezone.utc
            )
            scheduler = AsyncIOScheduler(timezone=timezone.utc)
            scheduler.add_job(
                self._enqueue_pending_instances,
                trigger=trigger,
                id=self._id,
                max_instances=1,
            )
            scheduler.start()
            logger.info("Scheduler started.")

            await self._enqueue_pending_instances()
            subscription = ModelInstance.subscribe(
                source="scheduler",
                event_types={EventType.CREATED},
                replay_existing=False,
            )
            async for event in subscription:
                if event.type == EventType.CREATED:
                    await self._enqueue_event_instance(event.data)
        except Exception:
            logger.exception("Scheduler failed")
            raise
        finally:
            try:
                if subscription is not None:
                    await subscription.aclose()
            finally:
                if scheduler is not None and scheduler.running:
                    scheduler.shutdown(wait=False)
                if consumer is not None:
                    consumer.cancel()
                    await asyncio.gather(consumer, return_exceptions=True)

    async def _enqueue_pending_instances(self):
        try:
            async with async_session() as session:
                instances = await ModelInstance.all(session)
                tasks = []
                for instance in instances:
                    if self._should_schedule(instance):
                        task = asyncio.create_task(self._evaluate(instance))
                        tasks.append(task)

                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed to enqueue pending model instances: {e}")

    async def _enqueue_event_instance(self, instance: Optional[ModelInstance]):
        if instance is None or instance.id is None:
            return
        try:
            if self._should_schedule(instance):
                await self._evaluate(instance)
        except Exception as e:
            logger.error(f"Failed to evaluate instance {instance.id} from event: {e}")

    async def _evaluate(self, instance: ModelInstance):  # noqa: C901
        """
        开始评估配置与资源，决定如何分配
        """
        async with async_session() as session:
            try:
                instance = await ModelInstance.one_by_id(session, instance.id)
                # Re-check against the freshly-fetched row: the caller's
                # snapshot may be stale (event payload, last full scan, etc.)
                # and the user may have deleted or transitioned the instance
                # between dispatch and now.
                if instance is None or not self._should_schedule(instance):
                    return

                model = await Model.one_by_id(session, instance.model_id)
                if model is None:
                    raise Exception("Model not found.")

                if instance.state != ModelInstanceStateEnum.ANALYZING:
                    instance.state = ModelInstanceStateEnum.ANALYZING
                    instance.state_message = "Evaluating resource requirements"
                    await ModelInstanceService(session).update(instance)

                # Get available workers for potential remote parsing
                workers = await Worker.all(session)

                should_update_model = False
                try:
                    should_update_model = await evaluate_pretrained_config(
                        model,
                        workers=workers,
                        raise_raw=True,
                    )
                except Exception as e:
                    # Even if the evaluation failed, we still want to proceed to deployment.
                    # Cases can be:
                    # 1. Model config is not valid, but is overridable by backend parameters.
                    # 2. It may not be required to be transformer-compatible for certain backends.
                    logger.error(
                        f"Failed to evaluate model {model.name or model.readable_source}: {e}"
                    )

                if should_update_model:
                    await ModelService(session).update(model)

                await self._queue.put(instance)
            except Exception as e:
                try:
                    instance.state = ModelInstanceStateEnum.ERROR
                    instance.state_message = str(e)
                    await ModelInstanceService(session).update(instance)
                except Exception as ue:
                    logger.error(
                        f"Failed to update model instance: {ue}. Original error: {e}"
                    )

    def _should_schedule(self, instance: ModelInstance) -> bool:
        """
        新建的 pending 立即处理；
        旧 pending 超过 90 秒可以重试；
        卡在 analyzing 或 scheduled 超过 3 分钟也会重新评估
        """
        newly_created = (instance.updated_at - instance.created_at) < timedelta(
            seconds=1
        )
        update_delta = datetime.now(timezone.utc) - instance.updated_at.replace(
            tzinfo=timezone.utc
        )
        return (
            (
                # When enqueueing pending state model instances, handle two cases:
                # 1. Newly created model instances (updated_at - created_at < 1 second),
                #    which will be updated to ANALYZING in _evaluate.
                # 2. Existing PENDING model instances periodically enqueued by the scheduler job.
                #    In this case, update_delta is longer than 90s, as the scheduler runs every 180s.
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.PENDING
                and (newly_created or update_delta > timedelta(seconds=90))
            )
            or (
                # Reschedule while it stays in anayzing state for too long,
                # maybe the server is restarted.
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.ANALYZING
                and update_delta > timedelta(minutes=3)
            )
            or (
                # Reschedule while it stays in scheduled state for too long,
                # maybe the worker is down.
                instance.worker_id is not None
                and instance.state == ModelInstanceStateEnum.SCHEDULED
                and update_delta > timedelta(minutes=3)
            )
        )

    async def _schedule_cycle(self):
        """消息队列、放置"""
        while True:
            try:
                item = await self._queue.get()
                try:
                    await self._schedule_one(item)
                except Exception as e:
                    logger.error(f"Failed to schedule model instance: {e}")
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Failed to get item from schedule queue: {e}")

    async def _schedule_one(self, instance: ModelInstance):  # noqa: C901
        """
        每次放置都重新读所有 Worker 和所有实例
        """
        logger.debug(f"Scheduling model instance {instance.name}")

        state_message = ""

        async with async_session() as session:
            model_instance = await ModelInstance.one_by_id(session, instance.id)
            if model_instance is None:
                logger.debug(
                    f"Model instance(ID: {instance.id}) was deleted before scheduling due"
                )
                return

            if model_instance.state != ModelInstanceStateEnum.ANALYZING:
                logger.debug(
                    f"Skipping queued instance {model_instance.name}: state is {model_instance.state}"
                )
                return

            workers = await Worker.all(session)
            if not workers:
                state_message = "No available workers"

            model = await Model.one_by_id(session, model_instance.model_id)
            if model is None:
                state_message = "Model not found"

            model_instances = await ModelInstance.all(
                session, options=[selectinload(ModelInstance.model)]
            )
            model_instances = [other for other in model_instances if other.id != model_instance.id]

            candidate = None
            messages = []
            if workers and model:
                try:
                    candidate, messages = await find_candidate(
                        session, self._config, model, workers, model_instances
                    )
                except Exception as e:
                    state_message = f"Failed to find candidate: {e}"

            if candidate is None:
                model_instance.state = ModelInstanceStateEnum.PENDING
                model_instance.state_message = (
                    "No suitable workers.\nDetails:\n" + "".join(messages)
                )
                model_instance.worker_id = None
                model_instance.worker_name = None
                model_instance.worker_ip = None
                model_instance.worker_advertise_address = None
                model_instance.worker_ifname = None
                model_instance.gpu_type = None
                model_instance.gpu_indexes = []
                model_instance.gpu_addresses = []
                model_instance.computed_resource_claim = None
                """为了将主从候选保存到模型实例"""
                model_instance.distributed_servers = None
                if state_message != "":
                    model_instance.state_message = state_message

                await ModelInstanceService(session).update(model_instance)
                logger.debug(
                    f"No suitable workers for model instance {model_instance.name}, state: {model_instance.state}"
                )
            else:
                # update model instance.
                model_instance.state = ModelInstanceStateEnum.SCHEDULED
                model_instance.state_message = ""
                model_instance.worker_id = candidate.worker.id
                model_instance.worker_name = candidate.worker.name
                model_instance.worker_ip = candidate.worker.ip
                model_instance.worker_advertise_address = (
                    candidate.worker.advertise_address
                )
                model_instance.worker_ifname = candidate.worker.ifname
                model_instance.computed_resource_claim = (
                    candidate.computed_resource_claim
                )
                model_instance.gpu_type = candidate.gpu_type
                model_instance.gpu_indexes = candidate.gpu_indexes
                model_instance.gpu_addresses = candidate.gpu_addresses
                """
                实例记录：多台机器共同服务一个 ModelInstance
                    - 所以比如两台机器参与一次分布式推理时，数据库中仍然是一条模型实例记录
                """
                model_instance.distributed_servers = DistributedServers(
                    subordinate_workers=candidate.subordinate_workers,
                )
                if get_backend(model) in (
                    BackendEnum.VLLM, BackendEnum.ASCEND_MINDIE, BackendEnum.SGLANG
                ):
                    model_instance.distributed_servers.mode = (
                        DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                    )

                await ModelInstanceService(session).update(model_instance)

                logger.debug(
                    f"Scheduled model instance {model_instance.name} to worker "
                    f"{model_instance.worker_name} gpu {candidate.gpu_indexes}"
                )


async def find_candidate(
    session: AsyncSession,
    config: Config,
    model: Model,
    workers: List[Worker],
    model_instances: List[ModelInstance],
) -> Tuple[Optional[ModelInstanceScheduleCandidate], List[str]]:
    """
    接上真正的资源匹配和评分
    """

    # Filter workers.
    filters = [
        ClusterFilter(model),
        GPUMatchingFilter(model),
        LabelMatchingFilter(model),
        StatusFilter(model),
    ]

    """哪些机器符合基本条件"""
    worker_filter_chain = WorkerFilterChain(filters)
    workers, filter_messages = await worker_filter_chain.filter(workers)
    messages = []
    if filter_messages:
        messages.append(str(ListMessageBuilder(filter_messages)) + "\n")
    if len(workers) == 0:
        return None, messages

    # Initialize candidate selector.
    try:
        # 阶段版：只有 vLLM。真实代码是一串 if/elif，第 19 章补全
        """哪些单卡、多卡方案放得下"""
        candidates_selector = VLLMResourceFitSelector(
            config, model, model_instances
        )
    except Exception as e:
        return None, [f"Failed to initialize {model.backend} candidates selector: {e}"]

    # Select candidates.
    candidates = await candidates_selector.select_candidates(workers)

    """可行方案中哪个更合适"""
    candidate_scorers = [
        PlacementScorer(model, model_instances),
    ]
    candidates = await CandidateScoreChain(candidate_scorers).score(candidates)

    """取最高分方案"""
    candidate = pick_highest_score_candidate(candidates)

    # Collect messages.
    if candidate is None and len(workers) > 0:
        resource_fit_messages = candidates_selector.get_messages() or [
            "No workers meet the resource requirements."
        ]
        messages.extend(resource_fit_messages)
    elif candidate and candidate.overcommit:
        messages.extend(candidates_selector.get_messages())

    # Return the candidate and messages.
    return candidate, messages


def pick_highest_score_candidate(candidates: List[ModelInstanceScheduleCandidate]):
    logger.debug(f"Pick highest score candidate from {len(candidates)} candidates")

    if len(candidates) == 0:
        return None

    candidate = candidates[0]
    for i in range(1, len(candidates)):
        if candidates[i].score > candidate.score:
            candidate = candidates[i]

    return candidate


async def evaluate_pretrained_config(
    model: Model,
    workers: Optional[List[Worker]] = None,
    raise_raw: bool = False,
) -> bool:
    """
    评估元数据
    """
    # 2) Check overrided architectures if specified in backend parameters.
    architectures = get_vllm_override_architectures(model)
    if not architectures:
        try:
            trust_remote_code = _extract_trust_remote_code(model)
            pretrained_config = await get_pretrained_config_with_workers(
                model,
                workers=workers,
                trust_remote_code=trust_remote_code,
            )
        except ValueError as e:
            # Skip value error exceptions and defaults to LLM catagory for certain cases.
            if should_skip_architecture_check(model):
                model.categories = model.categories or [CategoryEnum.LLM]
                return True

            if raise_raw:
                raise

            logger.debug(
                f"Failed to get config for model {model.name or model.readable_source}, ValueError: {e}"
            )
            raise simplify_auto_config_value_error(e)
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise Exception(
                f"Timeout while getting config for model {model.name or model.readable_source}: {e}."
            )
        except Exception as e:
            raise Exception(
                f"Failed to get config for model {model.name or model.readable_source}: {e}"
            )

        architectures = getattr(pretrained_config, "architectures", []) or []
        if not architectures and not model.backend_version:
            raise ValueError(
                "Unrecognized architecture. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
            )

    model_type = detect_model_type(architectures)

    # TODO : Additional checks for unsupported architectures for other backends.
    if (
        model.backend == BackendEnum.VLLM
        and model_type == CategoryEnum.UNKNOWN
        and not model.backend_version
    ):
        raise ValueError(
            f"Unsupported architecture: {architectures}. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
        )

    meta_modified = False
    if not model.meta and (known_meta := get_model_meta(pretrained_config)):
        model.meta = known_meta
        meta_modified = True

    categories_modified = set_model_categories(model, model_type)
    gpus_per_replica_modified = set_model_gpus_per_replica(model)
    return categories_modified or gpus_per_replica_modified or meta_modified


def _extract_trust_remote_code(model: Model) -> bool:
    """读取是否指定了 --trust-remote-code"""
    if model.backend_parameters and "--trust-remote-code" in model.backend_parameters:
        return True
    return False


def get_vllm_override_architectures(model: Model) -> List[str]:
    """
    解析 --hf-overrides 中覆盖的架构
    """
    backend = get_backend(model)
    if backend != BackendEnum.VLLM:
        return []

    hf_overrides = find_parameter(model.backend_parameters, ["hf-overrides"])
    if hf_overrides:
        overrides_dict = json.loads(hf_overrides)
        return overrides_dict.get("architectures", [])
    return []


def should_skip_architecture_check(model: Model) -> bool:
    """
    决定部分配置异常是否允许放宽处理
    """

    if (
        model.backend == BackendEnum.CUSTOM
        or model.backend_version
    ):
        # New model architectures may be added with custom backend/version.
        return True

    if model.backend_parameters and find_parameter(
        model.backend_parameters, ["tokenizer-mode"]
    ):
        # Models like Pixtral may not provide compatible config but still work with custom parameters.
        return True

    return False


def simplify_auto_config_value_error(e: ValueError) -> ValueError:
    """
    将依赖库异常整理成较清楚的提示
    """
    message = str(e)
    if "trust_remote_code=True" in message:
        return ValueError(
            "The model contains custom code that must be executed to load correctly. If you trust the source, please pass the backend parameter `--trust-remote-code` to allow custom code to be run."
        )

    if "pip install --upgrade transformers" in message:
        return ValueError(
            "Unsupported model. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
        )

    return ValueError(f"Not a supported model.\n\n{message}")


def set_model_categories(model: Model, model_type: CategoryEnum) -> bool:
    """类别为空时补充识别结果，保留已有类别"""
    if model.categories:
        return False

    if model_type == CategoryEnum.UNKNOWN:
        # Default to LLM for unknown architectures
        model.categories = [CategoryEnum.LLM]
    else:
        model.categories = [model_type]

    return True


def set_model_gpus_per_replica(model: Model) -> bool:
    """
    只在手动指定了 GPU ID、但尚未明确每副本卡数时进行补充
    """

    def calculate_gpus_per_replica(model: Model) -> int:
        if model.backend == BackendEnum.VOX_BOX.value:
            return 1

        replicas = model.replicas or 1
        gpus_per_replica = len(model.gpu_selector.gpu_ids) // replicas

        if should_skip_gpu_count_check(model):
            return gpus_per_replica

        # User-specified world size from backend parameters takes precedence.
        if model.backend_parameters is not None:
            selector_map = {
                BackendEnum.VLLM.value: VLLMResourceFitSelector,
            }
            selector = selector_map.get(model.backend or BackendEnum.VLLM)
            world_size = None
            if selector:
                result = selector.get_world_size_from_backend_parameters(model)
                world_size, _ = result if result is not None else (None, None)
            if world_size and world_size > 0:
                return world_size

        # The largest power of 2 less than or equal to (total GPUs / replicas), used as the initial per-replica GPU count.
        return largest_power_of_2_leq(gpus_per_replica)

    if not model.gpu_selector or not model.gpu_selector.gpu_ids:
        return False

    if model.gpu_selector.gpus_per_replica and model.gpu_selector.gpus_per_replica > 0:
        return False

    gpus_per_replica = calculate_gpus_per_replica(model)
    model.gpu_selector.gpus_per_replica = gpus_per_replica
    try:
        flag_modified(model, "gpu_selector")
    except AttributeError:
        # Ignore if the given model is not a SQLModel instance.
        pass
    return True
