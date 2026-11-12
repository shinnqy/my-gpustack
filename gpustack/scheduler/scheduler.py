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
        """先评估，在串行放置"""
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

                # 第 9 章在这里读取模型配置（config.json），估算资源需求

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

                await ModelInstanceService(session).update(model_instance)

                logger.debug(
                    f"Scheduled model instance {model_instance.name} to worker "
                    f"{model_instance.worker_name} gpu {candidate.gpu_indexes}"
                )


class _AnyGPUSelector:
    """阶段版：每张卡都是一个候选，不检查显存。第 9 章换成 VLLMResourceFitSelector。"""

    def __init__(self, model: Model):
        self._model = model

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        candidates = []
        for worker in workers:
            for gpu in (worker.status.gpu_devices or []) if worker.status else []:
                candidates.append(
                    ModelInstanceScheduleCandidate(
                        worker=worker,
                        gpu_indexes=[gpu.index],
                        gpu_type=gpu.type,
                        computed_resource_claim=ComputedResourceClaim(vram={gpu.index: 0}),
                        score=0,
                    )
                )
        return candidates

    def get_messages(self) -> List[str]:
        return ["No GPU found on matched workers."]


async def find_candidate(
    session: AsyncSession,
    config: Config,
    model: Model,
    workers: List[Worker],
    model_instances: List[ModelInstance],
) -> Tuple[Optional[ModelInstanceScheduleCandidate], List[str]]:
    """
    过滤 Worker，然后选 GPU
    """
    # Filter workers.
    filters = [
        ClusterFilter(model),
        GPUMatchingFilter(model),
        LabelMatchingFilter(model),
        StatusFilter(model),
    ]

    worker_filter_chain = WorkerFilterChain(filters)
    workers, filter_messages = await worker_filter_chain.filter(workers)
    messages = []
    if filter_messages:
        messages.append(str(ListMessageBuilder(filter_messages)) + "\n")
    if len(workers) == 0:
        return None, messages

    # 为每张可见 GPU 生成一个候选
    candidates_selector = _AnyGPUSelector(model)

    # Select candidates.
    candidates = await candidates_selector.select_candidates(workers)

    # Pick the highest score candidate.
    candidate = pick_highest_score_candidate(candidates)

    if candidate is None and len(workers) > 0:
        messages.extend(candidates_selector.get_messages())

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
