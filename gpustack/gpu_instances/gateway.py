from gpustack.gpu_instances import gateway_client
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.workers import Worker
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.bus import Event
from gpustack.server.bus import EventType
from gpustack.server.bus import event_field
from gpustack.server.bus import resolve_event_id
from typing import Dict
from typing import Optional
from typing import Set
import asyncio
from gpustack.schemas.clusters import is_gpu_service_cluster
from gpustack.server.db import async_session
import logging
from sqlmodel import select
from sqlalchemy import func


logger = logging.getLogger(__name__)

"""IMPORTANT: 告诉 wg 该管哪些集群：决定哪些集群需要订阅，何时订阅、更新或取消"""


_SOURCE = "gpustack_operator_subscription"


_GVK = [
    ("worker.gpustack.ai", "v1", "InstanceType"),
    ("worker.gpustack.ai", "v1", "Instance"),
]


async def reconcile_gpustack_operator_subscription():
    """Watch Cluster and Worker events and keep the operator gateway in sync."""
    await OperatorSubscriptionReconciler().start()


class OperatorSubscriptionReconciler:
    """
    集群订阅协调器：它负责维护本地 Operator 网关的“集群名单”：哪个集群启用了 GPU Service、现在是否有可用 Worker、是否应该接入或取消接入。
        - 需要这个是因为集群的状态是不断变化的，所以需要 reconciler
        - IMPORTANT: 本地 Operator 网关 指的是 “gpustack-operator 项目中的 worker-gateway 组件，简称 wg”

    (gateway_client.py 是它发送请求时使用的通信工具)
    """

    def __init__(self):
        # Cluster ids that should be subscribed once reachable, mapped to the
        # registration token to subscribe with — a Worker event carries the
        # cluster id but not the token.
        self._eligible: Dict[int, Optional[str]] = {}
        self._subscribed: Set[int] = set()
        self._active_tokens: Dict[int, Optional[str]] = {}
        # Both watchers mutate the state above across await points.
        self._lock = asyncio.Lock()

    async def start(self):
        watchers = [
            asyncio.create_task(
                self._watch(
                    "cluster",
                    Cluster.subscribe(source=_SOURCE),
                    self._reconcile_cluster,
                )
            ),
            # ``replay_existing=False``: the Cluster stream's own startup replay
            # already resolves every eligible cluster's reachability from the
            # database, so replaying each worker row would only repeat that. A
            # worker event that lands before its cluster is known is dropped, and
            # recovered by the cluster heartbeat's retry.
            asyncio.create_task(
                self._watch(
                    "worker",
                    Worker.subscribe(source=_SOURCE, replay_existing=False),
                    self._reconcile_worker,
                )
            ),
        ]
        try:
            await asyncio.gather(*watchers)
        finally:
            # gather leaves the other watcher running when one raises. Both are needed
            # to decide a subscription, so a half-dead reconciler would keep the process
            # up while silently subscribing nothing. The cancellation is joined so the
            # survivor is stopped, not merely asked to stop, before this returns.
            for watcher in watchers:
                watcher.cancel()
            await asyncio.gather(*watchers, return_exceptions=True)

    async def _watch(self, name, stream, reconcile):
        try:
            async for event in stream:
                try:
                    await reconcile(event)
                except Exception as error:
                    logger.exception(f"Failed to reconcile operator subscription on a {name} event: {error}")
        finally:
            close = getattr(stream, "aclose", None)
            if close:
                await close()

    async def _reconcile_cluster(self, event: Event):
        if event.type == EventType.HEARTBEAT:
            await self._retry_stale_subscriptions()
            return

        cluster: Cluster = event.data
        if cluster is None:
            return

        # A DELETED event can arrive as the raw row id rather than a Cluster
        # (see :func:`deleted_cluster_id`). The id is all this path needs — the
        # cluster is gone either way — and reading ``provider`` off that dict
        # would raise, be swallowed by the watcher, and leave the operator
        # proxying a deleted cluster forever: the retry sweep only unsubscribes
        # what left ``_eligible``, which is exactly what the raise skipped.
        if isinstance(cluster, dict):
            cluster_id = deleted_cluster_id(event)
            if cluster_id is not None:
                async with self._lock:
                    self._eligible.pop(cluster_id, None)
                    await self._unsubscribe(cluster_id)
            return

        if cluster.provider != ClusterProvider.Kubernetes:
            return

        async with self._lock:
            if event.type == EventType.DELETED or not _has_gpu_instances(cluster):
                self._eligible.pop(cluster.id, None)
                await self._unsubscribe(cluster.id)
                return

            self._eligible[cluster.id] = cluster.registration_token
            if cluster.id in self._subscribed:
                if self._active_tokens.get(cluster.id) != cluster.registration_token:
                    await self._subscribe(cluster.id, force=True)
                return
            if await count_ready_workers(cluster.id) > 0:
                await self._subscribe(cluster.id)

    async def _retry_stale_subscriptions(self):
        """心跳重新核对数据库状态，恢复失败调用和仅携带 ID 的删除事件。"""
        async with self._lock:
            for cluster_id in sorted(set(self._eligible) | self._subscribed):
                try:
                    reachable = cluster_id in self._eligible and await count_ready_workers(cluster_id) > 0
                    if not reachable:
                        await self._unsubscribe(cluster_id)
                    elif cluster_id not in self._subscribed:
                        await self._subscribe(cluster_id)
                    elif self._active_tokens.get(cluster_id) != self._eligible[cluster_id]:
                        await self._subscribe(cluster_id, force=True)
                except Exception as error:
                    logger.error(f"Failed to retry operator subscription for {cluster_id}: {error}")

    async def _reconcile_worker(self, event: Event):
        if event.type == EventType.HEARTBEAT:
            return
        worker: Worker = event.data
        # Same id-only payload as the cluster path above, for the same reason
        # (see :func:`deleted_cluster_id`) -- but here the id is not enough:
        # this reconciler is keyed on the worker's cluster, which a deleted
        # row can no longer name.
        #
        # ID-only deletes are reconciled by the next heartbeat database scan.
        cluster_id = event_field(worker, "cluster_id")
        if cluster_id is None:
            logger.warning(
                f"Worker {resolve_event_id(event)} {event.type} not reconciled: "
                f"the event carries only an id, so its cluster is unknown; a "
                f"subscription to it may be left in place"
            )
            return

        if event.type == EventType.UPDATED and "state" not in (
            event.changed_fields or {}
        ):
            # Every worker still posting status is republished every few seconds. Only
            # a state change can move a cluster in or out of reachability, so anything
            # else is dropped before it can cost a query.
            return

        async with self._lock:
            if cluster_id not in self._eligible:
                return

            if (
                event.type != EventType.DELETED
                and event_field(worker, "state") == WorkerStateEnum.READY
            ):
                if cluster_id not in self._subscribed:
                    await self._subscribe(cluster_id)
                return

            # A worker leaving READY only matters to a subscribed cluster, and
            # only when it was the last reachable one — so the count is queried
            # for that case alone, keeping worker status events cheap.
            if cluster_id not in self._subscribed:
                return
            if await count_ready_workers(cluster_id) == 0:
                await self._unsubscribe(cluster_id)

    async def _subscribe(self, cluster_id: int, force: bool = False):
        await gateway_client.subscribe_worker(
            str(cluster_id), self._eligible[cluster_id], gvk=_GVK, force=force,
        )
        # Recorded after the call so a failure is retried on the next event.
        self._subscribed.add(cluster_id)
        self._active_tokens[cluster_id] = self._eligible[cluster_id]
        logger.info(f"Subscribed cluster {cluster_id} to the operator gateway.")

    async def _unsubscribe(self, cluster_id: int):
        if cluster_id not in self._subscribed:
            return
        await gateway_client.unsubscribe_worker(str(cluster_id))
        self._subscribed.discard(cluster_id)
        self._active_tokens.pop(cluster_id, None)
        logger.info(f"Unsubscribed cluster {cluster_id} from the operator gateway.")


def _has_gpu_instances(cluster: Cluster) -> bool:
    # Over the bus the ``k8s_options`` JSON column can arrive as a plain dict
    # (nested pydantic_column_type isn't re-validated on replay). The schema-level
    # predicate reads that raw shape directly — both key spellings — so the dict no
    # longer has to be re-validated back into a model here just to read one field.
    return is_gpu_service_cluster(cluster)


async def count_ready_workers(cluster_id: int) -> int:
    """How many of a cluster's workers are READY — its reachability.

    Public because ``settings.py``'s reconciler decides reachability the same
    way and from the same query: a cluster with no READY worker has no reachable
    proxy, so every call would 503. Shared rather than copied so the two cannot
    drift apart on what "reachable" means.
    """
    async with async_session() as session:
        statement = select(func.count(Worker.id)).where(
            Worker.cluster_id == cluster_id, Worker.state == WorkerStateEnum.READY,
        )
        return (await session.exec(statement)).one()


def deleted_cluster_id(event: Event) -> Optional[int]:
    """The row id a cluster DELETED event carries when its payload never became
    a ``Cluster``, or ``None`` when the event is anything else.

    The bus enriches a DELETED event from the change-detector cache, and that
    cache is empty for clusters — the topic is not preloaded (see
    ``Server._preload_change_detector_cache``) — so the event is routed carrying
    nothing but ``{"id": ...}``. Every reconciler watching clusters therefore
    has to read the id off a raw dict, and a deletion is all the id is needed
    for; ``None`` for any other event type on such a payload, which has no model
    to read and nothing safe to do.
    """
    if event.type != EventType.DELETED:
        return None
    return resolve_event_id(event)
