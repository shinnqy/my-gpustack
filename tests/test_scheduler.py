import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpustack.config.config import Config
from gpustack.scheduler import scheduler as scheduler_module
from gpustack.scheduler.queue import AsyncUniqueQueue
from gpustack.scheduler.scheduler import Scheduler, find_candidate, pick_highest_score_candidate
from gpustack.schemas.models import GPUSelector, Model, ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.workers import (
    Worker, WorkerStateEnum, WorkerStatus, GPUDeviceStatus, MemoryInfo,
)
from gpustack.server import db
from gpustack.server.bus import event_bus
from gpustack.server.services import ModelInstanceService

from test_watch import eventually, project_context, seed


def instance_record(id=1, state="pending", worker_id=None, updated_seconds=0, lifetime_seconds=600):
    now = datetime.now(timezone.utc)
    return ModelInstance(
        id=id, name=f"instance-{id}", model_id=1, model_name="qwen", source="huggingface",
        huggingface_repo_id="Qwen/test", state=state, worker_id=worker_id,
        created_at=now - timedelta(seconds=lifetime_seconds),
        updated_at=now - timedelta(seconds=updated_seconds),
    )


def worker_record(id=1, cluster_id=1, ready=True, gpu=True, labels=None):
    return Worker(
        id=id, cluster_id=cluster_id, name=f"worker-{id}", hostname=f"worker-{id}",
        ip=f"10.0.0.{id}", advertise_address=f"10.1.0.{id}", ifname="eth0", port=10150,
        worker_uuid=f"worker-uuid-{id}", labels=labels or {"os": "linux", "gpu": "ascend"},
        state=WorkerStateEnum.READY if ready else WorkerStateEnum.NOT_READY,
        heartbeat_time=datetime.now(timezone.utc),
        status=WorkerStatus(gpu_devices=[GPUDeviceStatus(
            index=2, type="cann", name="Ascend910B", uuid=f"device-{id}",
            memory=MemoryInfo(total=64 << 30),
        )] if gpu else []),
    )


async def add_worker(cluster_id, **kwargs):
    async with db.async_session() as session:
        worker = worker_record(cluster_id=cluster_id, **kwargs)
        return await Worker.create(session, worker)


async def row(id):
    async with db.async_session() as session:
        return await ModelInstance.one_by_id(session, id)


async def wait_state(id, state, timeout=4):
    async def wait():
        while True:
            instance = await row(id)
            if instance is not None and instance.state == state:
                return instance
            await asyncio.sleep(0.01)
    return await asyncio.wait_for(wait(), timeout=timeout)


def test_queue_deduplicates_different_snapshots_of_one_instance():
    async def check():
        queue = AsyncUniqueQueue()
        await asyncio.gather(
            queue.put(instance_record()),
            queue.put(instance_record(state="analyzing")),
        )
        assert queue.qsize() == 1
        item = await queue.get()
        queue.task_done()
        await queue.put(instance_record(state="pending"))
        assert queue.qsize() == 1
        assert item.id == 1

    asyncio.run(check())


@pytest.mark.parametrize("state,worker_id,updated,lifetime,expected", [
    ("pending", None, 0, 0, True),
    ("pending", None, 30, 600, False),
    ("pending", None, 91, 600, True),
    ("pending", 1, 120, 600, False),
    ("analyzing", None, 60, 600, False),
    ("analyzing", None, 181, 600, True),
    ("scheduled", 1, 60, 600, False),
    ("scheduled", 1, 181, 600, True),
    ("running", 1, 600, 900, False),
    ("error", None, 600, 900, False),
])
def test_schedule_eligibility(tmp_path, state, worker_id, updated, lifetime, expected):
    scheduler = Scheduler(Config(data_dir=str(tmp_path)))
    instance = instance_record(
        state=state, worker_id=worker_id, updated_seconds=updated, lifetime_seconds=lifetime,
    )
    assert scheduler._should_schedule(instance) is expected


def test_candidate_filters_cluster_gpu_labels_and_readiness(tmp_path):
    async def check():
        model = Model(
            id=1, name="qwen", source="huggingface", huggingface_repo_id="Qwen/test",
            cluster_id=1, worker_selector={"gpu": "ascend"},
            gpu_selector=GPUSelector(gpu_ids=["worker-2:cann:2", "worker-3:cann:2", "worker-4:cann:2"]),
        )
        workers = [
            worker_record(1, cluster_id=2), worker_record(2, ready=False),
            worker_record(3, labels={"gpu": "h100"}), worker_record(4),
        ]
        candidate, messages = await find_candidate(None, Config(data_dir=str(tmp_path)), model, workers, [])
        assert candidate.worker.id == 4
        assert candidate.gpu_type == "cann"
        assert candidate.gpu_indexes == [2]
        assert candidate.computed_resource_claim.vram == {2: 0}
        assert "cluster selector" in "".join(messages)
        assert "gpu selector" in "".join(messages)
        assert "label selector" in "".join(messages)

    asyncio.run(check())


@pytest.mark.parametrize("worker_options,selector,reason", [
    ({"cluster_id": 2}, {}, "cluster selector"),
    ({"ready": False}, {}, "READY status"),
    ({}, {"gpu": "h100"}, "label selector"),
    ({"gpu": False}, {}, "No GPU found"),
])
def test_candidate_failure_explains_reason(tmp_path, worker_options, selector, reason):
    async def check():
        model = Model(
            id=1, name="qwen", source="huggingface", huggingface_repo_id="Qwen/test",
            cluster_id=1, worker_selector=selector,
        )
        candidate, messages = await find_candidate(
            None, Config(data_dir=str(tmp_path)), model, [worker_record(**worker_options)], [],
        )
        assert candidate is None
        assert reason in "".join(messages)

    asyncio.run(check())


def test_candidate_ties_keep_first():
    first = SimpleNamespace(score=10)
    second = SimpleNamespace(score=10)
    assert pick_highest_score_candidate([first, second]) is first
    assert pick_highest_score_candidate([first, SimpleNamespace(score=11)]) is not first
    assert pick_highest_score_candidate([]) is None


def test_queue_consumer_is_serial_and_accounts_for_failures(tmp_path, monkeypatch):
    async def check():
        scheduler = Scheduler(Config(data_dir=str(tmp_path)))
        completed = []
        active = maximum = 0

        async def place(instance):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.01)
                if instance.id == 1:
                    raise RuntimeError("one placement failed")
                completed.append(instance.id)
            finally:
                active -= 1

        monkeypatch.setattr(scheduler, "_schedule_one", place)
        for id in (1, 2, 3):
            await scheduler._queue.put(instance_record(id))
        task = asyncio.create_task(scheduler._schedule_cycle())
        try:
            await asyncio.wait_for(scheduler._queue.queue.join(), timeout=1)
            assert completed == [2, 3]
            assert maximum == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_pending_becomes_scheduled_and_records_assignment(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            model, instance = await seed(env.client)
            worker = await add_worker(model["cluster_id"])
            scheduler = Scheduler(env.cfg)
            await scheduler._evaluate(await row(instance["id"]))
            assert (await row(instance["id"])).state == ModelInstanceStateEnum.ANALYZING
            await scheduler._schedule_one(await scheduler._queue.get())
            scheduler._queue.task_done()
            scheduled = await row(instance["id"])
            assert scheduled.state == ModelInstanceStateEnum.SCHEDULED
            assert scheduled.worker_id == worker.id
            assert scheduled.worker_name == worker.name
            assert scheduled.worker_ip == worker.ip
            assert scheduled.worker_ifname == worker.ifname
            assert scheduled.worker_advertise_address == worker.advertise_address
            assert scheduled.gpu_type == "cann"
            assert scheduled.gpu_indexes == [2]
            assert scheduled.state_message == ""

    asyncio.run(check())


def test_stale_queue_item_cannot_overwrite_running_instance(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            model, instance = await seed(env.client)
            await add_worker(model["cluster_id"])
            scheduler = Scheduler(env.cfg)
            await scheduler._evaluate(await row(instance["id"]))
            queued = await scheduler._queue.get()
            async with db.async_session() as session:
                current = await ModelInstance.one_by_id(session, instance["id"])
                await ModelInstanceService(session).update(current, {"state": ModelInstanceStateEnum.RUNNING})
            await scheduler._schedule_one(queued)
            assert (await row(instance["id"])).state == ModelInstanceStateEnum.RUNNING

    asyncio.run(check())


def test_failed_reassignment_clears_old_worker_and_can_retry(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            model, instance = await seed(env.client, state="scheduled", worker_id=1)
            worker = await add_worker(model["cluster_id"], ready=False)
            async with db.async_session() as session:
                current = await ModelInstance.one_by_id(session, instance["id"])
                await ModelInstanceService(session).update(current, {
                    "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
                    "updated_at": datetime.now(timezone.utc) - timedelta(minutes=4),
                })
            scheduler = Scheduler(env.cfg)
            await scheduler._evaluate(await row(instance["id"]))
            await scheduler._schedule_one(await scheduler._queue.get())
            pending = await row(instance["id"])
            assert pending.state == ModelInstanceStateEnum.PENDING
            assert pending.worker_id is None
            assert "READY status" in pending.state_message

            async with db.async_session() as session:
                current = await ModelInstance.one_by_id(session, instance["id"])
                await ModelInstanceService(session).update(current, {
                    "updated_at": datetime.now(timezone.utc) - timedelta(seconds=100),
                })
                current_worker = await Worker.one_by_id(session, worker.id)
                await current_worker.update(session, {"state": WorkerStateEnum.READY})
            await scheduler._enqueue_pending_instances()
            await scheduler._schedule_one(await scheduler._queue.get())
            assert (await row(instance["id"])).state == ModelInstanceStateEnum.SCHEDULED

    asyncio.run(check())


def test_startup_and_created_events_schedule_without_processing_updates(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            model, first = await seed(env.client)
            await add_worker(model["cluster_id"])
            scheduler = Scheduler(env.cfg)
            events = []
            original = scheduler._enqueue_event_instance

            async def capture(instance):
                events.append(instance.id)
                await original(instance)

            monkeypatch.setattr(scheduler, "_enqueue_event_instance", capture)
            task = asyncio.create_task(scheduler.start())
            try:
                await wait_state(first["id"], ModelInstanceStateEnum.SCHEDULED)
                await eventually(lambda: any(
                    subscriber.source == "scheduler"
                    for subscriber in event_bus.subscribers.get("modelinstance", [])
                ))
                response = await env.client.post("/v2/model-instances", json={
                    **first, "id": None, "name": "created-after-start", "state": "pending",
                })
                assert response.status_code == 200, response.text
                second = await wait_state(response.json()["id"], ModelInstanceStateEnum.SCHEDULED)
                assert events == [second.id]
                response = await env.client.put(
                    f"/v2/model-instances/{second.id}",
                    json={**second.model_dump(mode="json"), "state_message": "progress"},
                )
                assert response.status_code == 200, response.text
                await asyncio.sleep(0.05)
                assert events == [second.id]
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_periodic_retry_places_old_pending_after_worker_joins(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            model, instance = await seed(env.client)
            scheduler = Scheduler(env.cfg, check_interval=0.05)
            task = asyncio.create_task(scheduler.start())
            try:
                async def failed_once():
                    while True:
                        current = await row(instance["id"])
                        if current.state_message == "No available workers":
                            return
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(failed_once(), timeout=3)
                await add_worker(model["cluster_id"])
                async with db.async_session() as session:
                    current = await ModelInstance.one_by_id(session, instance["id"])
                    await ModelInstanceService(session).update(current, {
                        "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
                        "updated_at": datetime.now(timezone.utc) - timedelta(seconds=100),
                    })
                await wait_state(instance["id"], ModelInstanceStateEnum.SCHEDULED)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_stopping_scheduler_stops_timer_and_consumer(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            timers = []
            timer_type = scheduler_module.AsyncIOScheduler

            def capture(*args, **kwargs):
                timer = timer_type(*args, **kwargs)
                timers.append(timer)
                return timer

            monkeypatch.setattr(scheduler_module, "AsyncIOScheduler", capture)
            task = asyncio.create_task(Scheduler(env.cfg).start())
            try:
                await eventually(lambda: bool(event_bus.subscribers.get("modelinstance")))
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await asyncio.sleep(0)
                assert not timers[0].running
                consumers = [
                    pending for pending in asyncio.all_tasks()
                    if pending.get_coro().__qualname__ == "Scheduler._schedule_cycle"
                ]
                assert not consumers
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                for timer in timers:
                    if timer.running:
                        timer.shutdown(wait=False)
                consumers = [
                    pending for pending in asyncio.all_tasks()
                    if pending.get_coro().__qualname__ == "Scheduler._schedule_cycle"
                ]
                for consumer in consumers:
                    consumer.cancel()
                await asyncio.gather(*consumers, return_exceptions=True)

    asyncio.run(check())
