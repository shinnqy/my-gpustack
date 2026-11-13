import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from gpustack import envs
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.workers import Maintenance, Worker, WorkerStateEnum
from gpustack.server import db
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import ModelController, WorkerController
from gpustack.server.init_db import create_db_and_tables
from gpustack.server.services import ModelInstanceService, WorkerService
from gpustack.server.worker_instance_cleaner import WorkerInstanceCleaner
from gpustack.server.worker_syncer import WorkerSyncer
from gpustack.worker import serve_manager as serve_module
from gpustack.worker import workload_cleaner as cleaner_module
from gpustack.worker.inference_backend_manager import InferenceBackendManager
from gpustack.worker.serve_manager import ServeManager, is_ready, _describe_workload_failure
from gpustack.worker.workload_cleaner import WorkloadCleaner
from gpustack_runtime.deployer import WorkloadStatusStateEnum as RuntimeState

from test_resource_fit import instance, model, worker, config_stub
from test_watch import project_context, seed


@pytest.fixture(autouse=True)
def simulated_runtime(monkeypatch):
    setup_logging()
    monkeypatch.setattr(serve_module, "get_workload", lambda name: None)
    monkeypatch.setattr(serve_module, "delete_workload", lambda name: None)
    monkeypatch.setattr(cleaner_module, "list_workloads", lambda: [])
    monkeypatch.setattr(cleaner_module, "delete_workload", lambda name: None)


def manager(tmp_path, rows):
    calls = []
    patches = []
    selected_model = model(backend_version="0.11.0")
    def listing(params=None, use_cache=True):
        calls.append((params, use_cache))
        return SimpleNamespace(items=rows)
    client = SimpleNamespace(
        model_instances=SimpleNamespace(list=listing, get=lambda id: next(row for row in rows if row.id == id)),
        models=SimpleNamespace(get=lambda id: selected_model),
        workers=SimpleNamespace(get=lambda id: worker(id)),
    )
    selected = ServeManager(lambda: 1, lambda: client, Config(data_dir=str(tmp_path)))
    selected._inference_backend_manager = InferenceBackendManager(client)
    selected._update_model_instance = lambda id, **kwargs: patches.append((id, kwargs))
    return selected, client, selected_model, calls, patches


@pytest.mark.parametrize("backend,path,expected", [
    ("vLLM", None, "/v1/models"), ("MindIE", None, "/info"),
    ("vLLM", "/health", "/health"), ("Custom", None, None),
])
def test_readiness_uses_backend_path_and_worker_ip(monkeypatch, backend, path, expected):
    urls = []
    monkeypatch.setattr(serve_module.requests, "get", lambda url, **kwargs: urls.append(url) or SimpleNamespace(status_code=200))
    mi = instance(worker_ip="10.0.0.9", port=40001)
    assert is_ready(backend, mi, path)
    assert urls == ([] if expected is None else ["http://10.0.0.9:40001" + expected])


@pytest.mark.parametrize("runtime_state,ready,expected", [
    (RuntimeState.INITIALIZING, True, None),
    (RuntimeState.RUNNING, False, None),
    (RuntimeState.RUNNING, True, ModelInstanceStateEnum.RUNNING),
    (RuntimeState.FAILED, False, ModelInstanceStateEnum.ERROR),
])
def test_runtime_state_and_readiness_drive_instance_state(tmp_path, monkeypatch, runtime_state, ready, expected):
    mi = instance(state="starting", port=40001, ports=[40001])
    selected, _, _, _, patches = manager(tmp_path, [mi])
    monkeypatch.setattr(serve_module, "get_workload", lambda name: SimpleNamespace(state=runtime_state, state_message="image failed", exits=[]))
    monkeypatch.setattr(serve_module, "is_ready", lambda *args: ready)
    selected.sync_model_instances_state()
    assert (patches[-1][1]["state"] if patches else None) == expected
    if expected == ModelInstanceStateEnum.ERROR:
        assert patches[-1][1]["state_message"] == "image failed"


def test_failure_message_preserves_runtime_reason_and_exit_code():
    status = SimpleNamespace(state_message="Out of memory", exits=[SimpleNamespace(name="engine", exit_code=137, reason="OOMKilled")])
    message = _describe_workload_failure(status)
    assert "Out of memory" in message and "137" in message
    assert _describe_workload_failure(None)


def test_reap_confirms_empty_cache_against_server_and_handles_real_deletion(tmp_path, monkeypatch):
    mi = instance(state="running")
    selected, client, _, calls, _ = manager(tmp_path, [])
    selected._model_instance_by_instance_id[mi.id] = mi
    stopped = []
    monkeypatch.setattr(selected, "_stop_model_instance", lambda mi, **kwargs: stopped.append(mi.id))
    client.model_instances.list = lambda params=None, use_cache=True: SimpleNamespace(items=[] if use_cache else [mi])
    monkeypatch.setattr(serve_module, "get_workload", lambda name: SimpleNamespace(state=RuntimeState.RUNNING))
    selected.sync_model_instances_state()
    assert stopped == []
    client.model_instances.list = lambda **kwargs: SimpleNamespace(items=[])
    selected.sync_model_instances_state()
    assert stopped == [mi.id]


def test_server_failure_cannot_reap_workloads(tmp_path, monkeypatch):
    mi = instance(state="running")
    selected, client, _, _, _ = manager(tmp_path, [])
    selected._model_instance_by_instance_id[mi.id] = mi
    stopped = []
    monkeypatch.setattr(selected, "_stop_model_instance", lambda mi, **kwargs: stopped.append(mi.id))
    def listing(params=None, use_cache=True):
        if not use_cache:
            raise RuntimeError("Server unavailable")
        return SimpleNamespace(items=[])
    client.model_instances.list = listing
    with pytest.raises(RuntimeError, match="unavailable"):
        selected.sync_model_instances_state()
    assert stopped == []


@pytest.mark.parametrize("count,delay", [(0, 0), (1, 10), (2, 20), (6, 300), (20, 300)])
def test_restart_exponential_backoff_and_limit(tmp_path, count, delay):
    mi = instance(state="error", restart_count=7, last_restart_time=datetime.now(timezone.utc) - timedelta(seconds=max(delay - 1, 0)))
    selected, _, _, _, patches = manager(tmp_path, [mi])
    selected._restart_backoff_counts[mi.id] = count
    selected._error_model_instances[mi.id] = mi
    selected._restart_error_model_instance(mi)
    if delay:
        assert patches == []
        mi.last_restart_time = datetime.now(timezone.utc) - timedelta(seconds=delay + 1)
        selected._restart_error_model_instance(mi)
    assert patches[-1][1]["state"] == ModelInstanceStateEnum.SCHEDULED
    assert patches[-1][1]["restart_count"] == 8
    assert selected._restart_backoff_counts[mi.id] == count + 1


def test_restart_respects_updated_model_setting_and_recovered_state(tmp_path):
    mi = instance(state="error")
    selected, client, selected_model, _, patches = manager(tmp_path, [mi])
    selected_model.restart_on_error = False
    selected._error_model_instances[mi.id] = mi
    selected._restart_error_model_instance(mi)
    assert patches == []
    selected_model.restart_on_error = True
    client.model_instances.get = lambda id: instance(state="running")
    selected._error_model_instances[mi.id] = mi
    selected._restart_error_model_instance(mi)
    assert patches == []


def test_success_clears_backoff_and_unreachable_waits_for_worker_recovery(tmp_path, monkeypatch):
    mi = instance(state="running")
    selected, client, _, _, patches = manager(tmp_path, [mi])
    selected._restart_backoff_counts[mi.id] = 5
    monkeypatch.setattr(serve_module, "get_workload", lambda name: SimpleNamespace(state=RuntimeState.RUNNING))
    monkeypatch.setattr(serve_module, "is_ready", lambda *args: True)
    selected.sync_model_instances_state()
    assert mi.id not in selected._restart_backoff_counts
    mi.state = ModelInstanceStateEnum.UNREACHABLE
    client.workers.get = lambda id: worker(state="unreachable", unreachable=True)
    selected.sync_model_instances_state()
    assert patches == []
    client.workers.get = lambda id: worker()
    selected.sync_model_instances_state()
    assert patches[-1][1]["state"] == ModelInstanceStateEnum.RUNNING


@pytest.mark.parametrize("mode,count,expected", [("enabled", 100, True), ("disabled", 1, False), ("auto", 50, True), ("auto", 51, False)])
def test_reachability_mode(monkeypatch, mode, count, expected):
    monkeypatch.setattr(envs, "WORKER_UNREACHABLE_CHECK_MODE", mode)
    assert WorkerSyncer(lambda: None, lambda: None)._should_check_unreachable(count) is expected


def test_worker_sync_heartbeat_recovery_and_maintenance_endpoint(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as context:
            stored_model, _ = await seed(context.client)
            async with db.async_session() as session:
                node = await Worker.create(session, worker(cluster_id=stored_model["cluster_id"], heartbeat_time=datetime.now(timezone.utc) - timedelta(seconds=160)))
            syncer = WorkerSyncer(lambda: None, lambda: None)
            async def reachable(node):node.unreachable = False
            monkeypatch.setattr(syncer, "_set_worker_unreachable", reachable)
            await syncer._sync_workers_states()
            async with db.async_session() as session:
                current = await Worker.one_by_id(session, node.id)
                assert current.state == WorkerStateEnum.NOT_READY
                await WorkerService(session).update(current, {"heartbeat_time": datetime.now(timezone.utc)})
            await syncer._sync_workers_states()
            response = await context.client.get(f"/v2/workers/{node.id}")
            assert response.json()["state"] == "ready"
            response = await context.client.put(f"/v2/workers/{node.id}", json={"name": node.name, "maintenance": {"enabled": True, "message": "planned"}})
            assert response.status_code == 200, response.text
            assert response.json()["state"] == "maintenance"

    asyncio.run(check())


def test_worker_controller_preserves_cluster_isolation(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as context:
            selected_model, first = await seed(context.client, state="running", worker_id=1)
            _, other = await seed(context.client, name="other-cluster", state="running", worker_id=1)
            async with db.async_session() as session:
                node = await Worker.create(session, worker(cluster_id=selected_model["cluster_id"], state="not_ready"))
            await WorkerController(context.cfg)._reconcile(Event(type=EventType.UPDATED, data=node, changed_fields={"state": ("ready", "not_ready")}))
            async with db.async_session() as session:
                assert (await ModelInstance.one_by_id(session, first["id"])).state == ModelInstanceStateEnum.UNREACHABLE
                assert (await ModelInstance.one_by_id(session, other["id"])).state == ModelInstanceStateEnum.RUNNING

    asyncio.run(check())


def test_offline_cleanup_respects_maintenance_and_rebuilds_replica(tmp_path, monkeypatch, config_stub):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as context:
            selected_model, instance_data = await seed(context.client, state="unreachable", worker_id=1)
            async with db.async_session() as session:
                node = await Worker.create(session, worker(cluster_id=selected_model["cluster_id"], state="not_ready", heartbeat_time=datetime.now(timezone.utc) - timedelta(seconds=301), maintenance={"enabled": True}))
                await Worker.create(session, worker(2, cluster_id=selected_model["cluster_id"]))
                stored = await Model.one_by_id(session, selected_model["id"])
                await stored.update(session, {"env": {"GPUSTACK_MODEL_VRAM_CLAIM": str(3 << 30)}})
            cleaner = WorkerInstanceCleaner()
            await cleaner._cleanup_offline_worker_instances()
            async with db.async_session() as session:
                assert await ModelInstance.one_by_id(session, instance_data["id"]) is not None
                node = await Worker.one_by_id(session, node.id)
                await WorkerService(session).update(node, {"maintenance": None})
            await cleaner._cleanup_offline_worker_instances()
            async with db.async_session() as session:
                assert await ModelInstance.all(session) == []
                stored = await Model.one_by_id(session, selected_model["id"])
            await ModelController(context.cfg)._reconcile(Event(type=EventType.UPDATED, data=stored))
            async with db.async_session() as session:
                rebuilt = (await ModelInstance.all(session))[0]
            from gpustack.scheduler.scheduler import Scheduler
            scheduler = Scheduler(context.cfg)
            await scheduler._evaluate(rebuilt)
            await scheduler._schedule_one(await scheduler._queue.get())
            async with db.async_session() as session:
                rebuilt = (await ModelInstance.all(session))[0]
                assert rebuilt.worker_id == 2
                assert rebuilt.state == ModelInstanceStateEnum.SCHEDULED

    asyncio.run(check())


def test_orphan_cleanup_uses_fresh_server_truth_and_skips_other_workload_types(tmp_path, monkeypatch):
    current = instance(name="live-instance")
    calls = []
    def listing(params=None, use_cache=True):
        calls.append(use_cache)
        return SimpleNamespace(items=[] if use_cache else [current])
    client = SimpleNamespace(model_instances=SimpleNamespace(list=listing))
    cleaner = WorkloadCleaner(lambda: 1, lambda: client)
    old = (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()
    fresh = datetime.now(timezone.utc).isoformat()
    workloads = [SimpleNamespace(name=name, created_at=created, labels=labels, state=RuntimeState.RUNNING) for name, created, labels in [
        ("live-instance", old, {}), ("orphan", old, {}), ("new-job", fresh, {}),
        ("benchmark", old, {"type": "benchmark"}), ("cache", old, {"type": "cache-service"}),
    ]]
    deleted = []
    monkeypatch.setattr(cleaner_module, "list_workloads", lambda: workloads)
    monkeypatch.setattr(cleaner_module, "delete_workload", deleted.append)
    cleaner.cleanup_orphan_workloads()
    assert calls == [False]
    assert deleted == ["orphan"]


def test_worker_schema_upgrade_preserves_old_rows(tmp_path):
    async def check():
        engine = create_async_engine("sqlite+aiosqlite:///" + str(tmp_path / "old.sqlite"))
        try:
            async with engine.begin() as connection:
                await connection.execute(text("CREATE TABLE workers (id INTEGER PRIMARY KEY, name VARCHAR)"))
                await connection.execute(text("INSERT INTO workers(id,name) VALUES (9,'keep-worker')"))
            await create_db_and_tables(engine)
            await create_db_and_tables(engine)
            async with engine.connect() as connection:
                columns = await connection.run_sync(lambda sync: {field["name"] for field in inspect(sync).get_columns("workers")})
                assert {"maintenance", "proxy_mode", "proxy_address"} <= columns
                assert (await connection.execute(text("SELECT name FROM workers WHERE id=9"))).scalar_one() == "keep-worker"
        finally:
            await engine.dispose()

    asyncio.run(check())


def test_optional_inference_probe_is_disabled_by_default_and_counts_failures(tmp_path, monkeypatch):
    mi = instance(state="running", port=40001, worker_ip="127.0.0.1")
    selected, _, selected_model, _, patches = manager(tmp_path, [mi])
    selected._model_instance_by_instance_id[mi.id] = mi
    requests = []
    monkeypatch.setattr(serve_module.requests, "post", lambda url, **kwargs: requests.append(url) or SimpleNamespace(status_code=500))
    selected.sync_model_instances_inference_health()
    assert requests == []
    selected_model.env = {
        "GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED": "true",
        "GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL": "0",
        "GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD": "2",
    }
    selected.sync_model_instances_inference_health()
    assert patches == []
    selected.sync_model_instances_inference_health()
    assert patches[-1][1]["state"] == ModelInstanceStateEnum.ERROR
    assert len(requests) == 2


@pytest.mark.parametrize("category,path", [("llm", "/v1/chat/completions"), ("embedding", "/v1/embeddings"), ("reranker", "/v1/rerank")])
def test_active_probe_uses_category_endpoint(monkeypatch, category, path):
    requests = []
    monkeypatch.setattr(serve_module.requests, "post", lambda url, **kwargs: requests.append((url, kwargs)) or SimpleNamespace(status_code=200))
    assert serve_module.is_inference_ready(instance(port=40001, worker_ip="127.0.0.1"), model(categories=[category]))
    assert requests[0][0].endswith(path)
