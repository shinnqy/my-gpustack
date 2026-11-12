import asyncio
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import threading

import httpx
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import selectinload
from tqdm import tqdm

from gpustack.client import ClientSet
from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.server import db
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.controllers import ensure_instance_model_file, ModelFileController
from gpustack.server.init_db import create_db_and_tables
from gpustack.server.services import ModelInstanceService
from gpustack.utils import hub, locks
from gpustack.worker.model_file_manager import ModelFileDownloadTask, ModelFileManager
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker import downloaders

from test_watch import eventually, live_server, project_context, seed


async def prepare(env, name="file-test", worker_id=1, **changes):
    model, instance = await seed(env.client, name=name, state="initializing", worker_id=worker_id)
    async with db.async_session() as session:
        current = await ModelInstance.one_by_id(session, instance["id"])
        if changes:
            await ModelInstanceService(session).update(current, changes)
        await ensure_instance_model_file(session, current)
    return model, instance


async def files_for(instance_id):
    async with db.async_session() as session:
        current = await ModelInstance.one_by_id_with_model_files(session, instance_id)
        return current, list(current.model_files), list(current.draft_model_files)


async def change_file(id, **changes):
    async with db.async_session() as session:
        file = await ModelFile.one_by_id(session, id)
        await file.update(session, changes)
    await ModelFileController()._reconcile(Event(type=EventType.UPDATED, data=file))


def test_file_reuse_per_worker_and_links_exist_before_event(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            subscriber = event_bus.subscribe("modelfile", event_types={EventType.CREATED})
            try:
                _, first = await prepare(env, name="first")
                event = await asyncio.wait_for(subscriber.receive(), timeout=1)
                async with db.async_session() as session:
                    file = await ModelFile.one_by_id(session, event.id, options=[selectinload(ModelFile.instances)])
                    assert [instance.id for instance in file.instances] == [first["id"]]
                _, second = await prepare(env, name="second")
                _, other = await prepare(env, name="other-worker", worker_id=2)
                a, primary_a, _ = await files_for(first["id"])
                b, primary_b, _ = await files_for(second["id"])
                _, primary_other, _ = await files_for(other["id"])
                assert a.state == b.state == ModelInstanceStateEnum.DOWNLOADING
                assert primary_a[0].id == primary_b[0].id
                assert primary_a[0].id != primary_other[0].id
                assert primary_a[0].source_index == primary_other[0].source_index
                assert len(await ModelFile.cached_all()) == 2
            finally:
                event_bus.unsubscribe("modelfile", subscriber)

    asyncio.run(check())


def test_progress_ready_reuse_and_error_retry(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            _, first = await prepare(env)
            _, files, _ = await files_for(first["id"])
            file_id = files[0].id
            await change_file(file_id, download_progress=37.5)
            current, _, _ = await files_for(first["id"])
            assert current.download_progress == 37.5
            await change_file(file_id, state=ModelFileStateEnum.ERROR, state_message="connection failed")
            current, _, _ = await files_for(first["id"])
            assert current.state == ModelInstanceStateEnum.ERROR
            assert current.state_message == "connection failed"
            async with db.async_session() as session:
                current = await ModelInstance.one_by_id(session, first["id"])
                await ModelInstanceService(session).update(current, {"state": ModelInstanceStateEnum.INITIALIZING})
                await ensure_instance_model_file(session, current)
            _, files, _ = await files_for(first["id"])
            assert files[0].state == ModelFileStateEnum.DOWNLOADING
            assert files[0].download_progress == 0
            await change_file(file_id, state=ModelFileStateEnum.READY, download_progress=100, resolved_paths=["/cache/qwen"])
            current, _, _ = await files_for(first["id"])
            assert current.state == ModelInstanceStateEnum.STARTING
            assert current.download_progress == 100
            assert current.resolved_path == "/cache/qwen"
            _, shared = await prepare(env, name="ready-reuse")
            current, files, _ = await files_for(shared["id"])
            assert current.state == ModelInstanceStateEnum.STARTING
            assert files[0].id == file_id

    asyncio.run(check())


def test_primary_and_draft_must_both_be_ready_and_draft_cannot_be_deleted(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            _, instance = await prepare(env, draft_model_source={"source": "huggingface", "huggingface_repo_id": "Qwen/draft"})
            _, primary, draft = await files_for(instance["id"])
            assert len(primary) == len(draft) == 1
            assert draft[0].huggingface_repo_id == "Qwen/draft"
            assert primary[0].source_index != draft[0].source_index
            response = await env.client.delete(f"/v2/model-files/{draft[0].id}")
            assert response.status_code == 409
            await change_file(primary[0].id, state=ModelFileStateEnum.READY, download_progress=100, resolved_paths=["/cache/primary"])
            current, _, _ = await files_for(instance["id"])
            assert current.state == ModelInstanceStateEnum.DOWNLOADING
            await change_file(draft[0].id, state=ModelFileStateEnum.READY, download_progress=100, resolved_paths=["/cache/draft"])
            current, _, _ = await files_for(instance["id"])
            assert current.state == ModelInstanceStateEnum.STARTING
            assert current.resolved_path == "/cache/primary"
            assert current.draft_model_resolved_path == "/cache/draft"

    asyncio.run(check())


def test_model_file_routes_watch_updates_and_cleanup_flag(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            _, instance = await prepare(env)
            _, files, _ = await files_for(instance["id"])
            file = files[0]
            async with live_server(env.app) as url:
                clientset = ClientSet(url)
                async with httpx.AsyncClient(base_url=url + "/v2", trust_env=False) as network:
                    clientset.http_client.set_async_httpx_client(network)
                    events = []
                    task = asyncio.create_task(clientset.model_files.awatch(callback=events.append))
                    try:
                        await eventually(lambda: file.id in clientset.model_files._cache)
                        current = (await env.client.get(f"/v2/model-files/{file.id}")).json()
                        response = await env.client.put(f"/v2/model-files/{file.id}", json={**current, "download_progress": 42})
                        assert response.status_code == 200, response.text
                        await eventually(lambda: clientset.model_files._cache[file.id].download_progress == 42)
                        response = await env.client.delete(f"/v2/model-files/{file.id}")
                        assert response.status_code == 409
                        await env.client.delete(f"/v2/model-instances/{instance['id']}")
                        response = await env.client.delete(f"/v2/model-files/{file.id}?cleanup=true")
                        assert response.status_code == 200, response.text
                        await eventually(lambda: file.id not in clientset.model_files._cache)
                        assert events[-1].type == EventType.DELETED
                        assert events[-1].data["cleanup_on_delete"] is True
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_local_download_task_reports_size_paths_and_ready(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            path = tmp_path / "local-model"
            path.mkdir()
            (path / "model.safetensors").write_bytes(b"test-weights")
            _, instance = await prepare(env, source="local_path", local_path=str(path))
            _, files, _ = await files_for(instance["id"])
            async with live_server(env.app) as url:
                cfg = Config(data_dir=str(tmp_path / "worker"), server_url=url, token="registration-token")
                monkeypatch.setattr(tqdm, "__init__", tqdm.__init__)
                monkeypatch.setattr(tqdm, "update", tqdm.update)
                monkeypatch.setattr(tqdm, "_original_init", tqdm.__init__, raising=False)
                monkeypatch.setattr(tqdm, "_original_update", tqdm.update, raising=False)
                task = ModelFileDownloadTask(files[0], cfg, threading.Event())
                await asyncio.to_thread(task.run)
            async with db.async_session() as session:
                file = await ModelFile.one_by_id(session, files[0].id)
                assert file.state == ModelFileStateEnum.READY, file.state_message
                assert file.size == len(b"test-weights")
                assert file.resolved_paths == [str(path)]
            await ModelFileController()._reconcile(Event(type=EventType.UPDATED, data=file))
            current, _, _ = await files_for(instance["id"])
            assert current.state == ModelInstanceStateEnum.STARTING

    asyncio.run(check())


def test_download_failure_and_cancel_do_not_publish_ready(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            _, instance = await prepare(env, source="local_path", local_path=str(tmp_path / "missing-model"))
            _, files, _ = await files_for(instance["id"])
            async with live_server(env.app) as url:
                cfg = Config(data_dir=str(tmp_path / "worker"), server_url=url, token="registration-token")
                task = ModelFileDownloadTask(files[0], cfg, threading.Event())
                await asyncio.to_thread(task.run)
                flag = threading.Event()
                flag.set()
                cancelled = ModelFileDownloadTask(files[0], cfg, flag)
                cancelled.prerun = lambda: pytest.fail("cancelled task started")
                cancelled.run()
            async with db.async_session() as session:
                file = await ModelFile.one_by_id(session, files[0].id)
                assert file.state == ModelFileStateEnum.ERROR
                assert "does not exist" in file.state_message

    asyncio.run(check())


@pytest.mark.parametrize("deleted", [False, True])
def test_manager_failure_updates_error_without_awaiting_sync_method(tmp_path, monkeypatch, deleted):
    async def check():
        cfg = Config(data_dir=str(tmp_path))
        manager = ModelFileManager(1, SimpleNamespace(), cfg)
        future = Future()
        future.set_exception(RuntimeError("download worker failed"))
        manager._mp_manager = SimpleNamespace(Event=threading.Event)
        manager._download_pool = SimpleNamespace(submit=lambda *args: future)
        errors = []
        updates = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        def update(*args, **kwargs):
            updates.append(kwargs)
            if deleted:
                raise NotFoundException()
        monkeypatch.setattr(manager, "_update_model_file", update)
        file = ModelFile(id=1, source="local_path", local_path="/test", worker_id=1)
        manager._create_download_task(file)
        await eventually(lambda: not manager._completion_tasks)
        assert updates[0]["state"] == ModelFileStateEnum.ERROR
        assert errors == []

    asyncio.run(check())


def test_stopping_file_watch_closes_pool_and_manager(tmp_path, monkeypatch):
    async def check():
        started = asyncio.Event()
        closed = []
        async def watch(**kwargs):
            started.set()
            await asyncio.Future()
        manager = ModelFileManager(1, SimpleNamespace(model_files=SimpleNamespace(awatch=watch)), Config(data_dir=str(tmp_path)))
        def prerun():
            manager._download_pool = SimpleNamespace(shutdown=lambda **kwargs: closed.append("pool"))
            manager._mp_manager = SimpleNamespace(shutdown=lambda: closed.append("manager"))
        monkeypatch.setattr(manager, "_prerun", prerun)
        task = asyncio.create_task(manager.watch_model_files())
        await started.wait()
        task.cancel()
        await task
        assert closed == ["pool", "manager"]

    asyncio.run(check())


def test_hugging_face_downloader_preserves_nested_repo_paths(tmp_path, monkeypatch):
    local_dir = tmp_path / "downloaded"
    monkeypatch.setattr(downloaders, "match_hugging_face_files", lambda *args: ["nested/model.bin", "config.json"])
    def download(repo_id, filename, token, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode())
        return str(path)
    monkeypatch.setattr(downloaders, "hf_hub_download", download)
    result = downloaders.HfDownloader.download(
        "Qwen/test", None, None, local_dir=str(local_dir), cache_dir=str(tmp_path / "cache"),
    )
    assert result == [str(local_dir)]
    assert (local_dir / "nested/model.bin").read_bytes() == b"nested/model.bin"
    assert (local_dir / "config.json").is_file()


def test_manager_delete_cancels_job_and_removes_only_requested_files(tmp_path):
    async def check():
        cfg = Config(data_dir=str(tmp_path / "worker"))
        manager = ModelFileManager(1, SimpleNamespace(), cfg)
        path = tmp_path / "cache"
        path.mkdir()
        (path / "weights.bin").write_bytes(b"weights")
        flag = threading.Event()
        future = Future()
        manager._active_downloads[1] = (future, flag)
        file = ModelFile(id=1, source="local_path", local_path=str(path), worker_id=1, resolved_paths=[str(path)], cleanup_on_delete=True)
        await manager._handle_deletion(file)
        assert flag.is_set()
        assert future.cancelled()
        assert not path.exists()

    asyncio.run(check())


def test_replayed_scheduled_instance_is_accepted(tmp_path, monkeypatch):
    cfg = Config(data_dir=str(tmp_path))
    manager = ServeManager(lambda: 1, lambda: None, cfg)
    started = []
    monkeypatch.setattr(manager, "_start_model_instance", lambda instance: started.append(instance.id))
    for event_type in (EventType.CREATED, EventType.UPDATED):
        manager._dispatch_model_instance_event(Event(type=event_type, data={
            "id": 3, "name": "scheduled", "model_id": 1, "model_name": "model",
            "source": "huggingface", "huggingface_repo_id": "Qwen/test",
            "worker_id": 1, "state": "scheduled",
        }))
    assert started == [3, 3]


def test_existing_instance_table_is_upgraded_without_deleting_rows(tmp_path):
    async def check():
        engine = create_async_engine("sqlite+aiosqlite:///" + str(tmp_path / "old.sqlite"))
        try:
            async with engine.begin() as connection:
                await connection.execute(text("CREATE TABLE model_instances (id INTEGER PRIMARY KEY, name VARCHAR)"))
                await connection.execute(text("INSERT INTO model_instances(id,name) VALUES (41,'keep-me')"))
            await create_db_and_tables(engine)
            await create_db_and_tables(engine)
            async with engine.connect() as connection:
                names = await connection.run_sync(lambda sync: {col["name"] for col in inspect(sync).get_columns("model_instances")})
                assert {"draft_model_source", "draft_model_download_progress", "draft_model_resolved_path", "distributed_servers"} <= names
                assert (await connection.execute(text("SELECT name FROM model_instances WHERE id=41"))).scalar_one() == "keep-me"
        finally:
            await engine.dispose()

    asyncio.run(check())


@pytest.mark.parametrize("soft", [False, True])
def test_download_lock_supports_nested_paths_and_reuse(tmp_path, monkeypatch, soft):
    monkeypatch.setattr(locks, "DISABLE_OS_FILELOCK", soft)
    path = tmp_path / "nested" / "repo.lock"
    lock = locks.HeartbeatSoftFileLock(str(path), heartbeat_seconds=0.01)
    for _ in range(2):
        with lock:
            assert path.exists()
        assert lock._hb_thread is None or not lock._hb_thread.is_alive()


def test_progress_is_capped_until_download_finishes(tmp_path, monkeypatch):
    file = ModelFile(id=1, source="local_path", local_path=str(tmp_path), size=100)
    task = ModelFileDownloadTask(file, Config(data_dir=str(tmp_path / "worker")), threading.Event())
    task._speed_lock = threading.Lock()
    task._model_file_size = 100
    task._model_downloaded_size = 0
    progress = []
    task._update_progress_func = progress.append
    monkeypatch.setattr(tqdm, "__init__", tqdm.__init__)
    monkeypatch.setattr(tqdm, "update", tqdm.update)
    monkeypatch.setattr(tqdm, "_original_init", tqdm.__init__, raising=False)
    monkeypatch.setattr(tqdm, "_original_update", tqdm.update, raising=False)
    task.hijack_tqdm_progress()
    with tqdm(total=100, unit="B", desc="model.bin") as bar:
        bar.update(25)
        bar.update(75)
    assert progress == [25, 99]
