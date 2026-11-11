import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import socket
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
import uvicorn

from gpustack.client import ClientSet
from gpustack.client.generated_http_client import HTTPClient
from gpustack.client.generated_model_instance_client import ModelInstanceClient
from gpustack.config import config as config_module
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.models import (
    Model, ModelInstance, ModelInstancePublic, ModelInstanceStateEnum,
)
from gpustack.server import cache as cache_module
from gpustack.server import db
from gpustack.server.app import create_app
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.init_db import create_db_and_tables
from gpustack.server.services import ModelInstanceService, ModelService
from gpustack.server.controllers import ModelInstanceController


@asynccontextmanager
async def project_context(tmp_path, monkeypatch):
    setup_logging()
    cfg = Config(data_dir=str(tmp_path / "server"), token="bootstrap")
    monkeypatch.setattr(config_module, "_config", cfg)
    engine = create_async_engine("sqlite+aiosqlite://")
    monkeypatch.setattr(db, "engine", engine)
    await cache_module.cache.clear()
    cache_module._cache_locks.clear()
    try:
        await create_db_and_tables(engine)
        app = create_app(cfg)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield SimpleNamespace(app=app, client=client, cfg=cfg)
    finally:
        for subscribers in list(event_bus.subscribers.values()):
            for subscriber in list(subscribers):
                event_bus.unsubscribe(subscriber.topic, subscriber)
        await cache_module.cache.clear()
        cache_module._cache_locks.clear()
        await engine.dispose()


async def seed(client, name="qwen-test", state="pending", worker_id=None):
    response = await client.post("/v2/clusters", json={"name": name + "-cluster"})
    assert response.status_code == 200, response.text
    cluster = response.json()
    response = await client.post("/v2/models", json={
        "name": name, "source": "huggingface", "huggingface_repo_id": "Qwen/test",
        "cluster_id": cluster["id"], "replicas": 1,
    })
    assert response.status_code == 200, response.text
    model = response.json()
    response = await client.post("/v2/model-instances", json={
        "name": name + "-instance", "model_name": name, "model_id": model["id"],
        "source": "huggingface", "huggingface_repo_id": "Qwen/test",
        "cluster_id": cluster["id"], "state": state, "worker_id": worker_id,
    })
    assert response.status_code == 200, response.text
    return model, response.json()


async def eventually(predicate, timeout=5):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout=timeout)


@asynccontextmanager
async def live_server(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app, log_level="critical", lifespan="off", timeout_graceful_shutdown=1,
    ))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await eventually(lambda: server.started or task.done())
        if task.done():
            await task
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        sock.close()


def test_snapshot_subscription_closes_during_replay(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            _, instance = await seed(env.client)
            stream = ModelInstance.subscribe(source="replay-close-test")
            first = await anext(stream)
            assert first.data.id == instance["id"]
            await stream.aclose()
            assert not event_bus.subscribers.get("modelinstance")

    asyncio.run(check())


def test_real_http_watch_updates_client_cache(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            model, instance = await seed(env.client)
            async with live_server(env.app) as url:
                clientset = ClientSet(url)
                async with httpx.AsyncClient(base_url=url + "/v2", trust_env=False) as network:
                    clientset.http_client.set_async_httpx_client(network)
                    events = []
                    watch = asyncio.create_task(clientset.model_instances.awatch(callback=events.append))
                    model_watch = asyncio.create_task(clientset.models.awatch())
                    try:
                        await eventually(lambda: instance["id"] in clientset.model_instances._cache)
                        await eventually(lambda: model["id"] in clientset.models._cache)
                        assert clientset.model_instances.get(instance["id"]).state == ModelInstanceStateEnum.PENDING
                        assert clientset.models.get(model["id"]).name == model["name"]

                        response = await env.client.put(
                            f"/v2/model-instances/{instance['id']}",
                            json={**instance, "state": "running"},
                        )
                        assert response.status_code == 200, response.text
                        await eventually(lambda: clientset.model_instances.get(instance["id"]).state == ModelInstanceStateEnum.RUNNING)
                        response = await env.client.delete(f"/v2/model-instances/{instance['id']}")
                        assert response.status_code == 200, response.text
                        await eventually(lambda: instance["id"] not in clientset.model_instances._cache)
                        assert [event.type for event in events] == [EventType.CREATED, EventType.UPDATED, EventType.DELETED]
                        assert ModelInstancePublic.model_validate(events[-1].data).name == instance["name"]
                    finally:
                        watch.cancel()
                        model_watch.cancel()
                        await asyncio.gather(watch, model_watch, return_exceptions=True)
                    assert not clientset.model_instances._watch_started
                    assert not clientset.model_instances._cache
                    await eventually(lambda: not event_bus.subscribers.get("modelinstance") and not event_bus.subscribers.get("model"))

    asyncio.run(check())


def test_fifty_http_watchers_share_one_snapshot_query(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            _, instance = await seed(env.client)
            query = ModelInstance._do_cached_all_query
            calls = 0

            async def counted(cls, options=None):
                nonlocal calls
                calls += 1
                await asyncio.sleep(0.05)
                return await query(options)

            monkeypatch.setattr(ModelInstance, "_do_cached_all_query", classmethod(counted))
            async with live_server(env.app) as url:
                async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=10) as client:
                    async def first_event():
                        async with client.stream("GET", "/v2/model-instances?watch=true") as response:
                            assert response.status_code == 200
                            async for line in response.aiter_lines():
                                if line:
                                    return json.loads(line)

                    events = await asyncio.gather(*(first_event() for _ in range(50)))
                assert all(event["data"]["id"] == instance["id"] for event in events)
                assert calls == 1
                await eventually(lambda: not event_bus.subscribers.get("modelinstance"))

    asyncio.run(check())


def test_bulk_delete_invalidates_replay_and_service_caches(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            model, instance = await seed(env.client, state="running")
            assert len(await ModelInstance.cached_all()) == 1
            async with db.async_session() as session:
                service = ModelInstanceService(session)
                assert await service.get_by_id(instance["id"]) is not None
                assert len(await service.get_running_instances(model["id"])) == 1
                rows = await ModelInstance.all(session)
                await service.batch_delete(rows)
                assert await service.get_by_id(instance["id"]) is None
                assert await service.get_running_instances(model["id"]) == []
            assert await ModelInstance.cached_all() == []

    asyncio.run(check())


def test_model_update_and_cascade_delete_invalidate_caches(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            model, instance = await seed(env.client, state="running")
            async with db.async_session() as session:
                models = ModelService(session)
                instances = ModelInstanceService(session)
                assert (await models.get_by_id(model["id"])).name == model["name"]
                assert await models.get_by_name(model["name"]) is not None
                assert await instances.get_by_id(instance["id"]) is not None
                assert len(await instances.get_running_instances(model["id"])) == 1
            await ModelInstance.cached_all()
            response = await env.client.put(
                f"/v2/models/{model['id']}", json={**model, "name": "renamed-model"}
            )
            assert response.status_code == 200, response.text
            async with db.async_session() as session:
                models = ModelService(session)
                assert await models.get_by_name(model["name"]) is None
                assert (await models.get_by_id(model["id"])).name == "renamed-model"
            response = await env.client.delete(f"/v2/models/{model['id']}")
            assert response.status_code == 200, response.text
            async with db.async_session() as session:
                assert await ModelService(session).get_by_id(model["id"]) is None
                assert await ModelInstanceService(session).get_by_id(instance["id"]) is None
                assert await ModelInstance.all(session) == []
            assert await ModelInstance.cached_all() == []

    asyncio.run(check())


def test_instance_rest_filters_match_pagination_total(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            _, first = await seed(env.client, name="Qwen-example", worker_id=7)
            await seed(env.client, name="other-model")
            response = await env.client.get(
                "/v2/model-instances", params={"search": "qWeN", "worker_id": 7}
            )
            assert response.status_code == 200, response.text
            assert [row["id"] for row in response.json()["items"]] == [first["id"]]
            assert response.json()["pagination"]["total"] == 1

    asyncio.run(check())


def test_controller_invalidates_service_cache_for_direct_deletes(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            model, instance = await seed(env.client, state="running")
            async with db.async_session() as session:
                service = ModelInstanceService(session)
                await service.get_by_id(instance["id"])
                await service.get_running_instances(model["id"])
                row = await ModelInstance.one_by_id(session, instance["id"])
                await row.delete(session)
            await ModelInstanceController(env.cfg)._reconcile(Event(
                type=EventType.DELETED, data={"id": instance["id"], "model_id": model["id"]},
            ))
            async with db.async_session() as session:
                service = ModelInstanceService(session)
                assert await service.get_by_id(instance["id"]) is None
                assert await service.get_running_instances(model["id"]) == []

    asyncio.run(check())


def public_instance(id, **changes):
    now = datetime.now(timezone.utc).isoformat()
    return ModelInstancePublic.model_validate({
        "id": id, "name": f"instance-{id}", "model_id": 1, "model_name": "qwen",
        "source": "huggingface", "huggingface_repo_id": "Qwen/test",
        "created_at": now, "updated_at": now, **changes,
    })


def test_client_cache_filter_matches_worker_and_state():
    client = ModelInstanceClient(HTTPClient(base_url="http://test"))
    client._watch_started = True
    client._cache = {
        1: public_instance(1),
        2: public_instance(2, worker_id=7, state="running"),
        3: public_instance(3, worker_id=8, state="pending"),
    }
    assert [row.id for row in client.list(params={"worker_id": "7"}).items] == [2]
    assert [row.id for row in client.list(params={"state": "running"}).items] == [2]


def test_unwatched_client_lists_all_rows_and_rest_does_not_overwrite_cache():
    requests = []
    client = ModelInstanceClient(HTTPClient(base_url="http://test"))

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/2"):
            client._cache[2] = public_instance(2, state="running")
            return httpx.Response(200, json=public_instance(2).model_dump(mode="json"))
        rows = [public_instance(i).model_dump(mode="json") for i in range(1, 102)]
        return httpx.Response(200, json={
            "items": rows, "pagination": {"page": -1, "perPage": 100, "total": 101, "totalPage": 2},
        })

    with httpx.Client(base_url="http://test/v2", transport=httpx.MockTransport(respond)) as network:
        client._client.set_httpx_client(network)
        assert len(client.list().items) == 101
        assert requests[-1].url.params["page"] == "-1"
        assert client.get(2, use_cache=False).state == ModelInstanceStateEnum.PENDING
        assert client._cache[2].state == ModelInstanceStateEnum.RUNNING


def test_id_only_delete_enriches_callback_and_closed_stream_clears_cache():
    async def check():
        instance = public_instance(1).model_dump(mode="json")
        payload = (
            json.dumps({"type": 1, "data": instance, "id": 1}) + "\n\n"
            + json.dumps({"type": 3, "data": {"id": 1}, "id": 1}) + "\n\n"
        )
        async with httpx.AsyncClient(
            base_url="http://test/v2", transport=httpx.MockTransport(lambda request: httpx.Response(200, text=payload))
        ) as network:
            client = ModelInstanceClient(HTTPClient(base_url="http://test").set_async_httpx_client(network))
            events = []
            await client.awatch(callback=events.append, stop_condition=lambda event: event.type == EventType.DELETED)
            assert events[-1].data.name == "instance-1"
            assert not client._watch_started
            assert client._cache == {}
            with pytest.raises(ConnectionError, match="closed"):
                await client.awatch()
            assert not client._watch_started

    asyncio.run(check())


def test_delete_filter_keeps_scope_and_id_only_wire_events():
    row = SimpleNamespace(id=1, deleted_at=datetime.now(timezone.utc), cluster_id=7)
    event = Event(type=EventType.DELETED, data=row)
    assert ModelInstance._match_fields(event, {"deleted_at": None, "cluster_id": 7})
    assert not ModelInstance._match_fields(event, {"deleted_at": None, "cluster_id": 8})
    assert ModelInstance._format_event(Event(type=EventType.CREATED, data={"id": 1})) is None
    deleted = ModelInstance._format_event(Event(type=EventType.DELETED, data={"id": 1}))
    assert json.loads(deleted)["data"] == {"id": 1}


def test_invalidated_in_flight_read_cannot_refill_stale_cache():
    async def check():
        setup_logging()
        await cache_module.cache.clear()
        cache_module._cache_locks.clear()
        started = asyncio.Event()
        release = asyncio.Event()
        current = 1

        class Reader:
            @cache_module.locked_cached()
            async def get(self, id):
                value = current
                started.set()
                await release.wait()
                return value

        reader = Reader()
        pending = asyncio.create_task(reader.get(1))
        await started.wait()
        current = 2
        await cache_module.delete_cache_by_key(reader.get, 1)
        release.set()
        assert await pending == 1
        assert await reader.get(1) == 2
        await cache_module.cache.clear()
        cache_module._cache_locks.clear()

    asyncio.run(check())
