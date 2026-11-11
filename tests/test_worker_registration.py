import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from gpustack.config import config as config_module
from gpustack.config.config import Config
from gpustack.detectors.base import GPUDetectExepction
from gpustack.logging import setup_logging
from gpustack.schemas.config import PredefinedConfigNoDefaults
from gpustack.schemas.workers import GPUDeviceStatus, MemoryInfo, Worker, WorkerStateEnum
from gpustack.server import db
from gpustack.server.app import create_app
from gpustack.server.bus import event_bus
from gpustack.server.init_db import create_db_and_tables
from gpustack.server import worker_status_buffer as buffer
from gpustack.worker.collector import WorkerStatusCollector
from gpustack.worker.worker import Worker as WorkerProcess


@asynccontextmanager
async def server_context(tmp_path, monkeypatch):
    setup_logging()
    cfg = Config(data_dir=str(tmp_path / "server"), token="bootstrap")
    monkeypatch.setattr(config_module, "_config", cfg)
    engine = create_async_engine("sqlite+aiosqlite://")
    monkeypatch.setattr(db, "engine", engine)
    buffer.heartbeat_flush_buffer.clear()
    buffer.worker_status_flush_buffer.clear()
    try:
        await create_db_and_tables(engine)
        app = create_app(cfg)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client
    finally:
        buffer.heartbeat_flush_buffer.clear()
        buffer.worker_status_flush_buffer.clear()
        await engine.dispose()


def worker_payload(name="gpu-node-1", **overrides):
    return {
        "name": name,
        "hostname": name,
        "ip": "10.0.0.5",
        "ifname": "eth0",
        "port": 10150,
        "worker_uuid": "",
        "status": None,
        "labels": {"os": "linux", "arch": "amd64"},
        **overrides,
    }


async def add_cluster(client, name="gpu-room", worker_config=None):
    response = await client.post(
        "/v2/clusters", json={"name": name, "worker_config": worker_config}
    )
    assert response.status_code == 200, response.text
    cluster = response.json()
    response = await client.get(f"/v2/clusters/{cluster['id']}/registration-token")
    assert response.status_code == 200, response.text
    return cluster, {"Authorization": f"Bearer {response.json()['token']}"}


def test_registration_and_restart_keep_identity(tmp_path, monkeypatch):
    async def check():
        async with server_context(tmp_path, monkeypatch) as client:
            cluster, registration_headers = await add_cluster(client)
            subscriber = event_bus.subscribe("worker", source="registration-test")
            try:
                response = await client.post(
                    "/v2/workers", headers=registration_headers, json=worker_payload(name="")
                    | {"hostname": "gpu-node-1", "cluster_id": 999, "worker_uuid": "forged"},
                )
                assert response.status_code == 200, response.text
                event = await asyncio.wait_for(subscriber.receive(), timeout=1)
                assert event.data.id == response.json()["id"]
            finally:
                event_bus.unsubscribe("worker", subscriber)
            assert response.status_code == 200, response.text
            first = response.json()
            assert first["name"] == "gpu-node-1"
            assert first["cluster_id"] == cluster["id"]
            assert first["worker_uuid"] != "forged"
            assert first["token"] != registration_headers["Authorization"][7:]

            async with db.async_session() as session:
                worker = await Worker.one_by_id(session, first["id"])
                await worker.update(session, {"labels": {**worker.labels, "rack": "a"}})

            response = await client.post(
                "/v2/workers", headers=registration_headers,
                json=worker_payload(worker_uuid=first["worker_uuid"], ip="10.0.0.6"),
            )
            assert response.status_code == 200, response.text
            second = response.json()
            assert second["id"] == first["id"]
            assert second["worker_uuid"] == first["worker_uuid"]
            assert second["token"] != first["token"]
            assert second["labels"]["rack"] == "a"
            assert second["ip"] == "10.0.0.6"

            response = await client.get(
                "/v2/workers?me=true", headers={"Authorization": f"Bearer {second['token']}"}
            )
            assert response.status_code == 200, response.text
            assert len(response.json()["items"]) == 1
            assert "worker_uuid" not in response.json()["items"][0]
            assert "token" not in response.json()["items"][0]

            response = await client.post(
                "/v2/worker-heartbeat", headers={"Authorization": f"Bearer {first['token']}"}
            )
            assert response.status_code == 403

    asyncio.run(check())


def test_registration_credentials_and_names_are_checked(tmp_path, monkeypatch):
    async def check():
        async with server_context(tmp_path, monkeypatch) as client:
            _, headers = await add_cluster(client)
            response = await client.post("/v2/workers", json=worker_payload())
            assert response.status_code == 404
            response = await client.post("/v2/workers", headers=headers, json=worker_payload())
            assert response.status_code == 200, response.text
            response = await client.post("/v2/workers", headers=headers, json=worker_payload())
            assert response.status_code == 409
            response = await client.post("/v2/worker-status", json=worker_payload())
            assert response.status_code == 403

    asyncio.run(check())


def test_status_and_heartbeat_flush_and_detection_recovery(tmp_path, monkeypatch):
    async def check():
        async with server_context(tmp_path, monkeypatch) as client:
            cluster, registration_headers = await add_cluster(client)
            response = await client.post(
                "/v2/workers", headers=registration_headers, json=worker_payload()
            )
            assert response.status_code == 200, response.text
            registered = response.json()
            headers = {"Authorization": f"Bearer {registered['token']}"}
            status = {
                "memory": {"total": 128 << 30},
                "gpu_devices": [{
                    "index": 0, "type": "cuda", "name": "NVIDIA A100",
                    "memory": {"total": 80 << 30, "used": 2 << 30},
                }],
            }
            payload = worker_payload(status=status, state_message=None)
            response = await client.post("/v2/worker-status", headers=headers, json=payload)
            assert response.status_code == 204
            assert registered["id"] in buffer.worker_status_flush_buffer
            await buffer.flush_worker_status()
            async with db.async_session() as session:
                worker = await Worker.one_by_id(session, registered["id"])
                assert worker.state == WorkerStateEnum.READY
                assert worker.status.gpu_devices[0].memory.used == 2 << 30
                previous = datetime.now(timezone.utc) - timedelta(minutes=10)
                await worker.update(session, {"heartbeat_time": previous})

            response = await client.post("/v2/worker-heartbeat", headers=headers)
            assert response.status_code == 204
            await buffer.flush_heartbeats()
            async with db.async_session() as session:
                worker = await Worker.one_by_id(session, registered["id"])
                assert worker.heartbeat_time > previous

            response = await client.post(
                "/v2/worker-status", headers=headers,
                json={**payload, "state_message": "GPU driver is unavailable"},
            )
            assert response.status_code == 204
            await buffer.flush_worker_status()
            async with db.async_session() as session:
                worker = await Worker.one_by_id(session, registered["id"])
                assert worker.state == WorkerStateEnum.NOT_READY
                assert worker.state_message == "GPU driver is unavailable"

            await client.post("/v2/worker-status", headers=headers, json=payload)
            await buffer.flush_worker_status()
            response = await client.get(f"/v2/clusters/{cluster['id']}")
            assert response.json()["workers"] == 1
            assert response.json()["ready_workers"] == 1
            assert response.json()["gpus"] == 1
            response = await client.get("/v2/clusters")
            assert response.json()["items"][0]["gpus"] == 1

    asyncio.run(check())


def test_cluster_configuration_is_applied_without_secrets(tmp_path, monkeypatch):
    async def check():
        async with server_context(tmp_path, monkeypatch) as client:
            _, headers = await add_cluster(client, worker_config={
                "service_port_range": "42000-42063",
                "huggingface_token": "private-hub-token",
            })
            response = await client.post("/v2/workers", headers=headers, json=worker_payload())
            assert response.status_code == 200, response.text
            assert response.json()["worker_config"]["service_port_range"] == "42000-42063"
            assert "private-hub-token" not in response.text

    asyncio.run(check())


@pytest.mark.parametrize("local_port", [None, 10150, 10200])
def test_worker_applies_registration_configuration(tmp_path, monkeypatch, local_port):
    overrides = {} if local_port is None else {"worker_port": local_port}
    cfg = Config(
        data_dir=str(tmp_path), server_url="http://test", token="registration-token",
        worker_ip="127.0.0.1", worker_ifname="lo0", **overrides,
    )
    worker = WorkerProcess(cfg)
    clientset = SimpleNamespace(workers=SimpleNamespace(list=lambda **kwargs: SimpleNamespace(
        items=[SimpleNamespace(id=7, name="gpu-node-1", worker_uuid=None)]
    )))

    async def register():
        return clientset, PredefinedConfigNoDefaults(
            service_port_range="42000-42063", worker_port=10300,
        )

    monkeypatch.setattr(worker._worker_manager, "register_with_server", register)
    asyncio.run(worker._register.__wrapped__(worker))
    assert cfg.service_port_range == "42000-42063"
    assert cfg.worker_port == (10300 if local_port is None else local_port)
    assert worker.worker_id() == 7


def test_collector_defers_gpu_probe_and_reports_failures(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    collector = WorkerStatusCollector(
        cfg, lambda: "10.0.0.5", lambda: "eth0", lambda: 1, lambda: "worker-uuid"
    )
    detector = Mock()
    detector.detect_system_info.return_value = SimpleNamespace(
        model_dump=lambda: {"cpu": {"total": 8}, "memory": {"total": 64 << 30}}
    )
    detector.detect_gpus.return_value = [GPUDeviceStatus(
        index=0, type="cuda", memory=MemoryInfo(total=80 << 30, is_unified_memory=True)
    )]
    collector._detector_factory = detector
    initial = collector.collect(initial=True)
    detector.detect_gpus.assert_not_called()
    assert initial.worker_uuid == "worker-uuid"
    status = collector.collect()
    assert status.status.gpu_devices[0].memory.total == 80 << 30
    assert status.status.memory.is_unified_memory is True
    detector.detect_gpus.side_effect = GPUDetectExepction("GPU probe failed")
    assert collector.collect().state_message == "GPU probe failed"


@pytest.mark.parametrize("vendor,backend", [("NVIDIA", "cuda"), ("ASCEND", "cann")])
def test_runtime_converts_gpu_and_npu_inventory(monkeypatch, vendor, backend):
    from gpustack.detectors.runtime import runtime

    device = SimpleNamespace(
        manufacturer=getattr(runtime.ManufacturerEnum, vendor), index=0,
        name="test-accelerator", uuid="device-uuid", driver_version="1.0",
        runtime_version="1.0", compute_capability="8.0", cores=100,
        cores_utilization=50.0, memory=80 * 1024, memory_used=1024,
        memory_utilization=1.25, temperature=45.0, power=300.0, power_used=100.0,
        appendix={"card_id": "2", "device_id": "1", "roce_ip": "10.0.1.5"},
    )
    monkeypatch.setattr(runtime, "detect_devices", lambda **kwargs: [device])
    monkeypatch.setattr(runtime, "supported_deployers", lambda: [])
    gpu = runtime.Runtime().gather_gpu_info()[0]
    assert gpu.type == backend
    assert gpu.memory.total == 80 << 30
    assert gpu.memory.used == 1 << 30
    assert (gpu.device_index, gpu.device_chip_index) == (2, 1)
    if vendor == "ASCEND":
        assert gpu.network.inet == "10.0.1.5"
        assert gpu.network.status == "up"
