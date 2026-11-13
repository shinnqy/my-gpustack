import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from gpustack.client import ClientSet
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.schemas.inference_backend import InferenceBackend, VersionConfig, VersionConfigDict, get_built_in_backend
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.utils import runtime as runtime_utils
from gpustack.utils.version import pick_runtime_version
from gpustack.worker.backends import vllm as vllm_module
from gpustack.worker.backends.base import ModelInstanceStateError
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker import serve_manager as serve_module
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker.inference_backend_manager import InferenceBackendManager
from gpustack_runtime.detector import ManufacturerEnum

from test_watch import project_context


def records(tmp_path, kind="cuda", family=None, count=1):
    now = datetime.now(timezone.utc).isoformat()
    model_path = str(tmp_path / "model")
    model = {
        "id": 1, "name": "qwen", "source": "local_path", "local_path": model_path,
        "backend": "vLLM", "backend_version": "0.11.0", "image_name": "registry.test/vllm:fixture",
        "meta": {"max_model_len": 32768}, "categories": ["llm"],
        "created_at": now, "updated_at": now,
    }
    vendor = ManufacturerEnum.ASCEND.value if kind == "cann" else ManufacturerEnum.NVIDIA.value
    worker = {
        "id": 1, "name": "node", "hostname": "node", "ip": "127.0.0.1", "ifname": "lo0",
        "port": 10150, "worker_uuid": "node-uuid", "state": "ready", "created_at": now, "updated_at": now,
        "status": {"gpu_devices": [
            {"index": i, "type": kind, "vendor": vendor, "name": family or "NVIDIA A100",
             "runtime_version": "8.2" if kind == "cann" else "12.8", "arch_family": family,
             "memory": {"total": 24 << 30}}
            for i in range(count + 1)
        ]},
    }
    instance = {
        "id": 1, "name": "qwen-instance", "model_name": "qwen", "model_id": 1,
        "source": "local_path", "local_path": model_path, "worker_id": 1, "worker_name": "node",
        "worker_ip": "127.0.0.1", "gpu_type": kind, "gpu_indexes": list(range(1, count + 1)),
        "port": 40001, "ports": [40001], "state": "starting", "resolved_path": model_path,
        "draft_model_resolved_path": None, "created_at": now, "updated_at": now,
    }
    return model, worker, instance


@pytest.fixture
def launch_server(tmp_path, monkeypatch):
    setup_logging()
    resources = []
    monkeypatch.setattr(runtime_utils.DockerDeployer, "is_supported", lambda self: False)

    def build(kind="cuda", family=None, count=1, model_changes=None):
        model, worker, instance = records(tmp_path, kind, family, count)
        model.update(model_changes or {})
        writes = []
        def request(request):
            if request.method == "PUT":
                body = json.loads(request.content)
                writes.append(body)
                if "/model-instances/" in request.url.path:
                    instance.update(body)
                else:
                    model.update(body)
            body = worker if "/workers/" in request.url.path else instance if "/model-instances/" in request.url.path else model
            return httpx.Response(200, json=body)
        sync = httpx.Client(base_url="http://test/v2", transport=httpx.MockTransport(request))
        async_client = httpx.AsyncClient(base_url="http://test/v2", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=json.dumps({"type": 1, "id": 1, "data": instance}) + "\n\n")
        ))
        resources.append((sync, async_client))
        clientset = ClientSet("http://test")
        clientset.http_client.set_httpx_client(sync)
        clientset.http_client.set_async_httpx_client(async_client)
        cfg = Config(data_dir=str(tmp_path / "worker"), server_url="http://test", token="registration")
        backend = get_built_in_backend()[0]
        server = VLLMServer(clientset, ModelInstance.model_validate(instance), cfg, 1, backend)
        return SimpleNamespace(server=server, writes=writes, model=model, worker=worker, instance=instance)

    yield build
    for sync, async_client in resources:
        sync.close()
        asyncio.run(async_client.aclose())


@pytest.mark.parametrize("kind,resource", [("cuda", "nvidia.com/devices"), ("cann", "huawei.com/devices")])
def test_runtime_plan_uses_only_selected_devices_and_mounts(launch_server, monkeypatch, kind, resource):
    env = {
        "GPUSTACK_MODEL_RUNTIME_UID": "1000", "GPUSTACK_MODEL_RUNTIME_GID": "1001",
        "GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB": "2.5", "OMP_NUM_THREADS": "3",
    }
    context = launch_server(kind=kind, family="910b" if kind == "cann" else None, model_changes={"env": env})
    plans = []
    monkeypatch.setattr(vllm_module, "create_workload", plans.append)
    context.server.start()
    assert len(plans) == 1
    plan = plans[0]
    assert plan.name == "qwen-instance"
    assert plan.host_network is True and plan.host_ipc is False
    assert plan.shm_size == int(2.5 * (1 << 30))
    container = plan.containers[0]
    assert dict(container.resources) == {resource: "1"}
    assert container.restart_policy.value.lower() == "never"
    assert container.execution.run_as_user == 1000
    assert container.execution.run_as_group == 1001
    assert [port.internal for port in container.ports] == [40001]
    assert container.mounts[0].path == str(Path(context.instance["resolved_path"]).parent)
    command = (container.execution.command or []) + (container.execution.args or [])
    assert command[:3] == ["vllm", "serve", context.instance["resolved_path"]]
    assert command[command.index("--max-model-len") + 1] == "8192"
    env = {item.name: item.value for item in container.envs}
    assert env["OMP_NUM_THREADS"] == "3"
    if kind == "cann":
        assert env["TASK_QUEUE_ENABLE"] == "1"
        assert env["HCCL_OP_EXPANSION_MODE"] == "AIV"


def test_user_args_override_defaults_and_multi_gpu_injects_tp(launch_server):
    context = launch_server(count=2, model_changes={"backend_parameters": [
        "--max-model-len=16384", "--host 0.0.0.0", "--port", "40123",
    ]})
    arguments, injected = context.server._build_command_args(40001, False, entrypoint=["vllm", "serve"])
    assert "8192" not in arguments
    assert "--max-model-len=16384" in arguments
    assert arguments.count("--host") == arguments.count("--port") == 1
    assert arguments[arguments.index("--port") + 1] == "40123"
    assert arguments[arguments.index("--tensor-parallel-size") + 1] == "2"
    assert "--max-model-len=16384" not in injected


def test_ascend_310p_uses_eager_float16(launch_server):
    server = launch_server(kind="cann", family="310p").server
    arguments, _ = server._build_command_args(40001, False, entrypoint=["vllm", "serve"])
    assert "--enforce-eager" in arguments
    assert arguments[arguments.index("--dtype") + 1] == "float16"
    env = server._get_configured_env(False)
    assert "HCCL_OP_EXPANSION_MODE" not in env


def test_missing_assigned_device_fails_instead_of_using_unselected_gpu(launch_server):
    server = launch_server().server
    server._model_instance.gpu_indexes = [99]
    with pytest.raises((ValueError, RuntimeError), match="99"):
        server._get_configured_resources()


def test_default_backend_is_resolved_and_runtime_failure_marks_error(launch_server, monkeypatch):
    context = launch_server(model_changes={"backend": None})
    assert context.server._model.backend == "vLLM"
    def fail(plan):
        raise RuntimeError("runtime rejected workload")
    monkeypatch.setattr(vllm_module, "create_workload", fail)
    with pytest.raises(RuntimeError, match="rejected workload"):
        context.server.start()
    assert context.writes[-1]["state"] == "error"
    assert "rejected workload" in context.writes[-1]["state_message"]


def test_image_precedence_and_packaged_ascend_catalog(launch_server, monkeypatch):
    context = launch_server(kind="cann", family="910b")
    server = context.server
    assert server._resolve_image()[0] == "registry.test/vllm:fixture"
    server._model.image_name = None
    server._model.backend_version = "custom-version"
    server.inference_backend = InferenceBackend(
        backend_name="vLLM", is_built_in=True,
        version_configs=VersionConfigDict(root={"custom-version": VersionConfig(image_name="registry.test/custom:v1")}),
    )
    assert server._resolve_image() == ("registry.test/custom:v1", "custom-version")
    server._model.backend_version = None
    server.inference_backend = get_built_in_backend()[0]
    monkeypatch.setattr("gpustack.worker.backends.base.platform.system_arch", lambda: "linux/arm64")
    image, version = server._resolve_image()
    assert image and "cann" in image and version


@pytest.mark.parametrize("candidates,host,expected", [
    (["12.1", "12.4", "12.8"], "12.6", "12.4"),
    (["12.4", "12.8"], "12.1", "12.8"),
    (["12.4", "12.8"], "11.8", "12.4"),
    (["12.4", "12.8"], None, "12.8"),
])
def test_runtime_version_selection(candidates, host, expected):
    assert pick_runtime_version(candidates, host) == expected


def test_wait_stops_on_error_deletion_and_rejects_missing_path(launch_server):
    context = launch_server()
    server = context.server
    with pytest.raises(ModelInstanceStateError):
        server._stop_when_starting(Event(type=EventType.UPDATED, data={**context.instance, "state": "error"}))
    with pytest.raises(ModelInstanceStateError):
        server._stop_when_starting(Event(type=EventType.DELETED, data=context.instance))
    with pytest.raises(ValueError, match="resolved"):
        server._stop_when_starting(Event(type=EventType.UPDATED, data={**context.instance, "resolved_path": None}))


def test_ports_and_spawn_are_idempotent_and_spawn_precedes_initializing(tmp_path, monkeypatch):
    setup_logging()
    cfg = Config(data_dir=str(tmp_path), service_port_range="40000-40002")
    model, worker, raw = records(tmp_path)
    raw.update({"state": "scheduled", "port": None, "ports": []})
    instance = ModelInstance.model_validate(raw)
    order = []
    class Process:
        pid = 4242
        def __init__(self, **kwargs):self.daemon = None
        def start(self):order.append("spawn")
        def is_alive(self):return True
        def join(self, **kwargs):pass
    manager = ServeManager(lambda: 1, lambda: SimpleNamespace(headers={}), cfg)
    manager._inference_backend_manager = InferenceBackendManager(None)
    monkeypatch.setattr(manager, "_get_model", lambda instance: SimpleNamespace(backend="vLLM"))
    monkeypatch.setattr(serve_module.multiprocessing, "Process", Process)
    monkeypatch.setattr(serve_module.registration, "determine_default_registry", lambda *args: None)
    monkeypatch.setattr(serve_module.network, "get_free_port", lambda **kwargs: min(set(range(40000, 40003)) - kwargs["unavailable_ports"]))
    monkeypatch.setattr(manager, "_update_model_instance", lambda id, **kwargs: order.append(kwargs))
    manager._start_model_instance(instance)
    manager._start_model_instance(instance)
    assert order[0] == "spawn"
    assert order[1]["state"] == ModelInstanceStateEnum.INITIALIZING
    assert order[1]["pid"] == 4242
    assert order.count("spawn") == 1
    assert manager._assigned_ports[1] == {40000}


def test_failed_spawn_releases_port_reservation(tmp_path, monkeypatch):
    setup_logging()
    model, worker, raw = records(tmp_path)
    raw.update({"state": "scheduled", "port": None, "ports": []})
    manager = ServeManager(lambda: 1, lambda: SimpleNamespace(headers={}), Config(data_dir=str(tmp_path)))
    manager._inference_backend_manager = InferenceBackendManager(None)
    patches = []
    class Process:
        def __init__(self, **kwargs):pass
        def start(self):raise RuntimeError("spawn failed")
    monkeypatch.setattr(manager, "_get_model", lambda instance: SimpleNamespace(backend="vLLM"))
    monkeypatch.setattr(manager, "_update_model_instance", lambda id, **kwargs: patches.append(kwargs))
    monkeypatch.setattr(serve_module.multiprocessing, "Process", Process)
    monkeypatch.setattr(serve_module.network, "get_free_port", lambda **kwargs: 40000)
    monkeypatch.setattr(serve_module.registration, "determine_default_registry", lambda *args: None)
    manager._start_model_instance(ModelInstance.model_validate(raw))
    assert patches[-1]["state"] == ModelInstanceStateEnum.ERROR
    assert manager._assigned_ports == {}


def test_replayed_existing_port_is_reserved(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    manager = ServeManager(lambda: 1, lambda: None, cfg)
    _, _, raw = records(tmp_path)
    raw["ports"] = []
    instance = ModelInstance.model_validate(raw)
    manager._assign_ports(instance, SimpleNamespace(), "vLLM")
    assert instance.ports == [40001]
    assert manager._assigned_ports[1] == {40001}


def test_worker_detail_route_returns_public_devices(tmp_path, monkeypatch):
    async def check():
        from test_scheduler import worker_record
        from gpustack.server import db
        from gpustack.schemas.workers import Worker
        async with project_context(tmp_path, monkeypatch) as env:
            response = await env.client.post("/v2/clusters", json={"name": "worker-lookup"})
            async with db.async_session() as session:
                node = await Worker.create(session, worker_record(cluster_id=response.json()["id"]))
            response = await env.client.get(f"/v2/workers/{node.id}")
            assert response.status_code == 200
            assert response.json()["status"]["gpu_devices"][0]["type"] == "cann"
            assert "token" not in response.json() and "worker_uuid" not in response.json()

    asyncio.run(check())
