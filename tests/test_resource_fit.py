import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from transformers import PretrainedConfig

from gpustack.policies import utils
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.candidate_selectors import base_candidate_selector
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import VLLMResourceFitSelector
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.scheduler import scheduler as scheduler_module
from gpustack.scheduler.calculator import get_pretrained_config_with_workers
from gpustack.schemas.models import (
    ComputedResourceClaim, Model, ModelInstance, ModelInstanceStateEnum,
)
from gpustack.schemas.workers import Worker, WorkerStatus, GPUDeviceStatus, MemoryInfo, SystemReserved
from gpustack.server import db
from gpustack.server.controllers import find_scale_down_candidates
from gpustack.server.services import ModelInstanceService
from gpustack.utils import hub

from test_watch import project_context, seed


GiB = 1 << 30


def model(**changes):
    return Model.model_validate({
        "id": 1, "name": "qwen", "source": "huggingface", "huggingface_repo_id": "Qwen/test",
        "backend": "vLLM", "categories": ["llm"], "cluster_id": 1,
        "distributed_inference_across_workers": True,
        "env": {"GPUSTACK_MODEL_VRAM_CLAIM": str(3 * GiB)}, **changes,
    })


def worker(id=1, sizes=(24, 24), gpu_type="cuda", indexes=None, **changes):
    indexes = list(range(len(sizes))) if indexes is None else indexes
    return Worker.model_validate({
        "id": id, "name": f"worker-{id}", "cluster_id": 1, "hostname": f"worker-{id}",
        "ip": f"10.0.0.{id}", "ifname": "eth0", "port": 10150, "worker_uuid": f"worker-{id}",
        "state": "ready", "system_reserved": {"ram": 0, "vram": 0},
        "status": {"memory": {"total": 128 * GiB}, "gpu_devices": [
            {"index": index, "type": gpu_type, "name": "test-accelerator", "uuid": f"gpu-{id}-{index}",
             "memory": {"total": int(size * GiB)}}
            for index, size in zip(indexes, sizes)
        ]}, **changes,
    })


def instance(id=1, worker_id=1, vram=None, ram=0, gpu_type="cuda", **changes):
    now = datetime.now(timezone.utc)
    vram = {0: int(24 * GiB * 0.9)} if vram is None else vram
    return ModelInstance.model_validate({
        "id": id, "name": f"instance-{id}", "model_id": 1, "model_name": "qwen",
        "source": "huggingface", "huggingface_repo_id": "Qwen/test", "worker_id": worker_id,
        "gpu_type": gpu_type, "gpu_indexes": list(vram), "state": "scheduled",
        "computed_resource_claim": {"ram": ram, "vram": vram},
        "created_at": now, "updated_at": now, **changes,
    })


@pytest.fixture
def config_stub(monkeypatch):
    async def load(model, *args, **kwargs):
        architectures = ["BertModel"] if model.categories == ["embedding"] else ["Qwen3ForCausalLM"]
        return PretrainedConfig(
            architectures=architectures, num_attention_heads=64, num_key_value_heads=8,
            num_hidden_layers=8, hidden_size=1024, vocab_size=128000,
            max_position_embeddings=32768,
        )

    monkeypatch.setattr(base_candidate_selector, "get_pretrained_config_with_workers", load)
    monkeypatch.setattr(scheduler_module, "get_pretrained_config_with_workers", load)


@pytest.mark.parametrize("categories,overhead", [(["llm"], 2 * GiB), (["embedding"], GiB // 2)])
def test_weight_estimation_and_explicit_override(monkeypatch, categories, overhead):
    calls = []

    def weights(*args):
        calls.append(True)
        return 10 * GiB

    monkeypatch.setattr(utils, "get_model_weight_size", weights)
    estimated = asyncio.run(utils.estimate_model_vram(model(categories=categories, env={})))
    assert estimated == int(10 * GiB * 1.2) + overhead
    assert calls == [True]
    calls.clear()
    assert asyncio.run(utils.estimate_model_vram(model())) == 3 * GiB
    assert calls == []


def test_weight_listing_uses_root_and_excludes_duplicate(monkeypatch):
    def list_repo(*args, **kwargs):
        assert kwargs["root_dir_only"] is True
        return [
            {"name": "model.safetensors", "size": 10 * GiB},
            {"name": "consolidated.safetensors", "size": 10 * GiB},
            {"name": "config.json", "size": 1024},
        ]

    monkeypatch.setattr(hub, "list_repo", list_repo)
    assert hub.get_model_weight_size(model()) == 10 * GiB


def test_allocation_ledger_and_reservations_ignore_reported_usage():
    node = worker(sizes=(80, 80), system_reserved={"ram": 4 * GiB, "vram": GiB})
    node.status.gpu_devices[0].memory.used = 79 * GiB
    assignments = [
        instance(1, vram={0: 72 * GiB}, ram=2 * GiB),
        instance(2, vram={1: 36 * GiB}, ram=3 * GiB),
        instance(3, worker_id=2, vram={0: 10 * GiB}),
    ]
    allocated = utils.compute_worker_allocated(assignments, 1)
    assert allocated.vram == {0: 72 * GiB, 1: 36 * GiB}
    assert allocated.ram == 5 * GiB
    available = utils.get_worker_allocatable_resource(assignments, node)
    assert available.vram == {0: 7 * GiB, 1: 43 * GiB}
    assert available.ram == 119 * GiB
    assert utils.get_worker_allocatable_resource([instance(vram={0: 100 * GiB})], node).vram[0] == 0


@pytest.mark.parametrize("index", [0, 3])
def test_unified_memory_does_not_count_gpu_memory_twice(index):
    node = worker(sizes=(32,), indexes=[index], system_reserved={"ram": GiB, "vram": 2 * GiB})
    node.status.memory = MemoryInfo(total=32 * GiB, is_unified_memory=True)
    available = utils.get_worker_allocatable_resource([instance(vram={index: 4 * GiB}, ram=2 * GiB)], node)
    assert available.ram == 23 * GiB
    assert available.vram == {index: 23 * GiB}


def test_missing_optional_memory_and_reservations_do_not_crash():
    node = worker(sizes=(24,))
    node.status.memory = None
    node.system_reserved = None
    available = utils.get_worker_allocatable_resource([], node)
    assert available.ram == 0
    assert available.vram == {0: 24 * GiB}


def test_grouping_heterogeneous_devices_preserves_original_inventory():
    node = worker(sizes=(24,))
    node.status.gpu_devices.append(GPUDeviceStatus(index=1, type="rocm", memory=MemoryInfo(total=40 * GiB)))
    grouped = utils.group_workers_by_gpu_type([node])
    assert [gpu.type for gpu in node.status.gpu_devices] == ["cuda", "rocm"]
    assert [gpu.type for gpu in grouped["cuda"][0].status.gpu_devices] == ["cuda"]
    assert [gpu.type for gpu in grouped["rocm"][0].status.gpu_devices] == ["rocm"]


def test_default_gmu_prevents_overbooking_and_chooses_multiple_cards(tmp_path, config_stub):
    async def check():
        from gpustack.config.config import Config
        cfg = Config(data_dir=str(tmp_path))
        node = worker()
        first = (await VLLMResourceFitSelector(cfg, model(), []).select_candidates([node]))[0]
        assert first.gpu_indexes == [0]
        assert first.computed_resource_claim.vram == {0: int(24 * GiB * 0.9)}
        large = model(env={"GPUSTACK_MODEL_VRAM_CLAIM": str(int(21.7 * GiB))})
        selector = VLLMResourceFitSelector(cfg, large, [instance()])
        assert await selector.select_candidates([node]) == []
        assert "21.7" in "".join(selector.get_messages())
        candidate = (await VLLMResourceFitSelector(cfg, large, []).select_candidates([node]))[0]
        assert candidate.gpu_indexes == [0, 1]
        assert candidate.computed_resource_claim.vram == {index: int(24 * GiB * 0.9) for index in (0, 1)}

    asyncio.run(check())


def test_multi_gpu_accepts_exact_gmu_capacity(tmp_path, config_stub):
    async def check():
        from gpustack.config.config import Config
        workload = model(
            env={"GPUSTACK_MODEL_VRAM_CLAIM": str(20 * GiB)},
            backend_parameters=["--tensor-parallel-size=2", "--gpu-memory-utilization=0.5"],
        )
        claims = [instance(vram={0: 12 * GiB, 1: 12 * GiB})]
        candidates = await VLLMResourceFitSelector(Config(data_dir=str(tmp_path)), workload, claims).select_candidates([worker()])
        assert len(candidates) == 1
        assert candidates[0].computed_resource_claim.vram == {0: 12 * GiB, 1: 12 * GiB}

    asyncio.run(check())


def test_head_divisibility_and_minimum_card_count(tmp_path, config_stub):
    async def check():
        from gpustack.config.config import Config
        cfg = Config(data_dir=str(tmp_path))
        large = model(env={"GPUSTACK_MODEL_VRAM_CLAIM": str(int(80.6 * GiB))})
        candidates = await VLLMResourceFitSelector(cfg, large, []).select_candidates([worker(sizes=(40, 40, 40, 40))])
        assert candidates[0].gpu_indexes == [0, 1, 2, 3]
        medium = model(env={"GPUSTACK_MODEL_VRAM_CLAIM": str(50 * GiB)})
        candidates = await VLLMResourceFitSelector(cfg, medium, []).select_candidates([
            worker(1, sizes=(24, 24, 24, 24)), worker(2, sizes=(40, 40)),
        ])
        assert all(candidate.worker.id == 2 and len(candidate.gpu_indexes) == 2 for candidate in candidates)

    asyncio.run(check())


def test_explicit_gpu_count_is_honored_without_card_ids(tmp_path, config_stub):
    async def check():
        from gpustack.config.config import Config
        cfg = Config(data_dir=str(tmp_path))
        selected = model(gpu_selector={"gpus_per_replica": 2})
        candidates = await VLLMResourceFitSelector(cfg, selected, []).select_candidates([worker()])
        assert candidates[0].gpu_indexes == [0, 1]
        invalid = model(backend_parameters=["--tensor-parallel-size=3"])
        candidates = await VLLMResourceFitSelector(cfg, invalid, []).select_candidates([worker(sizes=(24, 24, 24, 24))])
        assert candidates == []

    asyncio.run(check())


def test_parallelism_sets_default_backend_gpu_count():
    selected = model(
        backend=None, backend_parameters=["--tensor-parallel-size=2"],
        gpu_selector={"gpu_ids": [f"worker-1:cuda:{index}" for index in range(8)]},
    )
    assert scheduler_module.set_model_gpus_per_replica(selected)
    assert selected.gpu_selector.gpus_per_replica == 2


def test_embedding_claims_estimate_and_manual_gpu_selection(tmp_path, config_stub):
    async def check():
        from gpustack.config.config import Config
        cfg = Config(data_dir=str(tmp_path))
        embedding = model(categories=["embedding"], env={"GPUSTACK_MODEL_VRAM_CLAIM": str(8 * GiB)})
        candidates = await VLLMResourceFitSelector(cfg, embedding, [instance(vram={0: 16 * GiB})]).select_candidates([worker(sizes=(24,))])
        assert candidates[0].computed_resource_claim.vram == {0: 8 * GiB}
        selected = model(gpu_selector={"gpu_ids": ["worker-1:cuda:1"], "gpus_per_replica": 1})
        candidates = await VLLMResourceFitSelector(cfg, selected, []).select_candidates([worker()])
        assert candidates and all(candidate.gpu_indexes == [1] for candidate in candidates)

    asyncio.run(check())


@pytest.mark.parametrize("gmu", ["0", "-0.1", "1.1", "nan", "inf"])
def test_invalid_gmu_is_rejected(tmp_path, config_stub, gmu):
    async def check():
        from gpustack.config.config import Config
        selected = model(backend_parameters=[f"--gpu-memory-utilization={gmu}"])
        selector = VLLMResourceFitSelector(Config(data_dir=str(tmp_path)), selected, [])
        with pytest.raises(ValueError, match="gpu-memory-utilization"):
            await selector.select_candidates([worker()])

    asyncio.run(check())


def test_invalid_device_memory_is_skipped(tmp_path, config_stub):
    async def check():
        from gpustack.config.config import Config
        node = worker()
        node.status.gpu_devices[0].memory = None
        node.status.gpu_devices[1].memory.total = 0
        assert await VLLMResourceFitSelector(Config(data_dir=str(tmp_path)), model(), []).select_candidates([node]) == []

    asyncio.run(check())


def test_spread_and_binpack_choose_expected_worker():
    async def check():
        nodes = [worker(1, sizes=(24,)), worker(2, sizes=(24,))]
        assigned = [instance(vram={0: 4 * GiB})]
        def candidates():
            return [ModelInstanceScheduleCandidate(
                worker=node, gpu_indexes=[0], gpu_type="cuda",
                computed_resource_claim=ComputedResourceClaim(vram={0: 8 * GiB}, ram=0),
            ) for node in nodes]
        spread = await PlacementScorer(model(), assigned).score(candidates())
        assert spread[1].score > spread[0].score
        packed = await PlacementScorer(model(placement_strategy="binpack"), assigned).score(candidates())
        assert packed[0].score > packed[1].score

    asyncio.run(check())


def test_local_config_loading_and_model_metadata(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            directory = tmp_path / "local-model"
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps({
                "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 8,
                "hidden_size": 1024, "vocab_size": 128000, "max_position_embeddings": 32768,
            }))
            selected = model(source="local_path", local_path=str(directory), categories=[])
            config = await get_pretrained_config_with_workers(selected)
            assert config.num_attention_heads == 32
            assert await scheduler_module.evaluate_pretrained_config(selected, raise_raw=True)
            assert selected.categories == ["llm"]

    asyncio.run(check())


def test_rescheduling_does_not_charge_its_own_reservation(tmp_path, monkeypatch, config_stub):
    async def check():
        async with project_context(tmp_path, monkeypatch, database_file=True) as env:
            stored_model, stored_instance = await seed(env.client, state="scheduled", worker_id=1)
            async with db.async_session() as session:
                node = worker(cluster_id=stored_model["cluster_id"], sizes=(24,))
                await Worker.create(session, node)
                stored = await Model.one_by_id(session, stored_model["id"])
                await stored.update(session, {"env": {"GPUSTACK_MODEL_VRAM_CLAIM": str(3 * GiB)}})
                current = await ModelInstance.one_by_id(session, stored_instance["id"])
                await ModelInstanceService(session).update(current, {
                    "gpu_type": "cuda", "gpu_indexes": [0],
                    "computed_resource_claim": ComputedResourceClaim(vram={0: int(24 * GiB * 0.9)}),
                    "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
                    "updated_at": datetime.now(timezone.utc) - timedelta(minutes=4),
                })
            scheduler = scheduler_module.Scheduler(env.cfg)
            async with db.async_session() as session:
                current = await ModelInstance.one_by_id(session, stored_instance["id"])
            await scheduler._evaluate(current)
            await scheduler._schedule_one(await scheduler._queue.get())
            async with db.async_session() as session:
                current = await ModelInstance.one_by_id(session, stored_instance["id"])
                assert current.state == ModelInstanceStateEnum.SCHEDULED
                assert current.worker_id == 1

    asyncio.run(check())


def test_scale_down_prefers_unhealthy_instance(tmp_path, monkeypatch):
    async def check():
        async with project_context(tmp_path, monkeypatch) as env:
            stored_model, _ = await seed(env.client)
            async with db.async_session() as session:
                await Worker.create(session, worker(cluster_id=stored_model["cluster_id"]))
                selected = await Model.one_by_id(session, stored_model["id"])
                healthy = instance(1, model_id=selected.id, state="running")
                failed = instance(2, model_id=selected.id, state="error")
                ranked = await find_scale_down_candidates([healthy, failed], selected)
                assert [candidate.model_instance.id for candidate in ranked] == [2, 1]

    asyncio.run(check())
