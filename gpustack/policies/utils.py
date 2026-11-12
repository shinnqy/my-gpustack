from typing import List
from typing import Optional
from gpustack.schemas.models import CategoryEnum
from gpustack.schemas.models import Model
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.models import is_llm_model
from gpustack.schemas.workers import Worker
from gpustack.utils.hub import get_model_weight_size
from sqlmodel.ext.asyncio.session import AsyncSession
import asyncio
from gpustack.policies.base import Allocated
from gpustack.schemas.models import ModelInstance
from gpustack.policies.base import Allocatable
from gpustack.utils.hub import get_diffusion_model_weight_size
from gpustack.scheduler.calculator import calculate_local_model_weight_size
import os
import logging
from typing import Dict
from gpustack.schemas.workers import GPUDevicesStatus
from typing import Any


logger = logging.getLogger(__name__)


def compute_worker_allocated(
    all_model_instances: List[ModelInstance],
    worker_id: int,
    gpu_type: Optional[str] = None,
) -> Allocated:
    """
    统计已经分配出去的资源 (这里统计的是调度器承诺给实例的资源预算。即使实例还没有启动，只要已经绑定 Worker 并写入资源声明，也会计入占用)
        - 它遍历模型实例，找到绑定到指定 Worker 的实例，累加各自的 computed_resource_claim
    """
    allocated = Allocated(ram=0, vram={})

    def add_vram(claim):
        for gpu_index, vram in (claim.vram or {}).items():
            allocated.vram[gpu_index] = allocated.vram.get(gpu_index, 0) + vram

    for mi in all_model_instances:
        if mi.worker_id == worker_id and (
            gpu_type is None or mi.gpu_type is None or mi.gpu_type == gpu_type
        ):
            claim = mi.computed_resource_claim
            if claim is not None:
                allocated.ram += claim.ram or 0
                if mi.gpu_indexes:
                    add_vram(claim)

    return allocated


def get_worker_allocatable_resource(
    all_model_instances: List[ModelInstance],
    worker: Worker,
    gpu_type: Optional[str] = None,
) -> Allocatable:
    """
    计算还能分配多少:
        每张卡可分配显存 = 显存总量 − 已分配显存 − 系统显存预留
        Worker 可分配内存 = 内存总量 − 已分配内存 − 系统内存预留
    """

    status = worker.status
    memory = status.memory if status else None
    reserved = worker.system_reserved
    reserved_ram = (reserved.ram or 0) if reserved else 0
    reserved_vram = (reserved.vram or 0) if reserved else 0
    is_unified_memory = bool(memory and memory.is_unified_memory)
    model_instances = get_worker_model_instances(all_model_instances, worker)
    allocated = compute_worker_allocated(model_instances, worker.id, gpu_type)

    allocatable = Allocatable(ram=0, vram={})
    if status and status.gpu_devices:
        for gpu in status.gpu_devices:
            gpu_index = gpu.index

            if (
                gpu.memory is None
                or gpu.memory.total is None
                or gpu.memory.total <= 0
                or gpu_index is None
                or (gpu_type is not None and gpu.type != gpu_type)
            ):
                continue
            allocatable_vram = max(
                (
                    gpu.memory.total
                    - allocated.vram.get(gpu_index, 0)
                    - reserved_vram
                ),
                0,
            )
            allocatable.vram[gpu_index] = allocatable_vram

    allocatable.ram = max(
        ((memory.total or 0) if memory else 0) - allocated.ram - reserved_ram, 0
    )

    if is_unified_memory:
        allocatable.ram = max(
            allocatable.ram
            - reserved_vram
            - sum(allocated.vram.values()),
            0,
        )

        # For UMA, we need to set the gpu memory to the minimum of
        # the calculated with max allow gpu memory and the allocatable memory.
        for gpu_index in allocatable.vram:
            allocatable.vram[gpu_index] = min(allocatable.ram, allocatable.vram[gpu_index])

    logger.debug(
        f"Worker {worker.name} gpu_type {gpu_type}, "
        f"reserved memory: {reserved_ram}, "
        f"reserved gpu memory: {reserved_vram}, "
        f"allocatable memory: {allocatable.ram}, "
        f"allocatable gpu memory: {allocatable.vram}"
    )
    return allocatable


def group_workers_by_gpu_type(workers: List[Worker]) -> Dict[str, List[Worker]]:
    """
    按 cuda、`rocm`、`cann` 等类型分组。同一 Worker 有不同类型设备时，复制出各类型的设备视图，保留原始清单。
    """
    gpu_type_to_workers: Dict[str, List[Worker]] = {}

    for worker in workers:
        if not worker.status or not worker.status.gpu_devices:
            gpu_type_to_workers.setdefault(None, []).append(worker)
            continue

        # Collect unique GPU types for this worker
        gpus: Dict[str, GPUDevicesStatus] = {}
        for gpu in worker.status.gpu_devices:
            gpus.setdefault(gpu.type, []).append(gpu)

        # Add worker to each GPU type group
        for gpu_type in gpus.keys():
            w = worker.model_copy()
            w.status = worker.status.model_copy(update={"gpu_devices": gpus[gpu_type]})
            gpu_type_to_workers.setdefault(gpu_type, []).append(w)

    return gpu_type_to_workers


def get_vram_claim_from_model_env(model: Model) -> Optional[int]:
    """
    Get the VRAM claim from model environment variable 'GPUSTACK_MODEL_VRAM_CLAIM' if set.
    """
    if model.env and 'GPUSTACK_MODEL_VRAM_CLAIM' in model.env:
        try:
            return int(model.env['GPUSTACK_MODEL_VRAM_CLAIM'])
        except ValueError:
            logger.warning(
                f"Invalid VRAM claim value for model {model.name}: {model.env['GPUSTACK_MODEL_VRAM_CLAIM']}"
            )

    return None


def should_skip_gpu_count_check(model: Model) -> bool:
    """
    读取手动选卡时的开关，决定是否跳过“选卡数量与并行参数推导数量一致”的检查
    """
    if not (model.gpu_selector and model.gpu_selector.gpu_ids):
        return False
    if not model.env:
        return False
    value = model.env.get("GPUSTACK_SKIP_GPU_COUNT_CHECK")
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def _get_cached_model_size(
    session: Optional[AsyncSession],
    model: Model,
) -> Optional[int]:
    """
    Get cached file size from downloaded ModelFile.
    """
    if not session:
        return None

    return None


async def estimate_model_vram(
    model: Model,
    token: Optional[str] = None,
    workers: Optional[List[Worker]] = None,
    session: Optional[AsyncSession] = None,
) -> int:
    """
    估算新模型需要多少显存
    """
    env_vram_claim = get_vram_claim_from_model_env(model)
    if env_vram_claim is not None:
        # Use as a potential workaround if the empirical vram estimation is far beyond the expected value.
        return env_vram_claim

    weight_size = 0
    timeout_in_seconds = 15

    try:
        if model.categories and CategoryEnum.IMAGE in model.categories:
            weight_size = await asyncio.wait_for(
                estimate_diffusion_model_vram(model, token, workers, session),
                timeout=timeout_in_seconds,
            )
            return weight_size
        elif (
            model.source == SourceEnum.HUGGING_FACE
            or model.source == SourceEnum.MODEL_SCOPE
        ):
            weight_size = await asyncio.wait_for(
                asyncio.to_thread(get_model_weight_size, model, token),
                timeout=timeout_in_seconds,
            )
        elif model.source == SourceEnum.LOCAL_PATH:
            # Try to get cached size from ModelFile first
            cached_size = await _get_cached_model_size(session, model)
            if cached_size:
                weight_size = cached_size
            else:
                # Fall back to querying workers
                weight_size = await get_local_model_weight_size(
                    model.local_path, workers, is_diffusion=False
                )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when getting weight size for model {model.name}")
    except Exception as e:
        logger.warning(f"Cannot get weight size for model {model.name}: {e}")

    # Reference: https://blog.eleuther.ai/transformer-math/#total-inference-memory
    activation_overhead_factor = 1.2

    # CUDA graphs can take additional 1~3 GiB memory
    # https://github.com/vllm-project/vllm/blob/v0.6.1/vllm/worker/model_runner.py#L1313
    # For non-LLM models like embedding, set a smaller overhead
    framework_overhead = 2 * 1024**3 if is_llm_model(model) else 512 * 1024**2

    lora_extra = 0

    return (
        int((weight_size + lora_extra) * activation_overhead_factor)
        + framework_overhead
    )


async def estimate_diffusion_model_vram(
    model: Model,
    token: Optional[str] = None,
    workers: Optional[List[Worker]] = None,
    session: Optional[AsyncSession] = None,
) -> int:
    """
    图像模型的估算入口；当前主要返回统计到的权重大小。
        - diffusion 模型没有 KV Cache，显存约等于权重大小
    """
    if model.env and 'GPUSTACK_MODEL_VRAM_CLAIM' in model.env:
        # Use as a potential workaround if the empirical vram estimation is far beyond the expected value.
        return int(model.env['GPUSTACK_MODEL_VRAM_CLAIM'])
    weight_size = 0
    timeout_in_seconds = 15
    try:
        if (
            model.source == SourceEnum.HUGGING_FACE
            or model.source == SourceEnum.MODEL_SCOPE
        ):
            weight_size = await asyncio.wait_for(
                asyncio.to_thread(get_diffusion_model_weight_size, model, token),
                timeout=timeout_in_seconds,
            )
        elif model.source == SourceEnum.LOCAL_PATH:
            # Try to get cached size from ModelFile first
            cached_size = await _get_cached_model_size(session, model)
            if cached_size:
                weight_size = cached_size
            else:
                # Fall back to querying workers with is_diffusion=True
                weight_size = await get_local_model_weight_size(
                    model.local_path, workers, is_diffusion=True
                )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when getting weight size for model {model.name}")
    except Exception as e:
        logger.warning(f"Cannot get weight size for model {model.name}: {e}")

    return weight_size


def get_worker_model_instances(
    all_model_instances: List[ModelInstance], worker: Worker
) -> List[ModelInstance]:
    """
    筛出绑定到某个 Worker 的实例，供资源统计使用。
        - 从所有实例里挑出和这台机器有关的（主机器是它，或者从属机器里有它
    """
    # Filter to get only the relevant instances:
    # 1. Instances assigned to this worker (main worker)
    # 2. Instances that use this worker as a subordinate worker
    relevant_instances = []
    for model_instance in all_model_instances:
        # Check if this is a main worker instance
        if model_instance.worker_id == worker.id:
            relevant_instances.append(model_instance)
        # Check if this worker is used as a subordinate worker

    return relevant_instances


class ListMessageBuilder:
    def __init__(self, messages: Optional[str | List[str]]):
        if not messages:
            self._messages = []
        self._messages = messages if isinstance(messages, list) else [messages]

    def append(self, message: str):
        self._messages.append(message)

    def extend(self, message: List[str]):
        self._messages.extend(message)

    def __str__(self) -> str:
        return "\n".join([f"- {line}" for line in self._messages])


def get_model_num_attention_heads(pretrained_config) -> Optional[int]:
    """
    从不同模型配置结构中读取文本注意力头数，供后续检查 TP 卡数是否满足整除要求。
    """

    num_attention_heads = None
    try:
        # Helper to get num_attention_heads from config
        def get_heads_from(cfg, key="num_attention_heads"):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                return value
            return None

        thinker_config = getattr(pretrained_config, "thinker_config", None)
        thinker_text_config = (
            getattr(thinker_config, "text_config", None) if thinker_config else None
        )

        configs_by_priority = [
            getattr(pretrained_config, "llm_config", None),
            getattr(pretrained_config, "text_config", None),
            pretrained_config,
            thinker_text_config,
        ]

        for config in configs_by_priority:
            if not config:
                continue
            heads = get_heads_from(config)
            if heads is not None:
                num_attention_heads = heads
                break

    except Exception as e:
        logger.warning(f"Cannot get num_attention_heads: {e}")

    return num_attention_heads


def _get_config_value(config: Any, key: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _get_config_int(config: Any, key: str) -> Optional[int]:
    value = _get_config_value(config, key)
    if type(value) is int:
        return value
    return None


def get_model_vision_num_attention_heads(pretrained_config: Any) -> Optional[int]:
    """
    读取视觉注意力头数，并计入 num_dummy_heads
    """
    try:
        vision_config = _get_config_value(pretrained_config, "vision_config")
        if not vision_config:
            return None

        num_heads = _get_config_int(vision_config, "num_attention_heads")
        if num_heads is None:
            num_heads = _get_config_int(vision_config, "num_heads")
        if num_heads is None or num_heads <= 0:
            return None

        num_dummy_heads = _get_config_int(vision_config, "num_dummy_heads") or 0
        return num_heads + max(num_dummy_heads, 0)
    except Exception as e:
        logger.warning(f"Cannot get vision num_attention_heads: {e}")
        return None


async def get_local_model_weight_size(
    local_path: str,
    workers: Optional[List[Worker]] = None,
    is_diffusion: bool = False,
) -> int:
    """
    Get the local model weight size in bytes.

    If the model exists locally (on the server), calculate it locally.
    Otherwise, if workers are provided, check if the model exists on any worker and get the size from there.

    Args:
        local_path: Path to the model directory
        workers: Optional list of workers to check
        is_diffusion: Whether this is a diffusion model (default: False)

    Returns:
        Total size in bytes
    """
    if os.path.exists(local_path):
        if not os.path.isdir(local_path):
            raise NotADirectoryError(
                f"The specified path '{local_path}' is not a directory."
            )
        try:
            # Use utility function to calculate size
            return calculate_local_model_weight_size(local_path, is_diffusion)
        except Exception as e:
            logger.error(f"Failed to calculate size locally for {local_path}: {e}")
            raise e

    if not workers:
        raise FileNotFoundError(f"The specified path '{local_path}' does not exist.")
    raise FileNotFoundError(
        f"Model path '{local_path}' is not accessible on the Server; remote weight sizing is not configured."
    )


def ram_not_enough(ram_claim: int, allocatable: Allocatable) -> bool:
    """
    Check if the allocatable RAM is not enough for the claimed RAM.
    """
    if ram_claim <= 0:
        return False
    return allocatable.ram < ram_claim


def get_model_ram_claim(model: Model) -> int:
    """
    Get the RAM requirement for the model in bytes.
    """
    return 0


def get_computed_ram_claim(
    model: Model, vram_claim: Dict[int, int], static_ram: Optional[int] = None
) -> Optional[int]:
    """
    Get the computed RAM claim for the model based on the provided model and vram_claim.
    The priority is as follows:
    1. If static_ram is provided, use it.
    2. If RAM size for extended KV cache is available, use it.
    3. If RAM ratio for extended KV cache is set and vram_claim is available, calculate RAM as ram_ratio * total_vram_claim.
    4. If neither is available, return None.
    """
    if static_ram:
        return static_ram
