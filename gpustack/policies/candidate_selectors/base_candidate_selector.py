from gpustack.policies.utils import should_skip_gpu_count_check
from gpustack.utils.gpu import group_gpu_indexes_by_gpu_type_and_worker
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
from gpustack.config import Config
from gpustack.policies.base import Allocatable, ModelInstanceScheduleCandidate
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
)
from gpustack.schemas.models import (
    Model,
    ModelInstance,
)
from gpustack.schemas.workers import Worker
from gpustack.scheduler.calculator import get_pretrained_config_with_workers
from gpustack.utils.gpu import group_gpu_ids_by_worker
from dataclasses import fields
from gpustack.policies.utils import get_model_num_attention_heads
from gpustack.policies.utils import get_model_vision_num_attention_heads
from gpustack.schemas.models import CategoryEnum
from gpustack.utils.hub import get_hf_text_config
from gpustack.utils.hub import get_max_model_len
from typing import Any
import enum


logger = logging.getLogger(__name__)


EVENT_ACTION_DEFAULT = "default_scheduling_msg"


EVENT_ACTION_MANUAL_MULTI = "manual_multi_gpu_scheduling_msg"


EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU = "auto_multi_worker_multi_gpu_scheduling_msg"


EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU = (
    "auto_single_worker_multi_gpu_scheduling_msg"
)


EVENT_ACTION_AUTO_SINGLE_GPU = "auto_single_gpu_scheduling_msg"


@dataclass
class RequestEstimateUsage:
    ram: int
    vram: int


class ModelAttentionTypeEnum(enum.Enum):
    UNK = "unknown"
    MHA = "multi_head_attention"
    GQA = "grouped_query_attention"
    MQA = "multi_query_attention"
    MLA = "multi_head_latent_attention"


@dataclass
class ModelParameters:
    """把不同模型的配置整理成统一结构"""

    # 模型架构
    architectures: List[str] = None
    # 词表与长度
    derived_max_seq_len: int = 0
    # 模型维度
    num_hidden_layers: int = 0
    # 模型维度
    hidden_size: Optional[int] = None
    # 词表与长度
    vocab_size: Optional[int] = None
    # 注意力参数
    num_attention_heads: Optional[int] = None
    # 注意力参数
    num_key_value_heads: int = 1
    n_group: Optional[int] = None
    head_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    torch_dtype: str = "bfloat16"
    quantize: Optional[str] = None
    # 量化
    quantization_config: Optional[Dict] = None
    # MoE 参数
    moe_num_experts: Optional[int] = None
    # MoE 参数
    moe_num_shared_experts: Optional[int] = None
    # MoE 参数
    moe_intermediate_size: Optional[int] = None

    is_multimodel: bool = False
    # 多模态参数
    vision_config: Optional[Dict] = None
    # 多模态参数
    vision_num_attention_heads: Optional[int] = None

    def from_model_pretrained_config(  # noqa: C901
        self, model: Model, pretrained_config: Any
    ):
        """
        先保存外层架构和视觉配置。
        多模态模型取出内部文本配置。
        将配置中与 dataclass 同名的字段复制进来。
        补齐需要推导或兼容不同名称的字段。
        """
        if hasattr(pretrained_config, "vision_config"):
            self.is_multimodel = True
            self.vision_config = pretrained_config.vision_config
            self.vision_num_attention_heads = get_model_vision_num_attention_heads(
                pretrained_config
            )

        # Get architectures first, it is not available in text_config.
        if hasattr(pretrained_config, "architectures"):
            self.architectures = pretrained_config.architectures

        pretrained_config = get_hf_text_config(pretrained_config)
        if pretrained_config is None and CategoryEnum.LLM in model.categories:
            # Exclude empty dict cases, as they indicate the locally-sourced model is not local to the server node.
            raise ValueError(f"Failed to get model {model.name} pretrained config")

        for attr_name in [attr.name for attr in fields(self.__class__)]:
            try:
                attr_value = getattr(pretrained_config, attr_name, None)
                if attr_value is not None:
                    setattr(self, attr_name, attr_value)
            except AttributeError:
                # If reach here, that means the field is an internal property,
                # which would not register in the argument parser.
                pass

        # Default
        self.derived_max_seq_len = get_max_model_len(pretrained_config)
        if not self.num_attention_heads:
            # For backward compatibility, try to get num_attention_heads from llm_config.
            self.num_attention_heads = get_model_num_attention_heads(pretrained_config)
        if not self.head_dim and self.hidden_size and self.num_attention_heads:
            # 如果配置没有 head_dim，就计算：
            self.head_dim = self.hidden_size // self.num_attention_heads
        if not self.moe_num_experts:
            for key in [
                "n_routed_experts",
                "num_local_experts",
                "num_experts",
            ]:
                if value := getattr(pretrained_config, key, None):
                    setattr(self, "moe_num_experts", value)
                    break
        if self.moe_num_experts and not self.moe_num_shared_experts:
            for key in [
                "n_shared_experts",
                "num_shared_experts",
            ]:
                if value := getattr(pretrained_config, key, None):
                    setattr(self, "moe_num_shared_experts", value)
                    break

    def get_attention_type(self) -> ModelAttentionTypeEnum:
        """
        根据这些参数推断 MHA、GQA、MQA、MLA
            - 例如 32 个查询头、8 个 KV 头，在当前判断规则下返回 GQA
        """

        if self.num_attention_heads:
            if self.num_key_value_heads == 1:
                return ModelAttentionTypeEnum.MQA
            elif (
                1 < self.num_key_value_heads < self.num_attention_heads
                and self.num_attention_heads % self.num_key_value_heads == 0
            ):
                return ModelAttentionTypeEnum.GQA
            elif self.num_key_value_heads == self.num_attention_heads:
                if self.q_lora_rank and self.kv_lora_rank:
                    return ModelAttentionTypeEnum.MLA
                return ModelAttentionTypeEnum.MHA
        return ModelAttentionTypeEnum.UNK


"""
IMPORTANT:

这个文件是本次新增的候选选择器公共基类。它把模型参数解析、选卡数量处理、剩余资源查询和 TP 约束检查集中起来，供具体后端复用。
"""

class ScheduleCandidatesSelector(ABC):
    """定义所有选择器的共同接口"""

    # GPUStack configuration.
    _config: Config
    # Model to be scheduled.
    _model: Model
    # Model hyperparameters.
    _model_params: ModelParameters
    # Frequently used model parameter in selectors.
    _num_attention_heads: int
    _vision_num_attention_heads: int
    # Number of GPUs required by the model.
    # Derived from GPU selectors(manual) or Parallelism parameters(auto).
    _gpu_count: int
    # Selected worker names if manual GPU selection is used.
    _selected_gpu_workers: Optional[List[str]]
    # Worker allocatable resource cache by gpu type.
    _workers_allocatable_resource_by_gpu_type: Dict[str, Dict[int, Allocatable]]

    def __init__(
        self,
        config: Config,
        model: Model,
        model_instances: List[ModelInstance],
    ):
        self._config = config
        # 正在调度的模型
        self._model = model
        # 已有实例，用于统计资源占用
        self._model_instances = model_instances
        # 整理后的模型参数
        self._model_params = ModelParameters()
        self._num_attention_heads = 0
        self._vision_num_attention_heads = 0
        # 要求的卡数，0 表示尚未明确指定
        self._gpu_count = 0
        # 手动选卡涉及的 Worker
        self._selected_gpu_workers = None
        self._selected_gpu_indexes_by_gpu_type_and_worker = {}
        self._workers_allocatable_resource_by_gpu_type = {}

    @abstractmethod
    def get_messages(self) -> List[str]:
        """
        返回没有找到候选时的解释
        """
        pass

    @abstractmethod
    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        返回能运行模型的候选方案
        """
        pass

    @abstractmethod
    def _should_check_vision_tp_divisibility(self) -> bool:
        """
        决定是否检查视觉注意力头数的 TP 整除约束
        """
        pass

    async def _init_model_parameters(self, workers: List[Worker]):
        """负责读取配置，再调用前面的 ModelParameters 完成整理"""

        try:
            pretrained_config = await get_pretrained_config_with_workers(
                self._model,
                workers,
                trust_remote_code=True,
            )

            self._model_params.from_model_pretrained_config(
                self._model, pretrained_config
            )
            self._num_attention_heads = self._model_params.num_attention_heads
            self._vision_num_attention_heads = (
                self._model_params.vision_num_attention_heads
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse model {self._model.name} hyperparameters: {e}"
            )

    def _set_gpu_count(
        self,
        world_size: Optional[int] = None,
        strategies: Optional[List[str]] = None,
    ):
        """
        统一处理“这个副本需要几张卡”

        例如：
        gpu_ids = [
            "worker-a:cuda:0",
            "worker-a:cuda:2",
        ]
        会整理成：
        {
            "cuda": {
                "worker-a": [0, 2]
            }
        }
        """

        model = self._model
        if model.gpu_selector:
            requested = model.gpu_selector.gpus_per_replica
            if requested is not None and requested < 0:
                raise ValueError("gpus_per_replica cannot be negative")
            self._gpu_count = requested or 0
        if model.gpu_selector and model.gpu_selector.gpu_ids:
            gpu_indexes_by_gpu_type_and_worker = (
                group_gpu_indexes_by_gpu_type_and_worker(model.gpu_selector.gpu_ids)
            )
            gpu_ids_by_worker = group_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            self._selected_gpu_workers = list(gpu_ids_by_worker.keys())
            self._selected_gpu_indexes_by_gpu_type_and_worker = (
                gpu_indexes_by_gpu_type_and_worker
            )

            self._gpu_count = model.gpu_selector.gpus_per_replica or len(
                model.gpu_selector.gpu_ids
            )

        # When world_size is set.
        if world_size and world_size > 0:
            if self._gpu_count and self._gpu_count != world_size:
                # Both gpu selector and parallelism are set, validate they match.
                if should_skip_gpu_count_check(model):
                    logger.warning(
                        f"Model {model.name}: world size ({world_size}) does not match "
                        f"selected gpu count ({self._gpu_count}); skipping the check due "
                        f"to GPUSTACK_SKIP_GPU_COUNT_CHECK. Keeping the selected gpu count."
                    )
                    return
                strategies_str = "/".join(strategies) if strategies else "parallelism"
                raise ValueError(
                    f"Model {model.name} has {strategies_str} set, but the selected gpu count ({self._gpu_count}) does not match the world size ({world_size}). "
                    f"Adjust the selected GPUs or the parallelism backend parameters so they match. "
                    f"If the mismatch is intentional, add GPUSTACK_SKIP_GPU_COUNT_CHECK=1 to the model's environment variables to bypass this check and keep the selected gpu count."
                )

            self._gpu_count = world_size

    def get_worker_allocatable_resource(
        self, worker: Worker, gpu_type: Optional[str] = None
    ) -> Allocatable:
        """
        查询并缓存本轮剩余资源
        """

        allocatable = self._workers_allocatable_resource_by_gpu_type.get(
            gpu_type, {}
        ).get(worker.id)
        if allocatable is not None:
            return allocatable

        allocatable = get_worker_allocatable_resource(
            self._model_instances, worker, gpu_type
        )
        self._workers_allocatable_resource_by_gpu_type.setdefault(
            gpu_type, {}
        ).setdefault(worker.id, allocatable)
        return allocatable

    def _is_tp_size_divisible(self, tp_size: int) -> bool:
        """
        Check whether InferenceBackend's constraint of parameter divisibility is satisfied.
        1. num_attention_heads
        2. vision_num_attention_heads
        3. vocab_size

        Notes on `tp_size` (tensor parallel size) usage in auto scheduling:
        - Single-worker multi-GPU: `tp_size` is the number of GPUs currently selected
          in the traversal (i.e., `gpu_sum`). The scheduler grows the candidate set
          incrementally and validates divisibility at each step.
        - Multi-worker multi-GPU: selected workers are constrained to have the same
          number of GPUs, so `tp_size` equals the per-worker GPU count (i.e., `gpu_count`)
        """
        if not tp_size:
            return False
        if self._num_attention_heads and self._num_attention_heads % tp_size != 0:
            return False
        if (
            self._should_check_vision_tp_divisibility()
            and self._vision_num_attention_heads
            and self._vision_num_attention_heads % tp_size != 0
        ):
            return False

        if (
            self._model_params.vocab_size
            and self._model_params.vocab_size % tp_size != 0
        ):
            return False

        return True

    def _check_tp_size_divisibility(
        self,
        tp_size: int,
    ) -> Optional[str]:
        """
        Check whether InferenceBackend's constraint of parameter divisibility is satisfied.
        1. num_attention_heads
        2. vision_num_attention_heads
        3. vocab_size

        Return:
            None if divisibility is satisfied, otherwise an error message.
        """
        if not tp_size:
            return None
        if self._num_attention_heads and self._num_attention_heads % tp_size != 0:
            return (
                f"Total number of attention heads ({self._num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tp_size})."
            )
        if (
            self._should_check_vision_tp_divisibility()
            and self._vision_num_attention_heads
            and self._vision_num_attention_heads % tp_size != 0
        ):
            return (
                f"Total number of vision attention heads ({self._vision_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tp_size})."
            )

        if (
            self._model_params.vocab_size
            and self._model_params.vocab_size % tp_size != 0
        ):
            return (
                f"Vocabulary size ({self._model_params.vocab_size})"
                " must be divisible by tensor parallel size "
                f"({tp_size})."
            )

        return None
