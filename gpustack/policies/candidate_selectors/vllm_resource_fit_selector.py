from gpustack.config import Config
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.candidate_selectors.base_candidate_selector import EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU
from gpustack.policies.candidate_selectors.base_candidate_selector import EVENT_ACTION_AUTO_SINGLE_GPU
from gpustack.policies.candidate_selectors.base_candidate_selector import EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU
from gpustack.policies.candidate_selectors.base_candidate_selector import EVENT_ACTION_DEFAULT
from gpustack.policies.candidate_selectors.base_candidate_selector import EVENT_ACTION_MANUAL_MULTI
from gpustack.policies.candidate_selectors.base_candidate_selector import ScheduleCandidatesSelector
from gpustack.policies.event_recorder.recorder import EventCollector
from gpustack.policies.event_recorder.recorder import EventLevelEnum
from gpustack.policies.utils import ListMessageBuilder
from gpustack.policies.utils import estimate_model_vram
from gpustack.policies.utils import get_computed_ram_claim
from gpustack.policies.utils import get_model_ram_claim
from gpustack.policies.utils import group_workers_by_gpu_type
from gpustack.policies.utils import ram_not_enough
from gpustack.schemas.models import CategoryEnum
from gpustack.schemas.models import ComputedResourceClaim
from gpustack.schemas.models import Model
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.workers import GPUDevicesStatus
from gpustack.schemas.workers import Worker
from gpustack.utils.command import find_bool_parameter
from gpustack.utils.command import find_parameter
from gpustack.utils.unit import byte_to_gib
from gpustack.utils.vllm_topology import parse_user_parallelism
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import logging
from collections import defaultdict
from gpustack.policies.utils import sort_workers_by_gpu_count
from gpustack.schemas.models import ModelInstanceSubordinateWorker
from gpustack.utils.command import resolve_executor_backend
from gpustack.utils.vllm_topology import validate_multinode_topology
from gpustack.policies.candidate_selectors.base_candidate_selector import RequestEstimateUsage


logger = logging.getLogger(__name__)

"""
这个文件是本次新增的 vLLM 资源匹配选择器。前面的基类负责准备参数和通用约束，这里开始真正回答：

这个模型能放在哪台 Worker、用哪些 GPU、每张卡需要预留多少显存？

它输出候选方案，随后调度器再调用评分器择优。
"""

class VLLMResourceFitSelector(ScheduleCandidatesSelector):
    def __init__(
        self,
        cfg: Config,
        model: Model,
        model_instances: List[ModelInstance],
    ):
        super().__init__(cfg, model, model_instances)

        self._vram_claim = 0
        self._ram_claim = 0
        self._largest_single_gpu_vram = 0
        self._largest_single_gpu_vram_utilization = 0
        self._largest_multi_gpu_vram = 0
        self._largest_multi_gpu_total = 0
        self._largest_multi_gpu_utilization_satisfied_count = 0

        self._messages = []
        self._event_collector = EventCollector(self._model, logger)

        world_size, strategies = (
            VLLMResourceFitSelector.get_world_size_from_backend_parameters(model)
        )
        """GPU 数量计算结果交给基类 _set_gpu_count()，与手动选卡、`gpus_per_replica` 协调。"""
        self._set_gpu_count(world_size, strategies)

    async def _init_model_parameters(self, workers: List[Worker]):
        await super()._init_model_parameters(workers)
        # GMU relies on architecture info in model parameters. Set it after model parameters are initialized.
        self._set_gpu_memory_utilization()

    def _should_check_vision_tp_divisibility(self) -> bool:
        if not self._model.backend_parameters:
            return True

        language_only = find_bool_parameter(
            self._model.backend_parameters, ["language-model-only"]
        )
        return not language_only

    @staticmethod
    def get_world_size_from_backend_parameters(
        model: Model,
    ) -> Tuple[Optional[int], Optional[List[str]]]:
        """
        先确定需要几张卡
            - world_size = TP × PP × PCP × DP (即计算本次部署需要多少个 GPU 执行进程，通常也就是需要多少张 GPU)
                - TP: 张量并行 (把同一层的权重、矩阵计算拆到多张 GPU)
                - PP: 流水线并行 (把模型的不同层分配给不同执行阶段)
                - PCP: 预填充上下文并行 (将长输入在 Prefill 阶段的序列计算分摊到多个进程)
                - DP: 数据并行 (增加处理不同请求的并行执行组)
        """
        # 提取 vLLM 并行参数
        parallelism = parse_user_parallelism(model.backend_parameters)
        tp = parallelism.tp
        pp = parallelism.pp
        dp = parallelism.dp
        dpl = parallelism.dpl
        pcp = parallelism.pcp

        # A non-positive value would silently yield a zero/negative world_size below.
        for name, value in (
            ("tensor-parallel-size", tp),
            ("pipeline-parallel-size", pp),
            ("prefill-context-parallel-size", pcp),
            ("data-parallel-size", dp),
            ("data-parallel-size-local", dpl),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"vLLM: --{name} {value} must be positive.")

        # Hybrid-LB: --data-parallel-size is a GLOBAL count spanning separate
        # deployments (e.g. one per node), so it must NOT size this deployment's
        # local GPU need. Only `dpl` ranks run here, each taking tp*pp GPUs, so
        # the local world is tp*pp*dpl. Without dpl we can't infer it; leave
        # world_size unset and let the manual GPU selection stand.
        hybrid_lb = find_bool_parameter(
            model.backend_parameters, ["data-parallel-hybrid-lb"]
        )
        if hybrid_lb:
            if dpl is None:
                return None, None
            dp = dpl

        # Each parallel dimension multiplies the world size in a fixed order
        # (tp, pp, pcp, dp).
        # DP-Local is a strategy label only — its ranks live inside the dp
        # world, so it never multiplies.
        world_dimensions = (("tp", tp), ("pp", pp), ("pcp", pcp), ("dp", dp))
        if any(value for _, value in world_dimensions):
            world_size = 1
            strategies = []
            for name, value in world_dimensions:
                if value:
                    strategies.append(name)
                    world_size *= value

            # In hybrid-lb, dp already carries the dpl value above, so the "dp"
            # strategy already reflects the local world; don't double-count dpl.
            if dpl and not hybrid_lb:
                strategies.append("dpl")

            return world_size, strategies

        return None, None

    def _set_gpu_memory_utilization(self):
        """
        确定显存预留比例 GMU (--gpu-memory-utilization)
            - _vram_claim: 	    模型的估算显存需求
            - allocatable_vram: 扣除已有实例和系统预留后的可分配显存
            - GPU 总显存 × GMU:  这次部署在该卡上的预留预算
        """

        self._gpu_memory_utilization = 0.9
        model = self._model
        if self._disable_gpu_memory_utilization():
            # gpu memory utilization is not used for non-LLM models
            self._gpu_memory_utilization = 0

        self._gpu_memory_utilization_parameter_name = "gpu-memory-utilization"
        gmu = find_parameter(
            model.backend_parameters, [self._gpu_memory_utilization_parameter_name]
        )
        if gmu:
            self._gpu_memory_utilization = float(gmu)
            if not 0 < self._gpu_memory_utilization <= 1:
                raise ValueError("--gpu-memory-utilization must be greater than 0 and at most 1")

    def _disable_gpu_memory_utilization(self) -> bool:
        """
        Determine whether GPU memory utilization should be disabled. vLLM does not use --gpu-memory-utilization for non-LLM models
        like embedding and reranker, except for some specific models like Qwen3-Embedding and Qwen3-Reranker.

        Rules:
        1. For non-LLM models, GPU memory utilization is DISABLED (return True) unless they are in the exception list.
        2. Otherwise, GPU memory utilization is ENABLED (return False).
        """
        if not self._model.categories:
            return False

        architectures = self._model_params.architectures or []

        # Non-LLM models that vLLM still uses GPU memory utilization
        NON_LLM_GMU_EXCEPTIONS = {
            "Qwen3ForCausalLM",
            "Qwen3ForSequenceClassification",  # Qwen3-Embedding & Qwen3-Reranker
            "Qwen3VLForConditionalGeneration",  # Qwen3-VL-Embedding & Qwen3-VL-Reranker
        }

        use_gmu_categories = [CategoryEnum.LLM, CategoryEnum.SPEECH_TO_TEXT]
        if any(cat in self._model.categories for cat in use_gmu_categories):
            return False

        # Disable for non-LLM models unless they are in the exception list
        return not any(arch in NON_LLM_GMU_EXCEPTIONS for arch in architectures)

    def _cal_effective_vram(self) -> float:
        if self._largest_multi_gpu_total == 0:
            return 0.0
        return (
            byte_to_gib(self._largest_multi_gpu_vram)
            * self._gpu_memory_utilization
            * self._largest_multi_gpu_utilization_satisfied_count
            / self._largest_multi_gpu_total
        )

    def _set_messages(self):
        """记录为什么没有找到方案"""

        if self._messages:
            return

        event_messages = {
            EVENT_ACTION_DEFAULT: "",
            EVENT_ACTION_MANUAL_MULTI: "",
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU: "",
            EVENT_ACTION_AUTO_SINGLE_GPU: "",
        }

        for event in self._event_collector.events:
            event_messages[event.action] = event.message

        messages = event_messages[EVENT_ACTION_DEFAULT] + "\n"
        for action in [
            EVENT_ACTION_MANUAL_MULTI,
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            EVENT_ACTION_AUTO_SINGLE_GPU,
        ]:
            if event_messages[action]:
                messages += event_messages[action]
                break

        self._messages.append(messages)

    def get_messages(self) -> List[str]:
        return self._messages

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        处理指定的卡数、卡号
                ↓
        读取模型参数，确定显存预留比例
                ↓
        估算模型显存需求
                ↓
        尝试单 Worker、单 GPU
                ↓ 没有可行方案
        尝试单 Worker、多 GPU
                ↓
        返回候选，或返回失败原因
        """

        if self._selected_gpu_workers is not None:
            workers = [worker for worker in workers if worker.name in self._selected_gpu_workers]

        # Initialize model parameters.
        await self._init_model_parameters(workers)
        explicit_tp = parse_user_parallelism(self._model.backend_parameters).tp
        if explicit_tp and not self._is_tp_size_divisible(explicit_tp):
            self._messages.append(f"Tensor parallel size {explicit_tp} does not divide the model attention heads or vocabulary")
            return []

        self._vram_claim = await estimate_model_vram(
            self._model, self._config.huggingface_token, workers
        )
        self._ram_claim = get_model_ram_claim(self._model)
        logger.info(
            f"Calculated resource claim for model {self._model.readable_source}, "
            f"VRAM claim: {self._vram_claim}, RAM claim: {self._ram_claim}"
        )

        default_msg_list = ListMessageBuilder(
            f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM"
            f"{f' and {byte_to_gib(self._ram_claim)} GiB of RAM' if self._ram_claim > 0 else ''}."
        )
        if self._gpu_memory_utilization != 0:
            default_msg_list.append(
                f"With --{self._gpu_memory_utilization_parameter_name}={self._gpu_memory_utilization}, "
                f"all GPUs combined need to provide at least {byte_to_gib(int(self._vram_claim / self._gpu_memory_utilization))} GiB of total VRAM "
                f"and each GPU needs {int(self._gpu_memory_utilization * 100)}% of allocatable VRAM."
            )
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_DEFAULT,
            str(default_msg_list),
        )

        candidate_functions = [
            self.find_manual_gpu_selection_candidates,
            self.find_single_worker_single_gpu_full_offloading_candidates,
            self.find_single_worker_multi_gpu_full_offloading_candidates,
            self.find_multi_worker_multi_gpu_candidates,
        ]

        for candidate_func in candidate_functions:
            if self.should_skip_candidate_func(candidate_func):
                continue
            logger.debug(
                f"model {self._model.readable_source}, filter candidates with resource fit selector: {candidate_func.__name__}"
            )

            candidates = candidate_func(workers)

            if candidates:
                return candidates

        self._set_messages()
        return []

    def find_single_worker_single_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        单卡匹配：检查一张卡是否满足全部条件

        如果已经要求使用多张卡，直接跳过单卡阶段。否则，遍历 Worker 的 GPU，检查：
        - 设备是否有有效的卡号和显存总量。
        - 是否符合手动指定的 GPU 卡号。
        - 可分配资源是否满足要求。
        """
        if self._gpu_count is not None and self._gpu_count > 1:
            # Skip multi-GPU selection
            return []

        candidates = []
        workers_of_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_of_type.items():
            for worker in workers_of_type:
                if not worker.status.gpu_devices:
                    continue

                result = self._find_single_worker_single_gpu_full_offloading_candidates(
                    worker,
                    gpu_type,
                )
                if result:
                    candidates.extend(result)

        return candidates

    def _gpu_is_selected(self, worker: Worker, gpu) -> bool:
        if self._selected_gpu_workers is None:
            return True
        selected = self._selected_gpu_indexes_by_gpu_type_and_worker.get(gpu.type, {})
        return gpu.index in selected.get(worker.name, [])

    def _find_single_worker_single_gpu_full_offloading_candidates(
        self,
        worker: Worker,
        gpu_type: Optional[str] = None,
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        单卡匹配：检查一张卡是否满足全部条件
        """

        candidates = []

        allocatable = self.get_worker_allocatable_resource(worker, gpu_type)

        if ram_not_enough(self._ram_claim, allocatable):
            return []

        if not worker.status.gpu_devices:
            return []

        for _, gpu in enumerate(worker.status.gpu_devices):
            if (
                gpu.memory is None or not gpu.memory.total
                or gpu.memory.total <= 0 or gpu.index is None
                or not self._gpu_is_selected(worker, gpu)
            ):
                continue
            gpu_index = gpu.index
            allocatable_vram = allocatable.vram.get(gpu_index, 0)
            allocatable_gpu_memory_utilization = allocatable_vram / gpu.memory.total

            if allocatable_vram > self._largest_single_gpu_vram:
                self._largest_single_gpu_vram = allocatable_vram
                self._largest_single_gpu_vram_utilization = (
                    allocatable_gpu_memory_utilization
                )

            # exceeds_vram 为 true 表示：0.9 × 总量 装不下估算值
            exceeds_vram = (
                self._vram_claim > gpu.memory.total * self._gpu_memory_utilization
                if self._gpu_memory_utilization > 0  # LLMs
                else self._vram_claim > allocatable_vram  # non LLMs
            )
            # exceeds_memory_utilization 为 true 表示：剩余比例不到 0.9
            exceeds_memory_utilization = (
                self._gpu_memory_utilization > 0
                and allocatable_gpu_memory_utilization < self._gpu_memory_utilization
            )
            if exceeds_vram or exceeds_memory_utilization:
                continue

            vram_claim_bytes = (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0  # LLMs
                else int(self._vram_claim)  # non LLMs
            )

            vram_claim = {gpu_index: vram_claim_bytes}
            candidates.append(
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=[gpu_index],
                    gpu_type=gpu.type,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=get_computed_ram_claim(self._model, vram_claim),
                    ),
                )
            )

        if not candidates or (len(candidates) == 1 and candidates[0].overcommit):
            event_msg = f"The current available GPU only has {byte_to_gib(self._largest_single_gpu_vram)} GiB allocatable VRAM."
            if self._gpu_memory_utilization != 0:
                event_msg = (
                    event_msg.rstrip(".")
                    + f" ({(self._largest_single_gpu_vram_utilization * 100):.2f}%)."
                )
            self._event_collector.add(
                EventLevelEnum.INFO,
                EVENT_ACTION_AUTO_SINGLE_GPU,
                str(ListMessageBuilder(event_msg)),
            )

        return candidates

    def find_single_worker_multi_gpu_full_offloading_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        多卡匹配：在同一台 Worker 上逐张累加

        只有单卡没有可行方案时，才进入多卡阶段。当前实现按 GPU 类型分组，在每台 Worker 内：
        1. 筛出符合卡号限制、剩余容量满足 GMU 的 GPU。
        2. 按可分配显存从大到小排序。
        3. 逐张加入，累计显存预算。
        4. 检查当前卡数是否满足 TP 整除约束。
        5. 预算足够、卡数符合要求时生成候选。
        """
        if self._gpu_count == 1:
            return []

        candidates = []
        workers_of_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_of_type.items():
            for worker in workers_of_type:
                if not worker.status.gpu_devices:
                    continue

                result = self._find_single_worker_multi_gpu_full_offloading_candidates(
                    worker, gpu_type
                )
                if result:
                    candidates.extend(result)

        if not candidates:
            return []

        min_gpu_count = min(len(candidate.gpu_indexes) for candidate in candidates)
        final_candidates = [
            candidate
            for candidate in candidates
            if len(candidate.gpu_indexes) == min_gpu_count
        ]
        return final_candidates

    def _find_single_worker_multi_gpu_full_offloading_candidates(  # noqa: C901
        self, worker: Worker, gpu_type: Optional[str] = None
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Find single worker multi gpu full offloading candidates for the model instance.
        requires: worker.status.gpu_devices is not None
        """

        total_gpu = len(worker.status.gpu_devices)
        if total_gpu < 2:
            return None

        allocatable = self.get_worker_allocatable_resource(worker, gpu_type)

        if ram_not_enough(self._ram_claim, allocatable):
            return []

        gpu_list = []
        total_allocatable_vram = 0
        satisfied_gpu_count = 0

        for gpu in worker.status.gpu_devices:
            if (
                gpu.memory is None or not gpu.memory.total
                or gpu.memory.total <= 0 or gpu.index is None
                or not self._gpu_is_selected(worker, gpu)
            ):
                continue

            allocatable_vram = allocatable.vram.get(gpu.index, 0)
            total_allocatable_vram += allocatable_vram

            if allocatable_vram > 0 and allocatable_vram / gpu.memory.total >= self._gpu_memory_utilization:
                satisfied_gpu_count += 1
                gpu_list.append(gpu)

        if total_allocatable_vram > self._largest_multi_gpu_total:
            self._largest_multi_gpu_vram = total_allocatable_vram
            self._largest_multi_gpu_utilization_satisfied_count = satisfied_gpu_count
            self._largest_multi_gpu_total = len(worker.status.gpu_devices)

        # Sort by vram in descending order
        sorted_gpu_devices: GPUDevicesStatus = sorted(
            gpu_list,
            key=lambda gpu: allocatable.vram.get(gpu.index, 0),
            reverse=True,
        )

        vram_sum = 0
        gpu_sum = 0
        gpu_indexes = []
        vram_claim: Dict[int, int] = {}
        found_candidate = False
        for _, gpu in enumerate(sorted_gpu_devices):
            gpu_indexes.append(gpu.index)
            vram_claim[gpu.index] = (
                int(gpu.memory.total * self._gpu_memory_utilization)
                if self._gpu_memory_utilization > 0  # LLMs
                else allocatable.vram.get(gpu.index, 0)  # non LLMs
            )
            gpu_sum += 1
            vram_sum += vram_claim[gpu.index]

            if self._gpu_count and gpu_sum > self._gpu_count:
                break
            if not self._is_tp_size_divisible(gpu_sum):
                continue

            if self._gpu_count and gpu_sum >= self._gpu_count:
                if vram_sum >= self._vram_claim:
                    found_candidate = True
                # if self._gpu_count is set, cannot return more than gpu_count
                break

            if (not self._gpu_count) and vram_sum >= self._vram_claim:
                found_candidate = True
                break

        if found_candidate:
            return [
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_type=gpu_type,
                    gpu_indexes=gpu_indexes,
                    computed_resource_claim=ComputedResourceClaim(
                        vram=vram_claim,
                        ram=get_computed_ram_claim(self._model, vram_claim),
                    ),
                )
            ]
        event_msg_list = []
        if msg := self._check_tp_size_divisibility(
            self._largest_multi_gpu_utilization_satisfied_count
        ):
            event_msg_list.append(msg)
        event_msg = f"The largest available worker has {byte_to_gib(self._largest_multi_gpu_vram)} GiB allocatable VRAM."
        if self._gpu_memory_utilization != 0:
            event_msg = (
                event_msg.rstrip(".")
                + f", {self._largest_multi_gpu_utilization_satisfied_count}/{self._largest_multi_gpu_total} of GPUs meet the VRAM utilization ratio, providing {self._cal_effective_vram():.2f} GiB of allocatable VRAM."
            )
        event_msg_list.append(event_msg)

        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_AUTO_SINGLE_WORKER_MULTI_GPU,
            str(ListMessageBuilder(event_msg_list)),
        )

        return []

    def find_manual_gpu_selection_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        在用户指定的 GPU 范围内，先尝试单机部署，再尝试多机部署，生成符合资源要求的候选
        """

        """判断是否指定了具体 GPU"""
        if not self._selected_gpu_workers:
            return []
        """优先尝试在单台 Worker 上满足要求"""
        local = (
            self.find_single_worker_single_gpu_full_offloading_candidates(workers)
            or self.find_single_worker_multi_gpu_full_offloading_candidates(workers)
        )
        if local or len(self._selected_gpu_workers) == 1:
            return local
        """需要跨机器时，检查是否允许分布式推理"""
        if not self._model.distributed_inference_across_workers:
            self._event_collector.add(
                EventLevelEnum.INFO, EVENT_ACTION_MANUAL_MULTI,
                "Multi-worker GPU selection requires distributed_inference_across_workers.",
            )
            return []

        """按 GPU 类型分组，只保留用户指定的卡"""
        for gpu_type, group in group_workers_by_gpu_type(workers).items():
            selected_workers = []
            for worker in group:
                if not worker.status or not worker.status.gpu_devices:
                    continue
                selected = [gpu for gpu in worker.status.gpu_devices if self._gpu_is_selected(worker, gpu)]
                if not selected:
                    continue
                view = worker.model_copy()
                view.status = worker.status.model_copy(update={"gpu_devices": selected})
                allocatable = self.get_worker_allocatable_resource(view, gpu_type)
                """
                if RAM不足 or 任意所选GPU不满足要求:
                    break
                """
                if ram_not_enough(self._ram_claim, allocatable) or any(
                    not gpu.memory or not gpu.memory.total
                    or allocatable.vram.get(gpu.index, 0) < int(gpu.memory.total * self._gpu_memory_utilization)
                    for gpu in selected
                ):
                    break
                selected_workers.append(view)
            else:
                if len(selected_workers) < 2:
                    continue
                """检查总卡数，必要时截取，再检查总显存"""
                total_gpus = sum(len(worker.status.gpu_devices) for worker in selected_workers)
                if self._gpu_count and total_gpus < self._gpu_count:
                    continue
                if self._gpu_count and total_gpus > self._gpu_count:
                    remaining = self._gpu_count
                    subset = []
                    for worker in selected_workers:
                        if remaining == 0:
                            break
                        view = worker.model_copy()
                        gpus = worker.status.gpu_devices[:remaining]
                        view.status = worker.status.model_copy(update={"gpu_devices": gpus})
                        subset.append(view)
                        remaining -= len(gpus)
                    selected_workers = subset
                total_vram = sum(
                    int(gpu.memory.total * self._gpu_memory_utilization)
                    for worker in selected_workers for gpu in worker.status.gpu_devices
                )
                if total_vram < self._vram_claim:
                    continue
                candidate, reason = _create_candidate(
                    self._model, selected_workers, self._gpu_memory_utilization
                )
                if candidate:
                    return [candidate]
                if reason:
                    self._event_collector.add(EventLevelEnum.INFO, EVENT_ACTION_MANUAL_MULTI, reason)
        return []

    def should_skip_candidate_func(self, candidate_func) -> bool:
        """
        根据当前配置，决定传入的候选生成方法是否应该执行


        | 配置                    | 手动方法 | 自动单卡 | 自动单机多卡 | 自动多机 |
        |---                     |---      |---      |---         |---|
        | 未指定 GPU，关闭分布式    | 跳过     | 允许    | 允许        | 跳过 |
        | 未指定 GPU，开启分布式    | 跳过     | 允许    | 允许        | 允许 |
        | 已指定 GPU，任意分布式设置 | 允许    | 跳过     | 跳过        | 跳过 |
        """

        """没有指定具体 GPU，就跳过手动选卡"""
        if candidate_func == self.find_manual_gpu_selection_candidates:
            return not self._selected_gpu_workers
        """指定了具体 GPU，就跳过其他自动选卡方法。"""
        if (
            self._selected_gpu_workers
            and candidate_func != self.find_manual_gpu_selection_candidates
        ):
            return True

        """没开分布式推理，就跳过自动多机选卡"""
        if (
            not self._model.distributed_inference_across_workers
            and candidate_func == self.find_multi_worker_multi_gpu_candidates
        ):
            return True

        return False

    def find_multi_worker_multi_gpu_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        单机多卡也不行的时候，这个方法来选出一组机器
        """
        candidates = []
        workers_of_type = group_workers_by_gpu_type(workers)
        for gpu_type, workers_of_type in workers_of_type.items():
            result = self.auto_select_multi_worker_multi_gpu_candidates(
                workers_of_type, gpu_type
            )
            if result:
                candidates.extend(result)

        return candidates

    def auto_select_multi_worker_multi_gpu_candidates(  # noqa: C901
        self, workers: List[Worker], gpu_type: Optional[str] = None
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        自动选择多机候选，
        一个候选需要满足:
        1. 每台机器卡数相同：Ray 模式下 GPUStack 的默认切法是“机器内张量并行、机器间流水线并行”，即 TP = 每台卡数、PP = 机器数。卡数不同就切不平。
        2. 每台机器的每张卡都满足 GMU：多机部署会用满每台机器的所有卡（第 11 章说过 vLLM 会占住 90%），一张卡被别的实例占了，这台机器就不能用。
        3. 每台卡数能整除注意力头数：它就是 TP。
        4. 总量 ≥ 估算值。
        5. 用户写了并行度：凑够 tp × pp 张卡为止。

        做法：按卡数把机器分组（ sort_workers_by_gpu_count 先按卡数从多到少排序，卡多的机器优先，机器数就少），在每组里逐台加入，直到总量够。
            - 它采用逐台累加的贪心策略，每种 GPU 类型找到第一个可行组合就返回。
        """

        if not workers or len(workers) < 2:
            return []

        """
        先按“每台机器有几张卡”分组
            - 上层调用已经按 cuda、`rocm` 等 GPU 类型分组；这里进一步按卡数分组，例如：
                ```
                8 卡组：[Worker-A, Worker-B]
                4 卡组：[Worker-C, Worker-D, Worker-E]
                2 卡组：[Worker-F]
                ```
        """
        sort_workers_by_gpu_count(workers)

        workers_by_gpu_count_dict = defaultdict(list)
        for worker in workers:
            if not worker.status or not worker.status.gpu_devices:
                continue

            workers_by_gpu_count_dict[len(worker.status.gpu_devices)].append(worker)

        # Store the optimal combination info to show
        workers_combination: List[Worker] = []
        largest_vram = 0
        worker_count = 0
        device_count_per_worker = 0

        event_message = ListMessageBuilder([])
        seen_skip_reasons: set[str] = set()

        # Loop through worker groups with the same number of GPUs.
        for gpu_count, worker_group in workers_by_gpu_count_dict.items():
            if len(worker_group) < 2:
                continue

            """
            检查这组卡数是否满足 TP 整除约束

            例如模型有 40 个注意力头：
                每台 8 卡：40 % 8 == 0，继续
                每台 6 卡：40 % 6 != 0，跳过整组
            """
            if not self._is_tp_size_divisible(gpu_count):
                continue

            selected_workers: List[Worker] = []
            gpu_sum = 0
            vram_sum = 0
            for worker in worker_group:
                allocatable = self.get_worker_allocatable_resource(worker, gpu_type)

                if ram_not_enough(self._ram_claim, allocatable):
                    # The RAM resource(for extended KV cache) is required per worker.
                    # Skip the worker if it does not satisfy the RAM requirement.
                    continue

                if any(
                    gpu.memory is None
                    or gpu.memory.total is None
                    or (
                        allocatable.vram.get(gpu.index, 0) / gpu.memory.total
                        < self._gpu_memory_utilization
                    )
                    for gpu in worker.status.gpu_devices
                ):
                    # Skip the worker if any GPU does not satisfy the gpu_memory_utilization requirement.
                    continue
                """将合格机器加入组合，累计卡数和显存预算。"""
                selected_workers.append(worker)
                gpu_sum += gpu_count
                vram_sum += sum(
                    int(gpu.memory.total * (self._gpu_memory_utilization or 1))
                    for gpu in worker.status.gpu_devices
                )

                if self._gpu_count:
                    # Parallelism is set. Proceed until we match the exact GPU count.
                    if gpu_sum < self._gpu_count:
                        continue
                    elif gpu_sum > self._gpu_count:
                        break

                if vram_sum >= self._vram_claim:
                    candidate, skip_reason = _create_candidate(
                        self._model,
                        selected_workers,
                        self._gpu_memory_utilization,
                    )
                    if candidate is None:
                        if skip_reason and skip_reason not in seen_skip_reasons:
                            seen_skip_reasons.add(skip_reason)
                            event_message.append(skip_reason)
                        continue
                    return [candidate]
            if vram_sum > largest_vram:
                workers_combination = selected_workers
                largest_vram = vram_sum
                worker_count = len(worker_group)
                device_count_per_worker = gpu_count

        # Nothing can be return, construct scheduling message
        if self._gpu_memory_utilization == 0:
            event_message.append(
                f"The largest available worker has {byte_to_gib(largest_vram)} GiB of VRAM."
            )
        elif workers_combination:
            worker_names = [worker.name for worker in workers_combination]
            worker_names_msg = (
                str(worker_names[:3]).rstrip("]")
                + f"...(more {len(worker_names) - 3})]"
                if len(worker_names) > 3
                else str(worker_names)
            )
            message = f"The optimal combination {worker_names_msg} provides {byte_to_gib(largest_vram)} GiB of allocatable VRAM."
            if worker_count - len(workers_combination) > 0:
                message += f" There are {worker_count - len(workers_combination)} {'workers' if worker_count - len(workers_combination) > 1 else 'worker'} that can provide {device_count_per_worker} {'GPUs' if device_count_per_worker > 1 else 'GPU'}, as the workers in the combination, but some GPUs among them fail to meet requirements."
            event_message.append(message)

        event_message.append(
            "Cannot find a suitable worker combination to run the model in distributed mode. "
            "If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and GPUs."
        )
        self._event_collector.add(
            EventLevelEnum.INFO,
            EVENT_ACTION_AUTO_MULTI_WORKER_MULTI_GPU,
            str(event_message),
        )

        return []


def _create_candidate(
    model: Model,
    selected_workers: List[Worker],
    gpu_memory_utilization: float = 0.9,
) -> Tuple[Optional[ModelInstanceScheduleCandidate], Optional[str]]:
    """
    校验多节点 MP 并行布局，然后把选中的 Worker 包装成“主 Worker + 从 Worker”的调度候选
    """

    """多节点且使用 MP 时，先校验并行布局"""
    if len(selected_workers) > 1 and (
        resolve_executor_backend(model.backend_parameters, model.backend_version)
        == "mp"
    ):
        # 先统计每台机器参与部署的卡数
        gpu_per_node = [len(w.status.gpu_devices or []) for w in selected_workers]
        try:
            validate_multinode_topology(
                gpu_per_node, parse_user_parallelism(model.backend_parameters)
            )
        except ValueError as e:
            logger.info(
                f"Skipping worker combination {[w.name for w in selected_workers]} "
                f"for {model.name}: {e}"
            )
            return None, str(e)

    """IMPORTANT: Ray 和 MP 都会执行下面的代码"""

    """将第一台机器设为主 Worker，计算它的资源声明"""
    main_worker = selected_workers[0]
    vram_claim_main = {
        gpu.index: int(gpu.memory.total * gpu_memory_utilization)
        for gpu in main_worker.status.gpu_devices
    }
    gpu_type = main_worker.status.gpu_devices[0].type
    candidate = ModelInstanceScheduleCandidate(
        worker=main_worker,
        gpu_type=gpu_type,
        gpu_indexes=[gpu.index for gpu in main_worker.status.gpu_devices],
        computed_resource_claim=ComputedResourceClaim(
            vram=vram_claim_main,
            ram=get_computed_ram_claim(model, vram_claim_main),
        ),
    )
    """将其他机器包装成从 Worker"""
    candidate.subordinate_workers = []
    for worker in selected_workers[1:]:
        vram_claim_subworker = {
            # 显存预算仍然按同一个公式计算
            gpu.index: int(gpu.memory.total * gpu_memory_utilization)
            for gpu in worker.status.gpu_devices
        }
        candidate.subordinate_workers.append(
            ModelInstanceSubordinateWorker(
                worker_id=worker.id,
                worker_name=worker.name,
                worker_ip=worker.ip,
                worker_ifname=worker.ifname,
                total_gpus=len(worker.status.gpu_devices),
                gpu_type=gpu_type,
                gpu_indexes=[gpu.index for gpu in worker.status.gpu_devices],
                computed_resource_claim=ComputedResourceClaim(
                    vram=vram_claim_subworker,
                    ram=get_computed_ram_claim(model, vram_claim_subworker),
                ),
            )
        )

    """返回候选和失败原因组成的二元组"""
    return candidate, None
