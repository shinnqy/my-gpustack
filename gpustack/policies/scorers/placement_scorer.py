from collections import defaultdict
from gpustack import envs
from gpustack.policies.base import Allocatable
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.base import ModelInstanceScore
from gpustack.policies.base import ModelInstanceScorer
from gpustack.policies.base import ScheduleCandidatesScorer
from gpustack.policies.utils import get_worker_allocatable_resource
from gpustack.schemas.models import ComputedResourceClaim
from gpustack.schemas.models import Model
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.models import PlacementStrategyEnum
from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session
from typing import List
from typing import Optional
from enum import Enum
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class ResourceWeight:
    vram: float = 2.0
    ram: float = 1.0


@dataclass
class ModelWeight:
    current: float = 1.0
    others: float = 0.2


@dataclass
class InferenceServerTypeWeight:
    server: float = 5.0
    rpc_server: float = 1.0  # max rpc server count is 3


@dataclass
class SpreadScoreWeights:
    worker_weight: float = 0.85
    gpu_weight: float = 0.15
    zero_current_base_score: float = 0.85
    zero_current_others_weight: float = 0.15
    has_both_base_score: float = 0.45
    has_both_current_weight: float = 0.35
    has_both_others_weight: float = 0.20
    all_have_current_base_score: float = 0.50
    all_have_current_weight: float = 0.25
    all_have_others_weight: float = 0.15


class ScaleTypeEnum(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


class PlacementScorer(ScheduleCandidatesScorer, ModelInstanceScorer):
    def __init__(
        self,
        model: Model,
        model_instances: List[ModelInstance],
        scale_type: ScaleTypeEnum = ScaleTypeEnum.SCALE_UP,
        resource_weight: Optional[ResourceWeight] = None,
        model_weight: Optional[ModelWeight] = None,
        inference_server_type_weight: Optional[InferenceServerTypeWeight] = None,
        spread_score_weights: Optional[SpreadScoreWeights] = None,
        max_score: Optional[float] = None,
    ):
        self._model = model
        self._model_instances = model_instances
        self._resource_weight = resource_weight or ResourceWeight()
        self._model_weight = model_weight or ModelWeight()
        self._inference_server_type_weight = (
            inference_server_type_weight or InferenceServerTypeWeight()
        )
        self._spread_score_weights = spread_score_weights or SpreadScoreWeights()
        self._scale_type = scale_type
        self._max_score = (
            envs.SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE
            if max_score is None
            else max_score
        )

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        spread（分散，默认）：同一个模型的多个副本尽量放到不同的机器上。一台机器挂了，不会所有副本一起挂
            - 机器维度占 85%、卡维度占 15%。机器上已经有同一个模型的副本，扣分很多；有其他模型的副本，扣分较少；卡上已有的实例越多，分越低。
        binpack（紧凑）：尽量往已经用了一部分的机器、卡上塞。省出整台空机器，留给需要多卡的大模型。
            - 分数 = 放下之后这张卡（这台机器）的使用率。越满分越高。
        """

        logger.debug(
            f"model {self._model.readable_source}, score canidates with {self._scale_type} placement policy"
        )

        # 根据 placement_strategy 分发到 Binpack 或 Spread
        if self._model.placement_strategy == PlacementStrategyEnum.SPREAD:
            return await self.score_spread(candidates)
        elif self._model.placement_strategy == PlacementStrategyEnum.BINPACK:
            return await self.score_binpack(candidates)
        else:
            raise ValueError(
                f"Invalid placement strategy {self._model.placement_strategy}"
            )

    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        """
        Score the instances with placement strategy.
        """

        logger.debug(
            f"model {self._model.name}, score instances with {self._scale_type} placement policy"
        )

        async with async_session() as session:
            workers = await Worker.all(session)
            worker_map = {worker.id: worker for worker in workers}

            if self._model.placement_strategy == PlacementStrategyEnum.SPREAD:
                return await self.score_spread_instances(instances, worker_map)
            elif self._model.placement_strategy == PlacementStrategyEnum.BINPACK:
                return await self.score_binpack_instances(instances, worker_map)
            else:
                raise ValueError(
                    f"Invalid placement strategy {self._model.placement_strategy}"
                )

    async def score_binpack(  # noqa: C901
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        优先填满剩余空间较小的位置。
        """
        for candidate in candidates:
            allocatable = get_worker_allocatable_resource(
                self._model_instances, candidate.worker
            )

            final_score = 0
            score = await self._score_binpack_item(
                candidate.gpu_indexes,
                candidate.computed_resource_claim,
                allocatable,
                self._scale_type,
            )
            final_score = score

            candidate.score = final_score

        return candidates

    async def score_binpack_instances(  # noqa: C901
        self, instances: List[ModelInstance], worker_map: dict
    ) -> List[ModelInstanceScore]:
        """
        Score the candidates with the binpack strategy.
        """
        scored_instances = []

        for instance in instances:
            if instance.worker_id is None:
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            worker = worker_map.get(instance.worker_id)
            if worker is None:
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            allocatable = get_worker_allocatable_resource(self._model_instances, worker)

            final_score = 0
            score = await self._score_binpack_item(
                instance.gpu_indexes,
                instance.computed_resource_claim,
                allocatable,
                self._scale_type,
            )
            final_score = score

            scored_instances.append(
                ModelInstanceScore(model_instance=instance, score=final_score)
            )

        return scored_instances

    async def score_spread(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        优先把同一模型分散到不同位置
        """
        worker_model_instances_count_map = await self._get_worker_model_instance_count()
        workers = [candidate.worker for candidate in candidates if candidate.worker]
        spread_stats = self._build_spread_stats(
            worker_model_instances_count_map, workers
        )

        for candidate in candidates:
            candidate.score = await self._score_spread_item(
                candidate.gpu_indexes,
                candidate.worker,
                worker_model_instances_count_map,
                spread_stats,
            )

        return candidates

    async def score_spread_instances(
        self, instances: List[ModelInstance], worker_map: dict
    ) -> List[ModelInstanceScore]:
        """
        Score the candidates with the spread strategy.
        """
        worker_model_instances_count_map = await self._get_worker_model_instance_count()
        spread_stats = self._build_spread_stats(
            worker_model_instances_count_map, worker_map.values()
        )

        scored_instances = []
        for instance in instances:
            if instance.worker_id is None:
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            worker = worker_map.get(instance.worker_id)
            score = await self._score_spread_item(
                instance.gpu_indexes,
                worker,
                worker_model_instances_count_map,
                spread_stats,
            )
            scored_instances.append(
                ModelInstanceScore(model_instance=instance, score=score)
            )

        return scored_instances

    async def _score_spread_item(
        self,
        instance_gpu_indexes: List[int],
        worker: Worker,
        worker_model_instances_count_map: dict,
        spread_stats: dict,
    ) -> float:
        """
        优先把同一模型分散到不同位置

        GPU 分数 = 最高分 / (1 + current + 0.2 × others)
        Spread 总分 = Worker 分数 × 85% + GPU 分数 × 15%
        """

        if worker is None:
            return 0

        instance_count_map = self._get_instance_count_map(
            worker_model_instances_count_map, worker.id
        )

        if instance_gpu_indexes is not None and len(instance_gpu_indexes) > 0:
            return await self._score_spread_gpu(
                instance_count_map,
                instance_gpu_indexes,
                spread_stats,
            )
        else:
            return await self._score_spread_cpu(instance_count_map, spread_stats)

    async def _score_binpack_item(  # noqa: C901
        self,
        gpu_indexes: List[int],
        computed_resource_claim: ComputedResourceClaim,
        allocatable: Allocatable,
        scale_type: str,
    ) -> float:
        """
        优先填满剩余空间较小的位置。

        每卡分数 =
            最高分 × (
                RAM 申请量 / RAM 可分配量
                + 2 × VRAM 申请量 / VRAM 可分配量
            ) / 3
        """
        score = 0
        gpu_count = len(gpu_indexes) if gpu_indexes else 0

        def calculate_score(
            ram_claim: Optional[int],
            ram_allocatable: Optional[int],
            vram_claim: Optional[int],
            vram_allocatable: Optional[int],
        ):
            if ram_claim is None or ram_allocatable is None or ram_allocatable == 0:
                ram_score = 0
            else:
                ram_score = (
                    ram_claim
                    / ram_allocatable
                    * self._max_score
                    * self._resource_weight.ram
                )

            if vram_claim is None or vram_allocatable is None or vram_allocatable == 0:
                vram_score = 0
            else:
                vram_score = (
                    vram_claim
                    / vram_allocatable
                    * self._max_score
                    * self._resource_weight.vram
                )
            return (ram_score + vram_score) / (
                self._resource_weight.ram + self._resource_weight.vram
            )

        if gpu_count == 0:
            # computed_resource_claim.ram must have value when running cpu only model instance
            if scale_type == ScaleTypeEnum.SCALE_UP:
                score = computed_resource_claim.ram / allocatable.ram * self._max_score
            elif scale_type == ScaleTypeEnum.SCALE_DOWN:
                score = (
                    computed_resource_claim.ram
                    / (allocatable.ram + computed_resource_claim.ram)
                    * self._max_score
                )
        elif gpu_count > 0:
            for i in gpu_indexes:
                vram_claim = (computed_resource_claim.vram or {}).get(i, 0)
                allocatable_vram = (allocatable.vram or {}).get(i, 0)
                result = 0
                if scale_type == ScaleTypeEnum.SCALE_UP:
                    result = calculate_score(
                        computed_resource_claim.ram,
                        allocatable.ram,
                        vram_claim,
                        allocatable_vram,
                    )
                elif scale_type == ScaleTypeEnum.SCALE_DOWN:
                    result = calculate_score(
                        computed_resource_claim.ram,
                        allocatable.ram + (computed_resource_claim.ram or 0),
                        vram_claim,
                        allocatable_vram + vram_claim,
                    )
                if result > score:
                    score = result

        return score

    async def _score_spread_gpu(
        self,
        instance_count_map: dict,
        instance_gpu_indexes: List[int],
        spread_stats: dict,
    ) -> float:
        worker_score = self._score_spread_worker_score(instance_count_map, spread_stats)

        gpu_map = instance_count_map.get("gpu", {})
        per_gpu_scores = []
        for gpu_index in instance_gpu_indexes:
            gpu_count_map = gpu_map.get(gpu_index, {})
            current_count = gpu_count_map.get("current", 0)
            others_count = gpu_count_map.get("others", 0)
            per_gpu_scores.append(
                1 / (1 + current_count + others_count * self._model_weight.others)
            )

        gpu_score = sum(per_gpu_scores) / len(per_gpu_scores) if per_gpu_scores else 1
        gpu_score = gpu_score * self._max_score

        return (
            worker_score * self._spread_score_weights.worker_weight
            + gpu_score * self._spread_score_weights.gpu_weight
        )

    async def _score_spread_cpu(
        self, instance_count_map: dict, spread_stats: dict
    ) -> float:
        return self._score_spread_worker_score(instance_count_map, spread_stats)

    def _score_spread_worker_score(
        self, instance_count_map: dict, spread_stats: dict
    ) -> float:
        totals = instance_count_map.get("total", {})
        current_count = totals.get("current", 0)
        others_count = totals.get("others", 0)

        any_zero_current = spread_stats.get("any_zero_current", False)
        min_current = spread_stats.get("min_current", 0)
        max_current = spread_stats.get("max_current", 0)
        min_others = spread_stats.get("min_others", 0)
        max_others = spread_stats.get("max_others", 0)

        if any_zero_current:
            if current_count == 0:
                score = (
                    self._spread_score_weights.zero_current_base_score
                    + self._spread_score_weights.zero_current_others_weight
                    * inverse_norm(others_count, min_others, max_others)
                )
            else:
                score = (
                    self._spread_score_weights.has_both_base_score
                    + self._spread_score_weights.has_both_current_weight
                    * inverse_norm(current_count, min_current, max_current)
                    + self._spread_score_weights.has_both_others_weight
                    * inverse_norm(others_count, min_others, max_others)
                )
        else:
            score = (
                self._spread_score_weights.all_have_current_base_score
                + self._spread_score_weights.all_have_current_weight
                * inverse_norm(current_count, min_current, max_current)
                + self._spread_score_weights.all_have_others_weight
                * inverse_norm(others_count, min_others, max_others)
            )

        return score * self._max_score

    def _build_spread_stats(self, worker_model_instances_count_map, workers) -> dict:
        totals = []
        for worker in workers:
            if worker is None:
                continue
            instance_count_map = self._get_instance_count_map(
                worker_model_instances_count_map, worker.id
            )
            total_map = instance_count_map.get("total", {})
            totals.append((total_map.get("current", 0), total_map.get("others", 0)))

        if not totals:
            return {
                "any_zero_current": True,
                "min_current": 0,
                "max_current": 0,
                "min_others": 0,
                "max_others": 0,
            }

        current_counts = [current for current, _ in totals]
        others_counts = [others for _, others in totals]

        return {
            "any_zero_current": any(current == 0 for current in current_counts),
            "min_current": min(current_counts),
            "max_current": max(current_counts),
            "min_others": min(others_counts),
            "max_others": max(others_counts),
        }

    def _get_instance_count_map(self, worker_model_instances_count_map, worker_id):
        return worker_model_instances_count_map.get(
            worker_id,
            {
                "total": {"current": 0, "others": 0},
                "gpu": {},
            },
        )

    async def _get_worker_model_instance_count(self) -> dict:
        """
        Get current model and other models deployed model instance count for each worker/gpu.

        Returns:
            dict: A map of worker id to model instance count.

        Example:
            {
                "worker_1": {
                    "total": {"current": 2, "others": 3},
                    "gpu": {
                        0: {"current": 1, "others": 2},
                        1: {"current": 0, "others": 1}
                    }
                },
                "worker_2": {
                    "total": {"current": 1, "others": 1},
                    "gpu": {
                        0: {"current": 2, "others": 0}
                    }
                }
            }
        """

        if not hasattr(self._model, "id") or self._model.id is None:
            return {}

        model_id = self._model.id

        worker_model_instances_count_map = defaultdict(
            lambda: {
                "total": {"current": 0, "others": 0},
                "gpu": defaultdict(lambda: {"current": 0, "others": 0}),
            }
        )

        for model_instance in self._model_instances:
            if model_instance.worker_id is None:
                continue

            is_current_model = model_instance.model_id == model_id
            if model_instance.gpu_indexes:
                for gpu_index in model_instance.gpu_indexes:
                    update_count(
                        worker_model_instances_count_map,
                        model_instance.worker_id,
                        gpu_index,
                        is_current_model,
                    )
            else:
                update_count(
                    worker_model_instances_count_map,
                    model_instance.worker_id,
                    None,
                    is_current_model,
                )

        return worker_model_instances_count_map


def update_count(
    worker_model_instances_count_map, worker_id, gpu_index, is_current_model
):
    if gpu_index is not None:
        key = "current" if is_current_model else "others"
        worker_model_instances_count_map[worker_id]["gpu"][gpu_index][key] += 1

    key = "current" if is_current_model else "others"
    worker_model_instances_count_map[worker_id]["total"][key] += 1


def inverse_norm(value: int, min_value: int, max_value: int) -> float:
    """
    Inverse normalize a value into [0, 1] where smaller is better.

    Example:
        min=0, max=4:
        value=0 -> 1.0
        value=1 -> 0.75
        value=2 -> 0.5
        value=4 -> 0.0
    """

    if max_value <= min_value:
        return 1.0
    return 1.0 - (value - min_value) / (max_value - min_value)
