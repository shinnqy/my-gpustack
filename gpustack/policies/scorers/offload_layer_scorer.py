from gpustack.policies.base import ModelInstanceScore
from gpustack.policies.base import ModelInstanceScorer
from gpustack.schemas.models import Model
from gpustack.schemas.models import ModelInstance
from typing import List
import logging


logger = logging.getLogger(__name__)


class OffloadLayerScorer(ModelInstanceScorer):
    """
    看运行在 GPU 上的层占比，默认最高 10 分
        - 得分 = offload_layers / total_layers × 10
    """
    def __init__(self, model: Model, max_score: float = 100.0):
        self._model = model
        self._max_score = max_score

    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        """
        Score the instances with offload layers.
        """

        logger.debug(
            f"model {self._model.name}, score instances with offload layer policy"
        )

        scored_instances = []
        for instance in instances:

            if instance.computed_resource_claim is None:
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            if (
                instance.computed_resource_claim.total_layers is None
                or instance.computed_resource_claim.offload_layers is None
            ):
                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=0)
                )
                continue

            score = 0
            total_layers = instance.computed_resource_claim.total_layers
            offload_layers = instance.computed_resource_claim.offload_layers

            if total_layers == offload_layers:
                score = self._max_score
            else:
                score = offload_layers / total_layers * self._max_score

            scored_instances.append(
                ModelInstanceScore(model_instance=instance, score=score)
            )

        return scored_instances
