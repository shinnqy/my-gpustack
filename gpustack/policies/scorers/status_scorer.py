from gpustack.policies.base import ModelInstanceScore
from gpustack.policies.base import ModelInstanceScorer
from gpustack.schemas.models import Model
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.schemas.workers import Worker
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.db import async_session
from typing import List
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class StatusScorer(ModelInstanceScorer):
    """
    看实例和 Worker 的状态，默认最高 100 分

    | 条件                                | 得分 |
    |---                                  |---:|
    | 没有绑定 Worker，或 Worker 已不存在    | 0 |
    | Worker 为 `NOT_READY`               | 0 |
    | 实例为 `ERROR`                       | 0 |
    | Worker 为 `READY`，且实例为 `RUNNING` | 100 |
    | 其他情况，例如正在启动                  | 50 |
    """
    def __init__(
        self,
        model: Model,
        model_instance: Optional[ModelInstance] = None,
        max_score: float = 100.0,
    ):
        self._model = model
        self._model_instance = model_instance
        self._max_score = max_score

    async def score_instances(
        self, instances: List[ModelInstance]
    ) -> List[ModelInstanceScore]:
        """
        Score the instances with the worker and instance status.
        """

        logger.debug(f"model {self._model.name}, score instances with status policy")

        scored_instances = []
        async with async_session() as session:
            workers = await Worker.all(session)
            worker_map = {worker.id: worker for worker in workers}

            for instance in instances:
                if instance.worker_id is None:
                    scored_instances.append(
                        ModelInstanceScore(model_instance=instance, score=0)
                    )
                    continue

                score = 0
                worker = worker_map.get(instance.worker_id)
                if worker is None:
                    scored_instances.append(
                        ModelInstanceScore(model_instance=instance, score=0)
                    )
                    continue

                if worker.state == WorkerStateEnum.NOT_READY:
                    score = 0
                elif instance.state == ModelInstanceStateEnum.ERROR:
                    score = 0
                elif (
                    worker.state == WorkerStateEnum.READY
                    and instance.state == ModelInstanceStateEnum.RUNNING
                ):
                    score = self._max_score
                else:
                    score = self._max_score * 0.5

                scored_instances.append(
                    ModelInstanceScore(model_instance=instance, score=score)
                )

            return scored_instances
