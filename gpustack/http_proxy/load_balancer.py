from typing import List
from gpustack.http_proxy.strategies import LoadBalancingStrategy, RoundRobinStrategy
from gpustack.schemas.models import ModelInstance

"""统一选择推理副本的入口"""

class LoadBalancer:
    """包装一个负载均衡策略，默认使用轮询"""

    def __init__(self, strategy: LoadBalancingStrategy = None):
        if strategy is None:
            # 默认轮询
            strategy = RoundRobinStrategy()
        self._strategy = strategy

    def set_strategy(self, strategy: LoadBalancingStrategy):
        self._strategy = strategy

    async def get_instance(self, instances: List[ModelInstance]) -> ModelInstance:
        return await self._strategy.select_instance(instances)
