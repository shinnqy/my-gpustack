from gpustack.client import ClientSet
from gpustack.schemas.inference_backend import InferenceBackend
from typing import Optional
from gpustack.schemas.inference_backend import get_built_in_backend

"""提供当前阶段的后端配置查询"""

class InferenceBackendManager:
    """阶段版：只认内置后端。第 19 章换成 watch Server 上 inference_backends 表的版本。"""

    def __init__(self, clientset: ClientSet):
        self._clientset = clientset
        self._builtin = {b.backend_name: b for b in get_built_in_backend()}

    def get_backend_by_name(
        self, backend_name: str, owner_principal_id: Optional[int] = None
    ) -> Optional[InferenceBackend]:
        return self._builtin.get(backend_name)
