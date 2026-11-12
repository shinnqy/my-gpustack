import base64
from .generated_http_client import HTTPClient
from typing import Optional
from .generated_worker_client import WorkerClient
from .generated_model_client import ModelClient
from .generated_model_instance_client import ModelInstanceClient
from .generated_model_file_client import ModelFileClient


class ClientSet:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        headers: Optional[dict] = None,
        timeout: Optional[float] = 60.0,
        enable_cache: bool = True,
    ):
        if headers is None:
            headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif username and password:
            base64_credentials = base64.b64encode(
                f"{username}:{password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {base64_credentials}"
        self.base_url = base_url
        self.headers = headers
        http_client = (
            HTTPClient(
                base_url=base_url,
            )
            .with_headers(headers)
            .with_timeout(timeout)
        )
        self.http_client = http_client
        # 查看有哪些节点、节点状态和 GPU 信息。
        self.workers = WorkerClient(
            http_client,
            enable_cache=enable_cache,
        )
        # 创建或修改模型来源、推理后端、期望副本数等。
        self.models = ModelClient(
            http_client,
            enable_cache=enable_cache,
        )
        # 监听副本分配到哪个 Worker，查询和回报实例状态。
        self.model_instances = ModelInstanceClient(
            http_client,
            enable_cache=enable_cache,
        )
        # 监听文件准备任务，回报下载进度、实际路径和成功／失败状态。
        self.model_files = ModelFileClient(
            http_client,
            enable_cache=enable_cache,
        )
