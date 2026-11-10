import asyncio
import logging
from multiprocessing import Process

import uvicorn

from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.app import create_app

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, config: Config, worker_process: Process):
        self._config: Config = config
        self._async_tasks = []
        self._worker_process = worker_process

    def _create_async_task(self, coro):
        """
        `_create_async_task` + `gather`：把所有长期运行的协程收集起来，一起等。任何一个抛异常，`gather` 就会抛出，`run_server` 退出进程。以后每章往这里加任务，只需要一行 `self._create_async_task(...)`。
        """

        self._async_tasks.append(asyncio.create_task(coro))

    async def start(self):
        logger.info("Starting GPUStack server.")

        app = create_app(self._config)

        # 内嵌网关模式下，API 只监听本机，由网关对外；其他模式直接监听所有网卡
        serving_host = (
            "127.0.0.1"
            if self._config.gateway_mode == GatewayModeEnum.embedded
            else "0.0.0.0"
        )

        config = uvicorn.Config(
            app,
            host=serving_host,
            port=self._config.get_api_port(),
            access_log=False,
            log_level="error",
        )

        """
        为什么用 `uvicorn.Server(config).serve()`，而不是常见的 `uvicorn.run(app)`？
            - `uvicorn.run` 会自己创建并占住事件循环。我们已经在 `asyncio.run(server.start())` 里了，而且后面还要在同一个循环里并行跑调度器、控制器等十几个后台任务。`server.serve()` 是一个普通协程，可以和它们一起放进 `asyncio.gather`。
        """
        server = uvicorn.Server(config)
        self._create_async_task(server.serve())

        await asyncio.gather(*self._async_tasks)
