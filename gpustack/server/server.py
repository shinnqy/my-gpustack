import asyncio
import os
import logging
from multiprocessing import Process

import uvicorn

from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.app import create_app
from gpustack.server.init_db import init_db
from gpustack.server.controllers import (
    ModelController,
    ModelInstanceController
)

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

    @property
    def config(self):
        return self._config

    async def start(self):
        logger.info("Starting GPUStack server.")

        await self._prepare_data()

        app = create_app(self._config)
        self._app = app

        self._start_leader_tasks()

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

    async def _prepare_data(self):
        self._setup_data_dir(self._config.data_dir)

        await init_db(self._config.get_database_url())

        logger.debug("Data initialization completed.")

    @staticmethod
    def _setup_data_dir(data_dir: str):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _start_controllers(self):
        """Start all controllers and return the list of tasks."""
        tasks = []

        model_controller = ModelController(self._config)
        tasks.append(asyncio.create_task(model_controller.start()))

        model_instance_controller = ModelInstanceController(self._config)
        tasks.append(asyncio.create_task(model_instance_controller.start()))

        logger.debug("Controllers started.")
        return tasks

    def _start_leader_tasks(self):
        """Start tasks that run only on the leader.

        Note: If leadership is lost, the process exits directly (os._exit),
        so we don't need to track and cancel these tasks.
        """
        # Controllers
        self._start_controllers()