import asyncio
import os
import logging
import secrets
from sqlmodel.ext.asyncio.session import AsyncSession
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
from gpustack.envs import (
    DEFAULT_CLUSTER_KUBERNETES,
)
from gpustack.schemas.clusters import Cluster, ClusterProvider, ClusterStateEnum
from gpustack.security import (
    API_KEY_PREFIX,
)
from gpustack.server.db import async_session
from gpustack.config.registration import write_registration_token

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

        async with async_session() as session:
            await self._init_data(session)

        logger.debug("Data initialization completed.")

    async def _init_data(self, session: AsyncSession):
        # 按顺序调用（先建集群，再给它发令牌）
        init_data_funcs = [
            self._init_default_cluster,
            self._ensure_registration_token,
        ]
        for init_data_func in init_data_funcs:
            await init_data_func(session)

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

    def _should_create_default_cluster(self) -> bool:
        """纯 server 角色、且用户没有给 `--token` 时不建。这对应“Server 只做控制面，所有集群都由管理员手工创建”的部署方式。"""
        if self._config.server_role() == Config.ServerRole.BOTH:
            return True
        if self._config.token:
            return True
        return False

    async def _init_default_cluster(self, session: AsyncSession):
    # 真实版本：以“默认集群的系统主体是否存在”判断是否已初始化，并为集群创建系统主体
        if await Cluster.one_by_fields(session, {"is_default": True, "deleted_at": None}):
            return
        logger.info("Creating default cluster...")
        provider = ClusterProvider.Docker
        # Server 自己跑在 Kubernetes 里时（环境变量判断），默认集群就是 Kubernetes 类型
        if DEFAULT_CLUSTER_KUBERNETES:
            provider = ClusterProvider.Kubernetes
        hashed_suffix = secrets.token_hex(6)
        default_cluster = Cluster(
            name="Default Cluster",
            description="The default cluster for GPUStack",
            provider=provider,
            state=ClusterStateEnum.READY,
            hashed_suffix=hashed_suffix,
            # 先空着，由下面的 `_ensure_registration_token` 填
            registration_token="",
            is_default=True,
        )
        await Cluster.create(session, default_cluster)
        logger.debug("Default cluster created.")

    async def _ensure_registration_token(self, session: AsyncSession):
        # 真实版本：令牌是默认集群系统主体的 API Key（gpustack_<access>_<secret>）
        cluster = await Cluster.one_by_fields(session, {"is_default": True, "deleted_at": None})
        if cluster is None:
            logger.debug(
                "Default cluster user not exist, skipping registration token generation."
            )
            return
        token = cluster.registration_token
        if not token:
            token = f"{API_KEY_PREFIX}_{secrets.token_hex(8)}_{secrets.token_hex(16)}"
            await cluster.update(
                session=session,
                source={"registration_token": token},
            )

        write_registration_token(
            data_dir=self._config.data_dir,
            token=token,
        )

    def _should_create_default_cluster(self) -> bool:
        # only server or both will get into this logic
        if self._config.server_role() == Config.ServerRole.BOTH:
            return True
        if self._config.token:
            return True
        return False

    async def _init_default_cluster(self, session: AsyncSession):
        if not self._should_create_default_cluster():
            return
        # 真实版本：以“默认集群的系统主体是否存在”判断是否已初始化，并为集群创建系统主体
        if await Cluster.one_by_fields(session, {"is_default": True, "deleted_at": None}):
            return
        logger.info("Creating default cluster...")
        provider = ClusterProvider.Docker
        if DEFAULT_CLUSTER_KUBERNETES:
            provider = ClusterProvider.Kubernetes
        hashed_suffix = secrets.token_hex(6)
        default_cluster = Cluster(
            name="Default Cluster",
            description="The default cluster for GPUStack",
            provider=provider,
            state=ClusterStateEnum.READY,
            hashed_suffix=hashed_suffix,
            registration_token="",
            is_default=True,
        )
        await Cluster.create(session, default_cluster)
        logger.debug("Default cluster created.")