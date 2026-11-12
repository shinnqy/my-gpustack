import asyncio
import logging
from multiprocessing import Process
import uvicorn
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.app import create_app
import os
from gpustack.server.init_db import init_db
from gpustack.server.controllers import (
    ModelController,
    ModelInstanceController,
)
import secrets
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.envs import (
    DEFAULT_CLUSTER_KUBERNETES,
)
from gpustack.schemas.clusters import Cluster, ClusterProvider, ClusterStateEnum
from gpustack.security import (
    API_KEY_PREFIX,
)
from gpustack.server.db import async_session
from gpustack.config.registration import write_registration_token
from gpustack.server.worker_status_buffer import flush_worker_status_to_db
from gpustack.scheduler.scheduler import Scheduler


logger = logging.getLogger(__name__)


class Server:
    def __init__(self, config: Config, worker_process: Process):
        self._config: Config = config
        self._async_tasks = []
        self._worker_process = worker_process

    def _create_async_task(self, coro):
        self._async_tasks.append(asyncio.create_task(coro))

    @property
    def config(self):
        return self._config

    async def start(self):
        logger.info("Starting GPUStack server.")

        await self._prepare_data()

        app = create_app(self._config)
        self._app = app

        # 真实代码这里是 await self._start_leader_only_tasks()：
        # 单机（LocalCoordinator）时它直接调用 _start_leader_tasks()。第 23 章还原。
        self._start_leader_tasks()

        # These tasks can run on all instances
        self._start_worker_status_flusher()

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

        server = uvicorn.Server(config)
        self._create_async_task(server.serve())

        await asyncio.gather(*self._async_tasks)

    async def _prepare_data(self):
        self._setup_data_dir(self._config.data_dir)

        await init_db(self._config.get_database_url())

        async with async_session() as session:
            await self._init_data(session)

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
        self._start_scheduler()
        self._start_controllers()

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

    async def _init_data(self, session: AsyncSession):
        init_data_funcs = [
            self._init_default_cluster,
            self._ensure_registration_token,
        ]
        for init_data_func in init_data_funcs:
            await init_data_func(session)

    def _start_worker_status_flusher(self):
        self._create_async_task(flush_worker_status_to_db())

        logger.debug("Worker status flusher started.")

    def _start_scheduler(self):
        """Start the scheduler and return the task."""
        scheduler = Scheduler(self._config)
        task = asyncio.create_task(scheduler.start())
        logger.debug("Scheduler started.")
        return task
