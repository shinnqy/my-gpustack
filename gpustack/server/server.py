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
from gpustack.server.controllers import ModelFileController
from fastapi import FastAPI
from gpustack.server.worker_syncer import WorkerSyncer
from gpustack.server.worker_instance_cleaner import WorkerInstanceCleaner
from gpustack.server.controllers import ClusterController
from gpustack.server.controllers import WorkerController
from gpustack.gpu_instances import gateway_client
import shutil
import tempfile
from gpustack.gpu_instances.gateway import reconcile_gpustack_operator_subscription
import subprocess
import threading
import aiohttp
import signal


logger = logging.getLogger(__name__)


class _ExternalSubprocess:
    """
    用来管理外部程序
    """

    def __init__(self, args, name="", log=None, on_exit=None):
        self._args = args
        self.name = name or args[0]
        self._popen = None
        self._logger = log or logger
        self._log_threads = []
        self._on_exit = on_exit
        self._exit_callback_invoked = False
        self._exit_lock = threading.Lock()

    def start(self):
        try:
            self._popen = subprocess.Popen(
                self._args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except BaseException:
            self._invoke_on_exit()
            raise
        self._spawn_log_pump(self._popen.stdout, logging.INFO)
        self._spawn_log_pump(self._popen.stderr, logging.INFO)
        if self._on_exit is not None:
            self._spawn_exit_watcher()

    def stop(self):
        if self._popen and self._popen.poll() is None:
            self._popen.terminate()
            try:
                self._popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._popen.kill()
                self._popen.wait(timeout=5)
        self._invoke_on_exit()
        for thread in self._log_threads:
            thread.join(timeout=2)

    def _spawn_exit_watcher(self):
        def watch():
            try:
                self._popen.wait()
            finally:
                self._invoke_on_exit()

        t = threading.Thread(
            target=watch,
            name=f"{self.name}-exit-watcher",
            daemon=True,
        )
        t.start()

    def _invoke_on_exit(self):
        with self._exit_lock:
            if self._exit_callback_invoked:
                return
            self._exit_callback_invoked = True
        try:
            if self._on_exit is not None:
                self._on_exit()
        except Exception:
            self._logger.exception("[%s] on_exit callback failed", self.name)

    def _spawn_log_pump(self, stream, level):
        def pump():
            try:
                for line in iter(stream.readline, ""):
                    line = line.rstrip()
                    if line:
                        self._logger.log(level, "[%s] %s", self.name, line)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t = threading.Thread(
            target=pump,
            name=f"{self.name}-log-{'out' if level == logging.INFO else 'err'}",
            daemon=True,
        )
        t.start()
        self._log_threads.append(t)

    def is_alive(self) -> bool:
        if self._popen is None:
            return False
        return self._popen.poll() is None

    @property
    def exitcode(self):
        if self._popen is None:
            return None
        return self._popen.returncode


class Server:
    def __init__(self, config: Config, worker_process: Process):
        self._config: Config = config
        self._before_healthy_sub_processes = []
        self._after_healthy_sub_processes = []
        self._async_tasks = []
        self._worker_process = worker_process

    def _create_async_task(self, coro):
        self._async_tasks.append(asyncio.create_task(coro))

    @property
    def config(self):
        return self._config

    async def start(self):
        """
        Server 启动 wg 子进程

        ------------------
        准备数据库和初始数据
         |
        准备 Operator 启动配置
         |
        创建 API 应用
         |
        启动第一批子进程
         |
        启动调度器、控制器与订阅任务 (后台开始处理任务)
         |
        启动 http API
         |
        等待 API 就绪后启动第二批
         |
        持续检查子进程
        """
        logger.info("Starting GPUStack server.")
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        server = None
        previous_handlers = {}

        def request_shutdown():
            if server is None:
                current_task.cancel()
            else:
                server.should_exit = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(sig)
            try:
                loop.add_signal_handler(sig, request_shutdown)
            except (NotImplementedError, RuntimeError):
                continue
            previous_handlers[sig] = previous
        try:
            await self._prepare_data()
            # 准备 Operator 的启动命令、通信地址和清理动作，加入启动名单
            self._enqueue_operator_process()
            app = create_app(self._config)
            self._app = app
            self._start_sub_processes()

            # 真实代码这里是 await self._start_leader_only_tasks()：
            # 单机（LocalCoordinator）时它直接调用 _start_leader_tasks()。第 23 章还原。
            self._start_leader_tasks()

            self._start_worker_status_flusher()
            # 启动集群订阅协调器 (gateway_client 里的OperatorSubscriptionReconciler)
            self._start_gpustack_operator_subscription()
            serving_host = (
                "127.0.0.1"
                if self._config.gateway_mode == GatewayModeEnum.embedded
                else "0.0.0.0"
            )
            config = uvicorn.Config(
                app, host=serving_host, port=self._config.get_api_port(),
                access_log=False, log_level="error",
            )
            server = uvicorn.Server(config)
            api_task = asyncio.create_task(server.serve())
            self._async_tasks.append(api_task)
            pending = set(self._async_tasks)
            while api_task in pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if not task.cancelled():
                        task.result()
        finally:
            for task in self._async_tasks:
                task.cancel()
            await asyncio.gather(*self._async_tasks, return_exceptions=True)
            for process in self.all_processes:
                if isinstance(process, _ExternalSubprocess):
                    await asyncio.to_thread(process.stop)
            gateway_client.set_gateway_unix_path(None)
            for sig, previous in previous_handlers.items():
                loop.remove_signal_handler(sig)
                signal.signal(sig, previous)

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
        worker_controller = WorkerController(self._config)
        tasks.append(asyncio.create_task(worker_controller.start()))
        model_file_controller = ModelFileController()
        tasks.append(asyncio.create_task(model_file_controller.start()))
        cluster_controller = ClusterController(self._config)
        tasks.append(asyncio.create_task(cluster_controller.start()))
        logger.debug("Controllers started.")
        return tasks

    def _start_leader_tasks(self):
        """Start tasks that run only on the leader.
        Note: If leadership is lost, the process exits directly (os._exit),
        so we don't need to track and cancel these tasks.
        """
        self._async_tasks.append(self._start_scheduler())
        self._async_tasks.extend(self._start_controllers())
        self._start_worker_instance_cleaner()
        self._start_worker_syncer(self._app)

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

    def _start_worker_syncer(self, app: FastAPI):
        worker_syncer = WorkerSyncer(
            lambda: getattr(app.state, "http_client", None),
            lambda: getattr(app.state, "http_client_no_proxy", None),
        )
        self._create_async_task(worker_syncer.start())

        logger.debug("Worker syncer started.")

    def _start_worker_instance_cleaner(self):
        worker_instance_cleaner = WorkerInstanceCleaner()
        self._create_async_task(worker_instance_cleaner.start())

        logger.debug("Worker instance cleaner started.")

    @property
    def all_processes(self):
        return self._before_healthy_sub_processes + self._after_healthy_sub_processes

    def _start_gpustack_operator_subscription(self):
        """
        启动集群订阅协调器 (gateway_client 里的OperatorSubscriptionReconciler)
        """
        if shutil.which("gpustack-operator") is None:
            return
        # 决定哪些集群需要订阅，何时订阅、更新或取消
        self._create_async_task(reconcile_gpustack_operator_subscription())

        logger.debug("GPUStack operator subscription started.")

    async def _monitor_sub_processes(self):
        while self.all_processes:
            for process in self.all_processes[:]:
                if not process.is_alive():
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"Sub process {process.name} died with exit code {process.exitcode}"
                        )
                    if process in self._before_healthy_sub_processes:
                        self._before_healthy_sub_processes.remove(process)
                    elif process in self._after_healthy_sub_processes:
                        self._after_healthy_sub_processes.remove(process)
            await asyncio.sleep(5)

    def _enqueue_operator_process(self):
        """
        准备 Operator 的启动命令、通信地址和清理动作，加入启动名单
        """
        if shutil.which("gpustack-operator") is None:
            logger.debug("gpustack-operator binary not found on PATH, skipping start.")
            return

        socket_dir = tempfile.mkdtemp(prefix="gso-", dir="/tmp" if os.name == "posix" else None)
        os.chmod(socket_dir, 0o700)

        socket_path = os.path.join(socket_dir, f"{secrets.token_hex(8)}.sock")
        # 调用 wg 的客户端
        gateway_client.set_gateway_unix_path(socket_path)

        args = [
            "gpustack-operator",
            "wg",
            "--worker-conn-gpustack-api-port",
            str(self._config.get_api_port()),
            "--bind-unix-path",
            socket_path,
        ]
        if self._config.debug:
            args.append("--v=4")

        def _cleanup_socket_dir():
            shutil.rmtree(socket_dir, ignore_errors=True)

        self._before_healthy_sub_processes.append(
            _ExternalSubprocess(
                args, name="gpustack-operator", on_exit=_cleanup_socket_dir
            )
        )

    def _start_sub_processes(self):
        """
        立即启动第一批，再安排后台任务等待 API 就绪、启动第二批
        """

        for process in self._before_healthy_sub_processes:
            process.start()

        async def wait_and_start_after_healthy():
            # Wait for the API to be healthy before starting the rest of the processes,
            # so that they can rely on the API being available when they start.
            api_url = f"http://127.0.0.1:{self._config.api_port}/healthz"
            async with aiohttp.ClientSession() as session:
                while True:
                    try:
                        await asyncio.sleep(2)
                        async with session.get(api_url) as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    except asyncio.CancelledError:
                        return

            # Start the rest of the processes after the API is ready.
            for process in self._after_healthy_sub_processes:
                process.start()

            await self._monitor_sub_processes()

        if len(self.all_processes) == 0:
            return

        self._create_async_task(wait_and_start_after_healthy())
