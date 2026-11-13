from gpustack.client import ClientSet
from gpustack.config.config import Config
from gpustack.server.bus import Event
from typing import Callable
import asyncio
import os
from gpustack.schemas.models import ModelInstance
import logging
from typing import Dict
from typing import Set
import time
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.api.exceptions import NotFoundException
from gpustack.schemas.models import ModelInstanceUpdate
from gpustack.utils.attrs import set_attr
from gpustack.server.bus import EventType
from gpustack.config import registration
from gpustack.schemas.inference_backend import is_built_in_backend
from gpustack.schemas.models import get_backend
import multiprocessing
from gpustack.logging import RedirectStdoutStderr
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.models import BackendEnum
from gpustack.utils.process import add_signal_handlers
from gpustack.worker.backends.custom import CustomServer
from typing import Optional
import setproctitle
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.schemas.models import Model
from gpustack.utils import network
from gpustack.utils.process import terminate_process_tree
from gpustack_runtime.deployer import delete_workload
import threading
from gpustack.worker.inference_backend_manager import InferenceBackendManager


logger = logging.getLogger(__name__)


_port_lock = threading.Lock()


"""后端映射"""
_SERVER_CLASS_MAPPING = {
    BackendEnum.VLLM: VLLMServer,
}


class ServeManager:
    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    _inference_backend_manager: InferenceBackendManager

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._clientset_getter = clientset_getter
        self._provisioning_processes = {}
        self._error_model_instances = {}
        self._model_cache_by_instance = {}
        self._model_instance_by_instance_id = {}
        self._assigned_ports: Dict[int, Set[int]] = {}
        self._restart_backoff_counts: Dict[int, int] = {}
        os.makedirs(self._serve_log_dir, exist_ok=True)

    async def watch_models(self):
        """
        Loop to watch models to keep the cache updated.

        """

        logger.debug("Watching models.")

        while True:
            try:
                # Watch models without callback to keep the cache updated.
                await self._clientset.models.awatch(callback=None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching models: {e}")
                await asyncio.sleep(5)

    async def watch_model_instances_event(self):
        """
        Loop to watch model instances' event and handle.

        """

        logger.debug("Watching model instances event.")

        while True:
            try:
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching model instances: {e}")
                await asyncio.sleep(5)

    def _handle_model_instance_event(self, event: Event):
        """Handle a model instance event without ever crashing the watch stream.

        The awatch callback runs inline in the watch loop, so any exception
        that escapes here tears the stream down and forces a full reconnect
        plus cache reload. Swallow and log (with traceback) instead; the next
        event or the periodic state sync recovers the instance.
        """
        try:
            self._dispatch_model_instance_event(event)
        except Exception:
            logger.exception(
                f"Failed to handle {event.type} event for model instance "
                f"{getattr(event, 'id', None)}"
            )

    def _dispatch_model_instance_event(self, event: Event):  # noqa: C901
        """
        根据事件和实例状态，决定本机该做什么
        """
        mi = ModelInstance.model_validate(event.data)

        logger.trace(
            f"Received event: {str(event.type)}, id: {mi.id}, name: {mi.name}, state: {str(mi.state)}"
        )

        is_main_worker = mi.worker_id == self._worker_id

        if is_main_worker:
            self._model_instance_by_instance_id[mi.id] = mi
        else:
            # Return if it isn't a distribution serving.
            if not mi.distributed_servers:
                return

        if event.type == EventType.DELETED:
            # Teardown is left to the periodic reap in sync_model_instances_state,
            # which is the authoritative reconciler and must run anyway to catch
            # DELETEDs missed during a watch disconnect. Tearing down here too
            # would give a second concurrent caller racing the reap on
            # delete_workload, so just let the reap reap it (within one tick).
            logger.trace(
                f"DELETED event for model instance {mi.name}; "
                "teardown deferred to reap."
            )
            return

        if event.type == EventType.UPDATED:
            if mi.state == ModelInstanceStateEnum.ERROR:
                model = self._get_model(mi)
                if model.restart_on_error:
                    """如果配置了 restart_on_error ，先记录到错误实例缓存，等待后续重启处理。"""
                    self._error_model_instances[mi.id] = mi
                    logger.trace(
                        f"UPDATED event: cached error model instance {mi.name} for restart."
                    )
                return

            # Restart if scheduled and this is the assigned worker.
            if is_main_worker and mi.state == ModelInstanceStateEnum.SCHEDULED:
                """
                事件是 UPDATED，属于本机且为 SCHEDULED
                """
                self._restart_model_instance(mi)
                logger.trace(
                    f"UPDATED event: restarted scheduled model instance {mi.name}."
                )

            return

        if event.type == EventType.CREATED:
            if not is_main_worker:
                return
            if mi.state == ModelInstanceStateEnum.RUNNING:
                logger.warning(
                    f"Model instance {mi.name} is already running. Skipping start."
                )
                return
            """
            事件是 CREATED，属于本机主 Worker，且不是 RUNNING
            """
            self._start_model_instance(mi)
            logger.trace(f"CREATED event: started created model instance {mi.name}.")

    def _start_model_instance(self, mi: ModelInstance):  # noqa: C901
        """
        开始真正启动部署子进程


        检查是否已有部署子进程
        → 读取 Model 配置、确定后端
        → 分配端口
        → 准备镜像仓库配置和日志路径
        → 启动部署子进程
        → 回报 INITIALIZING、端口和 PID
        """

        # 先防止重复创建部署子进程
        if self._is_provisioning(mi):
            logger.warning(f"Model instance {mi.name} is provisioning. Skipping start.")
            return

        is_main_worker = mi.worker_id == self._worker_id

        log_file_path = self._get_numbered_log_path(mi)

        try:
            """准备子进程需要的信息"""
            model = self._get_model(mi)
            backend = get_backend(model)

            # 分配服务端口
            self._assign_ports(mi, model, backend)

            logger.debug(
                f"Starting model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

            fallback_registry = (
                registration.determine_default_registry(
                    self._config.system_default_container_registry
                )
                if is_built_in_backend(backend)
                else None
            )

            """创建并启动部署子进程"""
            process = multiprocessing.Process(
                target=ServeManager._serve_model_instance,
                args=(
                    mi,
                    backend,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                    self._worker_id,
                    self._inference_backend_manager.get_backend_by_name(
                        backend
                    ),
                    fallback_registry,
                ),
            )
            process.daemon = False
            process.start()
            self._provisioning_processes[mi.id] = process

            """子进程创建成功后，回报初始化状态"""
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.INITIALIZING,
                    "port": mi.port,
                    "ports": mi.ports,
                    "pid": process.pid,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.info(
                f"Started model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

        except Exception as e:
            # Clean up provisioning process if started.
            if mi.id in self._provisioning_processes:
                try:
                    self._stop_model_instance(mi)
                except Exception:
                    logger.exception(f"Failed to clean up provisioning instance {mi.name}")
            self._assigned_ports.pop(mi.id, None)

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.ERROR,
                    "state_message": f"Failed to start model instance: {e}",
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to start model instance {mi.name}: {e}")

    def _update_model_instance(self, id: int, **kwargs):
        """
        封装通过 API 更新实例的过程

        Args:
            id: The ID of the model instance to update.
            **kwargs: The fields to update, group by field name and value.
        """

        try:
            # 获取 Server 上的实例数据
            mi_public = self._clientset.model_instances.get(id=id)

            # 转成更新请求对象
            mi = ModelInstanceUpdate(**mi_public.model_dump())
            # 应用本次修改，例如 state=INITIALIZING
            for key, value in kwargs.items():
                set_attr(mi, key, value)

            # 发给 Server 保存
            self._clientset.model_instances.update(id=id, model_update=mi)
        except NotFoundException:
            logger.warning(
                f"Model instance with ID {id} not found when trying to update."
            )

    @staticmethod
    def _serve_model_instance(
        mi: ModelInstance,
        backend: BackendEnum,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        worker_id: int,
        inference_backend: InferenceBackend,
        fallback_registry: Optional[str] = None,
    ):
        """
        部署子进程的入口，负责：
            设置进程名称和信号处理。
            创建自己的 ClientSet，访问 Server。
            将标准输出和错误重定向到部署日志文件。
            根据后端选择 VLLMServer 等实现。
            创建后端对象并调用 start()。
        """

        setproctitle.setproctitle(f"gpustack_model_instance_{mi.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.get_server_url(),
            headers=client_headers,
        )

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                try:
                    server_cls = _SERVER_CLASS_MAPPING.get(backend, CustomServer)
                    server_ins = server_cls(
                        clientset,
                        mi,
                        cfg,
                        worker_id,
                        inference_backend,
                        fallback_registry,
                    )
                    logger.info(f"Provisioning model instance {mi.name}")
                    server_ins.start()
                    logger.info(f"Finished provisioning model instance {mi.name}")
                except Exception as e:
                    logger.exception(
                        f"Error provisioning model instance {mi.name}: {e}"
                    )
                    raise e

    def _get_numbered_log_path(self, mi: ModelInstance) -> str:
        """
        按实例 ID 和 restart_count 生成日志路径，例如 log/serve/1.0.log
        """
        restart_count = mi.restart_count or 0
        return f"{self._serve_log_dir}/{mi.id}.{restart_count}.log"

    def _assign_ports(
        self,
        mi: ModelInstance,
        model: Model,
        backend: BackendEnum,
    ) -> None:
        """
        复用已有端口或选择空闲的主服务端口，用锁和内部预留集合避免本管理器重复分配。
        """
        with _port_lock:
            if mi.port:
                mi.ports = mi.ports or [mi.port]
                used = set().union(*(ports for id, ports in self._assigned_ports.items() if id != mi.id))
                if used.intersection(mi.ports):
                    raise RuntimeError(f"Ports already reserved by another model instance: {mi.ports}")
                self._assigned_ports[mi.id] = set(mi.ports)
                return

            if self._assigned_ports:
                unavailable_ports = set.union(*self._assigned_ports.values())
            else:
                unavailable_ports = set()

            # Main serving port
            mi.port = network.get_free_port(
                port_range=self._config.service_port_range,
                unavailable_ports=unavailable_ports,
                host=mi.worker_ip,
            )
            mi.ports = [mi.port]
            unavailable_ports.add(mi.port)

            self._assigned_ports[mi.id] = set(mi.ports)

    def _restart_model_instance(self, mi: ModelInstance):
        """
        先停止旧部署，再重新启动，保留重启退避记录
        """

        self._stop_model_instance(mi, clear_restart_backoff=False)
        self._start_model_instance(mi)

    def _stop_model_instance(
        self,
        mi: ModelInstance,
        clear_restart_backoff: bool = True,
        delete_logs: bool = False,
    ):
        """
        终止仍存活的部署子进程、调用 delete_workload() 删除容器工作负载，再清理内部记录
        """

        logger.debug(f"Stopping model instance {mi.name or mi.id}")

        # Teardown provisioning process if still alive.
        if self._is_provisioning(mi):
            terminate_process_tree(self._provisioning_processes[mi.id].pid)

        # Delete workload.
        deployment_metadata = mi.get_deployment_metadata(self._worker_id)
        if deployment_metadata:
            delete_workload(deployment_metadata.name)

        # Cleanup internal states.
        self._provisioning_processes.pop(mi.id, None)
        self._assigned_ports.pop(mi.id, None)
        self._error_model_instances.pop(mi.id, None)
        self._model_cache_by_instance.pop(mi.id, None)
        self._model_instance_by_instance_id.pop(mi.id, None)
        if clear_restart_backoff:
            self._restart_backoff_counts.pop(mi.id, None)

        logger.info(f"Stopped model instance {mi.name or mi.id}")

    def _get_model(self, mi: ModelInstance) -> Model:
        """
        通过 API 获取实例对应的 Model，并按实例 ID 缓存
        """
        if model := self._model_cache_by_instance.get(mi.id):
            return model

        model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = model
        return model

    def _is_provisioning(self, mi: ModelInstance) -> bool:
        """
        检查该实例的部署子进程是否还存活，用来避免重复部署
        """
        if process := self._provisioning_processes.get(mi.id):
            if process.is_alive():
                process.join(timeout=0)
                return process.is_alive()
        return False
