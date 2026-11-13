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
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.utils import network
from gpustack.utils.process import terminate_process_tree
from gpustack_runtime.deployer import delete_workload
import threading
from gpustack.worker.inference_backend_manager import InferenceBackendManager
from gpustack import envs
from gpustack_runtime.deployer import WorkloadStatusStateEnum
from gpustack_runtime.deployer import get_workload
from typing import List
import contextlib
from gpustack.schemas.models import CategoryEnum
from gpustack_runtime.deployer.__utils__ import compare_versions
import requests
from datetime import datetime
from datetime import timezone
from gpustack.utils.convert import safe_int


logger = logging.getLogger(__name__)


_INFERENCE_HEALTH_CHECK_FAILED_MESSAGE = "Inference health check failed."


_WORKLOAD_FAILED_MESSAGE = "Inference server exited or unhealthy."


_port_lock = threading.Lock()


"""后端映射"""


_SERVER_CLASS_MAPPING = {
    BackendEnum.VLLM: VLLMServer,
}


def _describe_workload_failure(workload) -> str:
    """
    Explain why a workload stopped serving, for the instance's state message.

    The workload's `state_message` is the most specific text the runtime has --
    a Pod's admission rejection, or an image-pull reason plus the registry error
    behind it (gpustack/gpustack#5869) -- but it never carries an exit code, and
    a container that merely crashed leaves it empty on Kubernetes. The
    per-container `exits` carry both, so take the reason from there when there
    is no message, and append the exit code either way
    (gpustack/gpustack#4217).

    Args:
        workload: The runtime WorkloadStatus, None if the workload is gone.

    Returns:
        The failure message to surface on the model instance.
    """
    if not workload:
        return _WORKLOAD_FAILED_MESSAGE

    message = getattr(workload, "state_message", "") or ""
    # A container blocked from starting reports no exit code, and its reason is
    # already what the state message is built from, so it adds nothing here.
    exits = [
        exit_
        for exit_ in (getattr(workload, "exits", None) or [])
        if exit_.exit_code is not None
    ]
    if not exits:
        return message or _WORKLOAD_FAILED_MESSAGE

    if not message:
        # Deduplicated but order-preserving: sidecars usually die of one cause.
        message = ", ".join(
            dict.fromkeys(exit_.reason for exit_ in exits if exit_.reason)
        )
    codes = ", ".join(
        (
            f"{exit_.name} exit code {exit_.exit_code}"
            if len(exits) > 1
            else f"exit code {exit_.exit_code}"
        )
        for exit_ in exits
    )
    return f"{message or _WORKLOAD_FAILED_MESSAGE} ({codes})"


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
        self._last_state_reconcile_time: float = time.time()
        self._inference_health_check_failures: Dict[int, int] = {}
        self._last_health_check_time: Dict[int, float] = {}
        self._last_successful_inference: Dict[int, float] = {}
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
            if mi.state in {ModelInstanceStateEnum.RUNNING, ModelInstanceStateEnum.UNREACHABLE}:
                logger.warning(
                    f"Model instance {mi.name} is {mi.state}; health reconciliation will adopt its workload."
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
        self._inference_health_check_failures.pop(mi.id, None)
        self._last_health_check_time.pop(mi.id, None)
        self._last_successful_inference.pop(mi.id, None)
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

    async def watch_model_instances(self):
        """
        Loop to post process model instances, for example, restarting error instances.

        """

        logger.debug("Watching model instances.")

        while True:
            try:
                for mi in list(self._error_model_instances.values()):
                    self._restart_error_model_instance(mi)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error restarting model instances: {e}")
                await asyncio.sleep(5)

    def sync_model_instances_state(self):  # noqa: C901
        """
        Synchronize model instances' state.

        - If the model instance is scheduled but not initialized, skip.
        - If the provision process is still alive, skip.
        - If the workload is still launching, skip.
        - If the workload is not existed, unhealthy, inactive or failed, update the model instance state to ERROR.
        - If everything is fine, update the model instance state to RUNNING.
        """

        # Snapshot local state BEFORE the list call. Reversing this order
        # races with CREATED events and reaps freshly-assigned instances.
        local_assigned_ids = {
            mid
        for mid, mi in list(self._model_instance_by_instance_id.items())
            if mi.get_deployment_metadata(self._worker_id) is not None
        }

        # Read from the watch-backed cache. It holds the full set, so the
        # common (nothing-to-reap) path stays O(1) with no server/DB round
        # trip and scales independently of worker count — a direct per-worker
        # poll here is one SELECT over model_instances per worker every few
        # seconds, which does not scale to hundreds of workers.
        response = self._clientset.model_instances.list()
        all_items = response.items or []
        reap_ids = local_assigned_ids - {
            mi.id
            for mi in all_items
            if mi.get_deployment_metadata(self._worker_id) is not None
        }

        # Optional periodic reconciliation against DB truth even when the cache
        # reports nothing reapable. Disabled by default; opt in via the
        # GPUSTACK_MODEL_INSTANCE_STATE_RECONCILE_INTERVAL env var.
        reconcile_interval = envs.MODEL_INSTANCE_STATE_RECONCILE_INTERVAL
        now = time.time()
        force_authoritative = (
            reconcile_interval > 0
            and now - self._last_state_reconcile_time >= reconcile_interval
        )

        # Reaping tears down live workloads, so it must act on an authoritative
        # snapshot. The cache is not one at every instant: awatch clears it and
        # flips _watch_started before the replay snapshot arrives, so a read
        # during a reconnect window can see an empty-but-"authoritative" cache
        # and surface healthy instances as stale. So confirm reap candidates
        # against a fresh, unpaginated, uncached fetch — this only touches the
        # DB when something looks reapable or on the periodic reconciliation.
        if reap_ids or force_authoritative:
            response = self._clientset.model_instances.list(
                params={"page": -1},
                use_cache=False,
            )
            if reconcile_interval > 0:
                self._last_state_reconcile_time = now
            all_items = response.items or []
            reap_ids = local_assigned_ids - {
                mi.id
                for mi in all_items
                if mi.get_deployment_metadata(self._worker_id) is not None
            }

        # An empty authoritative result is legitimate (user stopped every
        # model); list() raises on API failure rather than returning empty, so
        # local_assigned_ids - ∅ correctly reaps everything still tracked.
        for stale_id in reap_ids:
            stale = self._model_instance_by_instance_id.get(stale_id)
            if stale is None:
                continue
            logger.info(
                f"Reaping stale model instance {stale.name} (id={stale_id}); "
                f"server no longer reports it assigned to this worker."
            )
            try:
                # Equivalent to a dropped DELETED event, so purge logs too —
                # otherwise a reused id would inherit this instance's stale logs.
                self._stop_model_instance(stale, delete_logs=True)
            except Exception as e:
                logger.warning(f"Failed to reap stale model instance {stale.name}: {e}")

        if not all_items:
            # Nothing left to sync; reap pass above already handled stale
            # local state.
            return

        model_instances: List[ModelInstance] = []
        for model_instance in all_items:
            # if the model instance is assigned to this worker, it must be scheduled.
            # But we don't need to sync the scheduled model when it is not initialized yet.
            if (
                model_instance.worker_id == self._worker_id
                and model_instance.state != ModelInstanceStateEnum.SCHEDULED
            ):
                model_instances.append(model_instance)

        for model_instance in model_instances:
            # Skip if the provision process has not exited yet.
            if self._is_provisioning(model_instance):
                logger.trace(
                    f"Model instance {model_instance.name} is provisioning. Skipping sync."
                )
                continue

            is_main_worker = model_instance.worker_id == self._worker_id

            # Skip if the workload is still launching.
            # Use deployment metadata name for subordinate workers (e.g., "model-f0")
            # since their workload name differs from the model instance name.
            if is_main_worker:
                workload = get_workload(model_instance.name)

            if workload and workload.state in [
                WorkloadStatusStateEnum.PENDING,
                WorkloadStatusStateEnum.INITIALIZING,
            ]:
                logger.trace(
                    f"Model instance {model_instance.name} workload is still launching. Skipping sync."
                )
                continue

            # Update model instance state to ERROR if the workload is not existed, unhealthy, inactive or failed.
            if not workload or workload.state in [
                WorkloadStatusStateEnum.UNKNOWN,  # Rare, but possible, for example, leaving pause container.
                WorkloadStatusStateEnum.UNHEALTHY,
                WorkloadStatusStateEnum.INACTIVE,
                WorkloadStatusStateEnum.FAILED,
            ]:
                # Only if not in ERROR state yet.
                if model_instance.state != ModelInstanceStateEnum.ERROR:
                    # Surface the workload's own diagnosis (e.g. a device-plugin
                    # admission rejection, an image-pull failure, an exit code)
                    # when available.
                    failure_message = _describe_workload_failure(workload)
                    with contextlib.suppress(NotFoundException):
                        # Get patch dict for main worker.
                        if is_main_worker:
                            patch_dict = {
                                "state": ModelInstanceStateEnum.ERROR,
                                "state_message": failure_message,
                            }
                        # Update model instance.
                        self._update_model_instance(model_instance.id, **patch_dict)
                continue

            # Otherwise, update model instance state to RUNNING if everything is fine.
            model = self._get_model(model_instance)
            if not model.backend_version:
                # backend version may be empty on initialization.
                # try to refresh to get updated model info on syncs.
                model = self._refresh_model(model_instance)

            backend = get_backend(model)
            health_check_path = self._get_health_check_path(
                backend
            )
            if model.env and 'GPUSTACK_MODEL_HEALTH_CHECK_PATH' in model.env:
                # NOTE: There is no known use case for now. Keep this in case the built-in backends
                # introduce breaking changes and the default health check path no longer works.
                health_check_path = model.env['GPUSTACK_MODEL_HEALTH_CHECK_PATH']

            with contextlib.suppress(NotFoundException):
                # Get patch dict for main worker.
                if is_main_worker:
                    subordinate_state = self._get_main_worker_distributed_state(
                        model_instance
                    )
                    if subordinate_state is None:
                        if model_instance.state == ModelInstanceStateEnum.RUNNING:
                            self._restart_backoff_counts.pop(model_instance.id, None)
                            continue

                        if model_instance.state == ModelInstanceStateEnum.UNREACHABLE:
                            worker = self._clientset.workers.get(self._worker_id)
                            if worker.state not in {WorkerStateEnum.READY, WorkerStateEnum.MAINTENANCE}:
                                continue

                        if (
                            model_instance.state == ModelInstanceStateEnum.ERROR
                            or not is_ready(
                                backend, model_instance, health_check_path, model
                            )
                        ):
                            continue

                        self._restart_backoff_counts.pop(model_instance.id, None)
                        patch_dict = {
                            "state": ModelInstanceStateEnum.RUNNING,
                            "state_message": "",
                        }
                    elif subordinate_state["should_update"]:
                        patch_dict = {
                            "state": subordinate_state["state"],
                            "state_message": subordinate_state["state_message"],
                        }
                    else:
                        continue
                # Update model instance.
                self._update_model_instance(model_instance.id, **patch_dict)

    @staticmethod
    def _get_main_worker_distributed_state(
        model_instance: ModelInstance,
    ) -> Optional[dict]:
        subordinate_workers = (
            model_instance.distributed_servers.subordinate_workers
            if (
                model_instance.distributed_servers
                and model_instance.distributed_servers.subordinate_workers
            )
            else []
        )

        if not subordinate_workers:
            return None

        error_sw = None
        unreachable_sw = None
        all_running = True
        for sw in subordinate_workers:
            if sw.state == ModelInstanceStateEnum.ERROR:
                error_sw = sw
                break
            if (
                sw.state == ModelInstanceStateEnum.UNREACHABLE
                and unreachable_sw is None
            ):
                unreachable_sw = sw
            if sw.state != ModelInstanceStateEnum.RUNNING:
                all_running = False

        if error_sw:
            return {
                "should_update": model_instance.state != ModelInstanceStateEnum.ERROR,
                "state": ModelInstanceStateEnum.ERROR,
                "state_message": (
                    f"Distributed serving error in subordinate worker "
                    f"{error_sw.worker_ip}: {error_sw.state_message}."
                ),
            }

        if unreachable_sw:
            return {
                "should_update": model_instance.state
                != ModelInstanceStateEnum.UNREACHABLE,
                "state": ModelInstanceStateEnum.UNREACHABLE,
                "state_message": (
                    f"Distributed serving unreachable in subordinate worker "
                    f"{unreachable_sw.worker_ip}: {unreachable_sw.state_message}."
                ),
            }

        if not all_running:
            return {"should_update": False}

        return None

    def _restart_error_model_instance(self, mi: ModelInstance):
        """
        Restart error model instance with exponential backoff,
        maximum delay 5 minutes.

        Args:
            mi: The model instance to restart.
        """
        try:
            current = self._clientset.model_instances.get(mi.id)
        except NotFoundException:
            self._error_model_instances.pop(mi.id, None)
            return
        if current.state != ModelInstanceStateEnum.ERROR or current.worker_id != self._worker_id:
            self._error_model_instances.pop(mi.id, None)
            return
        mi = current
        if not self._refresh_model(mi).restart_on_error:
            self._error_model_instances.pop(mi.id, None)
            return

        if self._is_provisioning(mi):
            logger.debug(f"Model instance {mi.name} is provisioning. Skipping restart.")
            return

        restart_count = mi.restart_count or 0
        backoff_count = self._restart_backoff_counts.get(mi.id, 0)
        last_restart_time = mi.last_restart_time or mi.updated_at

        current_time = datetime.now(timezone.utc)
        delay = min(10 * (2 ** (backoff_count - 1)), 300) if backoff_count > 0 else 0
        if backoff_count > 0 and last_restart_time:
            elapsed_time = (current_time - last_restart_time).total_seconds()
            if elapsed_time < delay:
                logger.trace(
                    f"Delaying restart of {mi.name} for {delay - elapsed_time:.2f} seconds."
                )
                return

        logger.info(
            f"Restarting model instance {mi.name} "
            f"(attempt {backoff_count + 1}) after {delay} seconds delay."
        )

        with contextlib.suppress(NotFoundException):
            self._restart_backoff_counts[mi.id] = backoff_count + 1
            self._update_model_instance(
                mi.id,
                restart_count=restart_count + 1,
                last_restart_time=current_time,
                state=ModelInstanceStateEnum.SCHEDULED,
                state_message="",
            )

        # Pop from error model instances,
        # if failed to restart next time, it will be added again in watch_model_instance_events().
        self._error_model_instances.pop(mi.id, None)

    def _refresh_model(self, mi: ModelInstance) -> Model:
        """
        Refresh the model information from the server.

        Args:
            mi: The model instance whose model to refresh.

        Returns:
            The refreshed model.
        """
        logger.debug(f"Refreshing model {mi.model_name} information from server.")
        refreshed_model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = refreshed_model
        return refreshed_model

    def _get_health_check_path(
        self, backend: str, owner_principal_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Get health check path for the given backend.

        Args:
            backend: The backend name.
            owner_principal_id: Owner of the model being served, used to
                resolve an Org-scoped backend row over the Platform one.
        Returns:
            The health check path if exists, else None.
        """
        inference_backend = self._inference_backend_manager.get_backend_by_name(
            backend, owner_principal_id
        )

        return inference_backend.health_check_path if inference_backend else None

    def sync_model_instances_inference_health(self):
        """
        Synchronize model instances' inference health by sending actual inference requests.

        Per-model configuration is read from model.env:
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED: "true"/"false" (default: false)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL: seconds (default: global env)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT: seconds (default: 15)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD: count (default: global env)

        If the model has received successful inference traffic recently
        (within the configured interval), the active health check is skipped.
        """

        # Use the event-driven local cache instead of an API call.
        model_instances = [
            mi
            for mi in list(self._model_instance_by_instance_id.values())
            if mi.state == ModelInstanceStateEnum.RUNNING
        ]
        if not model_instances:
            return

        now = time.time()

        for model_instance in model_instances:
            model = self._get_model(model_instance)
            if not model:
                continue

            # Read per-model config from model.env.
            config = _get_inference_health_check_config(model)
            if not config["enabled"]:
                continue

            interval = config["interval"]
            timeout = config["timeout"]
            threshold = config["threshold"]

            # Skip if the model is still provisioning.
            if self._is_provisioning(model_instance):
                continue

            # Skip if not enough time has passed since last check.
            last_check = self._last_health_check_time.get(model_instance.id, 0)
            if now - last_check < interval:
                continue

            self._last_health_check_time[model_instance.id] = now

            # Skip if recent successful inference was observed for this instance.
            last_success = self._last_successful_inference.get(model_instance.id, 0)
            if last_success > now - interval:
                logger.debug(
                    f"Model instance {model_instance.name} had recent successful "
                    f"inference, skipping health check."
                )
                # Reset failure count since real traffic is succeeding.
                self._inference_health_check_failures.pop(model_instance.id, None)
                continue

            # Perform inference health check.
            if not is_inference_ready(model_instance, model, timeout=timeout):
                failure_count = self._inference_health_check_failures.get(
                    model_instance.id, 0
                )
                failure_count += 1
                self._inference_health_check_failures[model_instance.id] = failure_count

                if failure_count >= threshold:
                    logger.warning(
                        f"Model instance {model_instance.name} inference health check failed "
                        f"{failure_count} times, updating state to ERROR."
                    )
                    patch_dict = {
                        "state": ModelInstanceStateEnum.ERROR,
                        "state_message": _INFERENCE_HEALTH_CHECK_FAILED_MESSAGE,
                    }
                    self._update_model_instance(model_instance.id, **patch_dict)
                    # Reset failure count after marking as error.
                    del self._inference_health_check_failures[model_instance.id]
                else:
                    logger.debug(
                        f"Model instance {model_instance.name} inference health check failed "
                        f"{failure_count}/{threshold} times."
                    )
            else:
                # Reset failure count on success.
                self._inference_health_check_failures.pop(model_instance.id, None)

    def record_successful_inference(self, instance_id: int):
        """
        记录实例最近一次成功推理的时间
        """
        self._last_successful_inference[instance_id] = time.time()

    def get_instance_port_by_model_instance_id(
        self, model_instance_id: int
    ) -> Optional[int]:
        """
        根据实例 ID 查询本地缓存，只有实例处于 RUNNING 才返回端口；优先使用 ports[0]，否则使用 port。

        Worker 代理靠它找到实际推理引擎
        """
        instance = self._model_instance_by_instance_id.get(
            model_instance_id
        )  # Ensure the model instance is cached.
        if not instance or instance.state != ModelInstanceStateEnum.RUNNING:
            return None
        return instance.ports[0] if instance.ports else instance.port


def is_ready(
    backend: str,
    mi: ModelInstance,
    health_check_path: Optional[str] = None,
    model: Model = None,
) -> bool:
    """
    Access the health endpoint of the given model instance to check if it is servable.
    """
    is_built_in = is_built_in_backend(backend)
    if (not is_built_in or backend == BackendEnum.CUSTOM) and (not health_check_path):
        # If custom backend does not have health check path, consider it always ready.
        return True

    if backend == BackendEnum.ASCEND_MINDIE and not health_check_path:
        # Ref: https://www.hiascend.com/document/detail/zh/mindie/21RC2/mindieservice/servicedev/mindie_service0066.html
        # /info provides metadata information and requires more time to respond. Use it for health check.
        health_check_path = "/info"
    elif (
        backend == BackendEnum.SGLANG
        and model
        and CategoryEnum.IMAGE in model.categories
    ):
        if not model.backend_version:
            # version may be empty at initialization, consider it not ready.
            return False
        elif compare_versions(model.backend_version, "0.5.5.post3") >= 0:
            # SGLang Diffusion supported health check path at v0.5.5.post3
            health_check_path = "/health"
        else:
            # Older versions do not support health check, consider it always ready.
            return True
    elif is_built_in and backend != BackendEnum.CUSTOM and not health_check_path:
        # Built-in backends (vLLM, SGLang, vox-box) except (Custom, MindIE) use /v1/models as health check path.
        health_check_path = "/v1/models"

    try:
        # Use the worker IP instead of localhost for health check.
        # Reasons:
        # 1. Connectivity to the loopback address does not work with Ascend MindIE.
        # 2. More adaptable to container networks.
        health_check_url = f"http://{mi.worker_ip}:{mi.port}{health_check_path}"
        response = requests.get(health_check_url, timeout=1)
        if response.status_code == 200:
            return True
    except Exception as e:
        logger.debug(f"Error checking model instance {mi.name} health: {e}")
        pass
    return False


def _get_inference_endpoint_and_payload(model: Model) -> tuple[str, dict] | None:
    """
    Get inference endpoint and payload for the model.
    Returns None if the model type should skip health check.
    """
    skip_categories = {
        CategoryEnum.IMAGE,
        CategoryEnum.SPEECH_TO_TEXT,
        CategoryEnum.TEXT_TO_SPEECH,
        CategoryEnum.UNKNOWN,
    }
    if not skip_categories.isdisjoint(model.categories):
        return None

    # Return endpoint and payload based on model type (priority order)
    if CategoryEnum.EMBEDDING in model.categories:
        return "/v1/embeddings", {"model": model.name, "input": "test"}

    if CategoryEnum.RERANKER in model.categories:
        return "/v1/rerank", {
            "model": model.name,
            "query": "test",
            "documents": ["test"],
        }

    return "/v1/chat/completions", {
        "model": model.name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "max_completion_tokens": 1,
    }


def _get_inference_health_check_config(model: Model) -> dict:
    """Read per-model inference health check config from model.env."""
    env = model.env or {}
    enabled = env.get(
        "GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED", "false"
    ).lower() in (
        "true",
        "1",
    )
    interval = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL"),
        300,
    )
    timeout = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT"),
        15,
    )
    threshold = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD"),
        3,
    )
    return {
        "enabled": enabled,
        "interval": interval,
        "timeout": timeout,
        "threshold": threshold,
    }


def is_inference_ready(mi: ModelInstance, model: Model, timeout: int = 15) -> bool:
    """
    Send a minimal inference request to verify the inference capability is working.
    """
    # Check Custom backend (no standard inference API)
    if (get_backend(model) == BackendEnum.CUSTOM or not is_built_in_backend(get_backend(model))):
        return True

    # Check port assignment
    if not mi.port:
        logger.debug(f"Model instance {mi.name} does not have port assigned yet.")
        return False

    # Get endpoint and payload, None means skip health check
    result = _get_inference_endpoint_and_payload(model)
    if not result:
        logger.debug(f"Skipping inference check for {mi.name}")
        return True

    endpoint_path, payload = result
    inference_url = f"http://{mi.worker_ip}:{mi.port}{endpoint_path}"

    try:
        response = requests.post(inference_url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return True
        else:
            logger.warning(
                f"Model instance {mi.name} inference health check failed "
                f"with status {response.status_code} for endpoint {endpoint_path}"
            )
    except Exception as e:
        logger.debug(
            f"Error checking model instance {mi.name} inference at {endpoint_path}: {e}"
        )

    return False
