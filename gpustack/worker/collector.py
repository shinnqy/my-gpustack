from gpustack.client.generated_clientset import ClientSet
import socket
import logging
from typing import Callable
from gpustack.config.config import Config
from gpustack.detectors.base import GPUDetectExepction
from gpustack.detectors.detector_factory import DetectorFactory
from gpustack.schemas.workers import (
    WorkerStatusPublic,
    WorkerStatus,
    SystemReserved,
)


logger = logging.getLogger(__name__)


class WorkerStatusCollector:
    _cfg: Config

    _worker_id_getter: Callable[[], int]

    _worker_ifname_getter: Callable[[], str]

    _worker_ip_getter: Callable[[], str]

    _worker_uuid_getter: Callable[[], str]

    def __init__(
        self,
        cfg: Config,
        worker_ip_getter: Callable[[], str],
        worker_ifname_getter: Callable[[], str],
        worker_id_getter: Callable[[], int],
        worker_uuid_getter: Callable[[], str],
    ):
        self._cfg = cfg
        self._worker_ip_getter = worker_ip_getter
        self._worker_ifname_getter = worker_ifname_getter
        self._worker_id_getter = worker_id_getter
        self._worker_uuid_getter = worker_uuid_getter
        # 真实版本先读配置里手动声明的设备/系统信息（cfg.get_gpu_devices() 等），
        # 有就用 Custom 探测器；都没有才用默认工厂
        self._detector_factory = DetectorFactory()

    def timed_collect(self, clientset: ClientSet = None, initial: bool = False):
        return self.collect(clientset=clientset, initial=initial)

    def collect(
        self, clientset: ClientSet = None, initial: bool = False
    ) -> WorkerStatusPublic:  # noqa: C901
        """Collect worker status information."""
        status = WorkerStatus.get_default_status()
        state_message = None
        try:
            system_info = self._detector_factory.detect_system_info()
            status = WorkerStatus.model_validate({**system_info.model_dump()})
        except Exception as e:
            logger.error(f"Failed to detect system info: {e}")

        if not initial:
            try:
                gpu_devices = self._detector_factory.detect_gpus()
                # status.gpu_devices = gpu_devices
                # FIXME: 第九章实验：只向 GPUStack 上报 0、1 两张卡；FIXME: 第十四章实验，每台机器只上报 1 张卡
                status.gpu_devices = [
                    gpu for gpu in gpu_devices
                    if gpu.index in {0}
                ]
            except GPUDetectExepction as e:
                state_message = str(e)
            except Exception as e:
                logger.error(f"Failed to detect GPU devices: {e}")
        self._inject_unified_memory(status)

        return WorkerStatusPublic(
            advertise_address=self._cfg.advertise_address or self._worker_ip_getter(),
            hostname=socket.gethostname(),
            ip=self._worker_ip_getter(),
            ifname=self._worker_ifname_getter(),
            port=self._cfg.worker_port,
            system_reserved=SystemReserved(**self._cfg.get_system_reserved()),
            state_message=state_message,
            status=status,
            worker_uuid=self._worker_uuid_getter(),
        )

    def _inject_unified_memory(self, status: WorkerStatus):
        is_unified_memory = False
        if status.gpu_devices is not None and len(status.gpu_devices) != 0:
            is_unified_memory = status.gpu_devices[0].memory.is_unified_memory

        if status.memory is not None:
            status.memory.is_unified_memory = is_unified_memory
