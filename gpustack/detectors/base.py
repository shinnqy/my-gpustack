from abc import ABC, abstractmethod
from gpustack.schemas.workers import GPUDevicesStatus, SystemInfo


class GPUDetector(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def gather_gpu_info(self) -> GPUDevicesStatus:
        pass


class SystemInfoDetector(ABC):
    @abstractmethod
    def gather_system_info(self) -> SystemInfo:
        pass


class GPUDetectExepction(Exception):
    pass
