from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Dict, Optional
from pydantic import ConfigDict, BaseModel
from sqlmodel import (
    Field,
    SQLModel,
    JSON,
    Column,
    Text,
)
from sqlalchemy import String
from gpustack import envs
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from typing import List
from gpustack.utils.network import is_offline
from gpustack.schemas.config import (
    PredefinedConfigNoDefaults,
)
from sqlmodel import ForeignKey
from sqlmodel import Integer
from sqlmodel import Relationship
from .clusters import Cluster
from gpustack.schemas.config import ModelInstanceProxyModeEnum


class UtilizationInfo(BaseModel):
    total: int = Field(default=None)
    utilization_rate: Optional[float] = Field(default=None)  # rang from 0 to 100


class MemoryInfo(UtilizationInfo):
    is_unified_memory: bool = Field(default=False)
    used: Optional[int] = Field(default=None)


class CPUInfo(UtilizationInfo):
    pass


class GPUCoreInfo(UtilizationInfo):
    pass


class GPUNetworkInfo(BaseModel):
    status: str = Field(default="down")  # Network status (up/down)
    inet: str = Field(default="")  # IPv4 address
    netmask: str = Field(default="")  # Subnet mask
    mac: str = Field(default="")  # MAC address
    gateway: str = Field(default="")  # Default gateway
    iface: Optional[str] = Field(default=None)  # Network interface name
    mtu: Optional[int] = Field(default=None)  # Maximum Transmission Unit


class SwapInfo(UtilizationInfo):
    used: Optional[int] = Field(default=None)


class GPUDeviceInfo(BaseModel):
    vendor: Optional[str] = Field(default="")
    type: Optional[str] = Field(default="")
    index: Optional[int] = Field(default=None)
    device_index: Optional[int] = Field(default=0)
    device_chip_index: Optional[int] = Field(default=0)
    arch_family: Optional[str] = Field(default=None)
    name: str = Field(default="")
    uuid: Optional[str] = Field(default="")
    driver_version: Optional[str] = Field(default=None)
    runtime_version: Optional[str] = Field(default=None)
    compute_capability: Optional[str] = Field(default=None)


class GPUDeviceStatus(GPUDeviceInfo):
    core: Optional[GPUCoreInfo] = Field(sa_column=Column(JSON), default=None)
    memory: Optional[MemoryInfo] = Field(sa_column=Column(JSON), default=None)
    temperature: Optional[float] = Field(default=None)
    power: Optional[float] = Field(default=None)
    power_used: Optional[float] = Field(default=None)
    network: Optional[GPUNetworkInfo] = Field(sa_column=Column(JSON), default=None)


GPUDevicesStatus = List[GPUDeviceStatus]


class MountPoint(BaseModel):
    name: str = Field(default="")
    mount_point: str = Field(default="")
    mount_from: str = Field(default="")
    total: int = Field(default=None)  # in bytes
    used: Optional[int] = Field(default=None)
    free: Optional[int] = Field(default=None)
    available: Optional[int] = Field(default=None)


FileSystemInfo = List[MountPoint]


class OperatingSystemInfo(BaseModel):
    name: str = Field(default="")
    version: str = Field(default="")


class KernelInfo(BaseModel):
    name: str = Field(default="")
    release: str = Field(default="")
    version: str = Field(default="")
    architecture: str = Field(default="")


class UptimeInfo(BaseModel):
    uptime: float = Field(default=None)  # in seconds
    boot_time: str = Field(default="")


class SystemReserved(BaseModel):
    ram: Optional[int] = Field(default=None)
    vram: Optional[int] = Field(default=None)


class WorkerStateEnum(str, Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    UNREACHABLE = "unreachable"
    PENDING = "pending"
    PROVISIONING = "provisioning"
    INITIALIZING = "initializing"
    DELETING = "deleting"
    ERROR = "error"
    MAINTENANCE = "maintenance"

    @property
    def is_provisioning(self) -> bool:
        return self in [
            WorkerStateEnum.PENDING,
            WorkerStateEnum.PROVISIONING,
            WorkerStateEnum.INITIALIZING,
            WorkerStateEnum.DELETING,
            WorkerStateEnum.ERROR,
        ]


class SystemInfo(BaseModel):
    cpu: Optional[CPUInfo] = Field(sa_column=Column(JSON), default=None)
    memory: Optional[MemoryInfo] = Field(sa_column=Column(JSON), default=None)
    swap: Optional[SwapInfo] = Field(sa_column=Column(JSON), default=None)
    filesystem: Optional[FileSystemInfo] = Field(sa_column=Column(JSON), default=None)
    os: Optional[OperatingSystemInfo] = Field(sa_column=Column(JSON), default=None)
    kernel: Optional[KernelInfo] = Field(sa_column=Column(JSON), default=None)
    uptime: Optional[UptimeInfo] = Field(sa_column=Column(JSON), default=None)


class Maintenance(BaseModel):
    enabled: bool = False
    message: Optional[str] = None


class WorkerStatus(SystemInfo):
    gpu_devices: Optional[GPUDevicesStatus] = Field(
        sa_column=Column(JSON), default=None
    )

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def get_default_status(cls) -> 'WorkerStatus':
        return WorkerStatus(
            cpu=CPUInfo(total=0),
            memory=MemoryInfo(total=0, is_unified_memory=False),
            swap=SwapInfo(total=0),
            filesystem=[],
            os=OperatingSystemInfo(name="", version=""),
            kernel=KernelInfo(name="", release="", version="", architecture=""),
            uptime=UptimeInfo(uptime=0, boot_time=""),
            gpu_devices=[],
        )


class WorkerStatusStored(BaseModel):
    advertise_address: Optional[str] = None

    hostname: str

    ip: str

    ifname: str

    port: int

    metrics_port: Optional[int] = None

    system_reserved: Optional[SystemReserved] = Field(
        default=None, sa_column=Column(pydantic_column_type(SystemReserved))
    )

    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    status: Optional[WorkerStatus] = Field(
        sa_column=Column(pydantic_column_type(WorkerStatus))
    )

    worker_uuid: str = Field(sa_column=Column(String(255), nullable=False))

    proxy_mode: Optional[ModelInstanceProxyModeEnum] = Field(
        default=ModelInstanceProxyModeEnum.WORKER,
    )


class WorkerStatusPublic(WorkerStatusStored):
    pass


class WorkerUpdate(SQLModel):
    name: str = Field(index=True, unique=True)

    labels: Dict[str, str] = Field(sa_column=Column(JSON), default={})

    maintenance: Optional[Maintenance] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(Maintenance), default=None),
    )


class WorkerCreate(WorkerStatusStored, WorkerUpdate):
    cluster_id: Optional[int] = Field(
        sa_column=Column(Integer, ForeignKey("clusters.id"), nullable=False),
        default=None,
    )

    worker_version: Optional[str] = Field(
        default=None, sa_column=Column(String(100), nullable=True)
    )


class WorkerBase(WorkerCreate):
    state: WorkerStateEnum = WorkerStateEnum.NOT_READY

    heartbeat_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )

    unreachable: bool = False

    def compute_state(self):
        if self.maintenance and self.maintenance.enabled:
            self.state = WorkerStateEnum.MAINTENANCE
            self.state_message = self.maintenance.message
            return
        if self.state.is_provisioning:
            return
        if (
            self.state == WorkerStateEnum.NOT_READY
            and self.state_message is not None
            and not self.state_message.startswith("Heartbeat lost (")
        ):
            return
        is_not_ready_flag, last_heartbeat_str = is_offline(
            self.heartbeat_time,
            envs.WORKER_HEARTBEAT_GRACE_PERIOD,
            datetime.now(timezone.utc),
        )
        if is_not_ready_flag:
            reschedule_minutes = envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD / 60
            self.state = WorkerStateEnum.NOT_READY
            self.state_message = (
                f"Heartbeat lost (last heartbeat: {last_heartbeat_str}). "
                f"If the worker remains unresponsive for more than {reschedule_minutes:.1f} minutes, "
                "the instances on this worker will be rescheduled automatically. "
                "If this downtime is planned maintenance, please enable maintenance mode. "
                "Otherwise, please <a href='https://docs.gpustack.ai/latest/troubleshooting/#view-gpustack-logs'>check the worker logs</a>."
            )
            return
        if self.unreachable:
            address = self.advertise_address or self.ip
            healthz_url = f"http://{address}:{self.port}/healthz"
            msg = (
                "Server cannot access the "
                f"worker's health check endpoint at {healthz_url}. "
                "Please verify the port requirements in the "
                "<a href='https://docs.gpustack.ai/latest/installation/requirements/#port-requirements'>documentation</a>"
            )
            self.state = WorkerStateEnum.UNREACHABLE
            self.state_message = msg
            return
        self.state = WorkerStateEnum.READY
        self.state_message = None

    proxy_address: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )

    def get_proxy_address(self) -> Optional[str]:
        """
        Get the proxy address for the worker. If the worker has a proxy_address, return it.
        Otherwise, return None to indicate that no proxy should be used.
        """
        if self.proxy_mode != ModelInstanceProxyModeEnum.TUNNEL:
            return None
        return self.proxy_address


class Worker(WorkerBase, BaseModelMixin, table=True):
    __tablename__ = 'workers'

    id: Optional[int] = Field(default=None, primary_key=True)

    cluster: Cluster = Relationship(
        back_populates="cluster_workers", sa_relationship_kwargs={"lazy": "noload"}
    )

    token: Optional[str] = Field(default=None, nullable=True)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, Worker):
            return self.id == other.id
        return False


class WorkerPublic(
    WorkerBase,
):
    id: int
    created_at: datetime
    updated_at: datetime
    me: Optional[bool] = None  # Indicates if the worker is the current worker
    worker_uuid: Optional[str] = Field(default=None, exclude=True)


class WorkerRegistrationPublic(WorkerPublic):
    token: str
    worker_uuid: str
    worker_config: Optional["PredefinedConfigNoDefaults"] = None


WorkersPublic = PaginatedList[WorkerPublic]
