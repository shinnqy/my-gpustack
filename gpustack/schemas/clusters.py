import secrets
from urllib.parse import urlparse
from enum import Enum
from typing import Optional, Dict, List
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)
from sqlmodel import (
    Field,
    Column,
    SQLModel,
    Text,
    String,
)

from gpustack.schemas.config import (
    PredefinedConfigNoDefaults,
    SensitivePredefinedConfig,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    pydantic_column_type,
    PaginatedList,
    PublicFields,
)


class ClusterProvider(Enum):
    Docker = "Docker"
    Kubernetes = "Kubernetes"
    DigitalOcean = "DigitalOcean"
    Shuihua = "Shuihua"


class ClusterStateEnum(str, Enum):
    PENDING = 'pending'
    PROVISIONING = 'provisioning'
    PROVISIONED = 'provisioned'
    READY = 'ready'


class ClusterUpdate(SQLModel):
    name: str
    description: Optional[str] = None
    gateway_endpoint: Optional[str] = None
    server_url: Optional[str] = None
    # Per-cluster default container registry. Promoted out of worker_config
    # so it can be referenced/overridden independently of the worker process
    # config (image rendering, registration token, worker config response
    # all read this directly). Falls back to the server's
    # GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY when unset.
    system_default_container_registry: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    worker_config: Optional[PredefinedConfigNoDefaults] = Field(
        default=None,
        sa_column=Column(
            pydantic_column_type(
                PredefinedConfigNoDefaults,
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
            )
        ),
    )

    @field_validator("server_url")
    def validate_server_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) == 0:
            return None
        if v is not None:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid server_url format")
            return v.rstrip("/")
        return v


class ClusterCreateBase(ClusterUpdate):
    provider: ClusterProvider = Field(default=ClusterProvider.Docker)
    region: Optional[str] = None


class ClusterCreate(ClusterCreateBase):
    pass


class ClusterBase(ClusterCreateBase):
    state: ClusterStateEnum = ClusterStateEnum.PROVISIONING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    reported_gateway_endpoint: Optional[str] = None
    is_default: bool = Field(default=False)


class Cluster(ClusterBase, BaseModelMixin, table=True):
    __tablename__ = "clusters"
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_suffix: str = Field(nullable=False, default=secrets.token_hex(6))
    registration_token: Optional[str] = Field(
        nullable=True, default=secrets.token_hex(16)
    )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, Cluster):
            return self.id == other.id
        return False


class ClusterPublic(ClusterBase, PublicFields):
    workers: int = Field(default=0)
    ready_workers: int = Field(default=0)
    gpus: int = Field(default=0)
    models: int = Field(default=0)


ClustersPublic = PaginatedList[ClusterPublic]


class SensitiveRegistrationConfig(SensitivePredefinedConfig):
    model_config = ConfigDict(extra="ignore")
    token: str


class ClusterRegistrationTokenPublic(BaseModel):
    """
    The arguments of docker run command to register a worker.
    The env attribute is basically a dict of environment variables parsed from SensitiveRegistrationConfig.
    """

    token: str
    server_url: str
    image: str
    env: Dict[str, str]
    args: List[str]