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
from typing import List
from pydantic import (
    PrivateAttr,
    computed_field,
)
from sqlmodel import (
    Relationship,
)
from typing import TYPE_CHECKING
from pydantic import Field as PydanticField
import re
from typing import Any
from sqlmodel import ForeignKey
from sqlmodel import Integer
from sqlmodel import JSON
import sqlalchemy as sa


class WorkerPoolUpdate(SQLModel):
    name: str
    batch_size: Optional[int] = Field(default=None, ge=1)
    replicas: int = Field(default=1, ge=0)
    labels: Optional[Dict[str, str]] = Field(sa_column=Column(JSON), default={})


class Volume(BaseModel):
    format: Optional[str] = None
    size_gb: Optional[int] = None
    name: Optional[str] = None


class ImageCredential(BaseModel):
    """
    A docker registry credential. At manifest render time each entry is
    materialized into a ``kubernetes.io/dockerconfigjson`` Secret named
    ``gpustack-image-pull-secret-<index>`` in the worker namespace, which
    every worker DaemonSet then references via ``imagePullSecrets``.

    ``username`` / ``password`` are optional so a credential entry can act
    as a placeholder Secret for a public or pre-configured registry — the
    rendered ``.dockerconfigjson`` falls back to an empty ``{"auths":{}}``
    payload when either is missing, matching the gpustack Helm chart
    convention.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    registry: str = PydanticField(
        ...,
        description="Docker registry host (e.g. docker.io, registry.example.com).",
    )
    username: Optional[str] = None
    password: Optional[str] = PydanticField(
        default=None,
        description="Registry password / access token. Stored as-is — handle as a secret.",
    )


class HostPathVolumeSource(BaseModel):
    """
    卷挂载部分，把 Kubernetes 的配置拆成了可校验的对象

    HostPathVolumeSource: 挂载宿主机目录
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    path: str = PydanticField(
        ...,
        description="Path of the directory on the host. If the path is a symlink, it will follow the link to the real path.",
    )
    type: Optional[str] = PydanticField(None, description="Type for HostPath Volume.")


class PersistentVolumeClaimVolumeSource(BaseModel):
    """
    卷挂载部分，把 Kubernetes 的配置拆成了可校验的对象

    PersistentVolumeClaimVolumeSource：挂载已有 PVC
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    claim_name: str = PydanticField(
        ...,
        alias="claimName",
        description="ClaimName is the name of a PersistentVolumeClaim in the same namespace as the pod using this volume.",
    )
    read_only: bool = PydanticField(
        False,
        alias="readOnly",
        description="Will force the ReadOnly setting in VolumeMounts.",
    )


class ConfigMapVolumeSource(BaseModel):
    """
    卷挂载部分，把 Kubernetes 的配置拆成了可校验的对象

    ConfigMapVolumeSource：挂载 ConfigMap，注释说明目前 UI 不开放
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    name: str = PydanticField(..., description="Name of the referent.")
    optional: Optional[bool] = PydanticField(
        None, description="Specify whether the ConfigMap or its keys must be defined."
    )


class VolumeSource(BaseModel):
    """
    VolumeSource 包装卷来源
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    host_path: Optional[HostPathVolumeSource] = PydanticField(None, alias="hostPath")
    persistent_volume_claim: Optional[PersistentVolumeClaimVolumeSource] = (
        PydanticField(None, alias="persistentVolumeClaim")
    )
    config_map: Optional[ConfigMapVolumeSource] = PydanticField(None, alias="configMap")


class K8sVolumeMount(BaseModel):
    """
    K8sVolumeMount 再把 VolumeSource 与卷名、容器内路径、只读标记组合起来
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    name: str
    mount_path: str = PydanticField(..., alias="mountPath")
    read_only: bool = PydanticField(False, alias="readOnly")
    volume_source: Optional[VolumeSource] = PydanticField(
        default=None,
        alias="volumeSource",
        description=(
            "Kubernetes VolumeSource definition. Examples:\n"
            '- hostPath: `{"hostPath": {"path": "/data", "type": "Directory"}}`\n'
            '- persistentVolumeClaim: `{"persistentVolumeClaim": {"claimName": "my-pvc"}}`\n'
            '- configMap: `{"configMap": {"name": "my-configmap"}}`'
        ),
    )

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            return v
        if len(v) > 63:
            raise ValueError("Volume name must be less than 64 characters")
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", v):
            raise ValueError(
                "Volume name must be a valid DNS-1123 label (e.g. 'my-name', or '123-abc'); "
                "it must consist of lower case alphanumeric characters or '-', "
                "and must start and end with an alphanumeric character."
            )
        return v


class GpuInstanceOptions(BaseModel):
    """
    GPU-instance support knobs for the operator. Its mere presence on
    ``k8s_options.gpu_instance_options`` signals "GPU instances enabled"
    for this cluster — leaving the field unset opts the cluster out, so
    no separate boolean flag is needed.

    Each knob mirrors a gpustack-operator ``Setting`` of the same name and is
    **tri-state**: ``None`` means GPUStack does not manage that setting and the
    cluster's own value is left alone, which is a different instruction from an
    explicit ``True`` or ``False``. The operator catalog is also administered by
    ``kubectl``, so an unmanaged knob must never be asserted.

    Every default stays ``None``. The column persists with ``exclude_none`` /
    ``exclude_unset`` / ``exclude_defaults`` (see ``ClusterUpdate.k8s_options``),
    so a non-``None`` default would be stripped back out of the JSON and read
    back as unmanaged. That same serialization is why an all-unset
    ``GpuInstanceOptions()`` still lands as a *present* ``{}`` rather than
    ``null`` — and its presence is the cluster-purpose signal
    (:func:`is_gpu_service_k8s_options`), so a knob-less GPU Service cluster
    must never serialize itself into a Model Service one.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    gpu_instances_access_static_address: Optional[str] = PydanticField(
        default=None,
        alias="gpuInstancesAccessStaticAddress",
        description=(
            "Static address surfaced to the operator for accessing GPU "
            "instances in this cluster (e.g. LoadBalancer VIP). Mirrors the "
            "operator's ``instance-access-static-address`` setting (operator "
            "default: blank, i.e. the address is generated from host IPs). "
            "Keeps its legacy name: renaming it to match the operator would "
            "break the payload for every existing client."
        ),
    )
    gpu_instance_type_derived_from_node: Optional[bool] = PydanticField(
        default=None,
        alias="gpuInstanceTypeDerivedFromNode",
        description=(
            "Whether the operator auto-derives InstanceTypes (and their backing "
            "ClusterQueues) from node hardware. Mirrors the operator's "
            "``instance-type-derived-from-node`` setting (operator default: "
            "true). Unset means GPUStack does not manage it and the cluster's "
            "own value stands."
        ),
    )
    gpu_instance_type_mixed_on_node: Optional[bool] = PydanticField(
        default=None,
        alias="gpuInstanceTypeMixedOnNode",
        description=(
            "Whether one node may surface both an accelerated and a CPU-only "
            "InstanceType. Mirrors the operator's "
            "``instance-type-mixed-on-node`` setting (operator default: true). "
            "Unset means GPUStack does not manage it and the cluster's own "
            "value stands."
        ),
    )


class OperatorOptions(BaseModel):
    """
    Operator-specific deployment options. Nested under ``k8s_options.operator``
    so operator concerns are isolated from worker process configuration.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    env: Optional[Dict[str, str]] = PydanticField(
        default=None,
        alias="env",
        description=(
            "Additional environment variables injected into the operator "
            "container. Keys are env var names, values are literal strings."
        ),
    )


class K8sOptions(BaseModel):
    """
    All Kubernetes-side deployment knobs for a cluster's worker DaemonSets:
    pod spec primitives (imageCredentials, nodeSelector, volumeMounts). The
    base ``nodeSelector`` applies to every rendered worker DaemonSet; each
    per-runtime DaemonSet additionally gets a vendor PCI-presence label merged
    on top at render time (see ``manifest_template``). Top-level on the cluster
    (parallel to ``worker_config``) so K8s deployment concerns are isolated
    from worker process behaviour; structure mirrors how Helm chart values are
    usually organised under ``worker.*`` for future chart migration.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    image_credentials: Optional[List[ImageCredential]] = PydanticField(
        default=None,
        alias="imageCredentials",
        description=(
            "Docker registry credentials. Each entry becomes a "
            "kubernetes.io/dockerconfigjson Secret named "
            "``gpustack-image-pull-secret-<index>`` in the worker namespace "
            "and is referenced from every worker DaemonSet's imagePullSecrets."
        ),
    )
    node_selector: Optional[Dict[str, str]] = PydanticField(
        default=None,
        alias="nodeSelector",
        description="Pod spec nodeSelector labels applied to manifests rendered for the cluster.",
    )
    volume_mounts: Optional[List[K8sVolumeMount]] = PydanticField(
        default=None,
        alias="volumeMounts",
        description=(
            "Pod spec volumes and volumeMounts applied to every worker "
            "DaemonSet. The first entry is reserved for the gpustack data "
            "dir; the route layer enforces that invariant."
        ),
    )
    operator_image: Optional[str] = PydanticField(
        default=None,
        alias="operatorImage",
        description=(
            "Override for the gpustack-operator container image. Falls back "
            "to the server's GPUSTACK_OPERATOR_IMAGE / built-in default when unset."
        ),
    )
    # IMPORTANT: 没有这个值的时候，用途是 Model Service，有这个值的时候，用途是 GPU Service
    gpu_instance_options: Optional[GpuInstanceOptions] = PydanticField(
        default=None,
        alias="gpuInstanceOptions",
        description=(
            "GPU-instance support knobs. Presence of this field enables GPU "
            "instance handling for the cluster; absence opts the cluster out."
        ),
    )
    namespace: Optional[str] = PydanticField(
        default=None,
        description=(
            "Kubernetes namespace this cluster's manifests render into. "
            "Falls back to ``gpustack-system`` at render time when unset."
        ),
    )
    operator: Optional[OperatorOptions] = PydanticField(
        default=None,
        alias="operator",
        description="Operator-specific deployment options for the cluster.",
    )


def is_gpu_service_k8s_options(k8s_options: Any) -> bool:
    """Whether a ``k8s_options`` value opts the cluster in to GPU Service.

    A cluster carries no purpose column: ``k8s_options.gpu_instance_options``
    being set *is* the signal — present means GPU Service, absent means Model
    Service. This is the single home for that test; the route layer
    (``routes/clusters.py``) and the gateway subscription
    (``gpu_instances/gateway.py``) both delegate here, so the two halves of the
    product can never disagree about what a cluster is for.

    Runs once per cluster on every list/watch tick, so we look at the raw shape
    instead of re-running ``K8sOptions.model_validate`` — a full nested parse on
    the hot path would also propagate any future schema drift as a
    request-level ``ValidationError``. The dict branch tolerates both serialized
    key forms (snake from ``model_dump``, camel from API/UI submissions).

    Presence, not validity, is what is read: a dict carrying an unparseable
    ``gpu_instance_options`` value still counts as GPU Service. Nothing writes
    such a row — the column is only ever written from a validated model — and
    answering by presence keeps a malformed row from raising on a watch tick.
    Any other shape reads as Model Service, so a cluster whose purpose cannot be
    established is excluded from GPU Service rather than included.
    """
    if isinstance(k8s_options, K8sOptions):
        return k8s_options.gpu_instance_options is not None
    if isinstance(k8s_options, dict):
        return (
            k8s_options.get("gpu_instance_options") is not None
            or k8s_options.get("gpuInstanceOptions") is not None
        )
    return False


def is_gpu_service_cluster(cluster: "Cluster") -> bool:
    """Whether the cluster is registered for GPU Service rather than Model Service.

    The cluster-shaped entry point onto :func:`is_gpu_service_k8s_options`,
    which carries the invariant and the reason for the shape.
    """
    return is_gpu_service_k8s_options(cluster.k8s_options)


class CloudOptions(BaseModel):
    volumes: Optional[List[Volume]] = None


class WorkerPoolCreate(WorkerPoolUpdate):
    instance_type: str
    os_image: str
    image_name: str
    # ``default_factory``, not ``default={}``: a default is not validated, so
    # an omitted ``cloud_options`` stayed a raw dict on the model and every
    # ``model_dump`` of it -- the create routes dump the input to build the row
    # -- warned ``PydanticSerializationUnexpectedValue``.
    cloud_options: Optional[CloudOptions] = Field(
        default_factory=CloudOptions,
        sa_column=Column(pydantic_column_type(CloudOptions)),
    )
    zone: Optional[str] = None
    # instance_spec is for UI to store the instance_type's extended specifications for display.
    instance_spec: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )


class WorkerPoolBase(WorkerPoolCreate):
    cluster_id: int = Field(
        sa_column=Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE"))
    )


class WorkerPool(WorkerPoolBase, BaseModelMixin, table=True):
    __tablename__ = "worker_pools"
    id: Optional[int] = Field(default=None, primary_key=True)
    cluster: Optional["Cluster"] = Relationship(
        back_populates="cluster_worker_pools",
        sa_relationship_kwargs={"lazy": "noload"},
    )
    pool_workers: list["Worker"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"},
        back_populates="worker_pool",
    )

    @computed_field
    @property
    def workers(self) -> int:
        return len([worker for worker in self.pool_workers or [] if worker.deleted_at is None and worker.state.value != "deleting"])

    @computed_field
    @property
    def ready_workers(self) -> int:
        return len([worker for worker in self.pool_workers or [] if worker.deleted_at is None and worker.state.value == "ready"])


class WorkerPoolPublic(WorkerPoolBase, PublicFields):
    workers: int = Field(default=0)
    ready_workers: int = Field(default=0)


WorkerPoolsPublic = PaginatedList[WorkerPoolPublic]


class ClusterProvider(Enum):
    Docker = "Docker"
    Kubernetes = "Kubernetes"
    DigitalOcean = "DigitalOcean"
    Shuihua = "Shuihua"


class CloudCredentialBase(SQLModel):
    """
    Supports providers other than Kubernetes and Docker.
    """

    name: str
    description: Optional[str] = None
    provider: ClusterProvider = Field(default=ClusterProvider.DigitalOcean)
    key: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class CloudCredentialUpdate(CloudCredentialBase):
    secret: Optional[str] = None


class CloudCredentialCreate(CloudCredentialUpdate):
    pass


class CloudCredential(CloudCredentialCreate, BaseModelMixin, table=True):
    __tablename__ = "cloud_credentials"
    __table_args__ = (
        sa.Index(
            "idx_cloud_credentials_deleted_at_created_at", "deleted_at", "created_at"
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, CloudCredential):
            return self.id == other.id
        return False


class CloudCredentialPublic(CloudCredentialBase, PublicFields):
    pass


CloudCredentialsPublic = PaginatedList[CloudCredentialPublic]


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
    """管 Worker 进程参数"""
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

    """
    用来存 Kubernetes 部署配置，并通过其中的 GPU 实例配置区分集群用途

    这个参数管命名空间、镜像凭据、挂载、Operator 等部署配置
    """
    k8s_options: Optional[K8sOptions] = Field(
        default=None,
        sa_column=Column(
            pydantic_column_type(
                K8sOptions,
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
            )
        ),
    )


class ClusterCreateBase(ClusterUpdate):
    provider: ClusterProvider = Field(default=ClusterProvider.Docker)
    region: Optional[str] = None

    credential_id: Optional[int] = Field(
        default=None, foreign_key="cloud_credentials.id"
    )


class ClusterCreate(ClusterCreateBase):
    worker_pools: Optional[List[WorkerPoolCreate]] = None


class ClusterBase(ClusterCreateBase):
    state: ClusterStateEnum = ClusterStateEnum.PROVISIONING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    reported_gateway_endpoint: Optional[str] = None
    is_default: bool = Field(default=False)


if TYPE_CHECKING:
    from gpustack.schemas.workers import Worker


class Cluster(ClusterBase, BaseModelMixin, table=True):
    __tablename__ = "clusters"

    id: Optional[int] = Field(default=None, primary_key=True)

    hashed_suffix: str = Field(nullable=False, default=secrets.token_hex(6))

    registration_token: Optional[str] = Field(
        nullable=True, default=secrets.token_hex(16)
    )

    cluster_workers: List["Worker"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="cluster",
    )

    _workers: int = PrivateAttr(default=-1)

    _ready_workers: int = PrivateAttr(default=-1)

    _gpus: int = PrivateAttr(default=-1)

    @computed_field()
    @property
    def workers(self) -> int:
        try:
            if self._workers >= 0:
                return self._workers
        except TypeError:
            pass
        return len(self.cluster_workers or [])

    @computed_field()
    @property
    def ready_workers(self) -> int:
        try:
            if self._ready_workers >= 0:
                return self._ready_workers
        except TypeError:
            pass
        return len([w for w in self.cluster_workers or [] if w.state.value == 'ready'])

    @computed_field(alias="gpus")
    @property
    def gpus(self) -> int:
        try:
            if self._gpus >= 0:
                return self._gpus
        except TypeError:
            pass
        count = 0
        for worker in self.cluster_workers or []:
            if worker.status is None or worker.status.gpu_devices is None:
                continue
            count += len(worker.status.gpu_devices)
        return count

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, Cluster):
            return self.id == other.id
        return False

    cluster_worker_pools: List[WorkerPool] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="cluster",
    )

    cluster_models: List["Model"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"}, back_populates="cluster"
    )


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


class CredentialType(str, Enum):
    SSH = "ssh"
    CA = "ca"
    X509 = "x509"


class SSHKeyOptions(BaseModel):
    algorithm: str = Field(default="RSA")
    length: int = Field(default=2048)


class CredentialBase(SQLModel):
    external_id: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )

    credential_type: CredentialType = Field(default=CredentialType.SSH)

    public_key: str = Field(sa_column=Column(Text, nullable=False))

    encoded_private_key: str = Field(default="", sa_column=Column(Text, nullable=False))

    ssh_key_options: Optional[SSHKeyOptions] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(SSHKeyOptions), nullable=True),
    )


class Credential(CredentialBase, BaseModelMixin, table=True):
    __tablename__ = "credentials"
    __table_args__ = (sa.Index("idx_credentials_external_id", "external_id"),)
    id: Optional[int] = Field(default=None, primary_key=True)


if TYPE_CHECKING:
    from gpustack.schemas.models import Model
