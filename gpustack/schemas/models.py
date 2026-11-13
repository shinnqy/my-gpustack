from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel, Text
from gpustack.schemas.common import (
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from gpustack.mixins import BaseModelMixin
from pydantic import model_validator
import hashlib
from typing import TYPE_CHECKING
from gpustack.schemas.links import ModelInstanceModelFileLink
from sqlalchemy.orm import selectinload
from sqlmodel import select
from gpustack.schemas.links import ModelInstanceDraftModelFileLink
from gpustack.utils.command import find_parameter
from pathlib import Path
from typing import Union
from dataclasses import dataclass
from gpustack.utils.command import find_bool_parameter


class SourceEnum(str, Enum):
    HUGGING_FACE = "huggingface"
    MODEL_SCOPE = "model_scope"
    LOCAL_PATH = "local_path"


class CategoryEnum(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    IMAGE = "image"
    RERANKER = "reranker"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    UNKNOWN = "unknown"


class PlacementStrategyEnum(str, Enum):
    SPREAD = "spread"
    BINPACK = "binpack"


class BackendEnum(str, Enum):
    VLLM = "vLLM"
    VOX_BOX = "VoxBox"
    ASCEND_MINDIE = "MindIE"
    SGLANG = "SGLang"
    CUSTOM = "Custom"


class BackendSourceEnum(str, Enum):
    """区分内置、社区、自定义后端"""

    CUSTOM = "custom"
    BUILT_IN = "built_in"
    COMMUNITY = "community"


class GPUSelector(BaseModel):
    # format of each element: "worker_name:device:gpu_index", example: "worker1:cuda:0"
    gpu_ids: Optional[List[str]] = None
    gpus_per_replica: Optional[int] = None


class ModelSource(BaseModel):
    source: SourceEnum

    huggingface_repo_id: Optional[str] = None

    huggingface_filename: Optional[str] = None

    model_scope_model_id: Optional[str] = None

    model_scope_file_path: Optional[str] = None

    local_path: Optional[str] = None

    @property
    def readable_source(self) -> str:
        values = []
        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend([self.model_scope_model_id, self.model_scope_file_path])
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])

        return "/".join([value for value in values if value is not None])

    @model_validator(mode="after")
    def check_huggingface_fields(self):
        if self.source == SourceEnum.HUGGING_FACE:
            if not self.huggingface_repo_id:
                raise ValueError(
                    "huggingface_repo_id must be provided "
                    "when source is 'huggingface'"
                )

        if self.source == SourceEnum.MODEL_SCOPE:
            if not self.model_scope_model_id:
                raise ValueError(
                    "model_scope_model_id must be provided when source is 'model_scope'"
                )

        if self.source == SourceEnum.LOCAL_PATH:
            if not self.local_path:
                raise ValueError(
                    "local_path must be provided when source is 'local_path'"
                )
        return self

    @property
    def model_source_index(self) -> str:
        """
        给模型来源生成标识
            - 将仓库 ID、文件名或本地路径拼接后计算 SHA-256，用来查找已有的 ModelFile，方便复用下载记录
        """
        values = []
        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend(
                [self.source, self.model_scope_model_id, self.model_scope_file_path]
            )
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])

        # Filter out None values and join
        filtered_values = [v for v in values if v is not None]
        source_string = "/".join(filtered_values)
        return hashlib.sha256(source_string.encode()).hexdigest()


class ModelSpecBase(SQLModel, ModelSource):
    name: str = Field(index=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )
    meta: Optional[Dict[str, Any]] = Field(sa_type=JSON, default={})

    replicas: int = Field(default=1, ge=0)
    ready_replicas: int = Field(default=0, ge=0)
    categories: List[str] = Field(sa_type=JSON, default=[])
    placement_strategy: PlacementStrategyEnum = PlacementStrategyEnum.SPREAD
    cpu_offloading: Optional[bool] = None
    distributed_inference_across_workers: Optional[bool] = None
    worker_selector: Optional[Dict[str, str]] = Field(sa_type=JSON, default={})
    gpu_selector: Optional[GPUSelector] = Field(
        sa_type=pydantic_column_type(GPUSelector), default=None
    )

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    image_name: Optional[str] = None
    run_command: Optional[str] = Field(sa_type=Text, default=None)

    env: Optional[Dict[str, str]] = Field(sa_type=JSON, default=None)
    restart_on_error: Optional[bool] = True
    distributable: Optional[bool] = False

    @model_validator(mode="after")
    def set_defaults(self):
        backend = get_backend(self)
        if self.distributed_inference_across_workers is None:
            self.distributed_inference_across_workers = (
                True
                if backend
                in [BackendEnum.VLLM, BackendEnum.ASCEND_MINDIE, BackendEnum.SGLANG]
                else False
            )
        return self


class ModelBase(ModelSpecBase):
    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")


class Model(ModelBase, BaseModelMixin, table=True):
    __tablename__ = 'models'
    id: Optional[int] = Field(default=None, primary_key=True)

    instances: list["ModelInstance"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="model",
    )


class ModelCreate(ModelBase):
    pass


class ModelUpdate(ModelBase):
    pass


class ModelPublic(
    ModelBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelsPublic = PaginatedList[ModelPublic]


class ModelInstanceStateEnum(str, Enum):
    r"""
    Enum for Model Instance State

    Transitions:

       |- - - - - Scheduler - - - - |- - ServeManager - -|- - - - Controller - - - -|- ServeManager -|
       |                            |                    |                          |                |
    PENDING ---> ANALYZING ---> SCHEDULED ---> INITIALIZING ---> DOWNLOADING ---> STARTING ---> RUNNING
                     |            ^  |               |                |               |          ^
                     |            |  |               |                |               |          |(Worker ready)
                     |------------|--|---------------|----------------|---------------|----------|
                     \____________|_____________________________________________________________/|
                                  |                  ERROR                                       |(Worker unreachable)
                                  └--------------------┘                                         v
                                    (Restart on Error)                                       UNREACHABLE
    """

    INITIALIZING = "initializing"
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    SCHEDULED = "scheduled"
    ERROR = "error"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class ComputedResourceClaim(BaseModel):
    is_unified_memory: Optional[bool] = False
    offload_layers: Optional[int] = None
    total_layers: Optional[int] = None
    ram: Optional[int] = Field(default=None)  # in bytes
    vram: Optional[Dict[int, int]] = Field(default=None)  # in bytes
    tensor_split: Optional[List[int]] = Field(default=None)
    vram_utilization: Optional[float] = Field(default=None)


class ModelInstanceSubordinateWorker(BaseModel):
    """保存每个从属 Worker 的 GPU、资源需求、下载进度、进程、端口和状态"""

    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    total_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    # - For model file preparation
    download_progress: Optional[float] = None
    # - For model instance serving preparation
    pid: Optional[int] = None
    ports: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    arguments: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class DistributedServerCoordinateModeEnum(Enum):
    # DELEGATED means that the subordinate workers' coordinate is by-pass to other framework.
    DELEGATED = "delegated"
    # INITIALIZE_LATER means that the subordinate workers' coordinate is handled by GPUStack,
    # all subordinate workers belong to one model instance SHOULD start after the main worker initializes.
    # For example, Ascend MindIE/vLLM/SGLang instances need to start their subordinate workers after the main worker initializes.
    INITIALIZE_LATER = "initialize_later"
    # RUN_FIRST means that the subordinate workers' coordinate is handled by GPUStack,
    # all subordinate workers belong to one model instance MUST get ready before the main worker starts.
    RUN_FIRST = "run_first"


class DistributedServers(BaseModel):
    """把这些从属 Worker 汇总起来，存入实例的 distributed_servers JSON 列"""
    # Indicates how the distributed servers coordinate with the main worker.
    mode: DistributedServerCoordinateModeEnum = (
        DistributedServerCoordinateModeEnum.DELEGATED
    )
    # Indicates if subordinate workers should download model files.
    download_model_files: Optional[bool] = True
    subordinate_workers: Optional[List[ModelInstanceSubordinateWorker]] = Field(
        sa_column=Column(JSON), default=[]
    )
    model_config = ConfigDict(from_attributes=True)


@dataclass
class ModelInstanceDeploymentMetadata:
    """
    Metadata for model instance deployment.
    """

    name: str
    """
    Name for model instance deployment.
    """
    distributed: bool = False
    """
    Whether the model instance is deployed in distributed mode.
    """
    distributed_leader: bool = False
    """
    Whether the model instance is the leader in distributed mode.
    """
    distributed_follower: bool = False
    """
    Whether the model instance is a follower in distributed mode.
    """
    distributed_follower_index: Optional[int] = None
    """
    Index of the follower in distributed mode.
    It is None for leader or non-distributed mode.
    """


class ModelInstanceBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)

    worker_id: Optional[int] = None

    worker_name: Optional[str] = None

    worker_advertise_address: Optional[str] = None

    worker_ip: Optional[str] = None

    worker_ifname: Optional[str] = None

    pid: Optional[int] = None

    port: Optional[int] = None

    ports: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])

    download_progress: Optional[float] = None

    resolved_path: Optional[str] = None

    # 单独记录草稿模型的准备状态 - 新增来源
    draft_model_source: Optional[ModelSource] = Field(
        sa_column=Column(pydantic_column_type(ModelSource)), default=None
    )

    # 单独记录草稿模型的准备状态 - 下载进度
    draft_model_download_progress: Optional[float] = None

    # 单独记录草稿模型的准备状态 - 本地实际路径
    draft_model_resolved_path: Optional[str] = None

    restart_count: Optional[int] = 0

    last_restart_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )

    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING

    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )

    gpu_type: Optional[str] = None

    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])

    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])

    model_id: int = Field(default=None, foreign_key="models.id")

    model_name: str

    backend: Optional[str] = None

    backend_version: Optional[str] = None

    """这里的 server 指参与推理的服务进程，不是管理 Worker 的 GPUStack Server"""
    distributed_servers: Optional[DistributedServers] = Field(
        sa_column=Column(pydantic_column_type(DistributedServers)), default=None
    )

    model_config = ConfigDict(protected_namespaces=())

    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")

    def get_deployment_metadata(
        self,
        worker_id: int,
    ) -> Optional[ModelInstanceDeploymentMetadata]:
        """
        判断指定 Worker 是否参与实例、是主节点还是从节点，并生成工作负载名称。
        比如主节点叫 example，第一个从节点叫 example-f0；无关 Worker 返回 None。
        这是运行时数据结构，不新增数据库列
        """

        dservers = self.distributed_servers
        subworkers = (
            dservers.subordinate_workers
            if dservers and dservers.subordinate_workers
            else []
        )

        name = self.name
        distributed = bool(subworkers)
        distributed_leader = distributed and self.worker_id == worker_id
        distributed_follower = distributed and not distributed_leader
        distributed_follower_index = None
        if distributed_follower:
            for idx, subworker in enumerate(subworkers):
                if subworker.worker_id == worker_id:
                    distributed_follower_index = idx
                    break
            if distributed_follower_index is not None:
                # Mutate the name to include the follower index,
                # so that each follower has a unique name.
                name += f"-f{distributed_follower_index}"

        if self.worker_id != worker_id and distributed_follower_index is None:
            # This model instance is not handling by the given worker.
            return None

        return ModelInstanceDeploymentMetadata(
            name=name,
            distributed=distributed,
            distributed_leader=distributed_leader,
            distributed_follower=distributed_follower,
            distributed_follower_index=distributed_follower_index,
        )


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    """
    表示“一个完整的模型服务副本”，这个副本可以由多个 Worker 共同运行
    """

    __tablename__ = 'model_instances'

    id: Optional[int] = Field(default=None, primary_key=True)

    model: Optional[Model] = Relationship(
        back_populates="instances",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    """让实例关联模型文件"""
    model_files: List["ModelFile"] = Relationship(
        back_populates="instances",
        link_model=ModelInstanceModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    draft_model_files: List["ModelFile"] = Relationship(
        back_populates="draft_instances",
        link_model=ModelInstanceDraftModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    @classmethod
    async def one_by_id_with_model_files(
        cls,
        session,
        instance_id: int,
        populate_existing: bool = True,
    ) -> Optional["ModelInstance"]:
        """
        提供显式加载关联数据的方法
        """
        stmt = (
            select(cls)
            .where(cls.id == instance_id)
            .options(
                selectinload(cls.model_files),
                selectinload(cls.draft_model_files),
                selectinload(cls.model),
            )
        )
        if populate_existing:
            stmt = stmt.execution_options(populate_existing=True)
        return (await session.exec(stmt)).first()

    def __hash__(self):
        return self.id


class ModelInstanceCreate(ModelInstanceBase):
    pass


class ModelInstanceUpdate(ModelInstanceBase):
    pass


class ModelInstancePublic(
    ModelInstanceBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelInstancesPublic = PaginatedList[ModelInstancePublic]


def is_gguf_model(model: Union[Model, ModelSource]):
    """
    根据文件名是否以 .gguf 结尾判断格式
    """
    return (
        (
            model.source == SourceEnum.HUGGING_FACE
            and model.huggingface_filename
            and model.huggingface_filename.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.MODEL_SCOPE
            and model.model_scope_file_path
            and model.model_scope_file_path.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.LOCAL_PATH
            and model.local_path
            and model.local_path.endswith(".gguf")
        )
    )


def is_audio_model(model: Model):
    """
    Check if the model is a STT or TTS model.
    Args:
        model: Model to check.
    """
    if model.backend == BackendEnum.VOX_BOX:
        return True

    if model.categories:
        return (
            'speech_to_text' in model.categories or 'text_to_speech' in model.categories
        )

    return False


def is_llm_model(model: Model):
    """
    Check if the model is an LLM model.
    Args:
        model: Model to check.
    """
    return not model.categories or CategoryEnum.LLM in model.categories


def is_omni_model(model: Model) -> bool:
    """
    根据 vLLM 的 --omni 参数，或 Image、Text-to-Speech 分类识别模型，供启动参数构造时选择处理分支
    """

    if model.backend == BackendEnum.VLLM and find_bool_parameter(
        model.backend_parameters, ["omni"]
    ):
        return True

    OMNI_CATEGORIES = (
        CategoryEnum.IMAGE,
        CategoryEnum.TEXT_TO_SPEECH,
    )
    return any(cat in model.categories for cat in OMNI_CATEGORIES)


def get_backend(model: Model) -> str:
    if model.backend:
        return model.backend

    return BackendEnum.VLLM


def get_mmproj_filename(model: Union[Model, ModelSource]) -> Optional[str]:
    """
    帮下载器寻找多模态投影文件
    """
    if not is_gguf_model(model):
        return None

    if hasattr(model, "backend_parameters"):
        mmproj = find_parameter(model.backend_parameters, ["mmproj"])
        if mmproj and Path(mmproj).name == mmproj:
            return mmproj

    return "*mmproj*.gguf"


if TYPE_CHECKING:
    from gpustack.schemas.model_files import ModelFile
