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

    model_config = ConfigDict(protected_namespaces=())

    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    __tablename__ = 'model_instances'
    id: Optional[int] = Field(default=None, primary_key=True)

    model: Optional[Model] = Relationship(
        back_populates="instances",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    # overwrite the hash to use in uniquequeue
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


def get_backend(model: Model) -> str:
    if model.backend:
        return model.backend

    return BackendEnum.VLLM
