from datetime import datetime
from enum import Enum
from typing import List, Optional
from sqlmodel import (
    JSON,
    BigInteger,
    Column,
    Field,
    Relationship,
    SQLModel,
    Text,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList
from gpustack.schemas.links import (
    ModelInstanceModelFileLink,
)
from gpustack.schemas.models import ModelSource, ModelInstance
from gpustack.schemas.links import ModelInstanceDraftModelFileLink


class ModelFileStateEnum(str, Enum):
    ERROR = "error"
    DOWNLOADING = "downloading"
    READY = "ready"


class ModelFileBase(SQLModel, ModelSource):
    # 用户可以指定下载到哪个目录；不指定就放在 `<cache_dir>/huggingface/<组织>/<仓库>`
    local_dir: Optional[str] = None
    # 这份文件在哪台机器上
    worker_id: Optional[int] = None
    # 删除这条记录时，要不要顺便删掉磁盘上的文件。默认不删（文件可能还会被再次使用）
    cleanup_on_delete: Optional[bool] = None

    # LoRA 适配器也是一种模型文件
    is_lora: bool = Field(default=False, nullable=False)

    # 下载前先查一次仓库得到总大小，界面上显示“16.4 GB”，进度也按它算
    size: Optional[int] = Field(sa_column=Column(BigInteger), default=None)
    download_progress: Optional[float] = None
    # 下载后的实际路径。整个仓库下载时是目录；GGUF 这种只下一个文件时是文件路径
    resolved_paths: List[str] = Field(sa_column=Column(JSON), default=[])
    state: ModelFileStateEnum = ModelFileStateEnum.DOWNLOADING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class ModelFile(ModelFileBase, BaseModelMixin, table=True):
    __tablename__ = 'model_files'

    id: Optional[int] = Field(default=None, primary_key=True)

    source_index: Optional[str] = Field(index=True, default=None)

    instances: list[ModelInstance] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"},
        back_populates="model_files",
        link_model=ModelInstanceModelFileLink,
    )

    draft_instances: list[ModelInstance] = Relationship(
        back_populates="draft_model_files",
        link_model=ModelInstanceDraftModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )


class ModelFileCreate(ModelFileBase):
    pass


class ModelFileUpdate(ModelFileBase):
    pass


class ModelFilePublic(
    ModelFileBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelFilesPublic = PaginatedList[ModelFilePublic]
