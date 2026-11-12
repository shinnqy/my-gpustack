from pydantic import ConfigDict
from sqlmodel import Column
from sqlmodel import Field
from sqlmodel import ForeignKey
from sqlmodel import Integer
from sqlmodel import SQLModel

"""这些是用来记录：哪个模型实例使用了哪份模型文件"""

class ModelInstanceModelFileLink(SQLModel, table=True):
    """实例 → 主模型等文件"""

    model_instance_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_instances.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    model_file_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_files.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )

    model_config = ConfigDict(protected_namespaces=())


class ModelInstanceDraftModelFileLink(SQLModel, table=True):
    """实例 → 草稿模型文件"""

    model_instance_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_instances.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    model_file_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_files.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )

    model_config = ConfigDict(protected_namespaces=())
