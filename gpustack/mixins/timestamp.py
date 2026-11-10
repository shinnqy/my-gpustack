from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field

from gpustack.schemas.common import UTCDateTime


def _datetime_func():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TimestampsMixin:
    """Mixin that define timestamp columns."""

    __abstract__ = True
    __config__ = None

    __created_at_name__ = "created_at"
    __updated_at_name__ = "updated_at"
    __deleted_at_name__ = "deleted_at"

    created_at: Optional[datetime] = Field(
        sa_type=UTCDateTime,
        sa_column_kwargs={
            "name": __created_at_name__,
            # 插入时由 SQLAlchemy 调用这个函数填值
            "default": _datetime_func,
            "nullable": False,
        },
        default=None,
    )

    updated_at: Optional[datetime] = Field(
        sa_type=UTCDateTime,
        sa_column_kwargs={
            "name": __updated_at_name__,
            # 插入时由 SQLAlchemy 调用这个函数填值
            "default": _datetime_func,
            # 每次 UPDATE 自动刷新 `updated_at`
            "onupdate": _datetime_func,
            "nullable": False,
        },
        default=None,
    )

    # 软删除
    deleted_at: Optional[datetime] = Field(
        sa_type=UTCDateTime,
        sa_column_kwargs={
            "name": __deleted_at_name__,
            "default": None,
            "nullable": True,
        },
        default=None,
    )