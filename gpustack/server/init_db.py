import logging
import re
from urllib.parse import urlparse, parse_qs, urlunparse
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)
from sqlmodel import SQLModel
from sqlalchemy import inspect
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.schema import CreateColumn
from gpustack import envs
from gpustack.server import db
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.clusters import (
    Cluster,
)
from gpustack.schemas.workers import Worker
from gpustack.schemas.links import (
    ModelInstanceModelFileLink,
    ModelInstanceDraftModelFileLink,
)
from gpustack.schemas.model_files import ModelFile


logger = logging.getLogger(__name__)


async def init_db(db_url: str):
    if db.engine is None:
        db.engine = await init_db_engine(db_url)
    await create_db_and_tables(db.engine)


async def init_db_engine(db_url: str):
    connect_args = {}
    if db_url.startswith("postgresql://"):
        db_url = re.sub(r'^postgresql://', 'postgresql+asyncpg://', db_url)
        parsed = urlparse(db_url)
        new_parsed = parsed._replace(query={})
        db_url = urlunparse(new_parsed)

    else:
        raise Exception(f"Unsupported database URL: {db_url}")

    engine = create_async_engine(
        db_url,
        echo=envs.DB_ECHO,
        pool_size=envs.DB_POOL_SIZE,
        max_overflow=envs.DB_MAX_OVERFLOW,
        pool_timeout=envs.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    return engine


async def create_db_and_tables(engine: AsyncEngine):
    async with engine.begin() as conn:
        await conn.run_sync(_upgrade_model_instance_columns)
        await conn.run_sync(_upgrade_worker_columns)
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                Model.__table__,
                ModelInstance.__table__,
                Worker.__table__,
                Cluster.__table__,
                ModelFile.__table__,
                ModelInstanceModelFileLink.__table__,
                ModelInstanceDraftModelFileLink.__table__,
            ],
        )


def _upgrade_model_instance_columns(connection):
    """用来修改已经存在的数据库表"""
    inspector = inspect(connection)
    table = ModelInstance.__table__
    if not inspector.has_table(table.name):
        return
    existing = {column["name"] for column in inspector.get_columns(table.name)}
    additions = {
        "draft_model_source", "draft_model_download_progress",
        "draft_model_resolved_path", "distributed_servers",
    }
    table_name = connection.dialect.identifier_preparer.quote(table.name)
    for column in table.columns:
        if column.name in additions and column.name not in existing:
            declaration = str(CreateColumn(column).compile(dialect=connection.dialect))
            connection.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {declaration}")


def _upgrade_worker_columns(connection):
    inspector = inspect(connection)
    table = Worker.__table__
    if not inspector.has_table(table.name):
        return
    existing = {column["name"] for column in inspector.get_columns(table.name)}
    table_name = connection.dialect.identifier_preparer.quote(table.name)
    for column in table.columns:
        if column.name in {"maintenance", "proxy_mode", "proxy_address"} and column.name not in existing:
            if connection.dialect.name == "postgresql" and isinstance(column.type, SQLAlchemyEnum):
                column.type.create(connection, checkfirst=True)
            declaration = str(CreateColumn(column).compile(dialect=connection.dialect))
            connection.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {declaration}")
