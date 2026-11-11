import logging
import re
from urllib.parse import urlparse, parse_qs, urlunparse
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)
from sqlmodel import SQLModel
from gpustack import envs
from gpustack.server import db
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.clusters import (
    Cluster,
)
from gpustack.schemas.workers import Worker


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
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                Model.__table__,
                ModelInstance.__table__,
                Worker.__table__,
                Cluster.__table__,
            ],
        )
