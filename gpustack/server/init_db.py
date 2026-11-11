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
        # 去掉 `?sslmode=disable` 这样的查询参数。它们是 libpq（PostgreSQL 的 C 客户端）的写法，asyncpg 不认识，留着会报错
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
        # 从连接池取连接时，先发一个轻量的 `SELECT 1`。数据库重启过、或者连接被防火墙掐断了，池里的连接其实已经死了。不检查的话，第一个拿到死连接的请求会失败
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    return engine

async def create_db_and_tables(engine: AsyncEngine):
    async with engine.begin() as conn:
        # `create_all` 是 SQLAlchemy 的同步 API，`run_sync` 让它在异步连接上运行。它只创建**不存在**的表，已有的表不动，所以每次启动都调用也安全。
        await conn.run_sync(
            SQLModel.metadata.create_all,
            # 只建列出来的表
            tables=[
                # 存：用户的期望：模型来源、副本数、后端、参数
                Model.__table__,
                # 存：每个副本的现实：在哪台机器、哪几张卡、什么状态、什么端口
                ModelInstance.__table__,
                Cluster.__table__,
            ]
        )