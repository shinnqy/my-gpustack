from sqlmodel.ext.asyncio.session import AsyncSession

engine = None

async def get_session():
    """
    它是一个**生成器**。FastAPI 的依赖注入在请求开始时执行到 `yield`，把 session 交给路由；请求结束后继续执行，退出 `async with`，关闭会话
    """
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session

def async_session() -> AsyncSession:
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return AsyncSession(engine, expire_on_commit=False)
