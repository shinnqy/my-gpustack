from fastapi import FastAPI

from gpustack import __version__
from gpustack.config.config import Config
from gpustack.routes import probes


def create_app(cfg: Config) -> FastAPI:
  app = FastAPI(
    title="GPUStack",
    # 返回 pydantic 对象时，没赋过值的字段不输出
    response_model_exclude_unset=True,
    version=__version__
  )

  app.include_router(probes.router)
  return app
