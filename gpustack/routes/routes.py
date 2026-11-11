from fastapi import APIRouter

from gpustack.api.exceptions import error_responses
from gpustack.routes import clusters, models, probes

versioned_prefix = "/v2"

api_router = APIRouter(responses=error_responses)
api_router.include_router(probes.router, tags=["Probes"])

v1_base_router = APIRouter()
v1_base_router.include_router(models.router, prefix="/models", tags=["Models"])
v1_base_router.include_router(clusters.router, prefix="/clusters", tags=["Clusters"])

api_router.include_router(v1_base_router, prefix=versioned_prefix)