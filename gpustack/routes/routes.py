from fastapi import APIRouter
from gpustack.api.exceptions import error_responses
from gpustack.routes import models, probes
from gpustack.routes import clusters, models, probes
from gpustack.routes import clusters, models, probes, workers


versioned_prefix = "/v2"


api_router = APIRouter(responses=error_responses)


v1_base_router = APIRouter()


v1_base_router.include_router(models.router, prefix="/models", tags=["Models"])


v1_base_router.include_router(clusters.router, prefix="/clusters", tags=["Clusters"])


v1_base_router.include_router(workers.router, prefix="/workers", tags=["Workers"])


v1_base_router.add_api_route(
    path="/worker-status",
    endpoint=workers.create_worker_status,
    methods=["POST"],
)


v1_base_router.add_api_route(
    path="/worker-heartbeat",
    endpoint=workers.heartbeat,
    methods=["POST"],
)


api_router.include_router(v1_base_router, prefix=versioned_prefix)


api_router.include_router(probes.router, tags=["Probes"])
