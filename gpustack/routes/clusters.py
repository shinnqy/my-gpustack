import secrets
from typing import Optional, Union
from fastapi import APIRouter, Depends, Request
from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    ConflictException,
)
from gpustack.schemas.config import (
    parse_base_model_to_env_vars,
)
from gpustack.server.deps import SessionDep
from gpustack.schemas.clusters import (
    ClusterUpdate,
    ClusterCreate,
    ClusterPublic,
    ClustersPublic,
    Cluster,
    ClusterStateEnum,
    ClusterProvider,
    SensitiveRegistrationConfig,
    ClusterRegistrationTokenPublic,
)
from gpustack.schemas.models import Model
from gpustack.security import (
    API_KEY_PREFIX,
)
from gpustack.config.config import (
    get_global_config,
    get_cluster_image_name,
)
from gpustack.schemas.common import ListParams
from sqlalchemy.orm import selectinload


router = APIRouter()


def get_server_url(request: Request, cluster_override: Optional[str]) -> str:
    """Resolve the worker-facing server URL.

    Priority:
    1. cluster override
    2. server_external_url
    3. request host (scheme + netloc). ForwardedHostPortMiddleware only gates
       X-Forwarded-Host rewriting by trusted_hosts; a directly supplied Host
       header still flows into request.url, so this is not full host-injection
       protection unless the server is reachable only via a trusted proxy.
    """
    if cluster_override:
        return cluster_override.rstrip("/")
    cfg = get_global_config()
    url = cfg.server_external_url if cfg else None
    if not url:
        url = f"{request.url.scheme}://{request.url.netloc}"
    return url.rstrip("/")


@router.get("", response_model=ClustersPublic, response_model_exclude_none=True)
async def get_clusters(session: SessionDep, params: ListParams = Depends()):
    return await Cluster.paginated_by_query(
        session=session,
        fields={"deleted_at": None},
        page=params.page,
        per_page=params.perPage,
        options=[selectinload(Cluster.cluster_workers)],
    )


@router.get("/{id}", response_model=ClusterPublic, response_model_exclude_none=True)
async def get_cluster(session: SessionDep, id: int):
    cluster = await Cluster.one_by_id(
        session, id, options=[selectinload(Cluster.cluster_workers)]
    )
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    return cluster


def hoist_system_default_container_registry(
    input: Union[ClusterCreate, ClusterUpdate],
):
    """
    Make ``cluster.system_default_container_registry`` the single source of
    truth at persistence time.

    If the caller put the value inside ``worker_config`` (the legacy shape),
    promote it onto the top-level column and null it out in ``worker_config``
    so downstream image-resolution helpers can read the cluster column
    exclusively without ever reaching back into ``worker_config``. We don't
    overwrite the column if the caller already set it.
    """
    if input.worker_config is None:
        return
    nested = input.worker_config.system_default_container_registry
    if nested is None:
        return
    if not input.system_default_container_registry:
        input.system_default_container_registry = nested
    input.worker_config.system_default_container_registry = None


@router.post("", response_model=ClusterPublic, response_model_exclude_none=True)
async def create_cluster(session: SessionDep, input: ClusterCreate):
    existing = await Cluster.one_by_fields(
        session,
        {
            'deleted_at': None,
            "name": input.name,
        },
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Cluster with name '{input.name}' already exists."
        )

    hoist_system_default_container_registry(input)

    target_state = ClusterStateEnum.READY
    state_message = None
    if input.provider not in [ClusterProvider.Kubernetes, ClusterProvider.Docker]:
        target_state = ClusterStateEnum.PENDING
        state_message = "No workers have been provisioned for this cluster yet."
    to_create_cluster = Cluster.model_validate(
        {
            **input.model_dump(),
            "state": target_state,
            "state_message": state_message,
            "hashed_suffix": secrets.token_hex(6),
            "registration_token": f"{API_KEY_PREFIX}_{secrets.token_hex(8)}_{secrets.token_hex(16)}",
        }
    )
    try:
        cluster = await Cluster.create(session, to_create_cluster)
        return cluster
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create cluster: {e}")


@router.put("/{id}", response_model=ClusterPublic, response_model_exclude_none=True)
async def update_cluster(session: SessionDep, id: int, input: ClusterUpdate):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster:
        raise NotFoundException(message=f"cluster {id} not found")

    hoist_system_default_container_registry(input)

    try:
        await cluster.update(session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update cluster: {e}")

    return await Cluster.one_by_id(session, id)


@router.delete("/{id}")
async def delete_cluster(session: SessionDep, id: int):
    existing = await Cluster.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    # check for models, if any are present, prevent deletion
    if await Model.all_by_field(session, "cluster_id", id):
        raise ConflictException(
            message=f"cluster {existing.name}(id: {id}) has models, cannot be deleted"
        )
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete cluster: {e}")


def get_registration_from_cluster(
    request: Request, cluster: Cluster
) -> ClusterRegistrationTokenPublic:
    config = cluster.worker_config.model_dump() if cluster.worker_config else {}
    sensitive_registration = SensitiveRegistrationConfig(
        token=cluster.registration_token, **config
    )
    return ClusterRegistrationTokenPublic(
        token=cluster.registration_token,
        server_url=get_server_url(request, cluster.server_url),
        image=get_cluster_image_name(
            cluster.worker_config, cluster.system_default_container_registry
        ),  # Default image, can be customized
        env=parse_base_model_to_env_vars(sensitive_registration),
        args=[],
    )


@router.get("/{id}/registration-token", response_model=ClusterRegistrationTokenPublic)
async def get_registration_token(request: Request, session: SessionDep, id: int):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    return get_registration_from_cluster(request, cluster)
