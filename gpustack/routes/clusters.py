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
    K8sVolumeMount, VolumeSource, HostPathVolumeSource,
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
from fastapi import Query
from fastapi import Response
from gpustack.api.exceptions import InvalidException
from gpustack.k8s.manifest_template import TemplateConfig
from gpustack.schemas.clusters import K8sOptions
from gpustack_runtime.detector import ManufacturerEnum
from typing import List
from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.schemas.workers import Worker
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.db import async_session
from gpustack.server.worker_request import stream_to_worker
import aiohttp
import logging
import random


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


def _normalize_k8s_options(options: Optional[K8sOptions]) -> K8sOptions:
    options = options.model_copy(deep=True) if options is not None else K8sOptions()
    if not options.volume_mounts:
        options.volume_mounts = [K8sVolumeMount(
            name="gpustack-data-dir", mount_path="/var/lib/gpustack",
            volume_source=VolumeSource(host_path=HostPathVolumeSource(path="/var/lib/gpustack")),
        )]
    first = options.volume_mounts[0]
    if first.volume_source is None or first.volume_source.host_path is None:
        raise InvalidException(message="The first k8s_options volume mount must use hostPath for the GPUStack data directory")
    first.name = "gpustack-data-dir"
    first.mount_path = "/var/lib/gpustack"
    first.read_only = False
    first.volume_source.host_path.type = "DirectoryOrCreate"
    first.volume_source.config_map = None
    first.volume_source.persistent_volume_claim = None
    names, paths = set(), set()
    for mount in options.volume_mounts:
        source = mount.volume_source
        if source is None or len(source.model_dump(by_alias=True, exclude_none=True)) != 1:
            raise InvalidException(message=f"Volume {mount.name} must have exactly one volume source")
        if not mount.name:
            raise InvalidException(message="Kubernetes volume names must not be empty")
        if not mount.mount_path.startswith("/") or source.host_path and not source.host_path.path.startswith("/"):
            raise InvalidException(message="Kubernetes mount paths and host paths must be absolute")
        if mount.name in names or mount.mount_path in paths:
            raise InvalidException(message="Kubernetes volume names and mount paths must be unique")
        names.add(mount.name)
        paths.add(mount.mount_path)
    return options


def enforce_data_dir_mounts(input: Union[ClusterCreate, ClusterUpdate]):
    """保证第一项为持久化的数据目录；省略挂载时使用教材默认值。"""
    input.k8s_options = _normalize_k8s_options(input.k8s_options)


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

    if input.provider == ClusterProvider.Kubernetes:
        enforce_data_dir_mounts(input)
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

    if cluster.provider == ClusterProvider.Kubernetes and "k8s_options" in input.model_fields_set:
        enforce_data_dir_mounts(input)
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


@router.get("/{id}/manifests")
async def get_cluster_manifests(
    request: Request,
    session: SessionDep,
    id: int,
    runtime: Optional[List[ManufacturerEnum]] = Query(
        None,
        description=(
            "GPU vendor runtimes to include in the manifest. Repeat the "
            "parameter for multiple vendors (e.g. ?runtime=nvidia&runtime=ascend). "
            "The CPU worker DaemonSet is always rendered regardless of this "
            "parameter."
        ),
    ),
):
    cluster = await Cluster.one_by_id(session, id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message=f"cluster {id} not found")
    if cluster.provider != ClusterProvider.Kubernetes:
        raise InvalidException(
            message=f"Cannot get manifests for cluster {cluster.name}(id: {id}) with provider {cluster.provider}"
        )

    # Resolve server-wide defaults onto a copy of k8s_options so the render
    # model only ever reads k8s_options. Copy (not mutate) the loaded cluster
    # so we don't risk persisting these derived values back to the DB.
    cfg = get_global_config()
    k8s_options = _normalize_k8s_options(cluster.k8s_options)
    if not k8s_options.namespace:
        k8s_options.namespace = cfg.namespace
    if not k8s_options.operator_image:
        k8s_options.operator_image = cfg.operator_image

    # Only forward worker ports when the cluster explicitly overrides them;
    # otherwise let TemplateConfig fall back to its own defaults rather than
    # duplicating the default constants here.
    worker_config = cluster.worker_config
    config_kwargs = {
        "registration": get_registration_from_cluster(request, cluster),
        "runtimes": runtime,
        "k8s_options": k8s_options,
        "system_default_container_registry": cluster.system_default_container_registry
        or cfg.system_default_container_registry,
    }
    if worker_config:
        if worker_config.worker_port is not None:
            config_kwargs["worker_port"] = worker_config.worker_port
        if worker_config.worker_metrics_port is not None:
            config_kwargs["worker_metrics_port"] = worker_config.worker_metrics_port

    config = TemplateConfig(**config_kwargs)
    """渲染 jinja 模版"""
    yaml_content = config.render()
    return Response(
        content=yaml_content,
        media_type="application/x-yaml",
        headers={"Content-Disposition": "attachment; filename=manifest.yaml"},
    )


_CLUSTER_PROXY_REQUEST_HEADER_SKIP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
    "authorization",
    "x-gpustack-model-instance",
    "cookie",
    "x-api-key",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
}


@router.api_route(
    "/{id}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def cluster_apiserver_proxy(
    request: Request,
    id: int,
    path: str,
):
    """
    为 Server 提供访问目标 Kubernetes 的通道
    """
    async with async_session() as session:
        cluster = await Cluster.one_by_id(session, id)
        if not cluster or cluster.deleted_at is not None:
            raise NotFoundException(message=f"cluster {id} not found")
        if cluster.provider != ClusterProvider.Kubernetes:
            raise InvalidException(
                message=(
                    f"cluster {cluster.name}(id: {id}) provider is "
                    f"{cluster.provider.value}; API server proxy is only supported "
                    "for Kubernetes-provider clusters."
                )
            )

        workers = await Worker.all_by_fields(
            session,
            fields={"cluster_id": id, "state": WorkerStateEnum.READY},
        )
        if not workers:
            # The gpustack-operator polls this proxy for worker API-service
            # readiness, so an unreachable cluster is an expected wait rather
            # than a fault — reporting each 503 at ERROR floods the log.
            raise ServiceUnavailableException(
                message=f"No reachable workers in cluster {cluster.name}(id: {id})",
                log_level=logging.WARNING,
            )
        worker = random.choice(workers)
        session.expunge(worker)

    connection_headers = {part.strip().lower() for part in request.headers.get("connection", "").split(",")}
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _CLUSTER_PROXY_REQUEST_HEADER_SKIP | connection_headers
    }

    body = None
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        body = await request.body()

    # request.query_params preserves order but a flat dict is sufficient for
    # the Kubernetes API surface we forward (no duplicate keys in practice).
    params = list(request.query_params.multi_items()) or None

    # No total timeout — Kubernetes watch and log-follow streams may be open
    # indefinitely. Connect timeout still bounds the upstream connect step.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)

    return StreamingResponseWithStatusCode(
        stream_to_worker(
            worker=worker,
            method=request.method,
            path=f"cluster-proxy/{path}",
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            params=params,
            data=body,
            headers=headers,
            timeout=timeout,
            raw=True,
        ),
    )
