import datetime
import secrets
from typing import Optional
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from gpustack.api.exceptions import (
    ForbiddenException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.config import PredefinedConfigNoDefaults
from gpustack.schemas.common import ListParams
from gpustack.schemas.workers import (
    Worker,
    WorkerCreate,
    WorkerPublic,
    WorkerRegistrationPublic,
    WorkerStateEnum,
    WorkerStatusStored,
    WorkersPublic,
)
from gpustack.security import API_KEY_PREFIX
from gpustack.server.db import async_session
from gpustack.server.worker_status_buffer import (
    heartbeat_flush_buffer,
    heartbeat_flush_buffer_lock,
    worker_status_flush_buffer,
    worker_status_flush_buffer_lock,
)
from gpustack.config.config import get_global_config
from gpustack.schemas.config import SensitivePredefinedConfig
from typing import Any
from typing import Dict
from gpustack.api.exceptions import AlreadyExistsException
from gpustack.api.exceptions import InvalidException
from typing import List
import asyncio
import uuid
from gpustack.schemas.clusters import ClusterProvider
import logging


router = APIRouter()


def _bearer(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    return auth[len("Bearer "):] if auth.startswith("Bearer ") else None


async def current_worker(request: Request) -> Worker:
    # 真实版本：get_current_user 解析 Worker 的 API Key，再取 user.worker
    token = _bearer(request)
    async with async_session() as session:
        worker = await Worker.one_by_field(session, "token", token) if token else None
    if worker is None:
        raise ForbiddenException(message="Failed to find related worker")
    return worker


logger = logging.getLogger(__name__)


create_worker_semaphore = asyncio.Semaphore(10)


@router.get("", response_model=WorkersPublic)
async def get_workers(request: Request, params: ListParams = Depends(), me: Optional[bool] = None):
    async with async_session() as session:
        fields = {}
        if me:
            fields["id"] = (await current_worker(request)).id
        result = await Worker.paginated_by_query(
            session=session, fields=fields, page=params.page, per_page=params.perPage
        )
        result.items = [WorkerPublic.model_validate(w) for w in result.items]
        return result


def update_worker_data(
    worker_in: WorkerCreate,
    existing: Optional[Worker] = None,
    **kwargs,
) -> Worker:
    to_create_worker = None
    cluster: Optional[Cluster] = kwargs.get("cluster")
    if existing is not None:
        incoming_data = worker_in.model_dump()

        to_create_worker = Worker.model_validate(
            {
                **existing.model_dump(),
                **incoming_data,
                "labels": {
                    **existing.labels,
                    **worker_in.labels,
                },
                "cluster_id": existing.cluster_id,
                "state": WorkerStateEnum.READY,
            }
        )
    else:
        # new worker should ignore the reported worker_uuid
        to_create_worker = Worker.model_validate(
            {
                **worker_in.model_dump(exclude={"name", "worker_uuid"}),
                "name": worker_in.name or worker_in.hostname,
                "worker_uuid": "",
                "state": WorkerStateEnum.READY,
                **kwargs,
            }
        )
        if cluster is not None:
            to_create_worker.cluster = cluster
    to_create_worker.compute_state()
    return to_create_worker


def _matches_exact_fields(worker: Worker, fields: Optional[Dict[str, Any]]) -> bool:
    if not fields:
        return True
    return all(getattr(worker, k, None) == v for k, v in fields.items())


def _matches_fuzzy_fields(worker: Worker, fuzzy_fields: Dict[str, str]) -> bool:
    if not fuzzy_fields:
        return True
    for k, v in fuzzy_fields.items():
        attr = getattr(worker, k, None)
        if not isinstance(attr, str) or v.lower() not in attr.lower():
            return False
    return True


def filter_workers_by_fields(
    workers: List[Worker],
    fields: Optional[Dict[str, Any]],
    fuzzy_fields: Optional[Dict[str, str]] = None,
) -> List[Worker]:
    if fuzzy_fields is None:
        fuzzy_fields = {}
    if not fields and not fuzzy_fields:
        return workers

    to_return = []
    for worker in workers:
        match = _matches_exact_fields(worker, fields) and _matches_fuzzy_fields(
            worker, fuzzy_fields
        )

        if match:
            to_return.append(worker)
    return to_return


def get_existing_worker(
    cluster_id: int, worker_in: WorkerCreate, workers: List[Worker]
) -> Optional[Worker]:
    static_fields = {
        "deleted_at": None,
        "cluster_id": cluster_id,
    }

    if worker_in.name == "":
        return None

    # find existing worker by external_id or worker_uuid
    for field in ["external_id", "worker_uuid"]:
        value = getattr(worker_in, field, None)
        if value is None:
            continue
        fields = {**static_fields, field: value}
        existing_worker = next(iter(filter_workers_by_fields(workers, fields)), None)
        if existing_worker is not None:
            return existing_worker

    # find existing worker by name
    if worker_in.labels and worker_in.labels.get("gpustack.existence-check"):
        fields = {"name": worker_in.name}
        existing_worker = next(iter(filter_workers_by_fields(workers, fields)), None)
        if existing_worker is not None:
            if existing_worker.cluster_id != cluster_id:
                raise AlreadyExistsException(
                    message=(
                        f"Worker with name '{worker_in.name}' already exists "
                        "in another cluster."
                    )
                )
            return existing_worker

    return None


def check_worker_name_conflict(
    name: str, workers: List[Worker], existing_id: Optional[int] = None
):
    if name == "":
        if existing_id is not None:
            raise InvalidException(message="worker name cannot be empty")
        return
    workers = [worker for worker in workers if worker.id != existing_id]
    name_conflict_fields = {"name": name}
    name_conflict_worker = next(
        iter(filter_workers_by_fields(workers, name_conflict_fields)), None
    )
    if name_conflict_worker is not None:
        raise AlreadyExistsException(
            message=f"Worker with name '{name}' already exists."
        )


def retry_create_unique_worker_uuid(workers: List[Worker]) -> str:
    current_uuids = set(
        worker.worker_uuid for worker in workers if worker.worker_uuid != ""
    )
    for i in range(5):
        new_uuid = str(uuid.uuid4())
        if new_uuid not in current_uuids:
            return new_uuid
        logger.warning(
            f"UUID collision detected for worker_uuid {new_uuid}, retrying... (attempt {i + 1}/5)"
        )
    # might not be necessary to retry so many times, but just in case, we want to make sure
    # the system can recover from such a rare event without manual intervention
    raise InternalServerErrorException(
        message="Failed to generate unique worker UUID after multiple attempts"
    )


def _build_worker_config_dict(cluster: Cluster) -> Dict[str, Any]:
    sensitive_fields = set(SensitivePredefinedConfig.model_fields.keys())
    worker_config = (
        {}
        if cluster.worker_config is None
        else cluster.worker_config.model_dump(exclude=sensitive_fields)
    )
    # ``system_default_container_registry`` now lives in its own column on
    # the cluster; inject the cluster value (then the server-wide default)
    # into the worker config dict so workers keep receiving the setting via
    # the existing PredefinedConfig channel.
    cfg = get_global_config()
    cluster_registry = (cluster.system_default_container_registry or "").strip()
    cfg_registry = (cfg.system_default_container_registry or "").strip()
    effective_registry = cluster_registry or cfg_registry
    if effective_registry:
        worker_config["system_default_container_registry"] = effective_registry
    return worker_config


@router.post("", response_model=WorkerRegistrationPublic)
async def create_worker(request: Request, worker_in: WorkerCreate):
    async with create_worker_semaphore:
        async with async_session() as session:
            # 真实版本：认证层把令牌解析成集群的系统主体，_resolve_create_worker_cluster_id 取 user.cluster
            token = _bearer(request)
            cluster = await Cluster.one_by_field(session, "registration_token", token) if token else None
            if cluster is None or cluster.deleted_at is not None:
                raise NotFoundException(message="Cluster not found")
            cluster_id = cluster.id
            all_workers = await Worker.all_by_fields(session, {"deleted_at": None})
            existing_worker = get_existing_worker(cluster_id, worker_in, all_workers)
            check_worker_name_conflict(
                worker_in.name,
                all_workers,
                existing_id=existing_worker.id if existing_worker else None,
            )
            if existing_worker is not None:
                existing_worker = await Worker.one_by_id(
                    session=session, id=existing_worker.id, for_update=True
                )

            worker_config = _build_worker_config_dict(cluster)

            new_token = f"{API_KEY_PREFIX}_{secrets.token_hex(8)}_{secrets.token_hex(16)}"

            new_worker = update_worker_data(
                worker_in,
                existing=existing_worker,
                # following args are only used when creating a new worker
                provider=cluster.provider,
                cluster=cluster,
                token=new_token,
            )
            if new_worker.worker_uuid == "":
                new_worker.worker_uuid = retry_create_unique_worker_uuid(all_workers)
            new_worker.token = new_token

            try:
                if existing_worker is not None:
                    await existing_worker.update(session, new_worker.model_dump(exclude={"id"}))
                    worker = existing_worker
                else:
                    worker = await Worker.create(session, new_worker)
                worker_dump = worker.model_dump()
                worker_dump["token"] = worker.token
                worker_dump["worker_config"] = (
                    PredefinedConfigNoDefaults.model_validate(worker_config)
                )
                return WorkerRegistrationPublic.model_validate(worker_dump)
            except Exception as e:
                await session.rollback()
                raise InternalServerErrorException(
                    message=f"Failed to create worker: {e}"
                )


async def create_worker_status(input: WorkerStatusStored, worker: Worker = Depends(current_worker)):
    heartbeat_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    input_dict = {key: getattr(input, key) for key in input.model_fields_set}
    input_dict["heartbeat_time"] = heartbeat_time

    # Add worker status to buffer for batch update
    async with worker_status_flush_buffer_lock:
        worker_status_flush_buffer[worker.id] = input_dict

    return Response(status_code=204)


async def heartbeat(worker: Worker = Depends(current_worker)):
    # Add worker ID to buffer for batch update
    async with heartbeat_flush_buffer_lock:
        heartbeat_flush_buffer.add(worker.id)

    return Response(status_code=204)
