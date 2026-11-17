from fastapi import APIRouter, Depends
from gpustack.api.exceptions import InternalServerErrorException, NotFoundException
from gpustack.api.exceptions import InvalidException, ConflictException
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.schemas.clusters import (
    CloudCredential,
    CloudCredentialCreate,
    CloudCredentialPublic,
    CloudCredentialsPublic,
    CloudCredentialUpdate,
)
from gpustack.schemas.common import ListParams
from gpustack.server.deps import SessionDep


router = APIRouter()


@router.get("", response_model=CloudCredentialsPublic)
async def list(session: SessionDep, params: ListParams = Depends()):
    return await CloudCredential.paginated_by_query(
        session=session, fields={"deleted_at": None}, page=params.page, per_page=params.perPage
    )


@router.get("/{id}", response_model=CloudCredentialPublic)
async def get(session: SessionDep, id: int):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")
    return existing


@router.post("", response_model=CloudCredentialPublic)
async def create(session: SessionDep, input: CloudCredentialCreate):
    if input.provider not in {ClusterProvider.DigitalOcean, ClusterProvider.Shuihua} or not input.secret:
        raise InvalidException(message="A supported cloud provider and non-empty secret are required")
    try:
        return await CloudCredential.create(session, CloudCredential.model_validate(input))
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create cloud credential: {e}")


@router.put("/{id}", response_model=CloudCredentialPublic)
async def update(session: SessionDep, id: int, input: CloudCredentialUpdate):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")
    if "secret" in input.model_fields_set and not input.secret:
        raise InvalidException(message="Cloud credential secret must not be empty")
    if "provider" in input.model_fields_set:
        if input.provider not in {ClusterProvider.DigitalOcean, ClusterProvider.Shuihua}:
            raise InvalidException(message="Unsupported cloud credential provider")
        if input.provider != existing.provider and await Cluster.all_by_field(session, "credential_id", id):
            raise ConflictException(message="Cannot change the provider of a credential used by a cluster")
    await existing.update(session=session, source=input)
    return existing


@router.delete("/{id}")
async def delete(session: SessionDep, id: int):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")
    if await Cluster.all_by_field(session, "credential_id", id):
        raise ConflictException(message="Cloud credential is still used by a cluster")
    await existing.delete(session=session)
