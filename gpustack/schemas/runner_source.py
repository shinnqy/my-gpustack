from gpustack_runner import BackendRunners
from gpustack_runner.runner import build_backend_runners
from typing import List
from datetime import datetime
from typing import Optional
from gpustack.mixins import BaseModelMixin
from sqlmodel import Field as SQLField
from gpustack_runner import Runners
from gpustack_runner import list_runners
from .source import SourceTypeEnum
from sqlalchemy import Column
from sqlalchemy import Text
from sqlmodel import SQLModel
from gpustack_runner.runner import Runner
from typing import Tuple

"""定义自定义 Runner 镜像目录"""

_KEY_FIELDS: Tuple[str, ...] = (
    "backend",
    "backend_version",
    "backend_variant",
    "service",
    "service_version",
    "platform",
)


_FILTER_KEYS = frozenset(_KEY_FIELDS)


class RunnerOverrideEntryBase(SQLModel):
    """
    A single per-platform runner override row, aligned field-for-field
    with the packaged ``gpustack_runner.Runner`` dataclass.
    """

    backend: str = SQLField(index=True)
    backend_version: str = SQLField(default="")
    original_backend_version: str = SQLField(default="")
    backend_variant: str = SQLField(default="")
    service: str = SQLField(index=True)
    service_version: str = SQLField(default="")
    platform: str = SQLField(default="")
    docker_image: str = SQLField(sa_column=Column(Text, nullable=False))
    deprecated: bool = SQLField(default=False)
    # Which source produced this row (last writer wins a key), same shape as
    # ``CatalogModelEntry``. Workers merge overrides without a DB session, so the
    # replace-or-layer decision has to travel with the entries themselves.
    source_name: str = SQLField(default="")
    source_type: SourceTypeEnum = SQLField(default=SourceTypeEnum.BUILTIN)
    # Reserved for future multi-tenancy; excluded from serialized responses.
    owner_principal_id: Optional[int] = SQLField(default=None, exclude=True)


class RunnerOverrideEntry(RunnerOverrideEntryBase, BaseModelMixin, table=True):
    __tablename__ = "runner_override_entries"
    id: Optional[int] = SQLField(default=None, primary_key=True)


class RunnerOverrideEntryPublic(RunnerOverrideEntryBase):
    id: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


def _to_runner(entry: RunnerOverrideEntry) -> Runner:
    """Convert an override row into a packaged ``Runner`` (lossless 1:1)."""
    return Runner(
        backend=entry.backend,
        backend_version=entry.backend_version,
        original_backend_version=entry.original_backend_version,
        backend_variant=entry.backend_variant,
        service=entry.service,
        service_version=entry.service_version,
        platform=entry.platform,
        docker_image=entry.docker_image,
        deprecated=entry.deprecated,
    )


def _filter_overrides(runners: Runners, **filters) -> Runners:
    """Apply the same exact-match filtering ``list_runners`` would, so
    overrides for an unrelated backend/service don't leak into a scoped query."""
    with_deprecated = filters.pop("with_deprecated", True)
    if with_deprecated is None:
        with_deprecated = True

    prefix_keys = [key for key in filters if key.endswith("_prefix")]
    if prefix_keys:
        raise ValueError(f"prefix filters are unsupported for overrides: {prefix_keys}")

    result: Runners = []
    for runner in runners:
        if not with_deprecated and runner.deprecated:
            continue
        if all(
            getattr(runner, key) == expected
            for key, expected in filters.items()
            if expected is not None and key in _FILTER_KEYS
        ):
            result.append(runner)
    return result


def merged_runners(overrides: List[RunnerOverrideEntry], **filters) -> Runners:
    """The runner catalog these override rows come to.

    - Whole-content replacement, as ``order_source_contents`` does it for the other
      two kinds: the rows stand in for the packaged catalog rather than layering
      over it per key, which is what lets a coordinate be *withdrawn*.
    - Decided here rather than in ``order_source_contents`` because runner's
      baseline is in-code, not a BUILTIN source row.
    - Decided on the *unfiltered* rows: a scoped query the document has nothing for
      comes back empty instead of falling back to images it replaced.
    """
    if overrides:
        return _filter_overrides(
            [_to_runner(override) for override in overrides], **filters
        )
    return list_runners(**filters)  # process-wide @lru_cache, static baseline


def merged_backend_runners(
    overrides: List[RunnerOverrideEntry], **filters
) -> BackendRunners:
    """Backend-grouped view of the merged runners (reuses the package grouping)."""
    return build_backend_runners(merged_runners(overrides, **filters))
