from datetime import datetime, timezone
import logging
import math
from typing import (
    Any,
    List,
    Optional,
    Union,
)
from sqlalchemy import func, inspect
from sqlmodel import SQLModel, and_, col, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm.exc import FlushError
from sqlalchemy.orm.state import InstanceState
from gpustack.schemas.common import PaginatedList, Pagination
import asyncio
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    AsyncGenerator,
    Iterable,
    List,
    Optional,
)
from sqlalchemy import event as sa_event, inspect
from sqlalchemy.orm import Session
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.db import async_session
from typing import Awaitable
from typing import Callable
import importlib
from fastapi.encoders import jsonable_encoder
import json
from gpustack.server.cache import class_key
from gpustack.server.cache import locked_cached
import anyio
from gpustack.server.cache import delete_cache_by_key


logger = logging.getLogger(__name__)


class CommitEvent:
    name: str
    event: Event

    def __init__(self, name: str, type: EventType, data: Any):
        self.name = name
        self.event = Event(type=type, data=data)


@sa_event.listens_for(Session, "before_flush")
def find_history(session: AsyncSession, flush_context, instances):
    events: List[CommitEvent] = session.info.get("pending_events", [])
    if len(events) == 0:
        return
    dirty_objs = [obj for obj in session.dirty if instances is None or obj in instances]
    for event in events:
        if event.event.data not in dirty_objs:
            continue
        obj = event.event.data
        relationship_keys = set(rel.key for rel in obj.__mapper__.relationships)
        state = inspect(obj)
        for attr in state.attrs:
            hist = attr.history
            if hist.has_changes():
                if attr.key in relationship_keys:
                    added_dump = [
                        obj.__class__.model_validate(obj.model_dump())
                        for obj in hist.added
                        if obj is not None
                    ]
                    deleted_dump = [
                        obj.__class__.model_validate(obj.model_dump())
                        for obj in hist.deleted
                        if obj is not None
                    ]
                    event.event.changed_fields[attr.key] = (deleted_dump, added_dump)
                else:
                    event.event.changed_fields[attr.key] = (hist.deleted, hist.added)


@sa_event.listens_for(Session, "after_commit")
def send_post_commit_events(session: AsyncSession):
    events: List[CommitEvent] = session.info.pop("pending_events", [])
    for event in events:
        # copy before submit to avoid mutation
        id = getattr(event.event.data, "id", None)
        logger.trace(f"Sending event {event.name} of type {event.event.type}, id {id}")
        bus_event = event.event
        try:
            bus_event.data = type(bus_event.data).model_validate(
                bus_event.data.model_dump(round_trip=True)
            )
            bus_event.refresh_id()
            asyncio.create_task(event_bus.publish(event.name, bus_event))
        except Exception as e:
            logger.exception(f"Failed to publish events: {e}")


class ActiveRecordMixin:
    """ActiveRecordMixin provides a set of methods to interact with the database."""

    __config__ = None

    @classmethod
    async def one_by_id(
        cls,
        session: AsyncSession,
        id: int,
        for_update: bool = False,
        options: Optional[List] = None,
    ):
        """Return the object with the given id. Return None if not found.

        If `for_update` is True, the row will be locked until the end of the transaction.
        If `options` is provided, it will be passed to the query for eager loading relationships.
        """

        return await session.get(cls, id, with_for_update=for_update, options=options)

    @classmethod
    async def one_by_field(
        cls,
        session: AsyncSession,
        field: str,
        value: Any,
        options: Optional[List] = None,
    ):
        """Return the object with the given field and value. Return None if not found."""

        return await cls.one_by_fields(session, {field: value}, options=options)

    @classmethod
    async def one_by_fields(
        cls, session: AsyncSession, fields: dict, options: Optional[List] = None
    ):
        """Return the object with the given fields and values. Return None if not found."""

        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(getattr(cls, key) == value)

        if options:
            statement = statement.options(*options)

        result = await session.exec(statement)
        return result.first()

    @classmethod
    async def all_by_field(
        cls, session: AsyncSession, field: str, value: Any, for_update: bool = False
    ):
        """
        Return all objects with the given field and value.
        Return an empty list if not found.
        """
        statement = select(cls).where(getattr(cls, field) == value)
        if for_update:
            statement = statement.with_for_update()

        result = await session.exec(statement)
        return result.all()

    @classmethod
    async def all_by_fields(
        cls,
        session: AsyncSession,
        fields: Optional[dict] = None,
        fuzzy_fields: Optional[dict] = None,
        extra_conditions: Optional[List] = None,
        options: Optional[List] = None,
    ):
        """
        Return all objects with the given fields and values.
        Return an empty list if not found.
        """
        if fields is None:
            fields = {}
        statement = select(cls)
        for key, value in fields.items():
            statement = statement.where(getattr(cls, key) == value)

        if fuzzy_fields:
            fuzzy_conditions = [
                func.lower(getattr(cls, key)).like(f"%{str(value).lower()}%")
                for key, value in fuzzy_fields.items()
            ]
            statement = statement.where(or_(*fuzzy_conditions))

        if extra_conditions:
            statement = statement.where(and_(*extra_conditions))

        if options:
            statement = statement.options(*options)

        result = await session.exec(statement)
        return result.all()

    @classmethod
    async def paginated_by_query(
        cls,
        session: AsyncSession,
        fields: Optional[dict] = None,
        fuzzy_fields: Optional[dict] = None,
        extra_conditions: Optional[List] = None,
        page: int = 1,
        per_page: int = 100,
        options: Optional[List] = None,
    ) -> PaginatedList[SQLModel]:
        statement = select(cls)
        if fields:
            conditions = [
                col(getattr(cls, key)) == value for key, value in fields.items()
            ]
            statement = statement.where(and_(*conditions))

        if fuzzy_fields:
            conditions = [
                func.lower(getattr(cls, key)).like(f"%{str(value).lower()}%")
                for key, value in fuzzy_fields.items()
            ]
            statement = statement.where(or_(*conditions))

        if extra_conditions:
            statement = statement.where(and_(*extra_conditions))

        if options:
            statement = statement.options(*options)

        if page is not None and page > 0 and per_page is not None:
            statement = statement.offset((page - 1) * per_page).limit(per_page)
        items = (await session.exec(statement)).all()

        count_statement = select(func.count(cls.id))
        if fields:
            conditions = [
                col(getattr(cls, key)) == value for key, value in fields.items()
            ]
            count_statement = count_statement.where(and_(*conditions))

        if fuzzy_fields:
            conditions = [
                func.lower(getattr(cls, key)).like(f"%{str(value).lower()}%")
                for key, value in fuzzy_fields.items()
            ]
            count_statement = count_statement.where(or_(*conditions))

        if extra_conditions:
            count_statement = count_statement.where(and_(*extra_conditions))

        count = (await session.exec(count_statement)).one()
        total_page = math.ceil(count / per_page)
        pagination = Pagination(
            page=page,
            perPage=per_page,
            total=count,
            totalPage=total_page,
        )

        return PaginatedList[cls](items=items, pagination=pagination)

    @classmethod
    def convert_without_saving(
        cls, source: Union[dict, SQLModel], update: Optional[dict] = None
    ) -> SQLModel:
        """
        Convert the source to the model without saving to the database.
        Return None if failed.
        """

        if isinstance(source, cls):
            obj = source
            if update:
                for k, v in update.items():
                    setattr(obj, k, v)
        elif isinstance(source, SQLModel):
            obj = cls.from_orm(source, update=update)
        elif isinstance(source, dict):
            obj = cls.parse_obj(source, update=update)
        return obj

    async def _refresh_related_objects(self, session: AsyncSession):
        """Refresh all related objects of the given object."""

        for rel in self.__mapper__.relationships:
            if rel.direction.name != "MANYTOONE":
                continue
            if not hasattr(self, rel.key):
                continue
            rel_obj = getattr(self, rel.key, None)
            if rel_obj is None:
                continue
            elif isinstance(rel_obj, list) and len(rel_obj) == 0:
                continue
            elif isinstance(rel_obj, InstanceState):
                continue
            await session.refresh(rel_obj)

    def _get_flush_targets(self, session: AsyncSession) -> List[SQLModel]:
        # always needs to flush self
        rtn = [self]
        state: InstanceState = inspect(self)
        dirty_objs = [obj for obj in session.dirty]
        for rel in self.__mapper__.relationships:
            if rel.direction.name != "MANYTOMANY":
                continue
            attr = state.attrs[rel.key]
            if not attr.history.has_changes():
                continue
            for obj in attr.value:
                if obj in dirty_objs:
                    rtn.append(obj)
            for obj in attr.history.deleted:
                if obj in dirty_objs:
                    rtn.append(obj)
        return rtn

    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        source: Union[dict, SQLModel],
        update: Optional[dict] = None,
        auto_commit: bool = True,
    ) -> Optional[SQLModel]:
        """Create and save a new record for the model."""

        obj = cls.convert_without_saving(source, update)
        if obj is None:
            return None

        cls._publish_event_after_commit(session, EventType.CREATED, obj)
        await obj.save(session, auto_commit=auto_commit)
        return obj

    async def refresh(self, session: AsyncSession):
        """Refresh the object from the database."""

        await session.refresh(self)

    async def save(self, session: AsyncSession, auto_commit=True):
        """Save the object to the database. Raise exception if failed."""

        session.add(self)
        try:
            targets = self._get_flush_targets(session)
            await session.flush(targets)

            if auto_commit:
                await session.commit()
                await session.refresh(self)
            if session.is_active:
                await self._refresh_related_objects(session)
            # Invalidate cached_all cache on successful write
            await self.__class__._invalidate_cached_all()
        except (IntegrityError, OperationalError, FlushError) as e:
            await session.rollback()
            raise e
        except Exception as e:
            await session.rollback()
            raise e

    async def update(
        self,
        session: AsyncSession,
        source: Union[dict, SQLModel, None] = None,
        auto_commit=True,
    ):
        """Update the object with the source and save to the database."""

        if isinstance(source, SQLModel):
            source = {
                key: getattr(source, key, None) for key in source.model_fields_set
            }
        elif source is None:
            source = {}

        for key, value in source.items():
            setattr(self, key, value)
        self._publish_event_after_commit(session, EventType.UPDATED, self)
        await self.save(session, auto_commit=auto_commit)

    async def delete(self, session: AsyncSession, soft=False, auto_commit=True):
        """Delete the object from the database."""
        self._publish_event_after_commit(session, EventType.DELETED, self)

        if soft or self._has_cascade_delete():
            if hasattr(self, "deleted_at"):
                # timestamp is stored without timezone in db
                self.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
                await self.save(session, auto_commit=False)
            await self._handle_cascade_delete(session, soft=soft, auto_commit=False)
        if not soft:
            await session.delete(self)
        if not auto_commit:
            return
        await session.commit()
        # Invalidate cached_all cache after successful delete
        await self.__class__._invalidate_cached_all()

    async def _handle_cascade_delete(
        self, session: AsyncSession, soft=False, auto_commit=True
    ):
        """Handle cascading deletes for all defined relationships."""
        for rel in self.__mapper__.relationships:
            if rel.cascade.delete:
                # Load the related objects
                await session.refresh(self)
                related_objects = getattr(self, rel.key)

                # Delete each related object
                if isinstance(related_objects, list):
                    for related_object in related_objects:
                        await related_object.delete(
                            session, soft=soft, auto_commit=auto_commit
                        )
                elif related_objects:
                    await related_objects.delete(
                        session, soft=soft, auto_commit=auto_commit
                    )

    def _has_cascade_delete(self):
        """Check if the model has cascade delete relationships."""
        return any(rel.cascade.delete for rel in self.__mapper__.relationships)

    @classmethod
    def _publish_event_after_commit(
        cls, session: AsyncSession, event_type: str, data: Any
    ):
        session.info.setdefault("pending_events", []).append(
            CommitEvent(
                name=cls.__name__.lower(),
                type=event_type,
                data=data,
            )
        )

    @classmethod
    async def all(cls, session: AsyncSession, options: Optional[List] = None):
        """Return all objects of the model."""
        statement = select(cls)
        if options:
            statement = statement.options(*options)
        result = await session.exec(statement)
        return result.all()

    @classmethod
    async def subscribe(
        cls,
        source: str,
        options: Optional[List] = None,
        event_types: Optional[Iterable[EventType]] = None,
        replay_existing: bool = True,
    ) -> AsyncGenerator[Event, None]:
        """Subscribe to bus events for this model.

        ``source`` labels the consumer in queue-full logs. ``event_types``
        whitelists pre-enqueue, so filtered events don't take queue slots.
        ``replay_existing=False`` skips the initial CREATED snapshot for
        consumers that bootstrap themselves.
        """
        topic = cls.__name__.lower()
        subscriber = event_bus.subscribe(topic, source=source, event_types=event_types)
        logger.info(
            "subscribed, source=%s topic=%s subscriber=%s",
            source,
            topic,
            id(subscriber),
        )

        try:
            if replay_existing:
                include_created = event_types is None or EventType.CREATED in event_types
                if include_created:
                    initial_items = await cls.cached_all(options=options)
                    for item in initial_items:
                        yield Event(type=EventType.CREATED, data=item)

            heartbeat_interval = timedelta(seconds=15)
            last_event_time = datetime.now(timezone.utc)

            while True:
                try:
                    event = await asyncio.wait_for(
                        subscriber.receive(), timeout=heartbeat_interval.total_seconds()
                    )
                    yield event
                except asyncio.TimeoutError:
                    if (
                        datetime.now(timezone.utc) - last_event_time
                        >= heartbeat_interval
                    ):
                        yield Event(type=EventType.HEARTBEAT, data=None)
                        last_event_time = datetime.now(timezone.utc)
        finally:
            event_bus.unsubscribe(cls.__name__.lower(), subscriber)

    @classmethod
    async def batch_update(
        cls,
        session: AsyncSession,
        updates: List[SQLModel],
        auto_commit: bool = True,
    ) -> int:
        if not updates:
            return 0

        try:
            for obj in updates:
                cls._publish_event_after_commit(session, EventType.UPDATED, obj)
                session.add(obj)

            if auto_commit:
                await session.commit()

            # Invalidate cached_all cache after successful batch update
            await cls._invalidate_cached_all()
            return len(updates)
        except Exception as e:
            await session.rollback()
            raise e

    @classmethod
    async def streaming(
        cls,
        fields: Optional[dict] = None,
        fuzzy_fields: Optional[dict] = None,
        filter_func: Optional[Callable[[Any], bool]] = None,
        options: Optional[List] = None,
        event_transform: Optional[Callable[[Event], Awaitable[None]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream events matching the given criteria as JSON strings.

        Args:
            fields: Exact match filters as key-value pairs
            fuzzy_fields: Fuzzy match filters
            filter_func: Optional filter function to apply to event data
            options: SQLAlchemy options for eager loading relationships (e.g., selectinload)
            event_transform: Optional async hook called with the public Event
                right before serialization. May mutate ``event.data`` in
                place to inject fields derived from outside the raw
                subscribe payload — used e.g. by /v2/workers to inject
                allocated computed from current ModelInstance bindings so
                the watch stream matches the REST response.
        """
        subscription = cls.subscribe(source="streaming", options=options)
        try:
            async for event in subscription:
                if event.type == EventType.HEARTBEAT:
                    yield "\n\n"
                    continue

                if not cls._match_fields(event, fields):
                    continue

                if not cls._match_fuzzy_fields(event, fuzzy_fields):
                    continue

                if filter_func and not filter_func(event.data):
                    continue

                public_event = Event(
                    type=event.type,
                    data=cls._convert_to_public_class(event.data),
                    changed_fields=event.changed_fields,
                    id=event.id,
                )
                if event_transform is not None:
                    try:
                        await event_transform(public_event)
                    except Exception as e:
                        logger.error(
                            f"event_transform failed for {cls.__name__} "
                            f"event {event.id}: {e}"
                        )
                formatted = cls._format_event(public_event)
                if formatted is not None:
                    yield formatted
        except asyncio.CancelledError:
            # Re-raise to ensure proper structured concurrency cancellation state
            raise
        except Exception as e:
            logger.error(f"Error in streaming {cls.__name__}: {e}")
        finally:
            await subscription.aclose()

    @classmethod
    def _match_fields(cls, event: Any, fields: Optional[dict]) -> bool:
        """Match fields using AND condition.

        ``deleted_at: None`` is exempt on a DELETED event. Callers pass it to
        mean "stream the live rows", but :meth:`delete` stamps ``deleted_at``
        before the after-commit hook publishes, so matching it here dropped the
        one event that says the row is gone — leaving it rendered on every open
        page until a reload. :meth:`_format_event` already assumes the opposite,
        keeping id-only payloads for DELETED alone so "clients can still remove
        by ID from their cache".

        The exemption is per-key, not per-event: every other filter still
        applies, so a stream scoped to one cluster (or one name) never learns
        about another's deletions. And only ``None`` is exempt — a concrete
        timestamp would mean "this retirement", not "live rows".
        """
        for key, value in (fields or {}).items():
            if (
                key == "deleted_at"
                and value is None
                and event.type == EventType.DELETED
            ):
                continue
            if getattr(event.data, key, None) != value:
                return False
        return True

    @classmethod
    def _match_fuzzy_fields(cls, event: Any, fuzzy_fields: Optional[dict]) -> bool:
        """Match fuzzy fields using OR condition."""
        for key, value in (fuzzy_fields or {}).items():
            attr_value = str(getattr(event.data, key, "")).lower()
            if str(value).lower() in attr_value:
                return True
        return not fuzzy_fields

    @classmethod
    def _convert_to_public_class(cls, data: Any) -> Any:
        """Convert the instance to the corresponding Public class if it exists."""
        # If data is a dict (e.g., ID-only event), return as-is
        if isinstance(data, dict):
            return data
        class_module = importlib.import_module(cls.__module__)
        public_class = getattr(class_module, f"{cls.__name__}Public", None)
        return public_class.model_validate(data) if public_class else data

    @staticmethod
    def _format_event(event: Any) -> Optional[str]:
        """Format the event as a JSON string."""
        # Skip ID-only events for CREATED/UPDATED - they can't be properly validated by clients
        # But allow ID-only for DELETED - clients can still remove by ID from their cache
        if (
            event.type != EventType.DELETED
            and isinstance(event.data, dict)
            and set(event.data.keys()) == {"id"}
        ):
            return None
        return json.dumps(jsonable_encoder(event), separators=(",", ":")) + "\n\n"

    @classmethod
    async def _do_cached_all_query(cls, options: Optional[List] = None):
        """Execute the cached_all query in a shielded context.

        This runs the entire database operation including session cleanup
        in a way that's protected from anyio cancellation.
        """
        session = async_session()
        try:
            results = await cls.all(session, options=options)
            for item in results:
                session.expunge(item)
            return results
        finally:
            await session.close()

    @classmethod
    @locked_cached(key=class_key("cached_all"))
    async def cached_all(cls, options: Optional[List] = None):
        """Return all objects with caching for subscribe() initial data loading."""
        logger.debug(f"Loading cached {cls.__name__} with options={options}")
        # Run the entire database operation in a shielded context to protect
        # from anyio cancellation. This prevents connection pool issues when
        # CancelledError interrupts database operations or session cleanup.
        with anyio.CancelScope(shield=True):
            return await cls._do_cached_all_query(options)

    @classmethod
    async def _invalidate_cached_all(cls):
        """Invalidate cached_all cache for this model class."""
        cache_key = class_key("cached_all")(None, cls)
        await delete_cache_by_key(_key=cache_key)
