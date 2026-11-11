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
        topic = cls.__name__.lower()
        subscriber = event_bus.subscribe(topic, source=source, event_types=event_types)

        if replay_existing:
            include_created = event_types is None or EventType.CREATED in event_types
            if include_created:
                # 阶段版：每次订阅都查一遍数据库。第 7 章换成带缓存的 cached_all
                async with async_session() as session:
                    initial_items = await cls.all(session, options=options)
                for item in initial_items:
                    yield Event(type=EventType.CREATED, data=item)

        heartbeat_interval = timedelta(seconds=15)
        last_event_time = datetime.now(timezone.utc)

        try:
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

            return len(updates)
        except Exception as e:
            await session.rollback()
            raise e
