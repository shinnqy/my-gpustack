import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional, Set

from gpustack.envs import EVENT_BUS_SUBSCRIBER_QUEUE_SIZE

# Re-export from coordinator.base for backward compatibility
from gpustack.server.coordinator.base import (
    Event,
    EventType,
    event_field,
    resolve_event_id,
)

logger = logging.getLogger(__name__)


def event_decoder(obj):
    if "type" in obj:
        obj["type"] = EventType[obj["type"]]
    return obj


class Subscriber:
    def __init__(
        self,
        topic: Optional[str] = None,
        source: Optional[str] = None,
        # 过滤：有的订阅者只关心删除事件，就在订阅时声明，不相关的事件**在入队前**就丢掉，不占队列位置
        event_types: Optional[Iterable[EventType]] = None,
        queue_size: Optional[int] = None,
    ):
        self.topic = topic
        self.source = source
        self.event_types: Optional[Set[EventType]] = (
            set(event_types) if event_types else None
        )
        self.queue: asyncio.Queue = asyncio.Queue(
            maxsize=(
                queue_size
                if queue_size is not None
                else EVENT_BUS_SUBSCRIBER_QUEUE_SIZE
            )
        )
        self.latest_by_key: Dict[Any, Event] = {}
        self.lock = asyncio.Lock()
        self._closed: bool = False

    def should_enqueue(self, event: Event) -> bool:
        """Pre-enqueue filter. Drops events the subscriber has opted out of."""
        if self.event_types is not None and event.type not in self.event_types:
            return False
        return True

    async def enqueue(self, event: Event):
        """
        这里**“先释放锁，再 `put`”**：如果拿着锁去 `put`，队列满时会一直拿着锁等待。别的行的事件也要拿这把锁，就全被卡住了。所以只在改字典时持锁。
        """
        if self._closed:
            return

        if not self.should_enqueue(event):
            return

        if event.type == EventType.UPDATED and event.id is not None:
            async with self.lock:
                if event.id in self.latest_by_key:
                    self.latest_by_key[event.id] = event
                    return
                self.latest_by_key[event.id] = event
            # Release the lock before awaiting put: a full queue would
            # otherwise serialize unrelated ids behind it.
            try:
                await self._put_with_backpressure(event)
            except BaseException:
                async with self.lock:
                    self.latest_by_key.pop(event.id, None)
                raise
            return

        await self._put_with_backpressure(event)

    async def _put_with_backpressure(self, event: Event):
        if self.queue.full():
            logger.warning(
                "Subscriber queue full, applying backpressure: "
                "source=%s topic=%s event_type=%s id=%s queue_size=%s",
                self.source,
                self.topic,
                event.type.name,
                event.id,
                self.queue.qsize(),
            )
        await self.queue.put(event)

    async def receive(self) -> Any:
        event = await self.queue.get()
        if event.type == EventType.UPDATED and event.id is not None:
            async with self.lock:
                return self.latest_by_key.pop(event.id, event)

        return event

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        putters = getattr(self.queue, "_putters", None)
        if putters is not None:
            for putter in list(putters):
                if not putter.done():
                    putter.cancel()
            putters.clear()


class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Subscriber]] = {}
        self._pending_tasks: Set[asyncio.Task] = set()

    def _spawn(self, coro) -> asyncio.Task:
        """``asyncio.create_task`` plus retain-and-discard bookkeeping."""
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    def subscribe(
        self,
        topic: str,
        source: Optional[str] = None,
        event_types: Optional[Iterable[EventType]] = None,
    ) -> Subscriber:
        subscriber = Subscriber(topic=topic, source=source, event_types=event_types)
        if topic not in self.subscribers:
            self.subscribers[topic] = []

        self.subscribers[topic].append(subscriber)
        return subscriber

    def _route_event(self, event: Event, topic: str):
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                self._spawn(subscriber.enqueue(event))

    def unsubscribe(self, topic: str, subscriber: Subscriber):
        if topic in self.subscribers:
            try:
                self.subscribers[topic].remove(subscriber)
            except ValueError:
                # Defensive: tolerate double-unsubscribe.
                pass
            if not self.subscribers[topic]:
                del self.subscribers[topic]
        subscriber.close()

    async def publish(self, topic: str, event: Event):
        self._route_event(event, topic)


event_bus = EventBus()