from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    CREATED = 1
    UPDATED = 2
    DELETED = 3
    UNKNOWN = 4
    HEARTBEAT = 5

    def __str__(self):
        return self.name


@dataclass
class Event:
    # 创建、更新、删除。`HEARTBEAT` 不是数据变化，是“我还活着”的信号
    type: EventType
    # 变化后的那一行（一个 `Model` 或 `ModelInstance` 对象）
    data: Any
    # 哪些字段变了，旧值和新值是什么，形如 `{"state": (["pending"], ["scheduled"])}`。
    # 为什么需要它？比如调度器只关心“状态变成 pending”，不关心“下载进度从 30% 变 31%”。有了它，订阅者可以快速跳过无关的更新
    changed_fields: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    id: Optional[Any] = None

    def __post_init__(self):
        if isinstance(self.type, int):
            self.type = EventType(self.type)

        self.refresh_id()

    def refresh_id(self) -> None:
        if self.id is None:
            # 从 `data.id` 推出事件的 `id`
            self.id = self._derive_id_from_data()

    def _derive_id_from_data(self) -> Optional[Any]:
        if self.data is None:
            return None

        if hasattr(self.data, "id"):
            return getattr(self.data, "id")

        if isinstance(self.data, dict):
            return self.data.get("id")

        return None


def event_field(data: Any, name: str, default: Any = None) -> Any:
    if isinstance(data, Event):
        raise TypeError(
            "event_field takes Event.data, not the Event; "
            "use event_field(event.data, ...) or resolve_event_id(event)"
        )
    if data is None:
        return default
    if isinstance(data, dict):
        value = data.get(name)
    else:
        value = getattr(data, name, None)
    return default if value is None else value


def resolve_event_id(event: "Event") -> Optional[Any]:
    return event.id