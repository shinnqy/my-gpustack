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
    type: EventType
    data: Any
    changed_fields: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    id: Optional[Any] = None

    def __post_init__(self):
        if isinstance(self.type, int):
            self.type = EventType(self.type)

        self.refresh_id()

    def refresh_id(self) -> None:
        if self.id is None:
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
