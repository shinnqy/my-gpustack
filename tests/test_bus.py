import asyncio

from gpustack.server.bus import EventBus, Event, EventType


class Row:
    def __init__(self, id, v):
        self.id, self.v = id, v


async def main():
    bus = EventBus()
    sub = bus.subscribe("t")
    for v in (10, 20, 30):
        await bus.publish("t", Event(type=EventType.UPDATED, data=Row(7, v)))
    await asyncio.sleep(0.1)  # 让投递任务跑完
    e = await sub.receive()
    assert e.data.v == 30, e.data.v  # 三次更新合并成最新一次
    assert sub.queue.empty()
    print("OK")


asyncio.run(main())