import asyncio
import logging


logger = logging.getLogger(__name__)


class AsyncUniqueQueue:
    """
    做去重，因为同一个实例可能被两条入口同时放进队列（刚创建时收到 CREATED 事件，同一时刻定时扫描也扫到了）。队列里有两份，就会被调度两次。所以用一个集合记住“队列里已经有谁”
    """
    def __init__(self):
        self.queue = asyncio.Queue()
        self.set = set()
        self.lock = asyncio.Lock()

    @staticmethod
    def _key(item):
        identifier = getattr(item, "id", None)
        return (type(item), identifier) if identifier is not None else item

    async def put(self, item):
        """在锁里检查集合，不在才放进去。锁是必要的：“检查”和“加入”之间如果被打断，两个协程可能都认为不在集合里。"""
        key = self._key(item)
        async with self.lock:
            if key not in self.set:
                self.set.add(key)
                await self.queue.put(item)
            else:
                logger.debug(f"Item {item.__hash__()} already in queue.")

    async def get(self):
        """取出后从集合里删掉。之后这个实例如果又需要调度（比如这次失败了），可以再次入队。"""
        item = await self.queue.get()
        async with self.lock:
            self.set.remove(self._key(item))
        return item

    def qsize(self):
        return self.queue.qsize()

    async def empty(self):
        return self.queue.empty()

    def task_done(self):
        self.queue.task_done()
