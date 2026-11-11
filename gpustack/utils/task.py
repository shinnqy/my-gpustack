import logging
import threading
import time
from typing import Callable, Optional
from gpustack.utils.process import threading_stop_event


logger = logging.getLogger(__name__)


def run_periodically(
    func: Callable[[], None],
    interval: float = 5,
    initial_delay: float = 0,
    stop_event: Optional[threading.Event] = None,
    *args,
    **kwargs,
) -> None:
    if stop_event is None:
        stop_event = threading.Event()

    if initial_delay > 0:
        time.sleep(initial_delay)

    while not stop_event.is_set():
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error running {func.__name__}: {e}")
            if stop_event.is_set():
                break
        time.sleep(interval)


def run_periodically_in_thread(
    func: Callable,
    interval: float,
    initial_delay: float = 0,
    stop_event: Optional[threading.Event] = threading_stop_event,
    *args,
    **kwargs,
) -> threading.Thread:
    thread = threading.Thread(
        target=run_periodically,
        args=(func, interval, initial_delay, stop_event) + args,
        kwargs=kwargs,
        daemon=True,
    )
    thread.start()
    return thread
