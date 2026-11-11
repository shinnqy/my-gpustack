from gpustack.client import ClientSet
from gpustack.config.config import Config
from gpustack.server.bus import Event
from typing import Callable
import asyncio
import os
from gpustack.schemas.models import ModelInstance
import logging
from typing import Dict
from typing import Set
import time


logger = logging.getLogger(__name__)


class ServeManager:
    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._clientset_getter = clientset_getter

        self._provisioning_processes = {}
        self._log_persistence_threads = {}
        self._log_persistence_stop_events = {}
        self._error_model_instances = {}
        self._model_cache_by_instance = {}
        self._model_instance_by_instance_id = {}

        # Instance-level port tracking to avoid conflicts
        self._assigned_ports: Dict[int, Set[int]] = {}
        self._restart_backoff_counts: Dict[int, int] = {}

        # Inference health check failure tracking
        # {model_instance_id: failure_count}
        self._inference_health_check_failures: Dict[int, int] = {}

        # Track last successful inference per port (set by worker proxy)
        self._last_successful_inference: Dict[int, float] = {}
        # Track last health check time per model instance
        self._last_health_check_time: Dict[int, float] = {}

        # Timestamp of the last authoritative (uncached) DB reconciliation in
        # the state sync, for the optional periodic backstop. Starts "now" so
        # the first forced reconciliation is one full period in, not immediate.
        self._last_state_reconcile_time: float = time.time()

        os.makedirs(self._serve_log_dir, exist_ok=True)

    async def watch_models(self):
        """
        Loop to watch models to keep the cache updated.

        """

        logger.debug("Watching models.")

        while True:
            try:
                # Watch models without callback to keep the cache updated.
                await self._clientset.models.awatch(callback=None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching models: {e}")
                await asyncio.sleep(5)

    async def watch_model_instances_event(self):
        """
        Loop to watch model instances' event and handle.

        """

        logger.debug("Watching model instances event.")

        while True:
            try:
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching model instances: {e}")
                await asyncio.sleep(5)

    def _handle_model_instance_event(self, event: Event):
        """Handle a model instance event without ever crashing the watch stream.

        The awatch callback runs inline in the watch loop, so any exception
        that escapes here tears the stream down and forces a full reconnect
        plus cache reload. Swallow and log (with traceback) instead; the next
        event or the periodic state sync recovers the instance.
        """
        try:
            self._dispatch_model_instance_event(event)
        except Exception:
            logger.exception(
                f"Failed to handle {event.type} event for model instance "
                f"{getattr(event, 'id', None)}"
            )

    def _dispatch_model_instance_event(self, event: Event):
        mi = ModelInstance.model_validate(event.data)

        logger.trace(
            f"Received event: {str(event.type)}, id: {mi.id}, name: {mi.name}, state: {str(mi.state)}"
        )

        is_main_worker = mi.worker_id == self._worker_id
        if is_main_worker:
            logger.info(f"Model instance {mi.name} is assigned to me, state: {mi.state}")
