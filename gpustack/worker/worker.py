import logging
from gpustack.config.config import Config
from gpustack.logging import setup_logging
from gpustack.utils.uuid import get_legacy_uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import aiohttp
from fastapi import FastAPI
import setproctitle
import tenacity
import uvicorn
from gpustack.config.config import (
    Config,
)
from gpustack import envs
from gpustack.routes import probes
from gpustack.client import ClientSet
from gpustack.utils.process import add_signal_handlers_in_loop
from gpustack.utils.task import run_periodically_in_thread
from gpustack.worker.worker_manager import WorkerManager
from gpustack.worker.collector import WorkerStatusCollector
from gpustack.config.registration import read_worker_token
from gpustack.utils.network import get_first_non_loopback_ip
from gpustack.utils.network import get_ifname_by_ip_hostname
from typing import Tuple
from urllib.parse import urlparse
from gpustack.utils.uuid import get_worker_name
from gpustack.worker.serve_manager import ServeManager
from gpustack.worker.model_file_manager import ModelFileManager
from gpustack.worker.inference_backend_manager import InferenceBackendManager


logger = logging.getLogger(__name__)


class Worker:
    _clientset: ClientSet

    _status_collector: WorkerStatusCollector

    _worker_manager: WorkerManager

    _config: Config

    _worker_ip: Optional[str] = None

    _worker_ifname: Optional[str] = None

    _worker_id: Optional[int] = None

    _worker_name: Optional[str] = None

    _worker_uuid: Optional[str] = None

    def worker_ip(self) -> str:
        return self._config.worker_ip or self._worker_ip

    def worker_ifname(self) -> str:
        return self._config.worker_ifname or self._worker_ifname

    def worker_name(self) -> Optional[str]:
        return (
            self._config.worker_name
            or self._worker_name
            or get_worker_name(self._config.data_dir)
        )

    def worker_uuid(self) -> str:
        return self._worker_uuid or get_legacy_uuid(self._config.data_dir) or ""

    def worker_id(self) -> int:
        return self._worker_id

    def clientset(self) -> ClientSet:
        return self._clientset

    def __init__(self, cfg: Config):
        self._config = cfg
        self._is_embedded = cfg.server_role() == Config.ServerRole.BOTH
        self._address = "0.0.0.0"
        self._async_tasks = []
        self._worker_ip, self._worker_ifname = self._detect_worker_ip_and_ifname()
        self._status_collector = WorkerStatusCollector(
            cfg=cfg,
            worker_ip_getter=self.worker_ip,
            worker_ifname_getter=self.worker_ifname,
            worker_id_getter=self.worker_id,
            worker_uuid_getter=self.worker_uuid,
        )
        self._worker_manager = WorkerManager(
            cfg=cfg,
            is_embedded=self._is_embedded,
            collector=self._status_collector,
        )
        self._serve_manager = ServeManager(
            worker_id_getter=self.worker_id,
            clientset_getter=self.clientset,
            cfg=self._config,
        )

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_fixed(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.debug(
            f"Retrying to get worker ID (attempt {retry_state.attempt_number}) due to: {retry_state.outcome.exception()}"
        ),
    )
    async def _register(self):
        self._clientset, self._default_config = (
            await self._worker_manager.register_with_server()
        )
        self._config.reload_worker_config(self._default_config)
        # Worker ID is available after the worker registration.
        worker_list = self._clientset.workers.list(
            params={"me": 'true'},
        )
        name = self.worker_name() or "<not specified>"
        if len(worker_list.items) != 1:
            raise Exception(f"Worker {name} not registered.")
        self._worker_id = worker_list.items[0].id
        self._worker_name = worker_list.items[0].name
        self._worker_uuid = worker_list.items[0].worker_uuid

    def _create_async_task(self, coro):
        self._async_tasks.append(asyncio.create_task(coro))

    def start(self):
        setup_logging(self._config.debug)

        if self._is_embedded:
            setproctitle.setproctitle("gpustack_worker")

        try:
            asyncio.run(self.start_async())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.error(f"Error serving worker APIs: {e}")
        finally:
            logger.info("Worker has shut down.")

    async def start_async(self):
        """
        Start the worker.
        """
        logger.info("Starting GPUStack worker.")
        add_signal_handlers_in_loop()
        await self._register()
        run_periodically_in_thread(self._heartbeat, envs.WORKER_HEARTBEAT_INTERVAL)
        run_periodically_in_thread(
            self._worker_manager.sync_worker_status, envs.WORKER_STATUS_SYNC_INTERVAL
        )
        self._create_async_task(self._serve_apis())
        inference_backend_manager = InferenceBackendManager(self._clientset)
        self._serve_manager._inference_backend_manager = inference_backend_manager
        self._create_async_task(self._serve_manager.watch_models())
        self._create_async_task(self._serve_manager.watch_model_instances_event())
        model_file_manager = ModelFileManager(
            worker_id=self._worker_id, clientset=self._clientset, cfg=self._config
        )
        self._create_async_task(model_file_manager.watch_model_files())
        await asyncio.sleep(0.5)
        logger.info("GPUStack worker startup completed.")
        await asyncio.gather(*self._async_tasks)

    async def _serve_apis(self):
        """
        Start the worker server to expose APIs.
        """

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            connector = aiohttp.TCPConnector(
                limit=envs.TCP_CONNECTOR_LIMIT,
                force_close=True,
            )
            app.state.http_client = aiohttp.ClientSession(
                connector=connector, trust_env=True
            )
            app.state.http_client_no_proxy = aiohttp.ClientSession(connector=connector)
            try:
                yield
            finally:
                await app.state.http_client.close()
                await app.state.http_client_no_proxy.close()

        app = FastAPI(
            title="GPUStack Worker",
            response_model_exclude_unset=True,
            lifespan=lifespan,
        )
        app.state.config = self._config
        app.state.token = read_worker_token(self._config.data_dir)
        app.state.worker_ip_getter = self.worker_ip
        app.include_router(probes.router)

        config = uvicorn.Config(
            app,
            host=self._address,
            port=self._config.get_api_port(self._is_embedded),
            access_log=False,
            log_level="error",
        )

        setup_logging()
        server = uvicorn.Server(config)

        await server.serve()

    def _detect_worker_ip_and_ifname(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect the worker IP and ifname.
        """
        static_worker_ip = self._config.worker_ip
        static_worker_ifname = self._config.worker_ifname
        detected_ifname = None
        detected_ip = None
        if static_worker_ip is not None and static_worker_ifname is not None:
            pass

        elif static_worker_ip is not None:
            # if ip is set, use it to detect ifname
            detected_ifname = get_ifname_by_ip_hostname(static_worker_ip)

        elif static_worker_ifname is not None:
            # if ifname is set, used it to detect ip
            detected_ip = get_first_non_loopback_ip(
                expected_ifname=static_worker_ifname
            )

        else:
            # detect both ip and ifname

            # detect_ifname may be None if the hostname resolves to a loopback address.
            # This typically happens when the worker and server run on the same host, or for embedded workers.
            detected_ifname = get_ifname_by_ip_hostname(
                urlparse(self._config.get_server_url()).hostname
            )

            try:
                # if the expected_ifname is none, it will scan all interfaces
                detected_ip = get_first_non_loopback_ip(expected_ifname=detected_ifname)
            except Exception:
                logger.warning(
                    f"Failed to detect worker IP from interface {detected_ifname}. Using first non-loopback IP."
                )
                # avoid edge case where the detected_ifname has no valid IPv4 address
                detected_ip = get_first_non_loopback_ip()

            if detected_ifname is None:
                detected_ifname = get_ifname_by_ip_hostname(detected_ip)
        return detected_ip, detected_ifname

    def _heartbeat(self):
        """
        Send heartbeat to the server to indicate the worker is alive.
        """
        if self._worker_id is None:
            logger.debug("Worker ID is not set, skipping heartbeat.")
            return
        try:
            resp = self._clientset.http_client.get_httpx_client().post(
                "/worker-heartbeat", json={}
            )
            if resp.status_code != 204:
                logger.error(
                    f"Failed to send heartbeat to server, status code: {resp.status_code}"
                )
        except Exception as e:
            logger.error(f"Failed to send heartbeat to server: {e}")
