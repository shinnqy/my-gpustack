from gpustack import envs
from gpustack.client.generated_clientset import ClientSet
from gpustack.utils import network
from gpustack.utils.datetimex import parse_iso8601_to_utc
from gpustack.utils.runtime import is_benchmark_workload
from gpustack.utils.runtime import is_cache_service_workload
from gpustack_runtime.deployer import delete_workload
from gpustack_runtime.deployer import list_workloads
from typing import Callable
import logging


logger = logging.getLogger(__name__)


class WorkloadCleaner:
    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
    ):
        self._worker_id_getter = worker_id_getter
        self._clientset_getter = clientset_getter

    def _current_model_instance_names(self) -> set:
        names = set()
        model_instances_page = self._clientset.model_instances.list(
            params={"page": -1}, use_cache=False
        )
        for model_instance in model_instances_page.items or []:
            deployment_metadata = model_instance.get_deployment_metadata(
                self._worker_id,
            )
            if deployment_metadata:
                names.add(deployment_metadata.name)
        return names

    def cleanup_orphan_workloads(self):
        current_instance_names = self._current_model_instance_names()

        workloads = list_workloads()
        for w in workloads:
            # 本章尚未接入这两类资源的登记表，不能将它们当作孤儿删除。
            if is_benchmark_workload(w) or is_cache_service_workload(w):
                continue
            if not w.created_at:
                continue
            try:
                create_at = parse_iso8601_to_utc(w.created_at)
            except (TypeError, ValueError):
                logger.warning(f"Skipping workload {w.name}: creation time is unknown")
                continue
            should_clean_orphan, _ = network.is_offline(
                create_at, envs.WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD
            )
            if w.name not in current_instance_names and should_clean_orphan:
                delete_workload(w.name)
                logger.info(
                    f"Deleted orphan workload {w.name}, created at {w.created_at}."
                )
