from typing import Optional
from gpustack_runtime import envs
from gpustack_runtime.deployer import DockerDeployer, WorkloadPlan
from gpustack_runtime.deployer.docker import DockerWorkloadPlan
from gpustack.config.config import Config
from gpustack.utils.config import apply_registry_override_to_image
from gpustack_runtime.deployer import WorkloadStatus


def transform_workload_plan(
    config: Config, workload: WorkloadPlan, fallback_registry: Optional[str] = None,
):
    """检测到 Docker 部署器可用时，将通用 WorkloadPlan 转换为 DockerWorkloadPlan ，补充 pause 等辅助镜像，并统一处理这些镜像的仓库地址"""

    if not DockerDeployer().is_supported():
        return workload
    return DockerWorkloadPlan(
        pause_image=apply_registry_override_to_image(
            config, envs.GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE, fallback_registry,
        ),
        unhealthy_restart_image=apply_registry_override_to_image(
            config, envs.GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE, fallback_registry,
        ),
        **workload.__dict__,
    )


def is_benchmark_workload(status: WorkloadStatus) -> bool:
    """
    Check if a workload is a benchmark workload.

    A workload is considered a benchmark workload if it has the 'type' label
    set to 'benchmark'.

    Args:
        status: The workload status to check.

    Returns:
        True if the workload is a benchmark workload, False otherwise.
    """
    if not status.labels:
        return False
    return status.labels.get("type") == "benchmark"


def is_cache_service_workload(status: WorkloadStatus) -> bool:
    """
    Check if a workload is a cache service workload.

    A workload is considered a cache service workload if it has the 'type'
    label set to 'cache-service'.

    Args:
        status: The workload status to check.

    Returns:
        True if the workload is a cache service workload, False otherwise.
    """
    if not status.labels:
        return False
    return status.labels.get("type") == "cache-service"
