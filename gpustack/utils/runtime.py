from typing import Optional

from gpustack_runtime import envs
from gpustack_runtime.deployer import DockerDeployer, WorkloadPlan
from gpustack_runtime.deployer.docker import DockerWorkloadPlan

from gpustack.config.config import Config
from gpustack.utils.config import apply_registry_override_to_image


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
