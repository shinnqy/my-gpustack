from gpustack.schemas.models import ModelInstanceDeploymentMetadata
from gpustack.utils.command import extend_args_no_exist
from typing import List
from typing import Optional
from typing import Tuple
from gpustack.schemas.models import ModelInstance
from gpustack.utils.command import find_parameter
from gpustack.utils.command import resolve_executor_backend
from gpustack.utils.vllm_topology import parse_user_parallelism
from gpustack.worker.backends.base import cal_distributed_parallelism_arguments
from gpustack.utils.envs import sanitize_env
from gpustack_runtime.deployer import Container
from gpustack_runtime.deployer import ContainerEnv
from gpustack_runtime.deployer import ContainerExecution
from gpustack_runtime.deployer import ContainerProfileEnum
from gpustack_runtime.deployer import ContainerRestartPolicyEnum
from gpustack_runtime.deployer import WorkloadPlan
from gpustack_runtime.deployer import create_workload
from typing import Dict
import os
from gpustack.worker.backends.base import is_ascend_310p
from gpustack.worker.backends.base import is_ascend
import logging
from gpustack.schemas.models import is_audio_model
from gpustack.schemas.models import is_omni_model
from gpustack.worker.backends.base import InferenceServer
from gpustack.utils.vllm_topology import MultinodeUserParallelism
from gpustack.utils.vllm_topology import validate_multinode_topology
from dataclasses import dataclass
from gpustack.utils.command import ExecutorBackend
from gpustack.utils.vllm_topology import MultinodeShape


logger = logging.getLogger(__name__)


@dataclass
class _VLLMArgsContext:
    """
    Derived inputs shared by every `_build_*_arguments` step.

    Computed once per `_build_command_args` invocation so individual builders
    stay side-effect-free and don't re-evaluate the same predicates.
    """

    port: int
    is_distributed: bool
    executor_backend: ExecutorBackend
    topology: Optional["MultinodeTopology"]
    is_omni: bool
    is_audio: bool
    entrypoint: Optional[List[str]]
    deployment_metadata: Optional[ModelInstanceDeploymentMetadata]


class VLLMServer(InferenceServer):
    """负责把模型部署配置转换成 vLLM 推理容器的启动计划"""

    def start(self):  # noqa: C901
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        """真正的启动流程"""

        logger.info(f"Starting vLLM model instance: {self._model_instance.name}")

        # Prepare distributed information.
        deployment_metadata = self._get_deployment_metadata()

        env = self._get_configured_env(
            is_distributed=deployment_metadata.distributed,
            deployment_metadata=deployment_metadata,
        )

        # Resolve image first so that backend_version is populated before
        # building command args (version-gated arguments depend on it).
        """取得镜像名，并准备 vllm serve 命令"""
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get vLLM backend image")

        command = ["vllm", "serve"]
        if self.inference_backend:
            entrypoint = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )
            if entrypoint:
                command = entrypoint

        command_script = self._get_serving_command_script(env)

        command_args, injected = self._build_command_args(
            port=self._get_serving_port(),
            is_distributed=deployment_metadata.distributed,
            entrypoint=command,
            deployment_metadata=deployment_metadata,
        )

        """将镜像、命令、GPU 资源和模型挂载目录放入容器配置，最终调用 create_workload()"""
        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=command,
            command_script=command_script,
            command_args=command_args,
            env=env,
            image=image,
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command: Optional[List[str]],
        command_script: Optional[str],
        command_args: List[str],
        env: Dict[str, str],
        image: str,
    ):
        command, command_args = self._override_entrypoint(
            command, command_args, command_script
        )

        resources = self._get_configured_resources()

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        # Read container config from environment variables
        container_config = self._get_container_env_config(env)

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
                command=command,
                command_script=command_script,
                args=command_args,
                run_as_user=container_config.user,
                run_as_group=container_config.group,
            ),
            envs=[
                ContainerEnv(
                    name=name,
                    value=value,
                )
                for name, value in env.items()
            ],
            resources=resources,
            mounts=mounts,
            ports=ports,
        )

        sidecar_container = None

        logger.info(f"Creating vLLM container workload: {deployment_metadata.name}")
        logger.info(
            f"With image: {image}, "
            f"command: [{' '.join(command) if command else ''}], "
            f"arguments: [{' '.join(command_args) if command_args else ''}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            # Host IPC only when a shared cache service is attached: its
            # CUDA-IPC transfer path imports the engine's KV buffers by IPC
            # handle. Everything else keeps the private /dev/shm sized by
            # shm_size.
            host_ipc=self._host_ipc_enabled(),
            shm_size=int(container_config.shm_size_gib * (1 << 30)),
            containers=(
                [run_container]
                if not sidecar_container
                else [run_container, sidecar_container]
            ),
            run_as_user=container_config.user,
            run_as_group=container_config.group,
        )
        create_workload(self._transform_workload_plan(workload_plan))

        logger.info(f"Created vLLM container workload: {deployment_metadata.name}")

    def _get_configured_env(
        self,
        is_distributed: bool,
        deployment_metadata: Optional[ModelInstanceDeploymentMetadata] = None,
    ) -> Dict[str, str]:
        """
        Get environment variables for vLLM service
        """
        env = super()._get_configured_env()
        env["OMP_NUM_THREADS"] = env.pop("OMP_NUM_THREADS", "1")
        env["SAFETENSORS_FAST_GPU"] = env.pop("SAFETENSORS_FAST_GPU", "1")
        env["RUNAI_STREAMER_MEMORY_LIMIT"] = env.pop("RUNAI_STREAMER_MEMORY_LIMIT", "0")
        env["RUNAI_STREAMER_LOG_TO_STDERR"] = env.pop(
            "RUNAI_STREAMER_LOG_TO_STDERR", "1"
        )
        env["RUNAI_STREAMER_LOG_LEVEL"] = env.pop("RUNAI_STREAMER_LOG_LEVEL", "INFO")
        if is_ascend(self._get_selected_gpu_devices()):
            self._set_ascend_env(env)
        return env

    def _set_ascend_env(self, env: Dict[str, str]):
        """
        Set up environment variables for Ascend devices.
        """

        # -- Optimize Pytorch NPU operations delivery performance.
        env["TASK_QUEUE_ENABLE"] = env.pop("TASK_QUEUE_ENABLE", "1")
        # -- Enable NUMA coarse-grained binding.
        env["CPU_AFFINITY_CONF"] = env.pop("CPU_AFFINITY_CONF", "1")
        # -- Reuse memory in multi-streams.
        env["PYTORCH_NPU_ALLOC_CONF"] = env.pop(
            "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"
        )
        # -- Increase HCCL connection timeout to avoid issues in large clusters.
        env["HCCL_CONNECT_TIMEOUT"] = env.pop("HCCL_CONNECT_TIMEOUT", "7200")
        # -- Enable RDMA PCIe direct post with no strict mode for better performance.
        env["HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT"] = env.pop(
            "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT", "TRUE"
        )
        if not is_ascend_310p(self._get_selected_gpu_devices()):
            # -- Disable HCCL execution timeout for better stability.
            env["HCCL_EXEC_TIMEOUT"] = env.pop("HCCL_EXEC_TIMEOUT", "0")
            # -- Enable the communication is scheduled by AI Vector directly with ROCE, instead of AI CPU.
            env["HCCL_OP_EXPANSION_MODE"] = env.pop("HCCL_OP_EXPANSION_MODE", "AIV")

    def _build_command_args(
        self,
        port: int,
        is_distributed: bool,
        entrypoint: Optional[List[str]] = None,
        deployment_metadata: Optional[ModelInstanceDeploymentMetadata] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        拼接 vLLM 启动参数
        """
        ctx = self._make_args_context(
            port, is_distributed, entrypoint, deployment_metadata
        )

        arguments: List[str] = [self._model_path]
        arguments = self.build_versioned_command_args(arguments)

        arguments.extend(self._build_max_model_len_arguments(ctx))
        arguments.extend(self._build_ascend_310p_arguments(ctx))
        arguments.extend(
            get_auto_parallelism_arguments(
                self._model.backend_parameters,
                self._model_instance,
                ctx.is_distributed,
                ctx.deployment_metadata,
                self._model.backend_version,
            )
        )

        # Inject user-defined backend parameters
        user_backend_parameters = self._flatten_backend_param()
        arguments.extend(user_backend_parameters)

        extend_args_no_exist(
            arguments,
            ("--host", self._worker.ip),
            ("--port", str(ctx.port)),
            ("--served-model-name", self._model_instance.model_name),
        )

        injected = self._get_injected_backend_parameters(
            arguments, user_backend_parameters, entrypoint
        )
        return arguments, injected

    def _make_args_context(
        self,
        port: int,
        is_distributed: bool,
        entrypoint: Optional[List[str]],
        deployment_metadata: Optional[ModelInstanceDeploymentMetadata],
    ) -> _VLLMArgsContext:
        executor_backend = resolve_executor_backend(
            self._model.backend_parameters, self._model.backend_version
        )
        topology: Optional[MultinodeTopology] = None
        if (
            is_distributed
            and executor_backend == "mp"
            and deployment_metadata is not None
        ):
            topology = cal_multinode_topology(
                self._model_instance,
                deployment_metadata,
                parse_user_parallelism(self._model.backend_parameters),
            )
        return _VLLMArgsContext(
            port=port,
            is_distributed=is_distributed,
            executor_backend=executor_backend,
            topology=topology,
            is_omni=is_omni_model(self._model),
            is_audio=is_audio_model(self._model),
            entrypoint=entrypoint,
            deployment_metadata=deployment_metadata,
        )

    def _build_max_model_len_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        if ctx.is_omni or ctx.is_audio:
            return []
        if (
            find_parameter(self._model.backend_parameters, ["max-model-len"])
            is not None
        ):
            return []
        derived_max_model_len = self._derive_max_model_len()
        if derived_max_model_len and derived_max_model_len > 8192:
            return ["--max-model-len", "8192"]
        return []

    def _build_ascend_310p_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        if not is_ascend_310p(self._get_selected_gpu_devices()):
            return []
        return ["--enforce-eager", "--dtype", "float16"]


@dataclass
class MultinodeTopology:
    """Resolved multi-node deployment topology from one node's perspective.

    Fields:
      - ``shape``: which set of vLLM arguments to emit (see
        :py:meth:`VLLMServer._build_mp_multinode_arguments`).
      - ``tp`` / ``pp``: cluster-wide, identical on every node.
      - ``dp``: total DP rank count across the whole cluster.
      - ``dpl``: ``--data-parallel-size-local`` value to emit on this node.
        In ``dp_only`` shape this may differ per node (heterogeneous clusters);
        in ``mp_only`` / ``nested`` it is always 1.
      - ``nnodes`` / ``node_rank``: physical cluster size and this node's
        position in it (0 = leader).
      - ``start_rank``: ``--data-parallel-start-rank`` to emit in ``dp_only``.
        In other shapes vLLM derives DP rank internally from
        ``node_rank % nnodes_within_dp``; we still surface the implied value
        for diagnostics.
      - ``is_follower``: whether this node should carry ``--headless``.
    """

    shape: MultinodeShape
    tp: int
    pp: int
    dp: int
    dpl: int
    nnodes: int
    node_rank: int
    start_rank: int
    is_follower: bool


def cal_multinode_topology(
    model_instance: ModelInstance,
    deployment_metadata: ModelInstanceDeploymentMetadata,
    user: Optional[MultinodeUserParallelism] = None,
) -> MultinodeTopology:
    """Translate user-supplied parallelism hints + physical topology into the
    vLLM parameter shape this node should emit.

    Delegates cluster-level validation and shape inference to
    :func:`validate_multinode_topology`, then layers on the node-perspective
    fields (``my_idx`` / ``start_rank`` / ``is_follower``).
    """
    g_main = len(model_instance.gpu_indexes) if model_instance.gpu_indexes else 1
    subordinate = (
        model_instance.distributed_servers.subordinate_workers
        if model_instance.distributed_servers
        and model_instance.distributed_servers.subordinate_workers
        else []
    )
    gpu_per_node = [g_main] + [len(s.gpu_indexes or []) for s in subordinate]
    nnodes = len(gpu_per_node)

    # This node's physical rank.
    if deployment_metadata.distributed_follower:
        my_idx = (deployment_metadata.distributed_follower_index or 0) + 1
        is_follower = True
    else:
        my_idx = 0
        is_follower = False
    if my_idx >= nnodes:
        raise ValueError(
            f"vLLM multi-node: follower index {my_idx} out of bounds "
            f"(cluster has {nnodes} workers)."
        )

    validated = validate_multinode_topology(gpu_per_node, user)

    if validated.shape == "dp_only":
        start_rank = sum(validated.dpl_per_node[:my_idx])
    else:
        # mp_only / nested: vLLM derives DP rank from node_rank // nnodes_within_dp.
        start_rank = my_idx // validated.nnodes_within_dp

    return MultinodeTopology(
        shape=validated.shape,
        tp=validated.tp,
        pp=validated.pp,
        dp=validated.dp,
        dpl=validated.dpl_per_node[my_idx],
        nnodes=nnodes,
        node_rank=my_idx,
        start_rank=start_rank,
        is_follower=is_follower,
    )


def get_auto_parallelism_arguments(
    backend_parameters: List[str],
    model_instance: ModelInstance,
    is_distributed: bool,
    deployment_metadata: Optional[ModelInstanceDeploymentMetadata] = None,
    backend_version: Optional[str] = None,
) -> List[str]:
    if (
        is_distributed
        and deployment_metadata is not None
        and resolve_executor_backend(backend_parameters, backend_version) == "mp"
    ):
        # MP multi-node: derive shape-specific defaults for tp/pp/dp/dpl
        # so the cluster works out-of-the-box; subordinate workers receive
        # the same backend_parameters, so any non-None field acts as a hard
        # cluster-wide constraint.
        user = parse_user_parallelism(backend_parameters)
        topology = cal_multinode_topology(model_instance, deployment_metadata, user)
        derived: List[str] = []
        if user.tp is None:
            derived.extend(["--tensor-parallel-size", str(topology.tp)])
        if user.pp is None and topology.pp > 1:
            derived.extend(["--pipeline-parallel-size", str(topology.pp)])
        # dp/dpl are only meaningful when the shape actually emits them
        # (dp_only and nested). mp_only operates with dp=1, dpl=1 implicit.
        if topology.shape in ("dp_only", "nested"):
            if user.dp is None:
                derived.extend(["--data-parallel-size", str(topology.dp)])
            if user.dpl is None:
                derived.extend(["--data-parallel-size-local", str(topology.dpl)])
        return derived

    parallelism = find_parameter(
        backend_parameters,
        [
            "tensor-parallel-size",
            "tp",
            "pipeline-parallel-size",
            "pp",
            "data-parallel-size",
            "dp",
        ],
    )

    if parallelism is not None:
        return []

    if is_distributed:
        # distributed across multiple workers (Ray sidecar path)
        (tp, pp) = cal_distributed_parallelism_arguments(model_instance)
        return [
            "--tensor-parallel-size",
            str(tp),
            "--pipeline-parallel-size",
            str(pp),
        ]

    if model_instance.gpu_indexes is not None and len(model_instance.gpu_indexes) > 1:
        # single worker with multiple GPUs
        return [
            "--tensor-parallel-size",
            str(len(model_instance.gpu_indexes)),
        ]

    return []
