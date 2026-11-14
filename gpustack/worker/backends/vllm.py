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
from gpustack.utils import network
from gpustack_runtime.deployer import ContainerPort
from gpustack.utils.command import find_int_parameter
from gpustack_runtime.deployer import ContainerMount
from gpustack.utils.command import find_bool_parameter


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

        # Resolve image first so that backend_version is populated before
        # building command args (version-gated arguments depend on it).
        """取得镜像名，并准备 vllm serve 命令"""
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get vLLM backend image")

        """准备通信环境"""
        env = self._get_configured_env(
            is_distributed=deployment_metadata.distributed,
            deployment_metadata=deployment_metadata,
        )

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

        """解析出是 ray 还是 mp"""
        executor_backend = resolve_executor_backend(
            self._model.backend_parameters, self._model.backend_version
        )

        # Adjust run container for distributed follower (Ray path only).
        # For MP multi-node followers, --headless and topology args injected
        # by _build_command_args, so no command swap is needed.
        if deployment_metadata.distributed_follower and executor_backend == "ray":
            """
            Ray 主从容器安排
                - Ray 从节点：把默认容器改成“加入 Ray 集群”
            """

            if os.path.abspath(self._model_path) != os.path.abspath(self._model_instance.resolved_path):
                raise ValueError(
                    "Ray distributed inference requires the same absolute model path on every worker; "
                    "configure matching cache directories on the participating machines."
                )
            ray_command, ray_command_args, ray_ports = self._build_ray_configuration(
                is_leader=False,
            )
            """这里是为了把新的 Ray 命令正确映射到容器的 ENTRYPOINT/CMD 语义"""
            ray_command, ray_command_args = self._override_entrypoint(
                ray_command,
                ray_command_args,
                command_script,
            )

            run_container.execution.command = ray_command
            # run_container.execution.command_script = command_script # already set
            run_container.execution.args = ray_command_args
            run_container.ports = ray_ports

        # Create sidecar container for distributed leader (Ray path only).
        # Headless leader runs vLLM directly with DP coordination args; no
        # ray-head sidecar is required.
        sidecar_container = None
        if deployment_metadata.distributed_leader and executor_backend == "ray":
            """
            Ray 主从容器安排
                - Ray 主节点：保留 vLLM，再增加共享 /tmp 和 Ray head
            """

            run_container.mounts.append(
                ContainerMount(
                    path="/tmp",
                    volume="tmp-volume",
                ),
            )

            ray_command, ray_command_args, ray_ports = self._build_ray_configuration(
                is_leader=True,
            )
            ray_command, ray_command_args = self._override_entrypoint(
                ray_command,
                ray_command_args,
                command_script,
            )

            # Copy envs and override RAY_LOG_TO_STDERR for the sidecar
            # so Ray head logs go to stderr (captured by container log stream),
            # while keeping RAY_LOG_TO_STDERR=0 in the main container to avoid
            # polluting vLLM's log output with Ray worker logs.
            """复制环境变量，让 Ray head 单独输出日志"""
            sidecar_envs = list(run_container.envs)
            ray_stderr_found = False
            for i, e in enumerate(sidecar_envs):
                if e.name == "RAY_LOG_TO_STDERR":
                    sidecar_envs[i] = ContainerEnv(name="RAY_LOG_TO_STDERR", value="1")
                    ray_stderr_found = True
                    break
            if not ray_stderr_found:
                sidecar_envs.append(ContainerEnv(name="RAY_LOG_TO_STDERR", value="1"))

            """IMPORTANT: 创建真正的 ray-head 容器对象"""
            sidecar_container = Container(
                image=image,
                name="ray-head",
                profile=ContainerProfileEnum.RUN,
                restart_policy=ContainerRestartPolicyEnum.NEVER,
                execution=ContainerExecution(
                    privileged=True,
                    command=ray_command,
                    command_script=command_script,
                    args=ray_command_args,
                    run_as_user=container_config.user,
                    run_as_group=container_config.group,
                ),
                envs=sidecar_envs,
                resources=run_container.resources,
                mounts=run_container.mounts,
                ports=ray_ports,
            )

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
        准备通信环境
        """
        env = super()._get_configured_env()
        env["OMP_NUM_THREADS"] = env.pop("OMP_NUM_THREADS", "1")
        env["SAFETENSORS_FAST_GPU"] = env.pop("SAFETENSORS_FAST_GPU", "1")
        env["RUNAI_STREAMER_MEMORY_LIMIT"] = env.pop("RUNAI_STREAMER_MEMORY_LIMIT", "0")
        env["RUNAI_STREAMER_LOG_TO_STDERR"] = env.pop(
            "RUNAI_STREAMER_LOG_TO_STDERR", "1"
        )
        env["RUNAI_STREAMER_LOG_LEVEL"] = env.pop("RUNAI_STREAMER_LOG_LEVEL", "INFO")
        if is_distributed:
            """IMPORTANT: 多机时准备分布式环境，额外配置通信地址、通信端口、Ray 日志和网卡"""
            self._set_distributed_env(env, deployment_metadata)
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
            - 将 Ray 和 MP 的参数生成器接入命令构造
        """
        ctx = self._make_args_context(
            port, is_distributed, entrypoint, deployment_metadata
        )

        arguments: List[str] = [self._model_path]
        arguments = self.build_versioned_command_args(arguments)

        arguments.extend(self._build_max_model_len_arguments(ctx))
        arguments.extend(self._build_ascend_310p_arguments(ctx))
        """IMPORTANT: 准备 ray 分布式参数"""
        arguments.extend(self._build_ray_distributed_arguments(ctx))
        """准备 mp 多机参数"""
        arguments.extend(self._build_mp_multinode_arguments(ctx))
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

    def _resolve_multinode_shape(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
    ) -> Optional[str]:
        """
        为环境配置查询 MP 布局
        """
        try:
            return cal_multinode_topology(
                self._model_instance,
                deployment_metadata,
                parse_user_parallelism(self._model.backend_parameters),
            ).shape
        except ValueError:
            return None

    def _set_distributed_env(
        self,
        env: Dict[str, str],
        deployment_metadata: Optional[ModelInstanceDeploymentMetadata] = None,
    ):
        """
        | 执行方式       | 注入 `VLLM_PORT` | 注入 `VLLM_DP_MASTER_PORT` |
        |---            |---              |---                        |
        | Ray           | `ports[-1]`     | 不注入                      |
        | MP：`dp_only` | `ports[3]`       | `ports[-1]`               |
        | MP：`mp_only` | 不注入            | 不注入                     |
        | MP：`nested`  | 不注入            | `ports[-1]`               |
        """
        # Configure Internal communication IP and port.
        # see https://docs.vllm.ai/en/stable/configuration/env_vars.html.
        env["VLLM_HOST_IP"] = self._worker.ip

        ports = self._model_instance.ports or []
        executor_backend = resolve_executor_backend(
            self._model.backend_parameters, self._model.backend_version
        )
        if executor_backend == "mp":
            # VLLM_DP_MASTER_PORT belongs to the DP coordinator path
            # (vllm-project/vllm#42585). mp_only has dp=1 and no DP master,
            # so injecting it would bind a port for nobody to use.
            shape = (
                # 为环境配置查询 MP 布局
                self._resolve_multinode_shape(deployment_metadata)
                if deployment_metadata is not None
                else None
            )
            if shape in ("dp_only", "nested") and ports:
                env["VLLM_DP_MASTER_PORT"] = str(ports[-1])
            # Only dp_only needs a reserved start port for vLLM's TCPStore
            # (#5657); with nnodes > 1 that port comes from --master-port, and a
            # shared start port makes cross-node WorkerProcs racing for their
            # XPUB socket collide (#6019).
            if shape == "dp_only" and len(ports) > 3:
                env["VLLM_PORT"] = str(ports[3])
        else:
            """RAY"""
            if ports:
                env["VLLM_PORT"] = str(ports[-1])
            env["RAY_LOG_TO_STDERR"] = env.pop("RAY_LOG_TO_STDERR", "0")
            env["RAY_BACKEND_LOG_LEVEL"] = env.pop("RAY_BACKEND_LOG_LEVEL", "warning")

        if is_ascend(self._get_selected_gpu_devices()):
            # See https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi-node_dsv3.2.html.
            if "HCCL_SOCKET_IFNAME" not in env:
                env.setdefault("HCCL_IF_IP", self._worker.ip)
                env["HCCL_SOCKET_IFNAME"] = f"={self._worker.ifname}"
                env.setdefault("GLOO_SOCKET_IFNAME", self._worker.ifname)
                env.setdefault("TP_SOCKET_IFNAME", self._worker.ifname)
            return

        """NCCL 和 Gloo 用哪块网卡建立连接"""
        if "NCCL_SOCKET_IFNAME" not in env:
            env["NCCL_SOCKET_IFNAME"] = f"={self._worker.ifname}"
            env.setdefault("GLOO_SOCKET_IFNAME", self._worker.ifname)

    def _build_ray_distributed_arguments(self, ctx: _VLLMArgsContext) -> List[str]:
        """
        生成交给 vLLM 的 Ray 参数

        给 vllm serve 添加：--distributed-executor-backend ray
        用户配置 DP > 1 且没有指定 DP 后端时，还会默认补：--data-parallel-backend ray
        """
        if not ctx.is_distributed or ctx.executor_backend != "ray":
            return []

        # 先准备 --distributed-executor-backend ray
        arguments: List[str] = ["--distributed-executor-backend", "ray"]
        dps = find_int_parameter(
            self._model.backend_parameters, ["data-parallel-size", "dp"]
        )
        if dps and dps > 1:
            dpb = find_parameter(
                self._model.backend_parameters, ["data-parallel-backend", "dpb"]
            )
            if dpb is None:
                arguments.extend(["--data-parallel-backend", "ray"])
            # ports[1] is reserved for DP RPC, see gpustack/worker/serve_manager.py.
            arguments.extend(
                ["--data-parallel-rpc-port", str(self._model_instance.ports[1])]
            )
        return arguments

    def _build_mp_multinode_arguments(
        self,
        ctx: _VLLMArgsContext,
    ) -> List[str]:
        """
        根据 MP 布局生成节点角色和连接参数。

        ==========================================================================================

        读取 ctx.topology，分成三条路径：

        | 布局       | 含义                                   | 这里补充的主要参数                  |
        |---        |---                                     |---                               |
        | `dp_only` | 每个模型并行组都能放在单节点内，跨节点扩展 DP | DP 起始 rank、DP 地址、DP RPC 端口  |
        | `mp_only` | 一个模型并行组跨多个节点，DP 为 1           | 节点数、节点 rank、master 地址和端口 |
        | `nested`  | 模型并行组跨节点，同时还有多个 DP 组        | 节点参数与 DP 连接参数              |
        """
        if (
            not ctx.is_distributed
            or ctx.executor_backend != "mp"
            or ctx.topology is None
        ):
            return []

        topology = ctx.topology
        leader_ip = self._model_instance.worker_ip
        arguments: List[str] = []
        if (
            find_parameter(
                self._model.backend_parameters, ["distributed-executor-backend"]
            )
            is None
        ):
            arguments.extend(["--distributed-executor-backend", "mp"])

        # serve_manager._assign_ports reserves ports[1] for the DP RPC
        # endpoint (ZMQ) and ports[2] for the PyTorch master endpoint
        # (TCPStore). The two channels use different protocols and can't
        # share one port, so each shape consumes the subset it needs.
        dp_rpc_port = str(self._model_instance.ports[1])
        master_port = str(self._model_instance.ports[2])
        if topology.shape == "dp_only":
            extend_args_no_exist(
                arguments,
                ("--data-parallel-start-rank", str(topology.start_rank)),
                ("--data-parallel-address", leader_ip),
                ("--data-parallel-rpc-port", dp_rpc_port),
            )
        elif topology.shape == "mp_only":
            extend_args_no_exist(
                arguments,
                ("--nnodes", str(topology.nnodes)),
                ("--node-rank", str(topology.node_rank)),
                ("--master-addr", leader_ip),
                ("--master-port", master_port),
            )
        else:  # nested
            extend_args_no_exist(
                arguments,
                ("--nnodes", str(topology.nnodes)),
                ("--node-rank", str(topology.node_rank)),
                ("--master-addr", leader_ip),
                ("--master-port", master_port),
                ("--data-parallel-address", leader_ip),
                ("--data-parallel-rpc-port", dp_rpc_port),
            )

        # Hybrid-LB requires every DP engine (leader and followers alike) to
        # run its own API server, so skip the headless injection when the user
        # opts into it.
        hybrid_lb = find_bool_parameter(
            self._model.backend_parameters, ["data-parallel-hybrid-lb"]
        )
        if topology.is_follower and not hybrid_lb:
            extend_args_no_exist(arguments, "--headless")
        return arguments

    def _build_ray_configuration(
        self,
        is_leader: bool,
    ) -> (List[str], List[str], Optional[List[ContainerPort]]):
        """
        IMPORTANT: 生成 Ray 节点的启动命令和端口描述

        # 主节点
        ray start --head --port=41000 ... --block

        # 从节点
        ray start --address=主WorkerIP:41000 ... --block
        """
        # Parse the Ray port range from configuration,
        # assign ports in order as below:
        # 1.  GCS server port (the first port of the range)
        # 2.  Client port (reserved for compatibility, not used anymore, see https://github.com/gpustack/gpustack/issues/4171)
        # 3.  Dashboard port
        # 4.  Dashboard gRPC port (no longer used, since Ray 2.45.0 kept for backward compatibility)
        # 5.  Dashboard agent gRPC port
        # 6.  Dashboard agent listen port
        # 7.  Metrics export port
        # 8.  Node Manager port
        # 9.  Object Manager port
        # 10. Raylet runtime env agent port
        # 11. Minimum port number for the worker
        # 12. Maximum port number for the worker (the last port of the range)

        start, end = network.parse_port_range(self._config.ray_port_range)
        gcs_server_port = start
        # client_port = start + 1
        dashboard_port = start + 2
        dashboard_grpc_port = start + 3
        dashboard_agent_grpc_port = start + 4
        dashboard_agent_listen_port = start + 5
        metrics_export_port = start + 6
        node_manager_port = start + 7
        object_manager_port = start + 8
        raylet_runtime_env_agent_port = start + 9
        worker_port_min = start + 10
        worker_port_max = end

        command = [
            "ray",
            "start",
        ]
        arguments = [
            "--block",
            "--disable-usage-stats",
            "--verbose",
            f"--node-manager-port={node_manager_port}",
            f"--object-manager-port={object_manager_port}",
            f"--runtime-env-agent-port={raylet_runtime_env_agent_port}",
            f"--dashboard-agent-grpc-port={dashboard_agent_grpc_port}",
            f"--dashboard-agent-listen-port={dashboard_agent_listen_port}",
            f"--metrics-export-port={metrics_export_port}",
            f"--min-worker-port={worker_port_min}",
            f"--max-worker-port={worker_port_max}",
            f"--node-ip-address={self._worker.ip}",
        ]
        ports = [
            ContainerPort(
                internal=port,
            )
            for port in [
                dashboard_grpc_port,
                dashboard_agent_grpc_port,
                dashboard_agent_listen_port,
                metrics_export_port,
                node_manager_port,
                object_manager_port,
                raylet_runtime_env_agent_port,
            ]
        ]

        if is_leader:
            arguments.extend(
                [
                    "--head",
                    f"--port={gcs_server_port}",
                    f"--dashboard-host={self._worker.ip}",
                    f"--dashboard-port={dashboard_port}",
                ]
            )
            ports.extend(
                [
                    ContainerPort(
                        internal=port,
                    )
                    for port in [gcs_server_port, dashboard_port]
                ]
            )
        else:
            arguments.extend(
                [
                    f"--address={self._model_instance.worker_ip}:{gcs_server_port}",
                ]
            )

        return command, arguments, ports


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
