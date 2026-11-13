from gpustack.client.generated_clientset import ClientSet
from gpustack.config.config import Config
from gpustack.config.config import set_global_config
from gpustack.logging import setup_logging
from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.models import BackendEnum
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.profiling import time_decorator
from gpustack_runtime.logging import setup_logging as setup_runtime_logging
from typing import Optional
import sys
from gpustack.server.bus import Event, EventType
from pathlib import Path
from gpustack.schemas.models import ModelInstanceUpdate
from gpustack.schemas.models import ModelInstanceDeploymentMetadata
from functools import lru_cache
from gpustack.schemas.workers import GPUDevicesStatus
from typing import Tuple
from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.deployer import ContainerResources
from gpustack_runtime.deployer import ContainerMount
from typing import List
import os
from gpustack_runtime.deployer import ContainerPort
from gpustack.schemas.inference_backend import ContainerEnvConfig
from typing import Dict
from gpustack.utils.envs import filter_env_vars
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.schemas.runner_source import merged_backend_runners
from gpustack.utils import platform
from gpustack.utils.version import pick_runtime_version
from gpustack_runner.runner import BackendVersionedRunner
from gpustack_runtime.detector import available_backends
from gpustack_runtime.detector.ascend import get_ascend_cann_variant
from gpustack.schemas.runner_source import RunnerOverrideEntryPublic
from gpustack_runtime.detector import ManufacturerEnum
from transformers import PretrainedConfig
import logging
from abc import ABC
from gpustack.schemas.models import ModelUpdate
from gpustack.schemas.models import get_backend
from gpustack.utils.hub import get_pretrained_config, get_hf_text_config, get_max_model_len
from gpustack.utils.runtime import transform_workload_plan
import shlex
from gpustack.utils.command import flatten_to_argv, is_parameter_key
from gpustack.schemas.inference_backend import ParameterFormatEnum


logger = logging.getLogger(__name__)


class ModelInstanceStateError(Exception):
    pass


def _normalize_param_format(
    tokens: List[str], target: ParameterFormatEnum
) -> List[str]:
    """
    Walk an argv-style token stream, regroup each ``--key [value...]`` cluster,
    and emit each cluster in ``target`` format. The key's leading dashes are
    preserved verbatim — ``-n`` / ``-ngl`` / ``-m`` are real llama.cpp short
    options and must not be coerced into ``--n`` / ``--ngl`` / ``--m``.

    Multi-value clusters (``--lora-modules v1 v2``) always stay in space form —
    ``--key=v1 --key=v2`` would change argparse semantics (the later value
    overwrites the earlier one), so equal form is unsafe to use for them.

    Stray positional tokens that do not follow a key (rare; only happens on
    malformed input) pass through verbatim — let the inference server reject
    them rather than guessing.
    """
    result: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not is_parameter_key(tok):
            result.append(tok)
            i += 1
            continue

        if "=" in tok:
            head_key, _, head_val = tok.partition("=")
            values = [head_val]
            i += 1
        else:
            head_key = tok
            values = []
            i += 1
            while i < n and not is_parameter_key(tokens[i]):
                values.append(tokens[i])
                i += 1

        if not values:
            result.append(head_key)
        elif len(values) == 1 and target == ParameterFormatEnum.EQUAL:
            result.append(f"{head_key}={values[0]}")
        elif len(values) == 1:
            result.extend([head_key, values[0]])
        else:
            result.append(head_key)
            result.extend(values)

    return result


class InferenceServer(ABC):
    """
    公共基类，供 VLLMServer 等推理后端复用
    最终为子类提供：镜像、GPU 请求、模型挂载、端口、环境变量和启动参数
    """

    _model_path: Optional[str] = None

    _draft_model_path: Optional[str] = None

    _pretrained_config: Optional[PretrainedConfig] = None

    _pretrained_config_initialized: bool = False

    _fallback_registry: Optional[str] = None

    _runner_overrides: Optional[List[RunnerOverrideEntryPublic]] = None

    @time_decorator
    def __init__(
        self,
        clientset: ClientSet,
        # 这个副本分到了哪些 GPU、端口和模型路径
        mi: ModelInstance,
        cfg: Config,
        worker_id: int,
        # 后端的镜像、命令和版本配置
        inference_backend: InferenceBackend,
        fallback_registry: Optional[str] = None,
    ):
        setup_logging(debug=cfg.debug)
        setup_runtime_logging()
        set_global_config(cfg)

        try:
            self._clientset = clientset
            self._model_instance = mi
            self._config = cfg
            self._fallback_registry = fallback_registry
            self._worker = self._clientset.workers.get(worker_id)
            if not self._worker:
                raise KeyError(f"Worker {worker_id} not found")

            self.get_model()
            self.inference_backend = inference_backend
            if (
                not inference_backend
                and self._model.image_name
                and self._model.run_command
            ):
                # Any deployment that directly specifies an image and command is treated as a Custom backend.
                # A basic InferenceBackend object is created to prevent exceptions in subsequent workflows.
                self.inference_backend = InferenceBackend(
                    backend_name=BackendEnum.CUSTOM.value,
                    run_command=self._model.run_command,
                )
            if not self.inference_backend:
                raise KeyError(
                    f"Inference backend {self._model.backend} not specified or not found"
                )

            logger.info("Preparing model files...")

            # 通过 Watch 等待实例变化
            self._until_model_instance_starting()

            logger.info("Model files are ready.")
        except ModelInstanceStateError:
            sys.exit(1)
        except Exception as e:
            error_message = f"Failed to initialize: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(mi.id, **patch_dict)
            except Exception as ue:
                logger.error(f"Failed to update model instance: {ue}")
            sys.exit(1)

    def _stop_when_starting(self, event: Event) -> bool:
        if event.type == EventType.DELETED:
            raise ModelInstanceStateError()
        instance = ModelInstance.model_validate(event.data)
        if instance.state == ModelInstanceStateEnum.ERROR:
            raise ModelInstanceStateError()
        elif instance.state == ModelInstanceStateEnum.STARTING:
            resolved_path = instance.resolved_path
            if not resolved_path:
                raise ValueError(
                    "Model instance reached STARTING without a resolved model path"
                )
            self._model_path = str(Path(resolved_path).absolute())
            if instance.draft_model_resolved_path:
                self._draft_model_path = str(
                    Path(instance.draft_model_resolved_path).absolute()
                )
            self._model_instance = instance
            return True

        return False

    def get_model(self):
        model = self._clientset.models.get(id=self._model_instance.model_id)
        data_dir = self._config.data_dir
        for i, param in enumerate(model.backend_parameters or []):
            model.backend_parameters[i] = param.replace("{data_dir}", data_dir)

        self._model = model
        self._model.backend = get_backend(model)

    def _derive_max_model_len(self, default: Optional[int] = None) -> Optional[int]:
        meta = self._model.meta or {}
        value = meta.get("max_model_len") or meta.get("context_length")
        if value:
            return int(value)
        try:
            config = get_pretrained_config(self._model)
            return get_max_model_len(get_hf_text_config(config))
        except Exception as error:
            logger.warning(f"Failed to derive max model length: {error}")
            return default

    def _host_ipc_enabled(self) -> bool:
        value = (self._model.env or {}).get("GPUSTACK_HOST_IPC", os.getenv("GPUSTACK_HOST_IPC", "0"))
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _get_serving_command_script(self, env: Dict[str, str]) -> Optional[str]:
        disabled = str(env.get("GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED", "0")).lower()
        if disabled in {"1", "true", "yes", "on"} or not env.get("PYPI_PACKAGES_INSTALL"):
            return None
        return '''#!/bin/sh
set -e
if command -v uv >/dev/null 2>&1; then
    uv pip install --system ${PYPI_PACKAGES_INSTALL}
else
    pip install ${PYPI_PACKAGES_INSTALL}
fi
unset PYPI_PACKAGES_INSTALL
exec "$@"
'''

    def _transform_workload_plan(self, workload):
        return transform_workload_plan(self._config, workload, self._fallback_registry)

    def _until_model_instance_starting(self):
        """通过 Watch 等待实例变化"""

        self._clientset.model_instances.watch(
            callback=None,
            stop_condition=self._stop_when_starting,
            params={"id": self._model_instance.id},
        )

    def _update_model_instance(self, id: int, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            setattr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)

    def _handle_error(self, error: Exception):
        """
        Handle errors during backend server startup in a unified way.
        Updates model instance state and re-raises the original error.
        """
        cause = getattr(error, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        error_message = f"Failed to run {self._model.backend}: {error}{cause_text}"

        try:
            is_main_worker = self._model_instance.worker_id == self._worker.id
            if is_main_worker:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(self._model_instance.id, **patch_dict)
        except Exception as ue:
            logger.error(f"Failed to update model instance: {ue}")

        raise error

    def _get_deployment_metadata(self) -> ModelInstanceDeploymentMetadata:
        """
        获取工作负载名称、分布式部署等元数据，并确认当前 Worker 参与该实例
        """
        deployment_metadata = self._model_instance.get_deployment_metadata(
            self._worker.id
        )
        if not deployment_metadata:
            raise RuntimeError(
                "Failed to get deployment metadata: model instance is not handling by the current worker"
            )
        return deployment_metadata

    def _get_configured_env(self, **kwargs) -> Dict[str, str]:
        """
        Get the environment variables for the model instance.
        Merge the model's env with the system env.
        If there are conflicts, the model's env takes precedence.

        Returns:
            A dictionary of environment variables for the model instance.
        """

        env = {}
        if not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT:
            env = filter_env_vars(os.environ)

        if self._model.env:
            env.update(self._model.env)

        return env

    @lru_cache
    def _get_selected_gpu_devices(self) -> GPUDevicesStatus:
        """
        从 Worker 硬件列表中找到分配给实例的设备；指定设备不存在时报错
        """
        minstance = self._model_instance
        dservers = minstance.distributed_servers
        gpu_type = None
        if (
            dservers
            and dservers.subordinate_workers
            and minstance.worker_id != self._worker.id
        ):
            subworker = next(
                (
                    w
                    for w in dservers.subordinate_workers
                    if w.worker_id == self._worker.id
                ),
                None,
            )
            gpu_indexes = sorted(subworker.gpu_indexes or [])
            gpu_type = subworker.gpu_type
        else:
            gpu_indexes = sorted(self._model_instance.gpu_indexes or [])
            gpu_type = self._model_instance.gpu_type

        gpu_devices: GPUDevicesStatus = []
        if gpu_indexes and self._worker.status.gpu_devices:
            for index in gpu_indexes:
                gpu_device = next(
                    (
                        d
                        for d in self._worker.status.gpu_devices
                        if d.index == index and (gpu_type is None or d.type == gpu_type)
                    ),
                    None,
                )
                if gpu_device:
                    gpu_devices.append(gpu_device)
        missing = set(gpu_indexes) - {device.index for device in gpu_devices}
        if missing:
            raise ValueError(f"Assigned accelerator indexes not found on Worker {self._worker.id}: {sorted(missing)}")
        return gpu_devices

    def _get_device_info(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        获取设备后端类型、运行时版本和架构，供镜像选择使用
        """
        gpu_devices = self._get_selected_gpu_devices()
        if gpu_devices:
            gpu_device = gpu_devices[0]
            return (
                gpu_device.type,
                gpu_device.runtime_version,
                gpu_device.arch_family,
            )
        elif self._worker.status.gpu_devices:
            gpu_device = self._worker.status.gpu_devices[0]
            return (
                gpu_device.type,
                gpu_device.runtime_version,
                gpu_device.arch_family,
            )
        return None, None, None

    def _get_configured_resources(
        self, mount_all_devices: bool = False
    ) -> ContainerResources:
        """
        将 GPU 索引转换成 Runtime 的资源请求，并检查设备类型一致
        """
        resources = ContainerResources()
        gpu_devices = self._get_selected_gpu_devices()
        if gpu_devices:
            gpu_type = gpu_devices[0].type
            for device in gpu_devices[1:]:
                if device.type != gpu_type:
                    raise RuntimeError(
                        "All GPUs assigned to the model instance must be of the same type."
                    )
            key = runtime_envs.GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY.get(
                gpu_type
            )
            if key:
                resources[key] = (
                    ",".join(str(d.index) for d in gpu_devices)
                    if not mount_all_devices
                    else "all"
                )
        return resources

    def _get_configured_mounts(self) -> List[ContainerMount]:
        """
        根据模型实际路径的父目
        """
        mounts: List[ContainerMount] = []
        if (
            self._model_path
            and not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT
        ):
            model_dir = os.path.dirname(self._model_path)
            mounts.append(
                ContainerMount(
                    path=model_dir,
                ),
            )
        return mounts

    def _get_configured_ports(self) -> List[ContainerPort]:
        """
        将实例端口列表转换成容器端口配置
        """
        return [
            ContainerPort(
                internal=port,
            )
            for port in self._model_instance.ports or []
        ]

    @staticmethod
    def _get_container_env_config(env: Dict[str, str]) -> ContainerEnvConfig:
        """
        Read container configuration from environment variables passed to the container.

        Args:
            env: The environment variables dictionary passed to the container.

        Returns:
            A ContainerEnvConfig containing container configuration:
            - user: Run as specific UID (int)
            - group: Run as specific GID (int)
            - shm_size_gib: Shared memory size in GiB (float, default 10.0)
        """
        config = ContainerEnvConfig()

        # Read user ID
        uid_str = env.get("GPUSTACK_MODEL_RUNTIME_UID")
        if uid_str:
            try:
                config.user = int(uid_str)
            except ValueError:
                logger.warning(
                    f"Invalid GPUSTACK_MODEL_RUNTIME_UID value: {uid_str}, ignoring"
                )

        # Read group ID
        gid_str = env.get("GPUSTACK_MODEL_RUNTIME_GID")
        if gid_str:
            try:
                config.group = int(gid_str)
            except ValueError:
                logger.warning(
                    f"Invalid GPUSTACK_MODEL_RUNTIME_GID value: {gid_str}, ignoring"
                )

        # Read shared memory size in GiB
        shm_str = env.get("GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB", "10")
        try:
            config.shm_size_gib = float(shm_str)
        except ValueError:
            logger.warning(
                f"Invalid GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB value: {shm_str}, using default 10.0"
            )
            config.shm_size_gib = 10.0

        return config

    def _get_serving_port(self) -> int:
        """
        Get the (main) serving port for the model instance.

        Returns:
            The (main) serving port for the model instance.
        """
        return (
            self._model_instance.ports[0]
            if self._model_instance.ports
            else self._model_instance.port
        )

    def _get_configured_image(
        self,
        backend: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve the container image to use for the current backend, then apply
        registry override once if needed.

        See _resolve_image for resolution details.
        """
        image_name, target_version = self._resolve_image(backend)
        if image_name is None:
            return None
        # Update model backend service version at upper layer if we detected it
        if target_version:
            self._update_model_backend_service_version(target_version)
        return apply_registry_override_to_image(
            self._config, image_name, self._fallback_registry
        )

    def _fetch_runner_overrides(self) -> List[RunnerOverrideEntryPublic]:
        # 真实版本：GET /runner-override-entries，读管理员配置的自定义镜像目录。
        # 空列表 = 用 gpustack-runner 包内置的目录。
        return []

    def _resolve_image(  # noqa: C901
        self,
        backend: Optional[str] = None,
    ) -> (Optional[str], Optional[str]):
        """
        优先使用模型显式指定的 image_name，否则按后端配置或 GPU 类型、平台、vLLM 版本等查询 Runner 目录。

        Resolve the container image to use for the current backend.

        This method returns the raw image name without applying any registry
        override. Callers should apply overrides as needed.

        Precedence:
        1) Explicitly configured image on the model (self._model.image_name)
        2) Prefer image name from the user's config when using custom backend or built-in backend with a custom version
        3) Auto-detected image from gpustack-runner based on device vendor/arch and backend

        Return:
            image_name, backend_version

        """
        # 1) Return directly if explicitly provided.
        if self._model.image_name:
            return self._model.image_name, None

        # 2) Configuration takes priority when backend_version is set
        if self._model and self.inference_backend:
            image_name, target_version = self.inference_backend.get_image_name(
                self._model.backend_version
            )
            if image_name and target_version:
                return image_name, target_version

        """
        Prepare queries for retrieving runners.
        """

        def get_docker_image(bvr: BackendVersionedRunner) -> str:
            return bvr.variants[0].services[0].versions[0].platforms[0].docker_image

        backend, runtime_version, arch_family = self._get_device_info()
        if not backend:
            # Return directly if there is not a valid device.
            # GPUStack-Runner does not provide CPU-only platform images.
            # To use a CPU-only version, user must configure in `Inference Backend` page.
            return None, None

        if backend not in available_backends():
            # Return directly if found backend is not within the available backends.
            return None, None

        """
        Retrieve runners by queries.

        For example, the queries of runners is as below.

        - backend: cuda
          backend_variant: None
          service: vllm
          service_version: 0.10.0
          platform: linux/amd64
        - backend: cann
          backend_variant: 910b
          service: vllm
          service_version: 0.10.0
          platform: linux/arm64
        """

        backend_variant = None
        service = self._model.backend.lower()
        # A blank backend version means "Auto", same as None. Legacy/migrated data
        # and API clients can store "", which would otherwise be used as an exact
        # version filter and match no runner at all.
        model_service_version = self._model.backend_version or None
        service_version = model_service_version

        # Default variant for some backends.
        if backend == "cann":
            if arch_family:
                backend_variant = _cann_variant(arch_family)
            if not backend_variant:
                backend_variant = "910b"

        logger.info(
            f"_resolve_image query: backend={backend}, backend_variant={backend_variant}, service={service}, service_version={service_version}, platform={platform.system_arch()}, runtime_version={runtime_version}"
        )

        overrides = self._fetch_runner_overrides()
        runners = merged_backend_runners(
            overrides,
            backend=backend,
            backend_variant=backend_variant,
            service=service,
            service_version=model_service_version,
            platform=platform.system_arch(),
            with_deprecated=model_service_version is not None,
        )
        if not runners:
            # Return directly if there is not a valid runner.
            return None, None

        """
        Pick the appropriate backend version from among the multiple versions.

        For example, the content of runners is as below.

        [
            {
                "backend": "cuda",
                "versions": [
                    {
                        "version": "12.8",
                        ...
                    },
                    {
                        "version": "12.6",
                        ...
                    },
                    {
                        "version": "12.4",
                        ...
                    }
                ]
            }
        ]
        """

        backend_versioned_runners = runners[0].versions

        # Try to update backend version for server model.
        if backend_versioned_runners and len(backend_versioned_runners) > 0:
            service_version = _get_service_version_from_versioned_runner(
                backend_versioned_runners[0]
            )

        # Return directly if there is only one versioned backend.
        if len(backend_versioned_runners) == 1:
            return get_docker_image(backend_versioned_runners[0]), service_version

        # The runtime-match rule (newest <= host, same-major then oldest
        # fallbacks) is shared with cache-provider runtime_images.
        """从多个运行时版本中选择镜像候选"""
        picked_version = pick_runtime_version(
            [candidate.version for candidate in backend_versioned_runners],
            runtime_version,
        )
        picked_runner = next(
            candidate
            for candidate in backend_versioned_runners
            if candidate.version == picked_version
        )
        service_version = _get_service_version_from_versioned_runner(picked_runner)
        return get_docker_image(picked_runner), service_version

    def _update_model_backend_service_version(
        self, service_version: Optional[str]
    ) -> None:
        """
        Update model backend (service) version back to server if not already set.

        This method is extracted from image resolution flow to be called from the upper
        layer after the version is detected.
        """
        if not service_version:
            return
        try:
            if not self._model.backend_version:
                self._model.backend_version = service_version
                self._clientset.models.update(
                    self._model.id, ModelUpdate(**self._model.model_dump())
                )
            if not self._model_instance.backend_version:
                self._update_model_instance(
                    self._model_instance.id, backend_version=service_version
                )
        except Exception as e:
            logger.error(
                f"Failed to update model service version {service_version}: {e}"
            )

    def build_versioned_command_args(
        self,
        default_args: List[str],
        model_path: Optional[str] = None,
        port: Optional[int] = None,
    ) -> List[str]:
        """
        Override default startup arguments based on version configuration
        when the version uses non-built-in version and defines a custom run_command

        Args:
        - default_args: The default command argument list.
        - model_path: Path used to replace {{model_path}}; if None, fall back to self._model_path.
        - port: Port used to replace {{port}}; if None, fall back to self._model_instance.port.

        Returns:
            The final command argument list used for container execution.
        """

        # if no version or inference backend is available, return default_args
        version = self._model.backend_version
        if not version or not self.inference_backend:
            return default_args

        # Load version configuration
        version_config = None
        try:
            version_config, version = self.inference_backend.get_version_config(version)
        except Exception:
            version_config = self.inference_backend.version_configs.root.get(version)

        # Only perform replacement when the version uses non-built-in version and defines run_command
        if (
            version_config
            and version_config.built_in_frameworks is None
            and version_config.run_command
        ):
            resolved_model_path = (
                model_path if model_path is not None else self._model_path
            )
            resolved_port = port if port is not None else self._model_instance.port
            resolved_model_name = self._model_instance.model_name
            selected_gpu_indexes = sorted(
                d.index for d in self._get_selected_gpu_devices()
            )

            command = self.inference_backend.replace_command_param(
                version=version,
                model_path=resolved_model_path,
                port=resolved_port,
                worker_ip=self._worker.ip,
                model_name=resolved_model_name,
                gpu_count=len(selected_gpu_indexes),
                gpu_ids=selected_gpu_indexes,
                command=version_config.run_command,
                env=self._model.env,
            )
            if command:
                return shlex.split(command)

        # Return original default_args by default
        return default_args

    @staticmethod
    def _override_entrypoint(
        command: Optional[List[str]],
        command_args: List[str],
        command_script: Optional[str],
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        Map command / command_args / command_script onto the container's
        ENTRYPOINT/CMD override semantics.

        - command_script becomes the entrypoint, so the original command is
          prepended to its args.
        - Otherwise merge command and args into command to override the image
          ENTRYPOINT (args alone would be appended to ENTRYPOINT, not replace it).
        - When there is no command to override the ENTRYPOINT, leave the args
          appended as-is.
        """
        if command_script:
            return None, (command or []) + command_args
        if not command:
            return None, command_args
        return command + command_args, None

    @staticmethod
    def _get_backend_parameter_start_index(
        arguments: List[str],
        entrypoint: Optional[List[str]] = None,
    ) -> int:
        """
        Return where backend parameters start in `arguments`, skipping any
        command prefix or model path that may precede the first option.
        """
        if not arguments:
            return 0

        if not entrypoint:
            command = os.path.basename(arguments[0])
            if (
                len(arguments) >= 3
                and command.startswith("python")
                and arguments[1] == "-m"
            ):
                return 3

        for index, argument in enumerate(arguments):
            if argument.startswith("-"):
                return index

        return len(arguments)

    def _get_injected_backend_parameters(
        self,
        arguments: List[str],
        user_backend_parameters: List[str],
        entrypoint: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Derive injected backend parameters from the final command line.

        The final command is the source of truth: remove the command prefix (or
        separate container entrypoint) and the user-specified backend
        parameters, and the remaining backend parameters are injected by
        GPUStack.
        """
        start_index = self._get_backend_parameter_start_index(arguments, entrypoint)
        candidates = arguments[start_index:]

        if not user_backend_parameters:
            return candidates

        user_param_len = len(user_backend_parameters)
        for start in range(len(candidates) - user_param_len, -1, -1):
            end = start + user_param_len
            if candidates[start:end] == user_backend_parameters:
                return candidates[:start] + candidates[end:]

        return candidates

    def _flatten_backend_param(self) -> List[str]:
        """
        Reduce ``backend_parameters`` to a flat argv-style token list.

        ``backend_parameters`` is semantically a concatenated argv: each element
        may be one token (``"--host"``, ``"0.0.0.0"``), one full
        ``--key value`` / ``--key=value`` string, or a whole pasted command line
        (``"--a 1 --b=2 --flag"``). ``flatten_to_argv`` recovers the underlying
        token stream uniformly.

        If the backend's ``parameter_format`` is set, each ``--key value(s)``
        cluster is normalized to that form (multi-value clusters always stay in
        space form — equal form can't safely express them).
        """
        tokens = flatten_to_argv(self._model.backend_parameters or [])

        parameter_format = (
            getattr(self.inference_backend, "parameter_format", None)
            if self.inference_backend
            else None
        )
        if parameter_format is None or not tokens:
            return tokens

        return _normalize_param_format(tokens, parameter_format)


def _get_service_version_from_versioned_runner(
    backend_versioned_runner: BackendVersionedRunner,
) -> Optional[str]:
    """
    Get the service version from the backend versioned runner.

    Args:
        backend_versioned_runner:
            The backend versioned runner.
    Returns:
        The service version string, or None if not found.
    """
    try:
        return backend_versioned_runner.variants[0].services[0].versions[0].version
    except Exception as e:
        logger.error(
            f"Failed to get service version from backend versioned runner: {e}"
        )
        return None


def is_ascend_310p(devices: GPUDevicesStatus) -> bool:
    """
    Check if the model instance is running on VLLM Ascend 310P.

    An empty device list is not a match: callers gate Ascend 310P specifics on
    this, and ``all()`` is vacuously true over nothing, so a model instance
    that named no device -- one scheduled by GPU type rather than by device
    index, for instance -- would be served as if it ran on 310P.
    """

    return bool(devices) and all(
        (gpu.type == "cann" or gpu.vendor == ManufacturerEnum.ASCEND.value)
        and _cann_variant(gpu.arch_family or gpu.name) == "310p"
        for gpu in devices
    )


def is_ascend(devices: GPUDevicesStatus) -> bool:
    """
    Check if all devices are Ascend.

    An empty device list is not a match, for the same reason as
    :func:`is_ascend_310p`.
    """

    return bool(devices) and all(
        gpu.type == "cann" or gpu.vendor == ManufacturerEnum.ASCEND.value for gpu in devices
    )


def _cann_variant(arch_family: Optional[str]) -> Optional[str]:
    normalized = (arch_family or "").lower()
    if normalized in {"910", "910b", "310p", "310b", "a3", "950"}:
        return normalized
    return get_ascend_cann_variant(arch_family)


def cal_distributed_parallelism_arguments(
    model_instance: ModelInstance,
) -> tuple[int, int]:
    pp = len(model_instance.distributed_servers.subordinate_workers) + 1
    tp = len(model_instance.gpu_indexes) if model_instance.gpu_indexes else 1
    uneven_pp = tp
    uneven = False
    for subordinate_worker in model_instance.distributed_servers.subordinate_workers:
        num_gpus = len(subordinate_worker.gpu_indexes)
        uneven_pp += num_gpus
        if num_gpus != tp:
            uneven = True

    if uneven:
        tp = 1
        pp = uneven_pp
        logger.warning(
            f"The number of GPUs selected for each worker is not equal: {num_gpus} != {tp}, fallback to using pipeline parallelism."
        )
    return tp, pp
