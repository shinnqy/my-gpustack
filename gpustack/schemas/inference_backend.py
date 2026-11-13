from typing import Optional
from gpustack.mixins import BaseModelMixin
from sqlalchemy import UniqueConstraint
from sqlmodel import Field as SQLField
from pydantic import BaseModel
from .models import BackendEnum
from typing import List
from .common import pydantic_column_type
from .models import BackendSourceEnum
from gpustack.schemas.source import SourceTypeEnum
from gpustack_runtime.deployer.__utils__ import compare_versions
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import Text
from typing import Dict
import re
import shlex
from sqlmodel import SQLModel
from enum import Enum
from pydantic import Field
from pydantic import RootModel


class ParameterFormatEnum(str, Enum):
    """Parameter format for backend parameters."""

    SPACE = "space"  # --key value format
    EQUAL = "equal"  # --key=value format


class ContainerEnvConfig(BaseModel):
    """Container environment configuration."""

    user: Optional[int] = None
    group: Optional[int] = None
    shm_size_gib: float = 10.0


class VersionConfig(BaseModel):
    """
    保存某个版本的镜像、运行命令、入口和环境变量
    """

    image_name: Optional[str] = Field(None)
    run_command: Optional[str] = Field(None)
    entrypoint: Optional[str] = Field(None)
    built_in_frameworks: Optional[List[str]] = Field(None)
    custom_framework: Optional[str] = Field(None)
    env: Optional[Dict[str, str]] = Field(None)
    # Per-version source origin: which managed source produced this version
    # (FILE-yaml / URL / BUILTIN). None for user-added versions and packaged
    # runner-catalog versions the UI renders without a badge.
    source_name: Optional[str] = Field(None)
    source_type: Optional[SourceTypeEnum] = Field(None)


class VersionConfigDict(RootModel[Dict[str, VersionConfig]]):
    """
    Wrapper model for version configs dictionary to enable proper JSON serialization.
    """

    root: Dict[str, VersionConfig] = Field(default_factory=dict)


class InferenceBackendBase(SQLModel):
    """
    Base model for inference backends.

    Attributes:
        backend_name: Name of the backend (e.g., 'SGLang')
        version_configs: Dictionary mapping version strings to their configurations
        default_version: Default version to use if not specified
        default_backend_param: Default parameters to pass to the backend
        default_run_command: Default command to run the inference server
        default_entrypoint: Default entrypoint to replace for the inference server
        description: Backend description
        health_check_path: Path for health check endpoint
        parameter_format: Parameter format for backend parameters (Optional)
        common_parameters: List of commonly used parameters for UI hints (Optional)

    """

    backend_name: str = SQLField(index=True)

    owner_principal_id: Optional[int] = SQLField(
        default=None,
    )

    version_configs: VersionConfigDict = SQLField(
        sa_column=Column(pydantic_column_type(VersionConfigDict)()),
        default_factory=lambda: VersionConfigDict(root={}),
    )

    default_version: Optional[str] = SQLField(default=None)

    default_backend_param: Optional[List[str]] = SQLField(
        sa_column=Column(JSON), default=[]
    )

    default_run_command: Optional[str] = SQLField(
        sa_column=Column(Text, nullable=True), default=""
    )

    default_entrypoint: Optional[str] = SQLField(
        sa_column=Column(Text, nullable=True), default=""
    )

    is_built_in: bool = SQLField(default=False)

    description: Optional[str] = SQLField(
        default=None, sa_column=Column(Text, nullable=True)
    )

    health_check_path: Optional[str] = SQLField(default=None)

    backend_source: Optional[BackendSourceEnum] = SQLField(default=None)

    enabled: Optional[bool] = SQLField(default=None)

    icon: Optional[str] = SQLField(default=None)

    source_name: Optional[str] = SQLField(default=None)

    source_type: Optional[SourceTypeEnum] = SQLField(default=None)

    default_env: Optional[Dict[str, str]] = SQLField(
        sa_column=Column(JSON), default=None
    )

    parameter_format: Optional[ParameterFormatEnum] = SQLField(default=None)

    common_parameters: Optional[List[str]] = SQLField(
        sa_column=Column(JSON), default=None
    )

    def resolve_target_version(self, version: Optional[str] = None) -> Optional[str]:
        """
        Resolve the target version to use based on the requested version, default version,
        and available version configs.

        Logic:
        - If requested/default version exists in version_configs, return it.
        - If using a non-built-in backend and version_configs exist, return the latest version
          (by compare_versions, falling back to lexicographical sort).
        - Otherwise, return None.
        """
        version_configs_dict = self.version_configs.root
        target_version = version or self.default_version

        # 1) Requested/default version exists
        if target_version in version_configs_dict:
            return target_version

        # 2) For non-built-in backends, auto-select the latest available version
        if version_configs_dict and not self.is_built_in:
            try:
                version_list = list(version_configs_dict.keys())
                latest_version = version_list[0]
                for ver in version_list[1:]:
                    if compare_versions(ver, latest_version) > 0:
                        latest_version = ver
                return latest_version
            except Exception:
                sorted_versions = sorted(version_configs_dict.keys())
                return sorted_versions[-1] if sorted_versions else None

        # 3) No suitable version found
        return None

    def get_version_config(self, version: Optional[str] = None) -> (VersionConfig, str):
        """
        Get configuration for a specific version.

        Args:
            version: Version string, uses default_version if None

        Returns:
            VersionConfig for the resolved version, and the resolved version string

        Raises:
            KeyError: If the version cannot be resolved from version_configs
        """
        target_version = self.resolve_target_version(version)
        if target_version is None:
            raise KeyError(
                f"Version '{version or self.default_version}' not found in backend '{self.backend_name}'"
            )
        return self.version_configs.root[target_version], target_version

    def get_run_command(self, version: Optional[str] = None) -> str:
        if not version:
            version = self.default_version
        version_config, _ = self.get_version_config(version)
        return version_config.run_command or self.default_run_command

    def get_backend_env(self, version: Optional[str] = None):
        """
        backend.version.env > backend.default_env
        """
        env_dict = {}
        if self.default_env:
            for k, v in self.default_env.items():
                env_dict[k] = v
        if version:
            try:
                version_config, _ = self.get_version_config(version)
                if version_config.env:
                    for k, v in version_config.env.items():
                        env_dict[k] = v
            except Exception:
                # built-in version may not include version config
                pass
        return env_dict

    def replace_command_param(
        self,
        version: Optional[str],
        model_path: Optional[str],
        port: Optional[int],
        worker_ip: Optional[str] = None,
        model_name: Optional[str] = None,
        gpu_count: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        command: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        if not command:
            command = self.get_run_command(version)
            if not command:
                return ""

        command = command.replace("{{model_path}}", model_path or "")
        command = command.replace("{{port}}", str(port))
        command = command.replace("{{worker_ip}}", worker_ip or "")
        command = command.replace("{{model_name}}", model_name or "")
        command = command.replace(
            "{{gpu_count}}", str(gpu_count) if gpu_count is not None else ""
        )
        command = command.replace(
            "{{gpu_ids}}", ",".join(str(i) for i in gpu_ids) if gpu_ids else ""
        )

        # Resolve environment variables using {{VAR_NAME}} syntax
        # Use provided env (from model) if available, otherwise fall back to backend env
        if env:
            command = self._resolve_env_vars(command, env)

        return command

    def _resolve_env_vars(self, command: str, env_dict: Dict[str, str]) -> str:
        """
        Resolve {{VAR_NAME}} placeholders in the command string using the provided environment dict.

        Args:
            command: The command string with {{VAR_NAME}} placeholders
            env_dict: Dictionary of environment variable names to values

        Returns:
            Command with placeholders replaced by their values.
            If a variable is not found in env_dict, the placeholder is left unchanged.
        """
        # Match valid variable names: start with letter or underscore, followed by alphanumeric or underscore
        pattern = r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}"

        def replace_var(match):
            var_name = match.group(1)
            return env_dict.get(var_name, match.group(0))

        return re.sub(pattern, replace_var, command)

    def get_container_entrypoint(
        self, version: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Get container entrypoint for the specified version.

        Args:
            version: Desired backend version; falls back to `default_version` when None.

        Returns:
            The container entrypoint string, or None if not configured.
        """
        if self.backend_name == BackendEnum.CUSTOM.value:
            return None
        try:
            # Resolve concrete version and fetch its configuration
            version_config, _ = self.get_version_config(version)
        except KeyError:
            # Version not found or cannot be resolved
            return None
        entrypoint = version_config.entrypoint or self.default_entrypoint
        if entrypoint:
            return shlex.split(entrypoint)
        else:
            return None

    def get_image_name(self, version: Optional[str] = None) -> (str, str):
        """
        Resolve a user-configured container image for the specified backend version.

        Args:
            version: Desired backend version; falls back to `default_version` when None.

        Returns:
            A tuple of (image_name, version). Empty strings indicate no user-configured image.
        """
        # CUSTOM backend does not resolve here; image/command come from the model configuration
        if self.backend_name == BackendEnum.CUSTOM.value:
            return "", ""
        try:
            # Resolve concrete version and fetch its configuration
            version_config, version = self.get_version_config(version)
        except KeyError:
            # Version not found or cannot be resolved
            return "", ""

        if not version_config or not version_config.image_name:
            return "", ""

        # Only return image for custom version configs (no built-in frameworks) with explicit image
        if (
            self.backend_source == BackendSourceEnum.BUILT_IN
            and version_config.built_in_frameworks
        ):
            return "", ""

        return version_config.image_name, version


class InferenceBackend(InferenceBackendBase, BaseModelMixin, table=True):
    """保存后端整体默认值及各版本配置"""

    __tablename__ = 'inference_backends'
    __table_args__ = (
        UniqueConstraint(
            "backend_name",
            "owner_principal_id",
            name="uix_inference_backends_name_org",
        ),
    )
    id: Optional[int] = SQLField(default=None, primary_key=True)


def get_built_in_backend() -> List[InferenceBackend]:
    return [
        InferenceBackend(backend_name=BackendEnum.VLLM.value, is_built_in=True),
        InferenceBackend(backend_name=BackendEnum.SGLANG.value, is_built_in=True),
        InferenceBackend(
            backend_name=BackendEnum.ASCEND_MINDIE.value, is_built_in=True
        ),
        InferenceBackend(backend_name=BackendEnum.VOX_BOX.value, is_built_in=True),
        InferenceBackend(backend_name=BackendEnum.CUSTOM.value, is_built_in=True),
    ]


def is_built_in_backend(backend_name: Optional[str]) -> bool:
    """
    Check if a backend is a built-in backend.

    Args:
        backend_name: The name of the backend to check

    Returns:
        True if the backend is built-in, False otherwise
    """
    if not backend_name:
        return False

    built_in_backends = get_built_in_backend()
    built_in_backend_names = {
        backend.backend_name.lower() for backend in built_in_backends
    }
    return backend_name.lower() in built_in_backend_names
