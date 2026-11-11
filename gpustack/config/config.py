import logging
import os
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from gpustack.schemas.config import (
    PredefinedConfig,
    GatewayModeEnum,
)
from gpustack import __version__
from gpustack.schemas.config import PredefinedConfigNoDefaults
from typing import Dict
from gpustack.config.registration import read_registration_token
from gpustack.config.registration import read_worker_token
from pydantic import model_validator


_config = None


logger = logging.getLogger(__name__)


class WorkerConfig(PredefinedConfig):
    # common config which should be dynamic or not configurable
    data_dir: Optional[str] = None
    advertise_address: Optional[str] = None
    # Worker options which are different for each worker
    token: Optional[str] = None
    server_url: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_name: Optional[str] = None


class Config(WorkerConfig, BaseSettings):
    api_port: Optional[int] = 30080

    database_port: Optional[int] = 5432

    database_url: Optional[str] = None

    disable_worker: Optional[bool] = None  # Deprecated

    enable_worker: bool = False

    disable_openapi_docs: bool = False

    server_external_url: Optional[str] = None

    _set_worker_fields = {}

    _data_dir_was_fresh = False

    model_config = SettingsConfigDict(
        env_prefix="GPUSTACK_", protected_namespaces=('settings_',), extra="allow"
    )

    def __init__(self, **values):
        super().__init__(**values)
        self._set_worker_fields = self.model_dump(
            exclude_unset=True,
            exclude_none=True,
            include=self.__pydantic_fields_set__
            & set(PredefinedConfig.model_fields.keys()),
        )
        def prepare_dir(dir_path: Optional[str], default: str) -> str:
            return default if dir_path is None else os.path.abspath(dir_path)
        self.data_dir = prepare_dir(self.data_dir, self.get_data_dir())
        self._data_dir_was_fresh = not os.path.isdir(self.data_dir) or not os.listdir(
            self.data_dir
        )
        self.cache_dir = prepare_dir(
            self.cache_dir, os.path.join(self.data_dir, "cache")
        )
        self.bin_dir = prepare_dir(self.bin_dir, os.path.join(self.data_dir, "bin"))
        self.log_dir = prepare_dir(self.log_dir, os.path.join(self.data_dir, "log"))
        if self.token is None:
            self.token = read_registration_token(self.data_dir)
        if (
            self._is_worker()
            and self.token is None
            and read_worker_token(self.data_dir) is None
        ):
            raise Exception("Token is required when running as worker")
        if self.system_reserved is None:
            self.system_reserved = {"ram": 0, "vram": 0}
        self.make_dirs()

    @model_validator(mode="after")
    def check_all(self):  # noqa: C901
        if 'PYTEST_CURRENT_TEST' in os.environ:
            # Skip validation during tests
            return self

        if self.server_url:
            self.server_url = self.server_url.rstrip("/")

        if self.service_port_range:
            self.check_port_range(self.service_port_range)

        if self.ray_port_range:
            self.check_port_range(self.ray_port_range, diff=20)

        return self

    @staticmethod
    def check_port_range(port_range: str, diff: Optional[int] = None):
        ports = port_range.split("-")
        if len(ports) != 2:
            raise Exception(f"Invalid port range: {port_range}")
        if not ports[0].isdigit() or not ports[1].isdigit():
            raise Exception("Port range must be numeric")
        if int(ports[0]) > int(ports[1]):
            raise Exception(f"Invalid port range: {ports[0]} > {ports[1]}")
        if diff is not None:
            if int(ports[1]) - int(ports[0]) + 1 < diff:
                raise Exception(
                    f"Port range is too small: {port_range}, at least {diff} ports are required"
                )

    def make_dirs(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.bin_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_database_url(self) -> str:
        if self.database_url is not None:
            return self.database_url
        return (
            f"postgresql://root@127.0.0.1:{self.database_port}/gpustack?sslmode=disable"
        )

    @staticmethod
    def get_data_dir():
        app_name = "gpustack"
        if os.name == "nt":  # Windows
            data_dir = os.path.join(os.environ["APPDATA"], app_name)
        elif os.name == "posix":
            data_dir = f"/var/lib/{app_name}"
        else:
            raise Exception("Unsupported OS")

        return os.path.abspath(data_dir)

    def _is_worker(self):
        return self.server_url is not None

    class ServerRole(Enum):
        SERVER = "server"
        WORKER = "worker"
        BOTH = "both"

    def server_role(self) -> ServerRole:
        if self._is_worker():
            return self.ServerRole.WORKER
        elif self._is_both_role():
            return self.ServerRole.BOTH
        else:
            return self.ServerRole.SERVER

    def _is_both_role(self) -> bool:
        if self._is_worker():
            return False
        elif self.enable_worker:
            return True
        elif self.disable_worker:
            return False

        if self._data_dir_was_fresh:
            return False

        bootstrap_version_path = os.path.join(self.data_dir, "bootstrap_version")
        if os.path.exists(bootstrap_version_path):
            return False

        return True

    def get_server_url(self) -> str:
        # returns server if not None else returns embedded server url
        return (
            self.server_url or f"http://127.0.0.1:{self.api_port}"
            if self.api_port
            else "http://127.0.0.1"
        )

    def get_api_port(self, embedded_worker: bool = False) -> int:
        if embedded_worker:
            return self.worker_port
        if self.server_role() != self.ServerRole.WORKER:
            return self.api_port
        return (
            self.api_port
            if self.gateway_mode == GatewayModeEnum.embedded
            else self.worker_port
        )

    def reload_token(self):
        token = read_registration_token(self.data_dir)
        if token:
            self.token = token

    def reload_worker_config(self, worker_config: Optional[PredefinedConfigNoDefaults]):
        if worker_config is None:
            return
        updated = {
            **worker_config.model_dump(exclude_none=True),
            **self._set_worker_fields,
        }
        for key, value in updated.items():
            if key in self.__class__.model_fields:
                setattr(self, key, value)
        self.check_all()

    def get_system_reserved(self) -> Dict[str, int]:
        system_reserved_in_bytes = {**(self.system_reserved or {})}
        ram = (
            system_reserved_in_bytes.get(
                "ram", system_reserved_in_bytes.pop("memory", 0)
            )
            or 0
        )
        vram = (
            system_reserved_in_bytes.get(
                "vram", system_reserved_in_bytes.pop("gpu_memory", 0)
            )
            or 0
        )
        system_reserved_in_bytes["ram"] = int(ram) << 30
        system_reserved_in_bytes["vram"] = int(vram) << 30
        return system_reserved_in_bytes


def get_image_name(
    image_name_override: Optional[str],
    registry: Optional[str] = None,
    image_repo: str = "gpustack/gpustack",
) -> str:
    if image_name_override:
        return image_name_override
    # 真实版本：版本号来自扩展插件提供的 resolve_version_info()
    version = __version__
    if version.removeprefix("v") == "0.0.0":
        version = "dev"
    prefix = f"{registry}/" if registry else ""
    return f"{prefix}{image_repo}:{version}"


def get_cluster_image_name(
    worker_config: Optional[PredefinedConfigNoDefaults],
    cluster_registry: Optional[str] = None,
) -> str:
    cfg = get_global_config()
    # 真实版本：determine_default_registry() 在没有配置时探测 Docker Hub / quay.io 哪个可达
    registry = cluster_registry or cfg.system_default_container_registry
    if worker_config is None:
        return get_image_name(
            image_repo=cfg.image_repo,
            image_name_override=cfg.image_name_override,
            registry=registry,
        )
    return get_image_name(
        image_name_override=worker_config.image_name_override
        or cfg.image_name_override,
        image_repo=worker_config.image_repo or cfg.image_repo,
        registry=registry,
    )


def get_global_config() -> Config:
    return _config


def set_global_config(cfg: Config):
    global _config
    _config = cfg
    return cfg
