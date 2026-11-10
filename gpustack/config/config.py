import logging
import os
from enum import Enum
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from gpustack.schemas.config import (
    PredefinedConfig,
    GatewayModeEnum,
)

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
    # The api_port is used in gpustack server/worker serving API requests.
    api_port: Optional[int] = 30080
    database_port: Optional[int] = 5432
    database_url: Optional[str] = None
    disable_worker: Optional[bool] = None  # Deprecated
    enable_worker: bool = False

    _data_dir_was_fresh = False

    model_config = SettingsConfigDict(
        env_prefix="GPUSTACK_",
        protected_namespaces=('settings_',),
        # YAML 里写了 `Config` 没有的字段也不报错
        extra="allow"
    )

    def __init__(self, **values):
        super().__init__(**values)

        def prepare_dir(dir_path: Optional[str], default: str) -> str:
            """用户给了路径就转成绝对路径"""
            return default if dir_path is None else os.path.abspath(dir_path)

        # common options
        self.data_dir = prepare_dir(self.data_dir, self.get_data_dir())
        self._data_dir_was_fresh = not os.path.isdir(self.data_dir) or not os.listdir(
            self.data_dir
        )
        self.cache_dir = prepare_dir(
            self.cache_dir, os.path.join(self.data_dir, "cache")
        )
        self.bin_dir = prepare_dir(self.bin_dir, os.path.join(self.data_dir, "bin"))
        self.log_dir = prepare_dir(self.log_dir, os.path.join(self.data_dir, "log"))

        if self.system_reserved is None:
            self.system_reserved = {"ram": 0, "vram": 0}

        self.make_dirs()

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
        """Linux 默认 `/var/lib/gpustack`，需要 root 权限。开发时请用 `--data-dir` 指到自己的目录"""

        app_name = "gpustack"
        if os.name == "nt":  # Windows
            data_dir = os.path.join(os.environ["APPDATA"], app_name)
        elif os.name == "posix":
            data_dir = f"/var/lib/{app_name}"
        else:
            raise Exception("Unsupported OS")

        return os.path.abspath(data_dir)

    def _is_worker(self):
        """
        配了 `server_url` 就是 worker。这就是分流的依据：Worker 必须知道去哪找 Server，Server 不需要
        """
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
        """
        没配 `server_url` 时，是“纯 server”还是“server + 内嵌 worker”？优先看用户的 `--enable-worker` / `--disable-worker`。
        都没给时，全新安装默认纯 server
        """
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
        """内嵌 worker 没有配 `server_url`，就连本机的 API 端口"""
        # returns server if not None else returns embedded server url
        return (
            self.server_url or f"http://127.0.0.1:{self.api_port}"
            if self.api_port
            else "http://127.0.0.1"
        )

    def get_api_port(self, embedded_worker: bool = False) -> int:
        """同一个函数回答“我该监听哪个端口”"""

        if embedded_worker:
            return self.worker_port
        if self.server_role() != self.ServerRole.WORKER:
            return self.api_port
        return (
            self.api_port
            if self.gateway_mode == GatewayModeEnum.embedded
            else self.worker_port
        )


def get_global_config() -> Config:
    return _config


def set_global_config(cfg: Config):
    global _config
    _config = cfg
    return cfg