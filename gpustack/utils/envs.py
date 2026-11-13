import os
from typing import Optional
from typing import Dict


def get_gpustack_env(env_var: str) -> Optional[str]:
    env_name = "GPUSTACK_" + env_var
    return os.getenv(env_name)


def get_gpustack_env_bool(env_var: str) -> Optional[bool]:
    env_name = "GPUSTACK_" + env_var
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value.lower() in ["true", "1"]
    return None


def sanitize_env(env: Dict[str, str]) -> Dict[str, str]:
    """
    用于打印日志前过滤变量，例如移除 GPUSTACK_* 和名称以 _TOKEN、`_PASSWORD` 等结尾的项
    """
    prefixes = ("GPUSTACK_",)
    suffixes = (
        "_KEY",
        "_key",
        "_TOKEN",
        "_token",
        "_SECRET",
        "_secret",
        "_PASSWORD",
        "_password",
        "_PASS",
        "_pass",
    )
    return {
        k: v
        for k, v in env.items()
        if not k.startswith(prefixes) and not k.endswith(suffixes)
    }


def filter_env_vars(env_dict: dict) -> dict:
    """
    用于整理准备传给容器的 Worker 环境，移除包管理器配置、设备可见性、`PATH`、`_PATH`、`_HOME` 等变量。
    基类随后再合并用户的模型环境配置。
    """
    return {
        k: v
        for k, v in env_dict.items()
        if not (
            k.startswith(
                (
                    "GPUSTACK_",
                    "PIP_",
                    "PIPX_",
                    "POETRY_",
                    "UV_",
                    "S6_",
                    "PGCONFIG_",
                    "POSTGRES_",
                )
            )
            or k.endswith(
                (
                    "_VISIBLE_DEVICES",
                    "_DISABLE_REQUIRE",
                    "_DRIVER_CAPABILITIES",
                    "_PATH",
                    "_HOME",
                )
            )
            or (
                k
                in (
                    "DEBIAN_FRONTEND",
                    "LANG",
                    "LANGUAGE",
                    "LC_ALL",
                    "PYTHON_VERSION",
                    "HOME",
                    "HOSTNAME",
                    "PWD",
                    "_",
                    "TERM",
                    "SHLVL",
                    "LS_COLORS",
                    "PATH",
                )
            )
        )
    }
