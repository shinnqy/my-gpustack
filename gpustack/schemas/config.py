from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GatewayModeEnum(str, Enum):
    """GPUStack 可以把推理流量交给一个内嵌的 Higress 网关"""

    auto = "auto"
    embedded = "embedded"
    incluster = "incluster"
    external = "external"
    disabled = "disabled"


class SensitivePredefinedConfig(BaseModel):
    # Common options
    huggingface_token: Optional[str] = Field(
        default=None, json_schema_extra={"env_var": "HF_TOKEN"}
    )


class PredefinedConfig(SensitivePredefinedConfig):
    # Common options
    debug: bool = False
    cache_dir: Optional[str] = None
    log_dir: Optional[str] = None
    bin_dir: Optional[str] = None
    benchmark_dir: Optional[str] = None
    gateway_mode: GatewayModeEnum = GatewayModeEnum.auto
    system_default_container_registry: Optional[str] = None
    image_name_override: Optional[str] = None
    image_repo: str = "gpustack/gpustack"

    # Worker options
    worker_ifname: Optional[str] = None
    # Worker 自己的 HTTP 端口
    worker_port: int = 10150
    # 每个推理容器要占一个端口，从这个范围里分配
    service_port_range: Optional[str] = "40000-40063"
    # 多机部署时 Ray 要用一批端口
    ray_port_range: Optional[str] = "41000-41999"
    # 给系统预留的内存和显存
    system_reserved: Optional[dict] = None

class PredefinedConfigNoDefaults(PredefinedConfig):
    debug: Optional[bool] = None
    worker_port: Optional[int] = None
    service_port_range: Optional[str] = None
    ray_port_range: Optional[str] = None
    image_repo: Optional[str] = None
    gateway_mode: Optional[str] = None

def parse_base_model_to_env_vars(
    config: BaseModel,
) -> dict[str, str]:
    """
    字段名转成 `GPUSTACK_<大写字段名>`，个别字段用 `json_schema_extra` 指定别的名字（`huggingface_token` → `HF_TOKEN`）
    """

    env_vars = {}
    for field_name, field in config.__class__.model_fields.items():
        extra = getattr(field, 'json_schema_extra', None) or {}
        env_var = extra.get("env_var")
        if env_var is None:
            # assuming the field name is in snake_case
            env_var = f"GPUSTACK_{field_name.upper()}"
        value = getattr(config, field_name)
        if value is not None:
            if isinstance(value, bool):
                env_vars[env_var] = "true" if value else "false"
            else:
                env_vars[env_var] = str(value)
    return env_vars