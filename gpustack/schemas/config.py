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