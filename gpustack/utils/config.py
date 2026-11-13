from gpustack.config.config import Config
import logging


logger = logging.getLogger(__name__)

"""统一处理镜像仓库替换"""

def apply_registry_override_to_image(
    _config: Config, image: str, fallback_registry: str
) -> str:
    """
    1) 如果镜像带有显式 registry（镜像仓库地址），则按原样返回。
    2) 如果镜像没有显式 registry，且配置了系统默认 registry，则使用配置中的系统默认 registry 给镜像添加前缀。
    3) 如果镜像没有显式 registry，且未配置系统默认 registry，则当镜像不带 "gpustack" 前缀时，默认使用 docker.io。
    4) 如果镜像没有显式 registry，且未配置系统默认 registry，并且镜像带有 "gpustack" 前缀，则当 docker.io 可达时默认使用 docker.io；否则使用 quay.io。
    """
    registry_cfg = (_config.system_default_container_registry or "").strip()

    parts = image.split("/", 1)
    # 1) If the image has an explicit registry, return it as is.
    has_explicit = len(parts) >= 2 and (
        "." in parts[0] or ":" in parts[0] or parts[0] == "localhost"
    )
    if has_explicit:
        return image
    # 2) If the image does not have an explicit registry and a system default registry is configured,
    #    prefix the image with the system default registry in config.
    if registry_cfg:
        registry = registry_cfg.rstrip("/")
        final = f"{registry}/{image}"
        logger.info(
            f"Using system default registry '{registry}'; image resolved to: {final}"
        )
        return final

    # 3) no explicit or configured, and not start with "gpustack" using "docker.io" as default.
    if not image.startswith("gpustack"):
        logger.info(
            f"Using Docker Hub for non-gpustack image; image resolved to: {image}"
        )
        return image

    # 4) Otherwise, use fallback registry if configured.
    #    The fallback registry is Docker Hub or Quay.io depending on reachability.
    #    If both are not reachable, use docker.io as default.
    prefix = fallback_registry + "/" if fallback_registry else ""
    return f"{prefix}{image}"
