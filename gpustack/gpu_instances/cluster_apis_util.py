from gpustack.schemas.principals import PLATFORM_PRINCIPAL_NAME
from typing import Optional


PREFIX = "gpustack"


def get_namespace_name(
    principal_identifier: Optional[str] = None,
) -> str:
    """
    生成所有者命名空间
    """

    if principal_identifier is None:
        principal_identifier = PLATFORM_PRINCIPAL_NAME

    return f"{PREFIX}-{principal_identifier}"
