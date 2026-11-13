from functools import cmp_to_key
from gpustack_runtime.deployer.__utils__ import compare_versions

"""从多个运行时版本中选择镜像候选"""

def major_version(version_str):
    """Major segment of a version string, e.g. "12" from "v12.8"."""
    if not version_str:
        return None
    return version_str.removeprefix("v").split(".", 1)[0]


def pick_runtime_version(candidates, host_version):
    """
    选择一个已声明的运行时版本，作为镜像构建针对某个主机时应使用的目标版本，
    该逻辑由推理后端 runner 解析和缓存提供者的 ``runtime_images`` 共用：

    - 未检测到主机运行时 -> 选择最新的候选版本；
    - 否则选择 <= 主机版本的最新候选版本（针对较旧运行时构建的镜像可在较新主机上运行，反之则不行）；
    - 所有候选版本都新于主机 -> 选择与主机主版本号相同的最新候选版本（同主版本号的次版本兼容性成立），否则选择整体最旧的候选版本；
    - 没有候选版本 -> None。

    比较使用 gpustack_runtime 的容错 compare_versions，而不是 PEP 440 解析：
    runner 和运行时版本字符串（例如 CANN 的 "8.1.RC1.alpha003"）不保证符合 PEP 440。
    """

    ordered = sorted(
        candidates,
        key=cmp_to_key(compare_versions),
        reverse=True,
    )
    if not ordered:
        return None
    if not host_version:
        return ordered[0]
    for candidate in ordered:
        if compare_versions(candidate, host_version) <= 0:
            return candidate
    host_major = major_version(host_version)
    return next(
        (candidate for candidate in ordered if major_version(candidate) == host_major),
        ordered[-1],
    )
