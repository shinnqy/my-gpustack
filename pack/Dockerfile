# Chapter-15 subset of GPUStack's pack/Dockerfile:
# base -> gpustack; build/install the wheel; prepare tools; copy operator;
# use the same image and entrypoint for Server and Worker.
ARG PYTHON_VERSION=3.11
ARG GPUSTACK_BASE_IMAGE=base
ARG GPUSTACK_OPERATOR_VERSION=v0.9.2

# 准备 Operator 二进制的来源
FROM gpustack/gpustack-operator:${GPUSTACK_OPERATOR_VERSION} AS gpustack-operator

# 准备 Python 和系统环境
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
# 指定后续 shell 形式的 RUN 使用 Bash
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=never
# 安装系统依赖
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 -o APT::Update::Error-Mode=any update \
    && apt-get install -y --no-install-recommends \
       bash ca-certificates curl git make pciutils libnuma1 libstdc++6 libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv==0.11.7

# 主阶段开始组装最终的 GPUStack 镜像
FROM ${GPUSTACK_BASE_IMAGE} AS gpustack
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
ARG TARGETOS
ARG TARGETARCH
ARG GPUSTACK_TOOLS_DOWNLOAD_BASE_URL
ARG GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS
ENV PYTHONUNBUFFERED=1 \
    GPUSTACK_TOOLS_DOWNLOAD_BASE_URL="${GPUSTACK_TOOLS_DOWNLOAD_BASE_URL}"

# Bind-mounted build inputs are discarded after the wheel has been installed.
# This follows the original package flow without copying host environments,
# database files, downloaded models or host-native tool binaries into the image.
# 临时提供项目源码
#  → 构建 wheel
#    → 安装到镜像的 Python 环境
#        → 检查程序和工具
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,target=/workspace/gpustack,rw <<EOF
    export UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy UV_NO_CACHE=0
    cd /workspace/gpustack
    make build
    uv pip install --system dist/*.whl
    cd /
    gpustack download-tools --system "${TARGETOS}" --arch "${TARGETARCH}"
    gpustack start --help >/dev/null
    python -c 'from gpustack.detectors.fastfetch.fastfetch import Fastfetch; info = Fastfetch().gather_system_info(); assert info.memory.total > 0'
    mkdir -p /var/lib/gpustack
EOF

# 给镜像写入默认运行配置：告诉程序到哪里找设备库、如何创建后续推理容器，以及如何输出日志
ENV ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest" \
    LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64" \
    NVIDIA_DISABLE_REQUIRE="true" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    GPUSTACK_RUNTIME_LOG_EXCEPTION="false" \
    GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY="/var/run/cdi" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT="true" \
    GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES="/var/run/cdi;/usr/local/Ascend/ascend-toolkit" \
    GPUSTACK_RUNTIME_DEPLOY_CORRECT_RUNNER_IMAGE="false" \
    GPUSTACK_RUNTIME_DOCKER_IMAGE_NO_PULL_VISUALIZATION="true" \
    GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS="${GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS}" \
    KUBE_FEATURE_WatchListClient="false"

COPY --from=gpustack-operator /usr/bin/gpustack-operator /usr/bin/gpustack-operator
COPY --chmod=755 pack/entrypoint.sh /usr/bin/entrypoint.sh
RUN gpustack-operator --version
WORKDIR /
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/bin/entrypoint.sh"]
