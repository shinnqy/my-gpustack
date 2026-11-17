#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PACKAGE_NAMESPACE=${PACKAGE_NAMESPACE:-gpustack}
PACKAGE_REPOSITORY=${PACKAGE_REPOSITORY:-gpustack}
PACKAGE_ARCH=${PACKAGE_ARCH:-$(uname -m)}
PACKAGE_TAG=${PACKAGE_TAG:-dev}
PACKAGE_PUSH=${PACKAGE_PUSH:-false}
PACKAGE_PULL=${PACKAGE_PULL:-true}
PACKAGE_NETWORK=${PACKAGE_NETWORK:-default}

case "$PACKAGE_ARCH" in
    aarch64|arm64) PACKAGE_ARCH=arm64 ;;
    x86_64|amd64) PACKAGE_ARCH=amd64 ;;
    *) printf 'Unsupported PACKAGE_ARCH: %s; use amd64 or arm64.\n' "$PACKAGE_ARCH" >&2; exit 1 ;;
esac
for package_boolean in "$PACKAGE_PUSH" "$PACKAGE_PULL"; do
    if [[ "$package_boolean" != true && "$package_boolean" != false ]]; then
        printf 'PACKAGE_PUSH and PACKAGE_PULL must be true or false.\n' >&2
        exit 1
    fi
done
case "$PACKAGE_NETWORK" in
    default|host|none) ;;
    *) printf 'Unsupported PACKAGE_NETWORK: %s; use default, host or none.\n' "$PACKAGE_NETWORK" >&2; exit 1 ;;
esac

command -v docker >/dev/null || { printf 'Docker is required for make package.\n' >&2; exit 1; }
docker buildx version >/dev/null

TAG="${PACKAGE_NAMESPACE}/${PACKAGE_REPOSITORY}:${PACKAGE_TAG}"
REVISION="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
LABELS=(
    "org.opencontainers.image.title=my-gpustack"
    "org.opencontainers.image.version=${PACKAGE_TAG}"
    "org.opencontainers.image.revision=${REVISION}"
    "org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
)
EXTRA_ARGS=(--network "$PACKAGE_NETWORK")
if [[ "$PACKAGE_NETWORK" == host ]]; then
    EXTRA_ARGS+=(--allow network.host)
fi
if [[ "$PACKAGE_PUSH" == true ]]; then
    EXTRA_ARGS+=(--push)
else
    EXTRA_ARGS+=(--load)
fi
if [[ "$PACKAGE_PULL" == true ]]; then
    EXTRA_ARGS+=(--pull)
fi
for package_label in "${LABELS[@]}"; do
    EXTRA_ARGS+=(--label "$package_label")
done
for package_arg in PYTHON_VERSION GPUSTACK_BASE_IMAGE GPUSTACK_OPERATOR_VERSION GPUSTACK_TOOLS_DOWNLOAD_BASE_URL; do
    package_value="${!package_arg:-}"
    if [[ -n "$package_value" ]]; then
        EXTRA_ARGS+=(--build-arg "${package_arg}=${package_value}")
    fi
done

printf "Building '%s' for linux/%s (network=%s)\n" "$TAG" "$PACKAGE_ARCH" "$PACKAGE_NETWORK"
docker buildx build \
    --platform "linux/${PACKAGE_ARCH}" \
    --target gpustack \
    --tag "$TAG" \
    --file "${ROOT_DIR}/pack/Dockerfile" \
    --progress plain \
    --build-arg "GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS=$(printf '%s;' "${LABELS[@]}")" \
    "${EXTRA_ARGS[@]}" \
    "$ROOT_DIR"
