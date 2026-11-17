SHELL := /bin/bash
.DEFAULT_GOAL := build
.PHONY: build package help

build:
	@bash hack/build.sh

package:
	@bash hack/package.sh

help:
	@printf '%s\n' \
	  'make build    Build the Python wheel and source distribution.' \
	  'make package  Build the unified GPUStack Server/Worker image.' \
	  'Image options: PACKAGE_NAMESPACE, PACKAGE_REPOSITORY, PACKAGE_TAG,' \
	  '               PACKAGE_ARCH, PACKAGE_PUSH, PACKAGE_PULL' \
	  'Build options: GPUSTACK_OPERATOR_VERSION, GPUSTACK_BASE_IMAGE,' \
	  '               PYTHON_VERSION, GPUSTACK_TOOLS_DOWNLOAD_BASE_URL'
