#!/usr/bin/env bash
set -euo pipefail

# Server/Worker selection belongs to the existing start command.
exec gpustack start "$@"
