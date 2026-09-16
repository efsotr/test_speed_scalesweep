#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_MODULE_LOADING=LAZY

cd "${SCRIPT_DIR}"
exec uv run python benchmark_nvfp4_tokens.py "$@"
