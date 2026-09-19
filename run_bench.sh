#!/usr/bin/env bash
# Run each benchmark with benchmark_nvfp4_tokens.py's default parameters.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_benchmark() {
  python benchmark_nvfp4_tokens.py "$@"
}

# Run the two quantized variants concurrently on separate GPUs.  Backend
# selection is passed to vLLM through --llm-kwargs-json; the dedicated
# activation argument below also labels the output and configures the local
# pre-Blackwell compatibility patch used by this benchmark.
# run_benchmark \
#   --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
#   --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"cutlass"}' 

run_benchmark \
  --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
  --act-quant-backend scalesweep_mse \
  --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"scalesweep_mse"}' 

run_benchmark \
  --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
  --act-quant-backend scalesweep_mse128 \
  --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"scalesweep_mse128"}' 

run_benchmark \
  --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
  --act-quant-backend scalesweep \
  --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"scalesweep"}' 

run_benchmark \
  --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
  --act-quant-backend scalesweep128 \
  --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"scalesweep128"}' 
