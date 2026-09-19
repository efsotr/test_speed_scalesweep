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

export SCALESWEEP_TUNE=0

run_benchmark \
  --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
  --act-quant-backend cutlass \
  --no-enforce-eager \
  --output-dir output1 \
  --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"cutlass","compilation_config":{"mode":3,"backend":"eager","cudagraph_mode":"PIECEWISE","pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false}}}'

run_benchmark \
  --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
  --act-quant-backend scalesweep_mse \
  --no-enforce-eager \
  --output-dir output1 \
  --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"scalesweep_mse","compilation_config":{"mode":3,"backend":"eager","cudagraph_mode":"PIECEWISE","pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false}}}'

# run_benchmark \
#   --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
#   --act-quant-backend scalesweep \
#   --no-enforce-eager \
#   --output-dir output_tune \
#   --llm-kwargs-json '{"max_num_seqs": 32, "linear_backend":"cutlass","act_quant_backend":"scalesweep","compilation_config":{"mode":3,"backend":"eager","cudagraph_mode":"PIECEWISE"}}'

# run_benchmark \
#   --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
#   --act-quant-backend scalesweep_mse128 \
#   --no-enforce-eager \
#   --output-dir output_tune \
#   --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"scalesweep_mse128","compilation_config":{"mode":3,"backend":"eager","cudagraph_mode":"PIECEWISE"}}'

# run_benchmark \
#   --model ../Llama-3.1-8B-Instruct-NVFP4-RTN \
#   --act-quant-backend scalesweep128 \
#   --no-enforce-eager \
#   --output-dir output_tune \
#   --llm-kwargs-json '{"linear_backend":"cutlass","act_quant_backend":"scalesweep128","compilation_config":{"mode":3,"backend":"eager","cudagraph_mode":"PIECEWISE"}}'
