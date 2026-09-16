#!/usr/bin/env python3
"""Offline vLLM throughput benchmark with vLLM-style random inputs.

For NVFP4 models on pre-Blackwell GPUs, the runtime patch keeps the CUTLASS
NVFP4 linear path but replaces its unsupported FP4 GEMM with a dequantize plus
torch.matmul fallback. Non-NVFP4 models run without this patch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Keep runtime monkey patches visible to the V1 engine and forked TP workers.
# These must be set before importing vllm.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

class LogStream:
    """Write a text stream to a log file without echoing to the terminal."""

    def __init__(self, path: Path) -> None:
        self.log = path.open("w", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self.log.write(data)
        return len(data)

    def flush(self) -> None:
        self.log.flush()

    def close(self) -> None:
        self.log.close()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self.log.fileno()


def safe_name(value: str) -> str:
    name = Path(value.rstrip("/")).name or "model"
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def model_is_nvfp4(model: str, trust_remote_code: bool) -> bool:
    """Detect NVFP4 from the model name or its Transformers configuration."""
    if "nvfp4" in model.lower():
        return True

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    quantization_config = getattr(config, "quantization_config", None)
    return "nvfp4" in json.dumps(quantization_config, default=str).lower()


def install_pre_blackwell_nvfp4_patch(act_quant_backend: str) -> None:
    """Force CUTLASS linear selection and emulate its unsupported FP4 GEMM."""
    import torch

    import vllm.model_executor.kernels.linear as linear_mod
    import vllm.model_executor.kernels.linear.nvfp4.cutlass as cutlass_mod
    from vllm.model_executor.kernels.linear.nvfp4.base import (
        NvFp4LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
        CutlassNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )

    def fallback_cutlass_scaled_fp4_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Numerical fallback for vllm._custom_ops.cutlass_scaled_fp4_mm.

        CUTLASS applies ``alpha`` after multiplying FP4 values and their FP8
        block scales. Dequantizing both operands with global scale 1 and then
        multiplying the matmul result by alpha is algebraically equivalent.
        Both scale tensors are in CUTLASS's swizzled layout here.
        """
        one = torch.ones((), dtype=torch.float32, device=a.device)
        a_dq = dequantize_to_dtype(
            a, block_scale_a, one, out_dtype, block_size=16, swizzle=True
        )
        b_dq = dequantize_to_dtype(
            b, block_scale_b, one, out_dtype, block_size=16, swizzle=True
        )
        return torch.matmul(a_dq, b_dq.t()) * alpha.to(dtype=out_dtype)

    def forced_init_nvfp4_linear_kernel(
        use_a16: bool = False,
        act_quant_backend: str | None = None,
    ) -> Any:
        if use_a16:
            raise RuntimeError(
                "This runtime patch is for W4A4 NVFP4 only; "
                "a CUTLASS W4A4 kernel cannot serve W4A16_NVFP4."
            )
        backend = act_quant_backend or act_quant_backend_from_cli
        config = NvFp4LinearLayerConfig(act_quant_backend=backend)
        return CutlassNvFp4LinearKernel(config)

    act_quant_backend_from_cli = act_quant_backend

    # cutlass.py imported this symbol directly, so patch its module-local name.
    cutlass_mod.cutlass_scaled_fp4_mm = fallback_cutlass_scaled_fp4_mm

    # Quantization schemes also imported init_nvfp4_linear_kernel directly.
    # Patch every alias before LLM/model construction.
    linear_mod.init_nvfp4_linear_kernel = forced_init_nvfp4_linear_kernel

    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4,
    )
    from vllm.model_executor.layers.quantization.quark.schemes import quark_nvfp4

    modelopt.init_nvfp4_linear_kernel = forced_init_nvfp4_linear_kernel
    compressed_tensors_w4a4_nvfp4.init_nvfp4_linear_kernel = (
        forced_init_nvfp4_linear_kernel
    )
    quark_nvfp4.init_nvfp4_linear_kernel = forced_init_nvfp4_linear_kernel


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def range_ratio(value: str) -> float | dict[str, float]:
    """Parse vLLM's --random-range-ratio syntax."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "must be a float or a JSON object with input/output keys"
        ) from exc
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        return float(parsed)
    if isinstance(parsed, dict) and set(parsed) == {"input", "output"}:
        try:
            return {"input": float(parsed["input"]), "output": float(parsed["output"])}
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "input and output range ratios must be numbers"
            ) from exc
    raise argparse.ArgumentTypeError(
        'must be a float or JSON such as \'{"input": 0.3, "output": 0.5}\''
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog="""Example:
  python benchmark_nvfp4_tokens.py \\
    --model /path/to/NVFP4-model \\
    --num-requests 256 --input-len 2048 --output-len 256
""",
    )
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument(
        "--num-requests", "--num-prompts", type=positive_int, default=256
    )
    parser.add_argument(
        "--random-input-len", "--input-len", type=positive_int, default=2048
    )
    parser.add_argument(
        "--random-output-len", "--output-len", type=positive_int, default=256
    )
    parser.add_argument(
        "--random-range-ratio",
        type=range_ratio,
        default=0.0,
        help=(
            "Uniform input/output length variation around their configured "
            "means. Accepts one float or JSON with input/output keys."
        ),
    )
    parser.add_argument(
        "--random-prefix-len",
        type=nonnegative_int,
        default=0,
        help="Number of shared random prefix tokens per request",
    )
    parser.add_argument("--tensor-parallel-size", type=positive_int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--max-model-len",
        type=positive_int,
        default=None,
        help="Defaults to the largest sampled prompt + output length",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup-requests",
        type=nonnegative_int,
        default=1,
        help="Warm-up requests excluded from timing; 0 disables warm-up",
    )
    parser.add_argument(
        "--act-quant-backend",
        default="cutlass",
        help="NVFP4 activation quantizer (ignored for non-NVFP4 models)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root directory for logs and JSON metrics (default: output)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use eager execution (default: true, safest for the fallback)",
    )
    parser.add_argument(
        "--llm-kwargs-json",
        default="{}",
        help='Extra keyword arguments for vllm.LLM, e.g. \'{"swap_space": 8}\'',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    is_nvfp4 = model_is_nvfp4(args.model, args.trust_remote_code)
    model_label = safe_name(args.model)
    if is_nvfp4:
        model_label = f"{model_label}-{safe_name(args.act_quant_backend)}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = args.output_dir / model_label / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    output_log = LogStream(run_dir / "output.log")
    sys.stdout = sys.stderr = output_log
    print(f"Run output: {run_dir.resolve()}")

    try:
        llm_extra = json.loads(args.llm_kwargs_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --llm-kwargs-json: {exc}") from exc
    if not isinstance(llm_extra, dict):
        raise SystemExit("--llm-kwargs-json must decode to a JSON object")

    imported_vllm = Path(__import__("vllm").__file__).resolve()
    print(f"vLLM import: {imported_vllm}")

    if is_nvfp4:
        install_pre_blackwell_nvfp4_patch(args.act_quant_backend)
        print(
            "Runtime patch installed: NVFP4 linear=CUTLASS, "
            "CUTLASS FP4 GEMM=dequantize+torch.matmul fallback, "
            f"activation backend={args.act_quant_backend}"
        )
    else:
        print("Non-NVFP4 model detected; runtime NVFP4 patch is disabled")

    import torch
    from vllm import LLM, SamplingParams
    from vllm.benchmarks.datasets import RandomDataset
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    def sample_requests(num_requests: int, seed: int) -> list[Any]:
        return RandomDataset(random_seed=seed).sample(
            tokenizer=tokenizer,
            num_requests=num_requests,
            prefix_len=args.random_prefix_len,
            range_ratio=args.random_range_ratio,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
        )

    requests = sample_requests(args.num_requests, args.seed)
    warmup_requests = sample_requests(args.warmup_requests, args.seed + 1)
    all_requests = requests + warmup_requests
    required_model_len = max(
        request.prompt_len + request.expected_output_len for request in all_requests
    )
    max_model_len = args.max_model_len or required_model_len
    if max_model_len < required_model_len:
        raise SystemExit(
            f"--max-model-len ({max_model_len}) must be at least the largest "
            f"sampled prompt + output length ({required_model_len})"
        )

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    overlap = sorted(llm_kwargs.keys() & llm_extra.keys())
    if overlap:
        raise SystemExit(
            "Do not repeat dedicated arguments in --llm-kwargs-json: "
            + ", ".join(overlap)
        )
    llm_kwargs.update(llm_extra)

    print(f"Loading model: {args.model}")
    llm = LLM(**llm_kwargs)

    def prepare_batch(batch: list[Any]) -> tuple[list[str], list[SamplingParams]]:
        prompts = [request.prompt for request in batch]
        sampling_params = [
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
            )
            for request in batch
        ]
        return prompts, sampling_params

    if warmup_requests:
        print(f"Warming up with {args.warmup_requests} request(s)...")
        warmup_prompts, warmup_sampling = prepare_batch(warmup_requests)
        llm.generate(
            warmup_prompts,
            warmup_sampling,
            use_tqdm=False,
        )

    prompts, sampling_params = prepare_batch(requests)
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    actual_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    actual_output_tokens = sum(
        len(completion.token_ids)
        for output in outputs
        for completion in output.outputs
    )
    expected_input_tokens = sum(request.prompt_len for request in requests)
    expected_output_tokens = sum(request.expected_output_len for request in requests)
    total_tokens = actual_input_tokens + actual_output_tokens
    request_throughput = len(requests) / elapsed
    total_token_throughput = total_tokens / elapsed
    output_token_throughput = actual_output_tokens / elapsed

    print(
        f"Throughput: {request_throughput:.2f} requests/s, "
        f"{total_token_throughput:.2f} total tokens/s, "
        f"{output_token_throughput:.2f} output tokens/s"
    )
    print(f"Total num prompt tokens:  {actual_input_tokens}")
    print(f"Total num output tokens:  {actual_output_tokens}")

    if (
        actual_input_tokens != expected_input_tokens
        or actual_output_tokens != expected_output_tokens
    ):
        print(
            "WARNING: actual token counts differ from requested counts: "
            f"input={actual_input_tokens}/{expected_input_tokens}, "
            f"output={actual_output_tokens}/{expected_output_tokens}",
            file=sys.stderr,
        )

    metrics = {
        "model": args.model,
        "model_name": model_label,
        "is_nvfp4": is_nvfp4,
        "act_quant_backend": args.act_quant_backend if is_nvfp4 else None,
        "elapsed_time": elapsed,
        "num_requests": len(requests),
        "total_num_tokens": total_tokens,
        "total_num_prompt_tokens": actual_input_tokens,
        "total_num_output_tokens": actual_output_tokens,
        "expected_num_prompt_tokens": expected_input_tokens,
        "expected_num_output_tokens": expected_output_tokens,
        "request_throughput": request_throughput,
        "total_token_throughput": total_token_throughput,
        "output_token_throughput": output_token_throughput,
        "random_input_len": args.random_input_len,
        "random_output_len": args.random_output_len,
        "random_range_ratio": args.random_range_ratio,
        "random_prefix_len": args.random_prefix_len,
        "warmup_requests": args.warmup_requests,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": max_model_len,
        "dtype": args.dtype,
        "seed": args.seed,
        "timestamp_utc": timestamp,
    }
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Metrics JSON: {metrics_path.resolve()}")

    sys.stdout, sys.stderr = original_stdout, original_stderr
    output_log.close()


if __name__ == "__main__":
    main()
