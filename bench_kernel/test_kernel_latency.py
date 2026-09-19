"""Measure CUDA-graph latency of vLLM NVFP4 activation quantizers.

Run with the project environment, for example:

    CUDA_MODULE_LOADING=LAZY uv run python test_kernel_latency.py

``act_quant_ops_latency`` and ``act_quant_ops_quant_error`` map backend name
and batch size to mean latency in milliseconds and dequantized MSE respectively.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph
try:
    from .custom import nvfp4_quant
except ImportError:  # Allows `uv run python bench_kernel/test_kernel_latency.py`.
    from custom import nvfp4_quant
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
    convert_swizzled_to_linear,
)


ACT_QUANT_BACKENDS = (
    "vllm",
    "scalesweep_mse",
    "scalesweep_mse_reference",
)
DEFAULT_BACKENDS = (
    "vllm",
    "scalesweep_mse",
    "scalesweep_mse_reference",
)
BATCH_SIZES = (1,) + tuple(1 << exponent for exponent in range(3, 14))
HIDDEN_SIZE = 8192
RESULT_PATH = Path(f"test_kernel_latency_dim{HIDDEN_SIZE}.json")
SCALESWEEP_BACKENDS = frozenset(
    {
        "scalesweep_mse",
        "scalesweep_mse_reference",
    }
)
BACKEND_OPS = {
    "vllm": ("_C", "scaled_fp4_quant"),
    "scalesweep_mse": ("vllm", "scalesweep_mse_nvfp4_quant"),
    "scalesweep_mse_reference": (
        "vllm",
        "scalesweep_mse_reference_nvfp4_quant",
    ),
}

# Populated by ``run_benchmark``. Values are mean latency in milliseconds.
act_quant_ops_latency: dict[str, dict[int, float]] = {}
act_quant_ops_quant_error: dict[str, dict[int, float]] = {}


def global_scale_inv(activations: torch.Tensor, backend: str) -> torch.Tensor:
    """Return the activation global-scale inverse expected by this backend."""
    fp8_max = fp8_scale_max(backend)
    return (fp8_max * scalar_types.float4_e2m1f.max()) / activations.abs().max().to(
        torch.float32
    )


def fp8_scale_max(backend: str) -> float:
    """Return the FP8 range used to derive a backend's global-scale inverse."""
    return 256.0 if backend in SCALESWEEP_BACKENDS else 448.0


def ensure_backends_available(backends: Sequence[str]) -> dict[str, bool]:
    """Verify the vLLM CUTLASS fallback; ScaleSweep variants are local."""
    availability: dict[str, bool] = {}
    for backend in backends:
        if backend in SCALESWEEP_BACKENDS:
            availability[backend] = True
            continue
        namespace_name, op_name = BACKEND_OPS[backend]
        namespace = getattr(torch.ops, namespace_name)
        exists = hasattr(namespace, op_name) and hasattr(getattr(namespace, op_name), "out")
        availability[backend] = exists
        if not exists:
            raise RuntimeError(
                f"vLLM backend {backend!r} is unavailable: "
                f"torch.ops.{namespace_name}.{op_name}.out is not registered."
            )
    return availability


def quantization_error_mse(
    activations: torch.Tensor,
    quantized: torch.Tensor,
    block_scales: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
) -> float:
    """Dequantize an NVFP4 result and return its mean squared error (MSE)."""
    m, n = activations.shape
    fp4_values = break_fp4_bytes(quantized, torch.float32).reshape(m, n // 16, 16)
    scales = convert_swizzled_to_linear(block_scales, m, n, 16).to(torch.float32)
    dequantized = fp4_values * (scales / input_global_scale_inv).unsqueeze(-1)
    return float(
        (dequantized.reshape_as(activations) - activations.float()).square().mean().item()
    )


def measure_latency(
    backend: str,
    batch_size: int,
    hidden_size: int,
    rep: int,
) -> tuple[float, float]:
    """Return CUDA-graph latency (ms) and dequantized MSE for one shape."""
    activations = torch.randn(
        (batch_size, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    input_global_scale_inv = global_scale_inv(activations, backend)

    quantized, block_scales = nvfp4_quant(
        activations,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend=backend,
    )
    error_mse = quantization_error_mse(
        activations, quantized, block_scales, input_global_scale_inv
    )

    # This is the same public vLLM dispatch path used by NVFP4 linear kernels.
    # The swizzled scale-factor layout is the CUTLASS NVFP4 GEMM layout.
    latency_ms = float(
        do_bench_cudagraph(
            lambda: nvfp4_quant(
                activations,
                input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend=backend,
            ),
            rep=rep,
            return_mode="mean",
        )
    )
    return latency_ms, error_mse


def run_benchmark(
    backends: Sequence[str] = DEFAULT_BACKENDS,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    hidden_size: int = HIDDEN_SIZE,
    rep: int = 20,
) -> dict[str, object]:
    """Run all requested shapes and return latency, error, and op availability."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for this benchmark.")
    if hidden_size <= 0 or hidden_size % 16:
        raise ValueError("hidden_size must be a positive multiple of 16 for NVFP4.")
    if rep <= 0:
        raise ValueError("rep must be positive.")

    for backend in backends:
        if backend not in ACT_QUANT_BACKENDS:
            raise ValueError(
                f"Unsupported backend {backend!r}; choose from {ACT_QUANT_BACKENDS}."
            )

    backend_available = ensure_backends_available(backends)
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            "NVFP4 activation quantization requires a Blackwell-or-newer "
            f"GPU (SM100+); found {torch.cuda.get_device_name()} (SM{major}{minor})."
        )
    act_quant_ops_latency.clear()
    act_quant_ops_quant_error.clear()
    for backend in backends:
        backend_latency: dict[int, float] = {}
        backend_error: dict[int, float] = {}
        act_quant_ops_latency[backend] = backend_latency
        act_quant_ops_quant_error[backend] = backend_error
        for batch_size in batch_sizes:
            if batch_size <= 0:
                raise ValueError("batch sizes must be positive.")
            latency_ms, error_mse = measure_latency(
                backend, batch_size, hidden_size, rep
            )
            backend_latency[batch_size] = latency_ms
            backend_error[batch_size] = error_mse
            print(
                f"{backend:18s} batch={batch_size:4d}: "
                f"{latency_ms:.4f} ms, MSE={error_mse:.8g}"
            )

    return {
        "backend_available": backend_available,
        "fp8_scale_max": {backend: fp8_scale_max(backend) for backend in backends},
        "act_quant_ops_latency": act_quant_ops_latency,
        "act_quant_ops_quant_error_mse": act_quant_ops_quant_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=ACT_QUANT_BACKENDS,
        default=DEFAULT_BACKENDS,
        help="Activation-quantization backends to measure.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=BATCH_SIZES,
        help="Batch sizes to measure (default: 1, 8, 16, ..., 8192).",
    )
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument(
        "--rep",
        type=int,
        default=20,
        help="Target benchmark time in milliseconds per Triton measurement.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_PATH,
        help="Path for the JSON result file (default: test_kernel_latency.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        backends=args.backends,
        batch_sizes=args.batch_sizes,
        hidden_size=args.hidden_size,
        rep=args.rep,
    )
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
