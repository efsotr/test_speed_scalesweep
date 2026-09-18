"""NVFP4 quantization dispatch for the local editable ScaleSweep kernels."""

from __future__ import annotations

import torch
from vllm import _custom_ops as vllm_ops
from vllm._custom_ops import create_fp4_output_tensors


def nvfp4_quant(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    backend: str = "scalesweep_mse",
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a local ScaleSweep variant or vLLM's CUTLASS NVFP4 quantizer."""
    if input.ndim < 1:
        raise ValueError("input must have at least one dimension")
    input = input.reshape(1 if input.ndim == 1 else -1, input.shape[-1])
    if input.shape[-1] % 16:
        raise ValueError("the last input dimension must be a multiple of 16")

    if backend == "cutlass":
        return vllm_ops.scaled_fp4_quant(
            input,
            input_global_scale,
            is_sf_swizzled_layout,
            backend=backend,
            padded_n=padded_n,
        )

    try:
        from . import scalesweep_mse_nvfp4_utils as mse
        from . import scalesweep_mse_nvfp4_utils_wo_acc as mse_wo_acc
        from . import scalesweep_nvfp4_utils as sweep
        from . import scalesweep_nvfp4_utils_wo_acc as sweep_wo_acc
    except ImportError:  # Allows `uv run python bench_kernel/test_kernel_latency.py`.
        import scalesweep_mse_nvfp4_utils as mse
        import scalesweep_mse_nvfp4_utils_wo_acc as mse_wo_acc
        import scalesweep_nvfp4_utils as sweep
        import scalesweep_nvfp4_utils_wo_acc as sweep_wo_acc

    implementations = {
        "scalesweep": sweep.scalesweep_nvfp4_quant_impl,
        "scalesweep_wo_acc": sweep_wo_acc.scalesweep_nvfp4_quant_impl,
        "scalesweep128": sweep.scalesweep128_nvfp4_quant_impl,
        "scalesweep_mse": mse.scalesweep_mse_nvfp4_quant_impl,
        "scalesweep_mse_wo_acc": mse_wo_acc.scalesweep_mse_nvfp4_quant_impl,
        "scalesweep_mse128": mse.scalesweep_mse128_nvfp4_quant_impl,
    }
    try:
        implementation = implementations[backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend: {backend}") from exc
    return implementation(input, input_global_scale, is_sf_swizzled_layout, padded_n)


__all__ = ["create_fp4_output_tensors", "nvfp4_quant"]
